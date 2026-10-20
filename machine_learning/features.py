"""One supervised row per stationary, uniformly sampled FFT window."""
import numpy as np
import pandas as pd
from scipy.signal.windows import hann

from .data import IDS, RX, TX, STATES, TARGETS
from .time_features import time_features, echo_features


def spectrum_features(signal, fs, bands):
    """Mean-removed Hann FFT, coherent-gain-corrected one-sided amplitudes."""
    x = np.asarray(signal, dtype=float)
    centered = x-x.mean()
    w = hann(len(x), sym=False)
    freq = np.fft.rfftfreq(len(x), 1/fs)
    amplitude = np.abs(np.fft.rfft(centered*w))/w.sum()
    amplitude[1:-1 if len(x) % 2 == 0 else None] *= 2
    # Periodogram power for fractions/centroid, with one-sided energy weighting.
    power = np.abs(np.fft.rfft(centered*w))**2
    power[1:-1 if len(x) % 2 == 0 else None] *= 2
    power[0] = 0
    total = power.sum()
    peak = np.argmax(power[1:])+1
    result = {"rms_V": np.sqrt(np.mean(centered**2)),
              "peak_amplitude_V": amplitude[peak],
              "peak_frequency_Hz": freq[peak] if total > 0 else 0.,
              "centroid_Hz": (freq @ power)/total if total > 0 else 0.}
    for i, (lo, hi) in enumerate(bands, 1):
        mask = (freq >= lo) & (freq < hi)
        result[f"band{i}_fraction"] = power[mask].sum()/total if total > 0 else 0.
    return result


def extract_features(samples, cfg):
    rows, audit = [], []
    n, hop = cfg["window_samples"], cfg["hop_samples"]
    for trial, frame in samples.groupby("trial_id", sort=False):
        frame = frame.reset_index(drop=True)
        times = frame.time_s.to_numpy()
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"{trial}: timestamps must be strictly increasing in acquisition order")
        dt = np.median(np.diff(times))
        fs = 1/dt
        if cfg["echo_features_enabled"]:
            if "excitation_trigger_time_s" not in frame or not np.isfinite(frame.excitation_trigger_time_s).all():
                raise ValueError(f"{trial}: measured excitation_trigger_time_s required for echo features")
            if frame.excitation_trigger_time_s.nunique() != 1:
                raise ValueError(f"{trial}: use one trial per triggered capture; trigger must be constant")
        if max(hi for _, hi in cfg["frequency_bands_Hz"]) > fs/2*(1+1e-8):
            raise ValueError(f"{trial}: frequency bands exceed Nyquist ({fs/2:.3f} Hz)")
        if len(frame) < n:
            audit.append(dict(trial_id=trial, start_sample=0, status="rejected", reason="short_recording"))
        for start in range(0, len(frame)-n+1, hop):
            window = frame.iloc[start:start+n]
            reasons = []
            if np.max(abs(np.diff(window.time_s)-dt)) > cfg["timestamp_relative_tolerance"]*dt:
                reasons.append("timestamp_gap_or_jitter")
            if not np.isfinite(window[RX+TX].to_numpy()).all():
                reasons.append("missing_waveform")
            if not window.quality_flag.isin(["ok", "unassessed"]).all():
                reasons.append("quality_flag")
            if cfg["echo_features_enabled"]:
                relative = window.time_s.to_numpy()-frame.excitation_trigger_time_s.iloc[0]
                lo, hi = cfg["echo_gate_s"]
                if relative[0] > lo+dt/2 or relative[-1] < hi-dt/2:
                    reasons.append("incomplete_echo_gate")
            for name, limit in [("contact_force_N", "max_force_range_N"),
                                ("contact_location_m", "max_location_range_m"),
                                ("excitation_frequency_Hz", "max_frequency_range_Hz")]:
                if cfg[limit] is not None and window[name].max()-window[name].min() > cfg[limit]:
                    reasons.append(f"nonstationary_{name}")
            audit.append(dict(trial_id=trial, start_sample=start,
                              status="rejected" if reasons else "accepted", reason=";".join(reasons)))
            if reasons:
                continue
            row = {name: window[name].iloc[0] for name in IDS}
            row.update(window_id=f"{trial}:{start}", time_start_s=window.time_s.iloc[0],
                       time_end_s=window.time_s.iloc[-1], sample_rate_Hz=fs,
                       fft_resolution_Hz=fs/n, samples=n)
            # Labels/states must already be aligned with a documented validity interval.
            # Do not invent missing labels or average only the available subset of a window.
            for name in STATES+TARGETS:
                row[name] = window[name].mean() if window[name].notna().all() else np.nan
            for channel, prefix in zip(RX+TX, [f"rx{i}" for i in range(1, 5)]+[f"tx{i}" for i in range(1, 5)]):
                row.update({f"{prefix}_{key}": value for key, value in
                            spectrum_features(window[channel], fs, cfg["frequency_bands_Hz"]).items()})
                if channel in RX and cfg["time_features_enabled"]:
                    row.update({f"{prefix}_{key}": value for key, value in time_features(window[channel], fs, cfg).items()})
                if channel in RX and cfg["echo_features_enabled"]:
                    row.update({f"{prefix}_{key}": value for key, value in echo_features(
                        window[channel], window.time_s, frame.excitation_trigger_time_s.iloc[0], cfg).items()})
            rows.append(row)
        trailing = len(frame)-((len(frame)-n)//hop*hop+n) if len(frame) >= n else 0
        if trailing:
            audit.append(dict(trial_id=trial, start_sample=len(frame)-trailing,
                              status="ignored", reason=f"incomplete_tail_{trailing}_samples"))
    if not rows:
        reasons = pd.Series([entry["reason"] for entry in audit]).value_counts().to_dict()
        raise ValueError(f"No valid FFT windows; rejection reasons: {reasons}")
    return pd.DataFrame(rows), pd.DataFrame(audit)
