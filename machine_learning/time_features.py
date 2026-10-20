"""WITS-inspired received-wave intensity and timing, with explicit timing reference.

The paper describes feature types but not an exact detector. The envelope,
smoothing, gates and peak thresholds here are our reproducible implementation.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, find_peaks


def signal_envelope(signal, smoothing_samples):
    x = np.asarray(signal, dtype=float)
    centered = x-x.mean()
    envelope = uniform_filter1d(np.abs(hilbert(centered)), size=smoothing_samples, mode="nearest")
    return centered, envelope


def time_features(signal, fs, cfg):
    """Window-relative descriptors. Centroid is NOT a propagation delay."""
    x, envelope = signal_envelope(signal, cfg["envelope_smoothing_samples"])
    power = x*x
    total = power.sum()
    result = dict(time_energy_V2_s=total/fs, time_peak_envelope_V=float(envelope.max()),
                  time_energy_centroid_fraction=float(np.linspace(0, 1, len(x)) @ power/total) if total > 0 else 0.)
    for i, segment in enumerate(np.array_split(x, cfg["time_segments"]), 1):
        result[f"time_segment{i}_rms_V"] = float(np.sqrt(np.mean(segment*segment)))
    return result


def echo_features(signal, sample_times, trigger_time, cfg):
    """Earliest prominent packet peaks in a measured-trigger-relative gate.

    Missing peaks have zero values AND a presence flag; zero is not asserted
    to be an observed arrival. Multiple cycles/sources are not disentangled.
    """
    times = np.asarray(sample_times, dtype=float)
    relative = times-float(trigger_time)
    dt = float(np.median(np.diff(times)))
    lo, hi = cfg["echo_gate_s"]
    if relative[0] > lo+dt/2 or relative[-1] < hi-dt/2:
        raise ValueError("incomplete_echo_gate")
    _, envelope = signal_envelope(signal, cfg["envelope_smoothing_samples"])
    gate = np.flatnonzero((relative >= lo-dt*1e-6) & (relative <= hi+dt*1e-6))
    if len(gate) < 3:
        raise ValueError("echo_gate_has_fewer_than_three_samples")
    values = envelope[gate]
    peaks, _ = find_peaks(values, height=cfg["echo_min_amplitude_V"],
                         prominence=max(cfg["echo_min_amplitude_V"], cfg["echo_prominence_fraction"]*values.max()),
                         distance=max(1, int(np.ceil(cfg["echo_min_peak_distance_s"]/dt))))
    result = {}
    for i in range(cfg["echo_peak_count"]):
        exists = i < len(peaks)
        index = gate[peaks[i]] if exists else None
        result[f"echo_peak{i+1}_present"] = float(exists)
        result[f"echo_peak{i+1}_time_s"] = float(relative[index]) if exists else 0.
        result[f"echo_peak{i+1}_amplitude_V"] = float(envelope[index]) if exists else 0.
    return result
