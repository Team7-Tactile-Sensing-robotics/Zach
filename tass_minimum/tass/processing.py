"""FFT features usable unchanged with simulated or measured piezo channels."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from .data import Measurement


@dataclass
class FrequencyResult:
    frequency_Hz: np.ndarray
    amplitude_V: np.ndarray
    excitation_amplitude_N: np.ndarray
    frf_V_N: np.ndarray
    frf_valid: np.ndarray
    features: dict


def extract_frequency_features(measurement: Measurement, *, start_s=None, end_s=None,
                               window="hann", max_peaks=5, frf_floor_relative=1e-3):
    """Single-record Y/U is approximate; only excited bins are reported.

    Choose a stationary interval for resonance interpretation. Pulse ringdown is
    suitable for FFT peak checks. FRF requires an input/output interval including
    the excitation and response, and preferably negligible response at the edges.
    """
    if window not in {"hann", "boxcar"}:
        raise ValueError("window must be hann or boxcar")
    if max_peaks < 1 or not 0 < frf_floor_relative < 1:
        raise ValueError("Invalid peak count or FRF threshold")
    t = measurement.time_s
    mask = np.ones(len(t), dtype=bool)
    if start_s is not None:
        mask &= t >= start_s
    if end_s is not None:
        mask &= t < end_s
    y = measurement.piezo_V[mask].copy()
    u = measurement.excitation_N[mask].copy()
    n = len(u)
    if n < 4:
        raise ValueError("Analysis interval requires at least four samples")
    fs = measurement.sample_rate_Hz
    y -= np.mean(y, axis=0)
    u -= np.mean(u)
    w = np.hanning(n) if window == "hann" else np.ones(n)
    Y = np.fft.rfft(y*w[:, None], axis=0)
    U = np.fft.rfft(u*w)
    f = np.fft.rfftfreq(n, 1/fs)
    one_sided = np.full(len(f), 2.0)
    one_sided[0] = 1.0
    if n % 2 == 0:
        one_sided[-1] = 1.0
    amplitude = np.abs(Y)*one_sided[:, None]/np.sum(w)
    input_amplitude = abs(U)*one_sided/np.sum(w)
    valid = (abs(U) > max(float(np.max(abs(U)))*frf_floor_relative, 1e-15))
    valid[0] = False
    frf = np.full(Y.shape, np.nan + 1j*np.nan, dtype=complex)
    frf[valid] = Y[valid]/U[valid, None]
    # Parseval energy of the unwindowed, demeaned record, in V^2 s.
    raw_fft = np.fft.rfft(y, axis=0)
    energy = np.sum(abs(raw_fft)**2 * one_sided[:, None], axis=0)/(n*fs)
    channels = []
    for i in range(y.shape[1]):
        spectrum = amplitude[:, i]
        peak_index = 1 + int(np.argmax(spectrum[1:]))
        peak_amp = float(spectrum[peak_index])
        indices, _ = find_peaks(spectrum, prominence=0.02*peak_amp)
        indices = indices[indices > 0]
        strongest = sorted(indices, key=lambda j: spectrum[j], reverse=True)[:max_peaks]
        ordered = sorted(strongest)
        channels.append({"channel": i+1,
                         "dominant_frequency_Hz": float(f[peak_index]) if peak_amp > 1e-15 else None,
                         "peak_amplitude_V": peak_amp,
                         "spectral_peak_frequencies_Hz": [float(f[j]) for j in ordered] if peak_amp > 1e-15 else [],
                         "spectral_peak_amplitudes_V": [float(spectrum[j]) for j in ordered] if peak_amp > 1e-15 else [],
                         "spectral_energy_V2_s": float(energy[i])})
    ratio = None
    if len(channels) >= 2 and channels[1]["peak_amplitude_V"] > 1e-15:
        ratio = channels[0]["peak_amplitude_V"] / channels[1]["peak_amplitude_V"]
    features = {"channels": channels, "channel_1_to_2_peak_amplitude_ratio": ratio,
                "frequency_resolution_Hz": fs/n, "sample_rate_Hz": fs,
                "start_s": float(t[mask][0]), "end_s": float(t[mask][-1]),
                "window": window, "frf_valid_bins": int(np.sum(valid)),
                "interpretation": "Spectral peaks are observations, not automatic modal identifications. "
                                  "Y/U is an approximate finite-record FRF only where excitation is nonzero."}
    return FrequencyResult(f, amplitude, input_amplitude, frf, valid, features)
