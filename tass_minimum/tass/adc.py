"""Configurable observation model from simulated sensor volts to ADC codes."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass(frozen=True)
class ADCConfig:
    enabled: bool = False
    resolution_bits: int | None = None
    attenuation_db: float | None = None  # Recorded metadata; not a voltage conversion.
    voltage_to_code_gain_counts_per_V: float | None = None
    zero_voltage_code: float | None = None
    analog_gain_V_per_V: float = 1.0
    analog_offset_V: float = 0.0
    voltage_noise_std_V: float = 0.0
    code_noise_std_counts: float = 0.0
    highpass_cutoff_Hz: float | None = None
    lowpass_cutoff_Hz: float | None = None
    parameter_source: str = "unconfigured"  # firmware, measured, calibrated, or assumed
    notes: str = ""
    calibration_points: int | None = None
    calibration_rmse_counts: float | None = None
    calibration_max_abs_error_counts: float | None = None

    def validate(self, sample_rate_hz):
        if not self.enabled:
            return self
        if type(self.resolution_bits) is not int or not 1 <= self.resolution_bits <= 24:
            raise ValueError("Enabled ADC model requires explicit resolution_bits in [1,24]")
        if self.voltage_to_code_gain_counts_per_V is None or self.voltage_to_code_gain_counts_per_V <= 0:
            raise ValueError("Enabled ADC model requires a positive measured/configured voltage_to_code_gain_counts_per_V")
        if self.zero_voltage_code is None or not np.isfinite(self.zero_voltage_code):
            raise ValueError("Enabled ADC model requires an explicit zero_voltage_code")
        if self.attenuation_db is not None and not np.isfinite(self.attenuation_db):
            raise ValueError("attenuation_db must be finite or null")
        for name in ("analog_gain_V_per_V", "voltage_noise_std_V", "code_noise_std_counts"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.analog_gain_V_per_V == 0 or not np.isfinite(self.analog_offset_V):
            raise ValueError("Analog gain must be nonzero and offset finite")
        nyquist = sample_rate_hz/2
        for name in ("highpass_cutoff_Hz", "lowpass_cutoff_Hz"):
            value = getattr(self, name)
            if value is not None and not 0 < value < nyquist:
                raise ValueError(f"{name} must lie between zero and Nyquist")
        if (self.highpass_cutoff_Hz is not None and self.lowpass_cutoff_Hz is not None
                and self.highpass_cutoff_Hz >= self.lowpass_cutoff_Hz):
            raise ValueError("ADC high-pass cutoff must be below low-pass cutoff")
        if self.parameter_source not in {"firmware", "measured", "calibrated", "assumed"}:
            raise ValueError("Enabled ADC parameter_source must be firmware, measured, calibrated, or assumed")
        return self

    @property
    def max_code(self):
        if self.resolution_bits is None:
            raise ValueError("ADC resolution is not configured")
        return 2**self.resolution_bits-1


def observe_adc(voltage_v, config: ADCConfig, sample_rate_hz, rng):
    """Apply conditioning, noise, affine ADC mapping, quantization and clipping."""
    config.validate(sample_rate_hz)
    signal = np.asarray(voltage_v, float) * config.analog_gain_V_per_V
    if config.highpass_cutoff_Hz is not None:
        signal = sosfilt(butter(1, config.highpass_cutoff_Hz, btype="highpass",
                                fs=sample_rate_hz, output="sos"), signal, axis=0)
    if config.lowpass_cutoff_Hz is not None:
        signal = sosfilt(butter(1, config.lowpass_cutoff_Hz, btype="lowpass",
                                fs=sample_rate_hz, output="sos"), signal, axis=0)
    signal = signal + config.analog_offset_V
    signal += rng.normal(0, config.voltage_noise_std_V, signal.shape)
    codes = config.zero_voltage_code + config.voltage_to_code_gain_counts_per_V*signal
    codes += rng.normal(0, config.code_noise_std_counts, signal.shape)
    return np.clip(np.rint(codes), 0, config.max_code).astype(np.int64)


def fit_affine_calibration(path):
    """Fit raw_count = offset + gain*known_ADC_input_voltage from paired data."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    required = {"adc_input_voltage_v", "raw_adc"}
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError("Calibration CSV requires adc_input_voltage_v,raw_adc")
    voltage = np.atleast_1d(data["adc_input_voltage_v"])
    counts = np.atleast_1d(data["raw_adc"])
    valid = np.isfinite(voltage) & np.isfinite(counts)
    voltage, counts = voltage[valid], counts[valid]
    if len(voltage) < 2 or np.ptp(voltage) <= 0:
        raise ValueError("Calibration requires at least two finite, distinct input voltages")
    gain, offset = np.polyfit(voltage, counts, 1)
    if gain <= 0:
        raise ValueError("Fitted ADC gain is not positive; check voltage/count pairing")
    predicted = offset + gain*voltage
    residual = counts-predicted
    return {"voltage_to_code_gain_counts_per_V": float(gain),
            "zero_voltage_code": float(offset),
            "calibration_points": int(len(voltage)),
            "calibration_rmse_counts": float(np.sqrt(np.mean(residual**2))),
            "calibration_max_abs_error_counts": float(np.max(abs(residual))),
            "parameter_source": "calibrated"}


def main():
    parser = argparse.ArgumentParser(description="Fit affine ESP32 ADC voltage-to-count calibration")
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with adc_input_voltage_v,raw_adc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-bits", type=int, required=True)
    parser.add_argument("--attenuation-db", type=float)
    args = parser.parse_args()
    result = {"enabled": True, "resolution_bits": args.resolution_bits,
              "attenuation_db": args.attenuation_db, **fit_affine_calibration(args.input)}
    ADCConfig(**{key: value for key, value in result.items()
                 if key in ADCConfig.__dataclass_fields__}).validate(1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
