"""Stable CSV measurement interchange and lossless NPZ simulation outputs."""
import json
from pathlib import Path
import numpy as np
from .data import Measurement


def save_measurement(measurement, path):
    columns = {"time_s": measurement.time_s, **measurement.channels,
               "excitation_N": measurement.excitation_N}
    for i in range(measurement.piezo_V.shape[1]):
        columns[f"piezo_{i+1}_V"] = measurement.piezo_V[:, i]
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=",",
               header=",".join(columns), comments="", fmt="%.12g")


def load_measurement(path):
    """Read this schema after converting ADC counts, timestamps and load-cell units."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.ndim == 0:
        raise ValueError("CSV must contain multiple samples")
    names = data.dtype.names or ()
    piezo_names = sorted((x for x in names if x.startswith("piezo_") and x.endswith("_V")),
                         key=lambda x: int(x.split("_")[1]))
    if "time_s" not in names or "excitation_N" not in names or not piezo_names:
        raise ValueError("CSV requires time_s, excitation_N, and piezo_1_V (optionally more channels)")
    excluded = {"time_s", "excitation_N", *piezo_names}
    return Measurement(data["time_s"], np.column_stack([data[x] for x in piezo_names]),
                       data["excitation_N"], {x: data[x] for x in names if x not in excluded})


def save_frequency_result(frequency, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cols = {"frequency_Hz": frequency.frequency_Hz,
            "excitation_amplitude_N": frequency.excitation_amplitude_N,
            "frf_valid": frequency.frf_valid.astype(int)}
    for i in range(frequency.amplitude_V.shape[1]):
        cols[f"piezo_{i+1}_amplitude_V"] = frequency.amplitude_V[:, i]
        cols[f"frf_{i+1}_real_V_N"] = frequency.frf_V_N[:, i].real
        cols[f"frf_{i+1}_imag_V_N"] = frequency.frf_V_N[:, i].imag
    np.savetxt(directory/"spectrum.csv", np.column_stack(list(cols.values())),
               delimiter=",", header=",".join(cols), comments="", fmt="%.12g")
    np.savez_compressed(directory/"spectrum.npz", frequency_Hz=frequency.frequency_Hz,
                        amplitude_V=frequency.amplitude_V, frf_V_N=frequency.frf_V_N,
                        frf_valid=frequency.frf_valid,
                        excitation_amplitude_N=frequency.excitation_amplitude_N)
    (directory/"features.json").write_text(json.dumps(frequency.features, indent=2, allow_nan=False)+"\n")


def save_run(result, frequency, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_measurement(result.measurement, directory/"timeseries.csv")
    np.savez_compressed(directory/"string_state.npz", time_s=result.measurement.time_s,
                        grid_m=result.grid_m, displacement_m=result.displacement_m,
                        velocity_m_s=result.velocity_m_s)
    (directory/"config.json").write_text(json.dumps(result.config, indent=2)+"\n")
    import yaml
    (directory/"config.yaml").write_text(yaml.safe_dump(result.config, sort_keys=False))
    (directory/"metadata.json").write_text(json.dumps(result.metadata, indent=2)+"\n")
    save_frequency_result(frequency, directory)
