"""Small, leakage-safe ML pipeline shared by simulation and captures."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ("contact_force_N", "contact_location_m", "load_cell_readout_N")
FORBIDDEN_INPUT_PATTERNS = (
    "contact_", "load_cell", "tension", "servo_", "force_label", "location_label",
)


@dataclass(frozen=True)
class FeatureConfig:
    window_s: float = 1.0
    stride_s: float = 1.0
    sensor_representation: str = "voltage"  # Simulation volts directly; real ADC converted if present.
    spectral_mode: str = "global"  # band_peaks avoids relying on the identity of the globally largest mode.
    adc_volts_per_count: float | None = None
    adc_voltage_offset_v: float = 0.0
    remove_sensor_mean: bool = False
    use_actuator_features: bool = False
    actuator_ids: tuple[int, ...] = (1, 2)
    actuator_command_min: float = -1.0
    actuator_command_max: float = 1.0
    actuator_drive_peak_v: float = 3.3
    sensor_ids: tuple[int, ...] = (1,)
    use_reference_features: bool = False
    physics_enabled: bool = False
    physics_sensor_pair: tuple[int, int] = (1, 2)
    physics_string_length_m: float | None = None
    physics_linear_density_kg_m: float | None = None
    physics_mode_number: int = 1
    confidence_snr_low_db: float = 3.0
    confidence_snr_high_db: float = 20.0
    confidence_frequency_change_scale_hz: float = 20.0
    confidence_amplitude_change_ratio_scale: float = 0.25
    confidence_tension_disagreement_scale: float = 0.20
    bands_Hz: tuple[tuple[float, float], ...] = ((20.0, 200.0), (200.0, 800.0), (800.0, 2000.0))
    context_columns: tuple[str, ...] = ("sample_rate_hz",)
    target_columns: tuple[str, ...] = TARGETS

    def validate(self):
        if self.window_s <= 0 or self.stride_s <= 0:
            raise ValueError("window_s and stride_s must be positive")
        if self.sensor_representation not in {"auto", "voltage", "raw_adc", "adc_voltage"}:
            raise ValueError("sensor_representation must be auto, voltage, raw_adc, or adc_voltage")
        if self.spectral_mode not in {"global", "band_peaks"}:
            raise ValueError("spectral_mode must be global or band_peaks")
        if self.sensor_representation == "adc_voltage" and self.adc_volts_per_count is None:
            raise ValueError("adc_voltage requires a positive finite adc_volts_per_count")
        if self.adc_volts_per_count is not None and (not np.isfinite(self.adc_volts_per_count) or self.adc_volts_per_count <= 0):
            raise ValueError("adc_volts_per_count must be positive and finite")
        if not np.isfinite(self.adc_voltage_offset_v):
            raise ValueError("adc_voltage_offset_v must be finite")
        if self.use_actuator_features:
            if not self.actuator_ids or len(set(self.actuator_ids)) != len(self.actuator_ids):
                raise ValueError("actuator_ids must be nonempty and unique")
            if any(type(i) is not int or i < 1 for i in self.actuator_ids):
                raise ValueError("actuator_ids must be positive integers")
            if (not np.isfinite([self.actuator_command_min, self.actuator_command_max,
                                 self.actuator_drive_peak_v]).all()
                    or self.actuator_command_min >= self.actuator_command_max or self.actuator_drive_peak_v <= 0):
                raise ValueError("Actuator command range and peak voltage must be finite and positive in width/amplitude")
        if not self.sensor_ids or any(int(number) < 1 for number in self.sensor_ids):
            raise ValueError("sensor_ids must contain positive channel numbers")
        if self.physics_enabled:
            if not self.use_reference_features:
                raise ValueError("physics features require use_reference_features")
            if len(self.physics_sensor_pair) != 2 or len(set(self.physics_sensor_pair)) != 2:
                raise ValueError("physics_sensor_pair requires two distinct sensor IDs")
            if not set(self.physics_sensor_pair).issubset(set(self.sensor_ids)):
                raise ValueError("physics_sensor_pair must be included in sensor_ids")
            if self.physics_string_length_m is None or self.physics_string_length_m <= 0:
                raise ValueError("physics_string_length_m must be positive")
            if self.physics_linear_density_kg_m is None or self.physics_linear_density_kg_m <= 0:
                raise ValueError("physics_linear_density_kg_m must be positive")
            if type(self.physics_mode_number) is not int or self.physics_mode_number < 1:
                raise ValueError("physics_mode_number must be a positive integer")
            if not self.confidence_snr_high_db > self.confidence_snr_low_db:
                raise ValueError("confidence SNR high threshold must exceed low threshold")
            if min(self.confidence_frequency_change_scale_hz,
                   self.confidence_amplitude_change_ratio_scale,
                   self.confidence_tension_disagreement_scale) <= 0:
                raise ValueError("confidence scales must be positive")
        for name in self.context_columns:
            lowered = name.lower()
            if any(token in lowered for token in FORBIDDEN_INPUT_PATTERNS):
                raise ValueError(f"Forbidden input feature: {name}")
        unknown = set(self.target_columns) - set(TARGETS)
        if unknown:
            raise ValueError(f"Unsupported targets: {sorted(unknown)}")
        return self


@dataclass
class Segment:
    segment_id: str
    time_s: np.ndarray
    columns: dict[str, np.ndarray]
    split: str | None = None
    source: str = "real"


def _sensor_columns(columns, representation, sensor_ids):
    patterns = {
        "voltage": re.compile(r"^piezo_sensor_(\d+)_(?:voltage_v|v)$", re.I),
        "raw_adc": re.compile(r"^piezo_sensor_(\d+)_raw_adc$", re.I),
    }
    found = {kind: [] for kind in patterns}
    for name in columns:
        for kind, pattern in patterns.items():
            match = pattern.match(name)
            if match:
                found[kind].append((int(match.group(1)), name))
    kind = representation
    if kind == "auto":
        kind = "raw_adc" if found["raw_adc"] else "voltage"
    if kind == "adc_voltage":
        kind = "raw_adc"
    # Prefer the normalized voltage name, selecting each receiver only once.
    selected_by_id = {}
    for number, name in sorted(found[kind],key=lambda item: (not item[1].lower().endswith('_voltage_v'),item)):
        if number in sensor_ids:
            selected_by_id.setdefault(number,name)
    selected = sorted(selected_by_id.items())
    if set(number for number, _ in selected) != set(sensor_ids):
        raise ValueError(f"Missing requested {kind} sensor columns for IDs {sensor_ids}")
    return kind, selected


def _sensor_inputs(segment, config):
    """Resolve voltage at the source boundary without changing recorded arrays."""
    if config.sensor_representation != "voltage":
        kind, selected = _sensor_columns(segment.columns,config.sensor_representation,config.sensor_ids)
        return [(number,name,kind,config.sensor_representation == 'adc_voltage') for number,name in selected]
    inputs = []
    lower_names = {name.lower() for name in segment.columns}
    for number in config.sensor_ids:
        use_adc = (segment.source != 'simulation' and f'piezo_sensor_{number}_raw_adc' in lower_names)
        kind = 'raw_adc' if use_adc else 'voltage'
        _, selected = _sensor_columns(segment.columns,kind,(number,))
        if use_adc and config.adc_volts_per_count is None:
            raise ValueError('Real raw ADC input requires --adc-volts-per-count or --adc-full-scale-v calibration')
        inputs.append((number,selected[0][1],kind,use_adc))
    return inputs


def iter_simulation_dataset(root: str | Path):
    root = Path(root)
    with (root / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    for row in manifest:
        with np.load(root / row["file"], allow_pickle=False) as data:
            time = np.asarray(data["time_s"])
            columns = {}
            for name in data.files:
                if name == "time_s":
                    continue
                values = np.asarray(data[name])
                if values.ndim == 1 and len(values) == len(time):
                    columns[name] = values
        # Normalize simulator names to the real-capture convention at the boundary.
        columns.update({name.replace("_V", "_voltage_v"): value for name, value in list(columns.items())
                        if re.match(r"^piezo_sensor_\d+_V$", name)})
        columns.update({name.replace("_reference_V", "_reference_voltage_v"): value
                        for name, value in list(columns.items())
                        if re.match(r"^piezo_sensor_\d+_reference_V$", name)})
        columns["sample_rate_hz"] = np.full(len(time), 1.0 / np.median(np.diff(time)))
        yield Segment(row["trial_id"], time, columns,
                      split=row.get("split"), source="simulation")


def load_simulation_dataset(root: str | Path) -> list[Segment]:
    return list(iter_simulation_dataset(root))


def load_capture_csv(path: str | Path) -> list[Segment]:
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.dtype.names is None or ("time_s" not in data.dtype.names and
            not {"sample_index","measured_sample_rate_hz"}.issubset(data.dtype.names)):
        raise ValueError("Capture CSV needs time_s or sample_index plus measured_sample_rate_hz")
    if data.ndim == 0:
        data = data.reshape(1)
    names = data.dtype.names
    ids = data["segment_id"].astype(str) if "segment_id" in names else np.full(len(data), path.stem)
    segments = []
    for segment_id in np.unique(ids):
        mask = ids == segment_id
        clock_name = "time_s" if "time_s" in names else "sample_index"
        order = np.argsort(np.asarray(data[clock_name][mask], float))
        columns = {name: np.asarray(data[name][mask])[order] for name in names
                   if name not in {"segment_id", "time_s"}}
        if "time_s" in names:
            time = np.asarray(data["time_s"][mask], float)[order]
        else:
            rate = _constant(columns["measured_sample_rate_hz"],"measured_sample_rate_hz")
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("measured_sample_rate_hz must be positive and finite")
            indices = np.asarray(columns["sample_index"],float)
            time = (indices-indices[0])/rate
        segments.append(Segment(str(segment_id),time,columns,source="real"))
    return segments


def load_segments(path: str | Path, *, stream=False):
    path = Path(path)
    if path.is_dir():
        return iter_simulation_dataset(path) if stream else load_simulation_dataset(path)
    return load_capture_csv(path)


def _constant(values, name):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"{name} has no finite values")
    if not np.allclose(finite, finite[0], rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} must be constant within a window")
    return float(finite[0])


def _signal_features(values, sample_rate, bands, *, spectral_mode="global"):
    x = np.asarray(values, float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Sensor windows must contain only finite values")
    centered = x - np.mean(x)
    spectrum = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    amplitude = 2.0 * np.abs(spectrum) / len(x)
    amplitude[0] *= 0.5
    if len(x) % 2 == 0:
        amplitude[-1] *= 0.5
    psd = np.abs(spectrum) ** 2 / (sample_rate * len(x))
    if len(psd) > 2:
        psd[1:-1] *= 2.0
    usable = freq > 0
    result = {
        "mean": float(np.mean(x)), "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x*x))), "peak_to_peak": float(np.ptp(x)),
        "dominant_frequency_hz": float(freq[np.flatnonzero(usable)[np.argmax(psd[usable])]]) if np.any(usable) else 0.0,
        "peak_amplitude": float(np.max(amplitude[usable])) if np.any(usable) else 0.0,
        "spectral_snr_db": float(10*np.log10((np.max(psd[usable])+1e-30)/(np.median(psd[usable])+1e-30))) if np.any(usable) else 0.0,
    }
    for low, high in bands:
        mask = (freq >= low) & (freq < high)
        result[f"band_power_{low:g}_{high:g}_hz"] = float(np.sum(psd[mask]) * (sample_rate/len(x)))
        if spectral_mode == "band_peaks":
            indices = np.flatnonzero(mask & usable)
            result[f"band_peak_frequency_{low:g}_{high:g}_hz"] = (
                float(freq[indices[np.argmax(psd[indices])]]) if len(indices) else 0.)
            result[f"band_fraction_{low:g}_{high:g}_hz"] = float(
                np.sum(psd[mask & usable])/max(np.sum(psd[usable]),1e-30))
    return result


def _sensor_signal(values, config, *, convert_adc=False):
    signal = np.asarray(values, float)
    if convert_adc:
        signal = signal*config.adc_volts_per_count + config.adc_voltage_offset_v
    if config.remove_sensor_mean:
        signal = signal - np.mean(signal)
    return signal


def _command_voltage(values, config):
    command = np.asarray(values, float)
    if (not np.isfinite(command).all() or np.any(command < config.actuator_command_min)
            or np.any(command > config.actuator_command_max)):
        raise ValueError("Actuator commands must be finite and inside the configured command range")
    return (2*(command-config.actuator_command_min) /
            (config.actuator_command_max-config.actuator_command_min)-1)*config.actuator_drive_peak_v


def _physics_features(contacted, reference, config):
    """Solve the two-span string equations for one ordered sensor pair.

    The first sensor observes the span of length x and the second the span L-x.
    Both dominant peaks are assumed to represent the configured same mode n.
    """
    left, right = config.physics_sensor_pair
    fc_left = contacted[left]["dominant_frequency_hz"]
    fc_right = contacted[right]["dominant_frequency_hz"]
    f0_left = reference[left]["dominant_frequency_hz"]
    f0_right = reference[right]["dominant_frequency_hz"]
    if min(fc_left, fc_right, f0_left, f0_right) <= 0:
        raise ValueError("Physics features require positive dominant frequencies")
    length = config.physics_string_length_m
    density = config.physics_linear_density_kg_m
    mode = config.physics_mode_number
    x = length * fc_right / (fc_left + fc_right)
    contacted_left = density * (2*x*fc_left/mode)**2
    contacted_right = density * (2*(length-x)*fc_right/mode)**2
    baseline_left = density * (2*length*f0_left/mode)**2
    baseline_right = density * (2*length*f0_right/mode)**2
    contacted_tension = 0.5*(contacted_left+contacted_right)
    baseline_tension = 0.5*(baseline_left+baseline_right)
    # Contacted-side estimates are algebraically identical after x is solved
    # from the same two frequencies. Use the two independent reference-tension
    # estimates for a non-tautological sensor-consistency check.
    disagreement = abs(baseline_left-baseline_right) / max(baseline_tension, 1e-30)
    consistency = float(np.exp(-disagreement/config.confidence_tension_disagreement_scale))

    snr = min(contacted[left]["spectral_snr_db"], contacted[right]["spectral_snr_db"])
    snr_score = float(np.clip((snr-config.confidence_snr_low_db) /
                              (config.confidence_snr_high_db-config.confidence_snr_low_db), 0, 1))
    frequency_change = 0.5*(abs(fc_left-f0_left)+abs(fc_right-f0_right))
    frequency_score = float(1-np.exp(-frequency_change/config.confidence_frequency_change_scale_hz))
    amplitude_ratios = []
    for sensor in (left, right):
        amplitude_ratios.append(abs(contacted[sensor]["peak_amplitude"]-reference[sensor]["peak_amplitude"])
                                / max(abs(reference[sensor]["peak_amplitude"]), 1e-30))
    amplitude_score = float(1-np.exp(-np.mean(amplitude_ratios)/config.confidence_amplitude_change_ratio_scale))
    confidence = float(np.mean([snr_score, frequency_score, amplitude_score, consistency]))
    delta_tension = contacted_tension-baseline_tension
    result = {
        "x_physics_m": float(x),
        "baseline_tension_estimate_N": float(baseline_tension),
        "baseline_tension_sensor_left_N": float(baseline_left),
        "baseline_tension_sensor_right_N": float(baseline_right),
        "contact_tension_estimate_N": float(contacted_tension),
        "contact_tension_sensor_left_N": float(contacted_left),
        "contact_tension_sensor_right_N": float(contacted_right),
        "tendon_delta_tension_N": float(delta_tension),
        "tension_relative_disagreement": float(disagreement),
        "tension_consistency": consistency,
        "physics_confidence": confidence,
    }
    # Explicit interactions let even a linear baseline emphasize physics when
    # confidence is high and vibration deltas when confidence is low.
    result.update({
        "confidence_weighted_x_physics_m": float(confidence*x),
        "confidence_weighted_delta_tension_N": float(confidence*delta_tension),
        "low_confidence_frequency_change_hz": float((1-confidence)*frequency_change),
        "low_confidence_amplitude_change_ratio": float((1-confidence)*np.mean(amplitude_ratios)),
    })
    return result


def extract_features(segments: list[Segment], config: FeatureConfig):
    config.validate()
    rows, provenance = [], {}
    for segment in segments:
        t = np.asarray(segment.time_s, float)
        if len(t) < 4 or np.any(np.diff(t) <= 0):
            raise ValueError(f"Segment {segment.segment_id} needs increasing timestamps and >=4 samples")
        deltas = np.diff(t)
        if not np.allclose(deltas, np.median(deltas), rtol=1e-4, atol=1e-12):
            raise ValueError(f"Segment {segment.segment_id} is not uniformly sampled; resample it first")
        inferred_rate = 1.0 / np.median(deltas)
        for clock_column in ("sample_rate_hz","measured_sample_rate_hz"):
            if clock_column in segment.columns:
                reported = _constant(segment.columns[clock_column],clock_column)
                if not np.isclose(reported,inferred_rate,rtol=1e-4,atol=1e-8):
                    raise ValueError(f"{clock_column} disagrees with timestamps in {segment.segment_id}; use the measured capture clock")
        sensors = _sensor_inputs(segment, config)
        count = max(1, int(round(config.window_s * inferred_rate)))
        stride = max(1, int(round(config.stride_s * inferred_rate)))
        if len(t) < count:
            raise ValueError(f"Segment {segment.segment_id} is shorter than one configured window")
        for start in range(0, len(t) - count + 1, stride):
            stop = start + count
            row = {"source": segment.source, "segment_id": segment.segment_id,
                   "window_index": len([r for r in rows if r["segment_id"] == segment.segment_id]),
                   "window_start_s": float(t[start]), "window_end_s": float(t[stop-1]),
                   "split": segment.split or ""}
            contacted_by_sensor, reference_by_sensor = {}, {}
            for sensor_number, name, sensor_kind, convert_adc in sensors:
                feature_kind = "voltage" if config.sensor_representation in {'voltage','adc_voltage'} else sensor_kind
                prefix = f"sensor_{sensor_number}_{feature_kind}"
                contacted = _signal_features(_sensor_signal(segment.columns[name][start:stop], config,convert_adc=convert_adc),
                                             inferred_rate, config.bands_Hz,spectral_mode=config.spectral_mode)
                if config.remove_sensor_mean:
                    contacted["mean"] = 0.0  # Avoid training on floating-point residuals of DC removal.
                contacted_by_sensor[sensor_number] = contacted
                for feature, value in contacted.items():
                    if config.spectral_mode == "band_peaks" and feature in {
                            "dominant_frequency_hz", "peak_amplitude", "spectral_snr_db"}:
                        continue
                    key = f"{prefix}__{feature}"
                    row[key] = value
                    if segment.source == "simulation":
                        source_kind = "simulated"
                    else:
                        source_kind = "calibrated" if sensor_kind == "voltage" else "measured_raw"
                    provenance[key] = {"kind": "derived", "source": name, "source_kind": source_kind}
                    if convert_adc:
                        provenance[key].update(kind="adc_to_voltage_derived",
                            volts_per_count=config.adc_volts_per_count,
                            voltage_offset_v=config.adc_voltage_offset_v)
                    provenance[key]["remove_sensor_mean"] = config.remove_sensor_mean
                if config.use_reference_features:
                    suffix = "reference_voltage_v" if sensor_kind == "voltage" else "reference_raw_adc"
                    reference_name = f"piezo_sensor_{sensor_number}_{suffix}"
                    if reference_name not in segment.columns and sensor_kind == 'voltage':
                        reference_name = f'piezo_sensor_{sensor_number}_reference_V'
                    if reference_name not in segment.columns:
                        raise ValueError(f"Reference features requested but {reference_name!r} is missing")
                    reference = _signal_features(_sensor_signal(segment.columns[reference_name][start:stop], config,convert_adc=convert_adc),
                                                 inferred_rate, config.bands_Hz,spectral_mode=config.spectral_mode)
                    if config.remove_sensor_mean:
                        reference["mean"] = 0.0
                    reference_by_sensor[sensor_number] = reference
                    changes = {
                        "dominant_frequency_change_hz": contacted["dominant_frequency_hz"]-reference["dominant_frequency_hz"],
                        "peak_amplitude_change": contacted["peak_amplitude"]-reference["peak_amplitude"],
                        "peak_amplitude_change_ratio": ((contacted["peak_amplitude"]-reference["peak_amplitude"])
                                                        / max(abs(reference["peak_amplitude"]), 1e-30)),
                    }
                    for feature, value in changes.items():
                        key = f"{prefix}__{feature}"
                        row[key] = value
                        provenance[key] = {"kind": "derived_contact_minus_reference",
                                           "source": [name, reference_name], "source_kind": source_kind}
            if config.use_actuator_features:
                for number in config.actuator_ids:
                    name = f"piezo_actuator_{number}_command"
                    if name not in segment.columns:
                        raise ValueError(f"Actuator command features require {name}")
                    voltage = _command_voltage(segment.columns[name][start:stop], config)
                    for feature, value in _signal_features(voltage, inferred_rate, config.bands_Hz).items():
                        key = f"actuator_{number}_command_voltage__{feature}"
                        row[key] = value
                        provenance[key] = {"kind": "command_derived", "source": name,
                            "command_min": config.actuator_command_min, "command_max": config.actuator_command_max,
                            "drive_peak_v": config.actuator_drive_peak_v, "measured_voltage": False}
            if config.physics_enabled:
                for feature, value in _physics_features(contacted_by_sensor, reference_by_sensor, config).items():
                    row[feature] = value
                    provenance[feature] = {
                        "kind": "physics_derived",
                        "source": [f"sensor_{number}_{'voltage' if config.sensor_representation in {'voltage','adc_voltage'} else sensor_kind}"
                                   for number in config.physics_sensor_pair],
                        "assumptions": "ordered opposite spans; same identified mode; ideal taut string",
                    }
            for name in config.context_columns:
                if name == "sample_rate_hz":
                    value = inferred_rate if name not in segment.columns else _constant(segment.columns[name][start:stop], name)
                    provenance[name] = {"kind": "timing_derived", "source": "time_s/sample_rate_hz"}
                else:
                    if name not in segment.columns:
                        raise ValueError(f"Configured context column {name!r} is missing from {segment.segment_id}")
                    value = _constant(segment.columns[name][start:stop], name)
                    kind = "derived" if "differential" in name else "measured_or_commanded"
                    provenance[name] = {"kind": kind, "source": name}
                row[name] = value
            for target in config.target_columns:
                if target in segment.columns:
                    row[target] = _constant(segment.columns[target][start:stop], target)
            rows.append(row)
    if not rows:
        raise ValueError("No windows extracted")
    feature_names = list(provenance)
    return rows, feature_names, provenance


def write_feature_set(rows, feature_names, provenance, config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    names = list(rows[0])
    with (output / "features.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    (output / "feature_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "feature_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "feature_order.json").write_text(json.dumps(feature_names, indent=2) + "\n")


def read_feature_set(path):
    root = Path(path)
    with (root / "features.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    features = json.loads((root / "feature_order.json").read_text())
    config = FeatureConfig(**json.loads((root / "feature_config.json").read_text()))
    return rows, features, config


def _matrix(rows, names):
    try:
        return np.asarray([[float(row[name]) for name in names] for row in rows], float)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or nonnumeric required column: {exc}") from exc


def _metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "mse": float(mean_squared_error(y, prediction)),
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "r2": float(r2_score(y, prediction)) if len(y) > 1 else None}


def _class_indices(values, classes, target):
    """Encode physical numeric labels, including noninteger locations, without rounding."""
    lookup = {value: index for index, value in enumerate(classes)}
    try:
        return np.asarray([lookup[float(value)] for value in values], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unseen class {exc.args[0]} for {target}; trained classes are {classes}") from exc


def _classification_metrics(actual, probabilities):
    labels = np.arange(probabilities.shape[1])
    predicted = np.argmax(probabilities, axis=1)
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "log_loss": float(log_loss(actual, probabilities, labels=labels)),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist()}


def _fit_classifier(X, y, target):
    classes = np.unique(y["train"]).tolist()
    if len(classes) < 2:
        raise ValueError(f"Classification requires at least two training classes for {target}")
    encoded = {split: _class_indices(values, classes, target) for split, values in y.items()}
    candidates = {f"logistic_C_{c:g}": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c, max_iter=3000, random_state=42))])
        for c in (.01, .1, 1., 10., 100.)}
    candidates.update({f"random_forest_depth_{depth}": RandomForestClassifier(
        n_estimators=200, max_depth=depth, min_samples_leaf=2, random_state=42, n_jobs=1)
        for depth in (2, None)})
    training, validation = {}, {}
    for name, model in candidates.items():
        model.fit(X["train"], encoded["train"])
        training[name] = _classification_metrics(encoded["train"], model.predict_proba(X["train"]))
        validation[name] = _classification_metrics(encoded["validation"], model.predict_proba(X["validation"]))
    best = min(validation, key=lambda name: validation[name]["log_loss"])
    chosen = candidates[best]
    priors = np.bincount(encoded["train"], minlength=len(classes)) / len(encoded["train"])
    result = {"classes": classes, "selected_model": best, "training_candidates": training,
              "validation_candidates": validation,
              "test": _classification_metrics(encoded["test"], chosen.predict_proba(X["test"])),
              "test_prior_baseline": _classification_metrics(
                  encoded["test"], np.tile(priors, (len(encoded["test"]), 1)))}
    return chosen, classes, priors.tolist(), result


def train(feature_dir, output, alphas=(0.01, 0.1, 1.0, 10.0, 100.0), *, plots=True,
          task="regression", regressor="auto"):
    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    if regressor not in {"auto", "linear", "ridge"}:
        raise ValueError("regressor must be auto, linear, or ridge")
    if task == "classification" and regressor != "auto":
        raise ValueError("regressor selection applies only to regression")
    if task == "regression" and regressor != "linear" and not alphas:
        raise ValueError("Ridge regression requires at least one alpha")
    rows, features, config = read_feature_set(feature_dir)
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "test")}
    if any(not splits[name] for name in splits):
        raise ValueError("Training requires nonempty train, validation, and test splits by segment")
    # A segment must never cross splits, even if upstream feature files were edited.
    membership = {}
    for row in rows:
        old = membership.setdefault(row["segment_id"], row["split"])
        if old != row["split"]:
            raise ValueError(f"Segment {row['segment_id']} appears in multiple splits")
    X = {name: _matrix(part, features) for name, part in splits.items()}
    target_names = [name for name in config.target_columns if all(name in row and row[name] != "" for row in rows)]
    if not target_names:
        raise ValueError("No configured labeled targets are present")
    models, baseline_values, class_values, class_priors = {}, {}, {}, {}
    report = {"task": task, "selection_split": "validation",
              "selection_metric": "log_loss" if task == "classification" else "rmse",
              "simulation_only": all(r["source"] == "simulation" for r in rows), "targets": {}}
    if task == "regression":
        report["regressor_family"] = regressor
    for target in target_names:
        y = {name: _matrix(part, [target]).ravel() for name, part in splits.items()}
        if task == "classification":
            models[target], class_values[target], class_priors[target], report["targets"][target] = _fit_classifier(X, y, target)
            continue
        candidates = {}
        if regressor in {"auto", "linear"}:
            candidates["linear"] = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
        if regressor in {"auto", "ridge"}:
            candidates.update({f"ridge_alpha_{alpha:g}": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=alpha))]) for alpha in alphas})
        validation, training = {}, {}
        for name, model in candidates.items():
            model.fit(X["train"], y["train"])
            training[name] = _metrics(y['train'],model.predict(X['train']))
            validation[name] = _metrics(y["validation"], model.predict(X["validation"]))
        best = min(validation, key=lambda name: validation[name]["rmse"])
        chosen = candidates[best]
        dummy = DummyRegressor(strategy="mean").fit(X["train"], y["train"])
        models[target] = chosen
        baseline_values[target] = float(np.mean(y["train"]))
        report["targets"][target] = {"selected_model": best, "validation_candidates": validation,
            "training_candidates":training,
            "test": _metrics(y["test"], chosen.predict(X["test"])),
            "test_mean_baseline": _metrics(y["test"], dummy.predict(X["test"]))}
    bundle = {"bundle_version": 1, "task": task, "models": models, "feature_order": features,
              "target_names": target_names, "feature_config": asdict(config),
              "training_mean_baselines": baseline_values, "class_values": class_values,
              "training_class_priors": class_priors}
    if task == "regression":
        bundle["regressor_family"] = regressor
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / "model_bundle.pkl").open("wb") as stream: pickle.dump(bundle, stream)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    split_definition = {name: sorted({row["segment_id"] for row in part}) for name, part in splits.items()}
    (output / "splits.json").write_text(json.dumps(split_definition, indent=2) + "\n")
    if plots:
        from .ml_plotting import plot_training_loss
        plot_training_loss(report,output/'training_loss')
    return report


def load_bundle(path):
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle.get("bundle_version") != 1:
        raise ValueError("Unsupported model bundle version")
    return bundle


def predict(bundle_path, input_path, output, *, split=None, adc_volts_per_count=None,
            adc_voltage_offset_v=None, plots=True):
    _protect_capture(input_path,[output,Path(output).with_suffix('.preprocessing.json'),
                                 Path(output).with_suffix('.png'),Path(output).with_suffix('.svg')])
    bundle = load_bundle(bundle_path)
    config = FeatureConfig(**bundle["feature_config"])
    if adc_volts_per_count is not None:
        config = replace(config,adc_volts_per_count=adc_volts_per_count)
    if adc_voltage_offset_v is not None:
        config = replace(config,adc_voltage_offset_v=adc_voltage_offset_v)
    segments = load_segments(input_path,stream=True)
    if split is not None and Path(input_path).is_dir():
        segments = (segment for segment in segments if segment.split == split)
    rows, features, provenance = extract_features(segments, config)
    if features != bundle["feature_order"]:
        raise ValueError(f"Feature mismatch: model expects {bundle['feature_order']}, capture produced {features}")
    X = _matrix(rows, features)
    output_rows = []
    for index, row in enumerate(rows):
        result = {name: row[name] for name in ("source", "segment_id", "window_index", "window_start_s", "window_end_s")}
        for target, model in bundle["models"].items():
            if bundle.get("task", "regression") == "classification":
                probabilities = model.predict_proba(X[index:index+1])[0]
                result[f"predicted_{target}"] = bundle["class_values"][target][int(np.argmax(probabilities))]
                for class_index, probability in enumerate(probabilities):
                    result[f"probability_{target}_class_{class_index}"] = float(probability)
            else:
                result[f"predicted_{target}"] = float(model.predict(X[index:index+1])[0])
            if target in row:
                result[target] = row[target]
        output_rows.append(result)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    output.with_suffix('.preprocessing.json').write_text(json.dumps({
        'feature_config':asdict(config),'feature_provenance':provenance,
        'task':bundle.get('task','regression'),'class_values':bundle.get('class_values',{})},indent=2)+'\n')
    if plots:
        from .ml_plotting import plot_predictions, plot_classification_predictions
        if bundle.get('task','regression') == 'classification':
            plot_classification_predictions(output_rows,bundle['class_values'],output.with_suffix(''))
        else:
            plot_predictions(output_rows,bundle['target_names'],output.with_suffix(''))
    return output_rows


def _protect_capture(input_path, outputs):
    if Path(input_path).is_file() and any(Path(path).resolve() == Path(input_path).resolve() for path in outputs):
        raise ValueError('Output must not overwrite the original capture recording')


def evaluate(bundle_path, input_path, output, *, adc_volts_per_count=None, adc_voltage_offset_v=None, plots=True):
    _protect_capture(input_path,[output])
    rows = predict(bundle_path, input_path, Path(output).with_suffix(".predictions.csv"), split="test",
                   adc_volts_per_count=adc_volts_per_count,adc_voltage_offset_v=adc_voltage_offset_v,plots=plots)
    bundle = load_bundle(bundle_path)
    task = bundle.get("task", "regression")
    report = {"domain": rows[0]["source"], "task": task, "targets": {}}
    for target in bundle["target_names"]:
        if not all(target in row for row in rows):
            continue
        actual = np.asarray([float(row[target]) for row in rows])
        if task == "classification":
            classes = bundle["class_values"][target]
            encoded = _class_indices(actual, classes, target)
            probabilities = np.asarray([[row[f"probability_{target}_class_{i}"]
                                         for i in range(len(classes))] for row in rows])
            report["targets"][target] = {"classes": classes,
                "model": _classification_metrics(encoded, probabilities),
                "training_prior_baseline": _classification_metrics(encoded,
                    np.tile(bundle["training_class_priors"][target], (len(actual), 1)))}
            continue
        predicted = np.asarray([row[f"predicted_{target}"] for row in rows])
        baseline = np.full(len(actual), bundle["training_mean_baselines"][target])
        report["targets"][target] = {"model": _metrics(actual, predicted),
                                      "training_mean_baseline": _metrics(actual, baseline)}
    if not report["targets"]:
        raise ValueError("Evaluation input has no labels; use predict for unlabeled captures")
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
