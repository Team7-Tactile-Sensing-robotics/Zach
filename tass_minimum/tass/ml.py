"""Small, leakage-safe regression pipeline shared by simulation and captures."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    sensor_representation: str = "auto"  # auto prefers raw ADC when present, then voltage
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
        if self.sensor_representation not in {"auto", "voltage", "raw_adc"}:
            raise ValueError("sensor_representation must be auto, voltage, or raw_adc")
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
    selected = sorted(item for item in found[kind] if item[0] in sensor_ids)
    if not selected:
        raise ValueError(f"No {kind} piezo sensor columns found")
    return kind, selected


def load_simulation_dataset(root: str | Path) -> list[Segment]:
    root = Path(root)
    with (root / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    segments = []
    for row in manifest:
        with np.load(root / row["file"], allow_pickle=False) as data:
            columns = {name: np.asarray(data[name]) for name in data.files
                       if np.asarray(data[name]).ndim == 1 and len(np.asarray(data[name])) == len(data["time_s"])}
        # Normalize simulator names to the real-capture convention at the boundary.
        columns.update({name.replace("_V", "_voltage_v"): value for name, value in list(columns.items())
                        if re.match(r"^piezo_sensor_\d+_V$", name)})
        columns.update({name.replace("_reference_V", "_reference_voltage_v"): value
                        for name, value in list(columns.items())
                        if re.match(r"^piezo_sensor_\d+_reference_V$", name)})
        columns["sample_rate_hz"] = np.full(len(columns["time_s"]), 1.0 / np.median(np.diff(columns["time_s"])))
        segments.append(Segment(row["trial_id"], columns.pop("time_s"), columns,
                                split=row.get("split"), source="simulation"))
    return segments


def load_capture_csv(path: str | Path) -> list[Segment]:
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.dtype.names is None or "time_s" not in data.dtype.names:
        raise ValueError("Capture CSV must have a header and time_s")
    if data.ndim == 0:
        data = data.reshape(1)
    names = data.dtype.names
    ids = data["segment_id"].astype(str) if "segment_id" in names else np.full(len(data), path.stem)
    segments = []
    for segment_id in np.unique(ids):
        mask = ids == segment_id
        order = np.argsort(np.asarray(data["time_s"][mask], float))
        columns = {name: np.asarray(data[name][mask])[order] for name in names
                   if name not in {"segment_id", "time_s"}}
        segments.append(Segment(str(segment_id), np.asarray(data["time_s"][mask], float)[order],
                                columns, source="real"))
    return segments


def load_segments(path: str | Path) -> list[Segment]:
    path = Path(path)
    return load_simulation_dataset(path) if path.is_dir() else load_capture_csv(path)


def _constant(values, name):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"{name} has no finite values")
    if not np.allclose(finite, finite[0], rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} must be constant within a window")
    return float(finite[0])


def _signal_features(values, sample_rate, bands):
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
    return result


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
        sensor_kind, sensors = _sensor_columns(segment.columns, config.sensor_representation, config.sensor_ids)
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
            for sensor_number, name in sensors:
                prefix = f"sensor_{sensor_number}_{sensor_kind}"
                contacted = _signal_features(segment.columns[name][start:stop], inferred_rate, config.bands_Hz)
                contacted_by_sensor[sensor_number] = contacted
                for feature, value in contacted.items():
                    key = f"{prefix}__{feature}"
                    row[key] = value
                    if segment.source == "simulation":
                        source_kind = "simulated"
                    else:
                        source_kind = "calibrated" if sensor_kind == "voltage" else "measured_raw"
                    provenance[key] = {"kind": "derived", "source": name, "source_kind": source_kind}
                if config.use_reference_features:
                    suffix = "reference_voltage_v" if sensor_kind == "voltage" else "reference_raw_adc"
                    reference_name = f"piezo_sensor_{sensor_number}_{suffix}"
                    if reference_name not in segment.columns:
                        raise ValueError(f"Reference features requested but {reference_name!r} is missing")
                    reference = _signal_features(segment.columns[reference_name][start:stop], inferred_rate, config.bands_Hz)
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
            if config.physics_enabled:
                for feature, value in _physics_features(contacted_by_sensor, reference_by_sensor, config).items():
                    row[feature] = value
                    provenance[feature] = {
                        "kind": "physics_derived",
                        "source": [f"sensor_{number}_{sensor_kind}" for number in config.physics_sensor_pair],
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
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "r2": float(r2_score(y, prediction)) if len(y) > 1 else None}


def train(feature_dir, output, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
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
    models, baseline_values = {}, {}
    report = {"selection_split": "validation", "simulation_only": all(r["source"] == "simulation" for r in rows), "targets": {}}
    for target in target_names:
        y = {name: _matrix(part, [target]).ravel() for name, part in splits.items()}
        candidates = {"linear": Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])}
        candidates.update({f"ridge_alpha_{alpha:g}": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=alpha))]) for alpha in alphas})
        validation = {}
        for name, model in candidates.items():
            model.fit(X["train"], y["train"])
            validation[name] = _metrics(y["validation"], model.predict(X["validation"]))
        best = min(validation, key=lambda name: validation[name]["rmse"])
        chosen = candidates[best]
        dummy = DummyRegressor(strategy="mean").fit(X["train"], y["train"])
        models[target] = chosen
        baseline_values[target] = float(np.mean(y["train"]))
        report["targets"][target] = {"selected_model": best, "validation_candidates": validation,
            "test": _metrics(y["test"], chosen.predict(X["test"])),
            "test_mean_baseline": _metrics(y["test"], dummy.predict(X["test"]))}
    bundle = {"bundle_version": 1, "models": models, "feature_order": features,
              "target_names": target_names, "feature_config": asdict(config),
              "training_mean_baselines": baseline_values}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / "model_bundle.pkl").open("wb") as stream: pickle.dump(bundle, stream)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    split_definition = {name: sorted({row["segment_id"] for row in part}) for name, part in splits.items()}
    (output / "splits.json").write_text(json.dumps(split_definition, indent=2) + "\n")
    return report


def load_bundle(path):
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle.get("bundle_version") != 1:
        raise ValueError("Unsupported model bundle version")
    return bundle


def predict(bundle_path, input_path, output, *, split=None):
    bundle = load_bundle(bundle_path)
    config = FeatureConfig(**bundle["feature_config"])
    segments = load_segments(input_path)
    if split is not None and any(segment.split for segment in segments):
        segments = [segment for segment in segments if segment.split == split]
        if not segments:
            raise ValueError(f"No segments found for split {split!r}")
    rows, features, _ = extract_features(segments, config)
    if features != bundle["feature_order"]:
        raise ValueError(f"Feature mismatch: model expects {bundle['feature_order']}, capture produced {features}")
    X = _matrix(rows, features)
    output_rows = []
    for index, row in enumerate(rows):
        result = {name: row[name] for name in ("source", "segment_id", "window_index", "window_start_s", "window_end_s")}
        for target, model in bundle["models"].items():
            result[f"predicted_{target}"] = float(model.predict(X[index:index+1])[0])
            if target in row:
                result[target] = row[target]
        output_rows.append(result)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    return output_rows


def evaluate(bundle_path, input_path, output):
    rows = predict(bundle_path, input_path, Path(output).with_suffix(".predictions.csv"), split="test")
    bundle = load_bundle(bundle_path)
    report = {"domain": rows[0]["source"], "targets": {}}
    for target in bundle["target_names"]:
        if not all(target in row for row in rows):
            continue
        actual = np.asarray([float(row[target]) for row in rows])
        predicted = np.asarray([row[f"predicted_{target}"] for row in rows])
        baseline = np.full(len(actual), bundle["training_mean_baselines"][target])
        report["targets"][target] = {"model": _metrics(actual, predicted),
                                      "training_mean_baseline": _metrics(actual, baseline)}
    if not report["targets"]:
        raise ValueError("Evaluation input has no labels; use predict for unlabeled captures")
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
