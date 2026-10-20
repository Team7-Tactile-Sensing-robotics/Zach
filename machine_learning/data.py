"""Configuration and synchronized wide-table schema; never alter raw recordings."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RX = [f"piezo_sensor_{i}_V" for i in range(1, 5)]
TX = [f"piezo_actuator_{i}_V" for i in range(1, 5)]
STATES = ["servo_angle_rad", "servo_torque_Nm", "load_cell_N", "excitation_frequency_Hz"]
TARGETS = ["contact_force_N", "contact_location_m"]
IDS = ["session_id", "trial_id", "data_source"]


def load_config(path=None):
    default = Path(__file__).with_name("config.json")
    cfg = json.loads(default.read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied)-set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(supplied)
    for key in ("window_samples", "hop_samples", "cv_folds", "min_inference_groups",
                "min_rows_per_parameter", "logistic_classes", "max_experiments", "max_subset_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("time_segments", "envelope_smoothing_samples", "echo_peak_count", "comparison_pca_components"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["time_segments"] > cfg["window_samples"] or cfg["envelope_smoothing_samples"] > cfg["window_samples"]:
        raise ValueError("Time segmentation/smoothing must fit inside a window")
    for key in ("time_features_enabled", "echo_features_enabled"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not 0 < cfg["echo_prominence_fraction"] <= 1:
        raise ValueError("echo_prominence_fraction must be in (0, 1]")
    for key in ("echo_min_peak_distance_s", "echo_min_amplitude_V"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["echo_features_enabled"]:
        gate = cfg["echo_gate_s"]
        if gate is None or len(gate) != 2 or not np.isfinite(gate).all() or not 0 <= gate[0] < gate[1]:
            raise ValueError("Echo features require an explicit nonnegative echo_gate_s after the trigger")
    for key in ("svr_C", "svr_epsilon", "svr_gamma"):
        if not cfg[key] or any(not (key == "svr_gamma" and v == "scale") and
                              (not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0) for v in cfg[key]):
            raise ValueError(f"{key} must contain positive values (or 'scale' for gamma)")
    if cfg["window_samples"] < 8 or cfg["cv_folds"] < 2 or cfg["logistic_classes"] < 2:
        raise ValueError("Need >=8 samples, >=2 CV folds and >=2 logistic classes")
    if cfg["group_column"] not in ("session_id", "trial_id"):
        raise ValueError("group_column must be session_id or trial_id")
    for key in ("holdout_fraction", "alpha"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"{key} must lie between 0 and 1")
    if not 0 <= cfg["timestamp_relative_tolerance"] < 0.1:
        raise ValueError("timestamp_relative_tolerance must be in [0, 0.1)")
    if not np.isfinite(cfg["max_condition_number"]) or cfg["max_condition_number"] <= 1:
        raise ValueError("max_condition_number must be finite and >1")
    for key in ("max_force_range_N", "max_location_range_m", "max_frequency_range_Hz"):
        if cfg[key] is not None and (not np.isfinite(cfg[key]) or cfg[key] < 0):
            raise ValueError(f"{key} must be nonnegative or null")
    if not cfg["frequency_bands_Hz"]:
        raise ValueError("At least one frequency band is required")
    for lo, hi in cfg["frequency_bands_Hz"]:
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo < hi):
            raise ValueError("Invalid frequency band")
    for target, edges in cfg["target_bin_edges"].items():
        if target not in TARGETS or (edges is not None and
                (len(edges) < 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0))):
            raise ValueError("Target bin edges must be increasing finite internal boundaries")
    return cfg


def read_samples(path, cfg):
    frame = pd.read_csv(path, dtype={name: str for name in IDS})
    frame = frame.rename(columns=cfg["column_map"])
    required = IDS + ["time_s"] + TARGETS + RX + TX + STATES
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}; use templates/samples.csv or column_map")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Input must have rows and unique column names")
    for name in IDS:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {name}; independent recording IDs are required")
        frame[name] = frame[name].astype(str)
    if not set(frame.data_source) <= {"hardware", "synthetic"}:
        raise ValueError("data_source must be hardware or synthetic")
    for name in ["time_s"] + TARGETS + RX + TX + STATES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name]).any():
            raise ValueError(f"Infinite values in {name}")
    if frame.time_s.isna().any():
        raise ValueError("Every waveform row needs a timestamp")
    if cfg["echo_features_enabled"]:
        if "excitation_trigger_time_s" not in frame:
            raise ValueError("Echo features require measured excitation_trigger_time_s for each capture")
        frame["excitation_trigger_time_s"] = pd.to_numeric(frame.excitation_trigger_time_s, errors="raise")
        if not np.isfinite(frame.excitation_trigger_time_s).all():
            raise ValueError("Excitation trigger times must be finite")
    if "quality_flag" not in frame:
        frame["quality_flag"] = "unassessed"
    frame["quality_flag"] = frame.quality_flag.fillna("unassessed")
    if not set(frame.quality_flag) <= {"ok", "unassessed", "clipped", "dropped", "suspect"}:
        raise ValueError("Unrecognized quality_flag")
    # Reject re-used trial IDs across sessions/sources; they otherwise leak across splits.
    if (frame.groupby("trial_id")[["session_id", "data_source"]].nunique() > 1).any().any():
        raise ValueError("trial_id must identify exactly one session and data source")
    return frame
