"""Run with python -m machine_learning --help from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import load_config, read_samples
from .features import extract_features
from .experiments import run_experiments
from .synthetic import make_samples, make_echo_samples
from .comparison import compare_features, ComparisonModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "train", "run", "demo", "compare", "compare-demo"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path)
        cmd.add_argument("--output", type=Path, required=True, help="New output directory")
        if name not in ("demo", "compare-demo"):
            cmd.add_argument("--input", type=Path, required=True,
                             help="samples CSV for extract/run; features CSV for train")
        if name == "compare-demo":
            cmd.add_argument("--signal-mode", choices=["steady", "echo"], default="steady")
    predict = sub.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True, help="Extracted feature CSV")
    predict.add_argument("--output", type=Path, required=True, help="Prediction CSV")
    args = parser.parse_args()
    try:
        if args.command == "predict":
            if args.output.exists():
                raise ValueError("Prediction output exists; choose another path")
            # Load only model files you produced/trust: joblib uses pickle.
            model = joblib.load(args.model)
            frame = pd.read_csv(args.input)
            if isinstance(model, ComparisonModel):
                result = pd.DataFrame(dict(predicted=model.predict(frame)))
                probabilities = None
            else:
                pred, labels, probabilities = model.predict(frame)
                result = pd.DataFrame(dict(predicted=pred, predicted_class=labels))
            if "window_id" in frame:
                result.insert(0, "window_id", frame.window_id)
            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    result[f"probability_class_{i}"] = probabilities[:, i]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            return
        cfg = load_config(args.config)
        if args.command == "compare-demo" and args.signal_mode == "echo":
            # Fixed published fixture protocol, not inferred from labels/holdout scores.
            cfg.update(time_features_enabled=True, echo_features_enabled=True,
                       echo_gate_s=[.008, .105], echo_min_peak_distance_s=.006)
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory is not empty; choose a fresh directory")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command not in ("demo", "compare-demo"):
            digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
            (args.output/"input_provenance.json").write_text(json.dumps(
                {"path": str(args.input.resolve()), "sha256": digest}, indent=2)+"\n")
        if args.command == "train":
            frame = pd.read_csv(args.input, dtype={"session_id": str, "trial_id": str})
        else:
            if args.command in ("demo", "compare-demo"):
                print("Generating TOY SYNTHETIC signals, not calibrated simulator/hardware data.", flush=True)
                raw = (make_echo_samples(cfg) if args.command == "compare-demo" and args.signal_mode == "echo"
                       else make_samples(cfg))
                raw.to_csv(args.output/"synthetic_samples.csv", index=False, float_format="%.10g")
            else:
                raw = read_samples(args.input, cfg)
            frame, audit = extract_features(raw, cfg)
            frame.to_csv(args.output/"features.csv", index=False)
            audit.to_csv(args.output/"window_audit.csv", index=False)
            (args.output/"config_used.json").write_text(json.dumps(cfg, indent=2)+"\n")
            if args.command == "extract":
                print(f"Extracted {len(frame)} valid windows into {args.output}")
                return
        summary = (compare_features(frame, cfg, args.output/"results") if args.command in ("compare", "compare-demo")
                   else run_experiments(frame, cfg, args.output/"results"))
        print(json.dumps(summary, indent=2))
        print(f"Report: {args.output/'results'/'REPORT.md'}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
