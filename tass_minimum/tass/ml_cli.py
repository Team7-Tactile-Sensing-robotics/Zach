"""Command-line interface for the simulation-to-real regression baseline."""
import argparse
import json
from pathlib import Path

from .ml import FeatureConfig, evaluate, extract_features, load_segments, predict, train, write_feature_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--window-s", type=float, default=1.0)
    extract.add_argument("--stride-s", type=float, default=1.0)
    extract.add_argument("--sensor-representation", choices=["auto", "voltage", "raw_adc"], default="auto")
    extract.add_argument("--sensor-id", action="append", type=int, dest="sensor_ids")
    extract.add_argument("--reference-features", action="store_true")
    extract.add_argument("--physics-features", action="store_true")
    extract.add_argument("--physics-sensor-pair", type=int, nargs=2, default=(1, 2), metavar=("LEFT", "RIGHT"))
    extract.add_argument("--string-length-m", type=float)
    extract.add_argument("--linear-density-kg-m", type=float)
    extract.add_argument("--mode-number", type=int, default=1)
    extract.add_argument("--context-column", action="append", dest="context_columns")
    training = sub.add_parser("train")
    training.add_argument("--features", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--input", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        context = tuple(args.context_columns) if args.context_columns else ("sample_rate_hz",)
        sensor_ids = tuple(args.sensor_ids) if args.sensor_ids else (1,)
        config = FeatureConfig(window_s=args.window_s, stride_s=args.stride_s,
                               sensor_representation=args.sensor_representation,
                               sensor_ids=sensor_ids,
                               use_reference_features=args.reference_features,
                               physics_enabled=args.physics_features,
                               physics_sensor_pair=tuple(args.physics_sensor_pair),
                               physics_string_length_m=args.string_length_m,
                               physics_linear_density_kg_m=args.linear_density_kg_m,
                               physics_mode_number=args.mode_number,
                               context_columns=context)
        rows, names, provenance = extract_features(load_segments(args.input), config)
        write_feature_set(rows, names, provenance, config, args.output)
        result = {"windows": len(rows), "features": len(names), "output": str(args.output)}
    elif args.command == "train":
        result = train(args.features, args.output)
    elif args.command == "evaluate":
        result = evaluate(args.model, args.input, args.output)
    else:
        result = {"predictions": len(predict(args.model, args.input, args.output)), "output": str(args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
