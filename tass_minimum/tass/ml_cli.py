"""Command-line interface for simulation-to-real regression and classification."""
import argparse
import json
from pathlib import Path

from .ml import (FeatureConfig, evaluate, extract_features, load_segments, predict, train,
                 write_feature_set, _protect_capture)


def _add_calibration_arguments(parser):
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument('--adc-volts-per-count',type=float,help='Real ADC calibration slope in V/count')
    mapping.add_argument('--adc-full-scale-v',type=float,help='Nominal real ADC span; slope = span / (2^bits-1)')
    parser.add_argument('--adc-resolution-bits',type=int,default=12)
    parser.add_argument('--adc-voltage-offset-v',type=float,
                        help='Real ADC affine offset in V (extract default: 0; predict/evaluate: saved value)')


def _calibration(args, parser):
    if not 1 <= args.adc_resolution_bits <= 24:
        parser.error('--adc-resolution-bits must be in [1,24]')
    slope = args.adc_volts_per_count
    if args.adc_full_scale_v is not None:
        slope = args.adc_full_scale_v/(2**args.adc_resolution_bits-1)
    return dict(adc_volts_per_count=slope,adc_voltage_offset_v=args.adc_voltage_offset_v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--window-s", type=float, default=1.0)
    extract.add_argument("--stride-s", type=float, default=1.0)
    extract.add_argument("--sensor-representation", choices=["auto", "voltage", "raw_adc", "adc_voltage"], default="voltage",
                        help='voltage (default): direct simulation volts; calibrated real ADC volts. Other modes retain legacy ADC workflows.')
    _add_calibration_arguments(extract)
    extract.add_argument("--remove-sensor-mean", action="store_true", help="Remove each sensor window's DC bias before all features")
    extract.add_argument("--actuator-features", action="store_true", help="Include derived drive features from sample-aligned commands")
    extract.add_argument("--actuator-id", action="append", type=int, dest="actuator_ids")
    extract.add_argument("--actuator-command-min", type=float, default=-1.0)
    extract.add_argument("--actuator-command-max", type=float, default=1.0)
    extract.add_argument("--actuator-drive-peak-v", type=float, default=3.3)
    extract.add_argument("--target", action="append", dest="targets", choices=["contact_force_N", "contact_location_m", "load_cell_readout_N"])
    extract.add_argument("--sensor-id", action="append", type=int, dest="sensor_ids")
    extract.add_argument("--spectral-mode", choices=["global","band_peaks"], default="global",
                         help="band_peaks uses per-band peak frequencies and normalized powers instead of global-peak features")
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
    training.add_argument("--task", choices=["regression", "classification"], default="regression",
                          help="Classification treats each configured target's exact values as discrete classes")
    training.add_argument("--regressor", choices=["auto", "linear", "ridge", "mlp"], default="auto",
                          help="Regression family for each independent target model; auto compares Linear and Ridge")
    training.add_argument('--classifier',choices=['auto','mlp'],default='auto',
                          help='auto compares logistic/random forest; mlp uses dense networks on the same features')
    training.add_argument('--mlp-max-epochs',type=int,default=600)
    training.add_argument('--mlp-patience',type=int,default=60)
    training.add_argument('--mlp-seed',type=int,default=42)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--input", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    _add_calibration_arguments(evaluation)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    _add_calibration_arguments(inference)
    args = parser.parse_args()
    if args.command == "extract":
        _protect_capture(args.input,[args.output/name for name in (
            'features.csv','feature_config.json','feature_order.json','feature_provenance.json')])
        calibration = _calibration(args,parser)
        context = tuple(args.context_columns) if args.context_columns else ("sample_rate_hz",)
        sensor_ids = tuple(args.sensor_ids) if args.sensor_ids else (1,)
        config = FeatureConfig(window_s=args.window_s, stride_s=args.stride_s,
                               sensor_representation=args.sensor_representation,
                               spectral_mode=args.spectral_mode,
                               adc_volts_per_count=calibration['adc_volts_per_count'],
                               adc_voltage_offset_v=calibration['adc_voltage_offset_v'] if calibration['adc_voltage_offset_v'] is not None else 0.,
                               remove_sensor_mean=args.remove_sensor_mean,
                               use_actuator_features=args.actuator_features,
                               actuator_ids=tuple(args.actuator_ids or (1, 2)),
                               actuator_command_min=args.actuator_command_min,
                               actuator_command_max=args.actuator_command_max,
                               actuator_drive_peak_v=args.actuator_drive_peak_v,
                               target_columns=tuple(args.targets or FeatureConfig().target_columns),
                               sensor_ids=sensor_ids,
                               use_reference_features=args.reference_features,
                               physics_enabled=args.physics_features,
                               physics_sensor_pair=tuple(args.physics_sensor_pair),
                               physics_string_length_m=args.string_length_m,
                               physics_linear_density_kg_m=args.linear_density_kg_m,
                               physics_mode_number=args.mode_number,
                               context_columns=context)
        rows, names, provenance = extract_features(load_segments(args.input,stream=True), config)
        write_feature_set(rows, names, provenance, config, args.output)
        result = {"windows": len(rows), "features": len(names), "output": str(args.output)}
    elif args.command == "train":
        result = train(args.features,args.output,task=args.task,regressor=args.regressor,
            classifier=args.classifier,mlp_max_epochs=args.mlp_max_epochs,
            mlp_patience=args.mlp_patience,mlp_seed=args.mlp_seed)
    elif args.command == "evaluate":
        result = evaluate(args.model, args.input, args.output,**_calibration(args,parser))
    else:
        result = {"predictions": len(predict(args.model, args.input, args.output,**_calibration(args,parser))), "output": str(args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
