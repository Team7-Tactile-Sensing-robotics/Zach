"""Run with python -m tass; no package installation needed from project root."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
from .config import ModelConfig
from .signals import Inputs, Constant, Pulse, SmoothStep, Chirp
from .simulation import simulate
from .processing import extract_frequency_features
from .io import load_measurement, save_run, save_frequency_result
from .plotting import plot_run


def main():
    parser = argparse.ArgumentParser(description="TASS minimum analytical simulator")
    parser.add_argument("--config", type=Path, help="JSON parameters; missing fields use defaults")
    parser.add_argument("--scenario", choices=["baseline", "contact", "coupled"], default="baseline")
    parser.add_argument("--tension-mode", choices=["measured", "simulated"], help="Override config")
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--process-csv", type=Path, help="Process hardware/saved CSV without simulation")
    args = parser.parse_args()
    if args.process_csv:
        frequency = extract_frequency_features(load_measurement(args.process_csv))
        save_frequency_result(frequency, args.output)
    else:
        config = ModelConfig.load(args.config) if args.config else ModelConfig()
        if args.tension_mode:
            config = replace(config, axial=replace(config.axial, mode=args.tension_mode))
        if args.scenario == "coupled":
            # Explicit --tension-mode overrides the example's default simulated mode.
            if args.tension_mode is None:
                config = replace(config, axial=replace(config.axial, mode="simulated"))
            duration = config.numerics.duration_s
            inputs = Inputs(motor_angle_rad=SmoothStep(0.12, 0.18, 0.05*duration, 0.4*duration),
                            contact_force_N=SmoothStep(0, 0.4, 0.55*duration, 0.65*duration),
                            excitation_N=Chirp(0.002, 80, 1800, duration))
        else:
            inputs = Inputs(contact_force_N=Constant(0.4 if args.scenario == "contact" else 0.0),
                            excitation_N=Pulse())
        result = simulate(config, inputs)
        frequency = extract_frequency_features(result.measurement)
        save_run(result, frequency, args.output)
        plot_run(result, frequency, args.output/"overview.png", f"TASS / {args.scenario} / {config.axial.mode} tension")
    print(json.dumps({"output": str(args.output.resolve()), "features": frequency.features}, indent=2))


if __name__ == "__main__":
    main()
