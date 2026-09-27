"""Run with python -m tass; no package installation needed from project root."""
import argparse
from dataclasses import replace, asdict
from pathlib import Path
import json
import numpy as np
from .config import ModelConfig
from .signals import Inputs, Constant, Pulse, SmoothStep, Chirp
from .simulation import simulate
from .processing import extract_frequency_features
from .io import load_measurement, save_run, save_frequency_result
from .plotting import plot_run
from .modal_sweep import run_modal_sweep, save_modal_sweep


def scenario_inputs(config):
    """Build the demo's time-varying signals from the resolved configuration."""
    e, duration = config.experiment, config.numerics.duration_s
    common = {"tension_N": Constant(e.measured_tension_N)}
    if e.scenario == "coupled":
        return Inputs(
            motor_angle_rad=SmoothStep(e.motor_initial_rad, e.motor_final_rad,
                                      e.motor_ramp_start_fraction*duration, e.motor_ramp_end_fraction*duration),
            contact_force_N=SmoothStep(0, e.contact_force_N,
                                      e.contact_ramp_start_fraction*duration, e.contact_ramp_end_fraction*duration),
            excitation_N=Chirp(e.excitation_amplitude_N, e.chirp_start_Hz, e.chirp_end_Hz, duration),
            **common)
    return Inputs(motor_angle_rad=Constant(e.motor_initial_rad),
                  contact_force_N=Constant(e.contact_force_N if e.scenario == "contact" else 0),
                  excitation_N=Pulse(e.excitation_amplitude_N, e.pulse_start_s, e.pulse_width_s), **common)


def main():
    parser = argparse.ArgumentParser(description="TASS minimum analytical simulator")
    parser.add_argument("--config", type=Path, help="YAML or JSON parameters; missing fields use defaults")
    parser.add_argument("--scenario", choices=["baseline", "contact", "coupled"], help="Override experiment.scenario")
    parser.add_argument("--tension-mode", choices=["measured", "simulated"], help="Override config")
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--process-csv", type=Path, help="Process hardware/saved CSV without simulation")
    operation.add_argument("--modal-sweep", action="store_true", help="Held-pose routed modal sweep (empirical amplitudes, no contact)")
    parser.add_argument("--joint-angle-deg", type=float, help="Held joint angle for modal sweep")
    parser.add_argument("--motor-angle-rad", type=float, help="Spool angle for modal sweep elastic tension")
    parser.add_argument("--tension-N", type=float, help="Prescribed modal sweep tension; otherwise calculate elastic tension")
    parser.add_argument("--start-Hz", type=float)
    parser.add_argument("--stop-Hz", type=float)
    parser.add_argument("--step-Hz", type=float)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--modal-damping-ratio", type=float)
    args = parser.parse_args()
    config = ModelConfig.load(args.config) if args.config else ModelConfig()
    if args.modal_sweep:
        overrides = {name: getattr(args, name) for name in (
            "joint_angle_deg", "motor_angle_rad", "tension_N", "start_Hz", "stop_Hz", "step_Hz", "modes")
            if getattr(args, name) is not None}
        if args.modal_damping_ratio is not None:
            overrides["damping_ratio"] = args.modal_damping_ratio
        config = replace(config, modal_sweep=replace(config.modal_sweep, **overrides)).validate()
        sweep = config.modal_sweep
        if args.scenario is not None or args.tension_mode:
            parser.error("--modal-sweep uses a held pose without contact; use --tension-N to prescribe tension")
        count = int(np.floor((sweep.stop_Hz-sweep.start_Hz)/sweep.step_Hz))+1
        frequencies = sweep.start_Hz + np.arange(count)*sweep.step_Hz
        result = run_modal_sweep(config, frequencies,
                                 joint_angle_rad=np.deg2rad(sweep.joint_angle_deg),
                                 motor_angle_rad=sweep.motor_angle_rad, tension_N=sweep.tension_N,
                                 modes=sweep.modes, damping_ratio=sweep.damping_ratio)
        save_modal_sweep(result, config, args.output)
        print(json.dumps({"output": str(args.output.resolve()), **result.metadata}, indent=2))
        return
    if args.process_csv:
        frequency = extract_frequency_features(load_measurement(args.process_csv), **asdict(config.analysis))
        save_frequency_result(frequency, args.output)
    else:
        scenario = args.scenario or config.experiment.scenario
        config = replace(config, experiment=replace(config.experiment, scenario=scenario))
        if args.tension_mode:
            config = replace(config, axial=replace(config.axial, mode=args.tension_mode))
        if scenario == "coupled":
            # Explicit --tension-mode overrides the example's default simulated mode.
            if args.tension_mode is None:
                config = replace(config, axial=replace(config.axial, mode="simulated"))
        inputs = scenario_inputs(config)
        result = simulate(config, inputs)
        frequency = extract_frequency_features(result.measurement, **asdict(config.analysis))
        save_run(result, frequency, args.output)
        plot_run(result, frequency, args.output/"overview.png", f"TASS / {scenario} / {config.axial.mode} tension")
    print(json.dumps({"output": str(args.output.resolve()), "features": frequency.features}, indent=2))


if __name__ == "__main__":
    main()
