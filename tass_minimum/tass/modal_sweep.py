"""Held-pose routed-tendon sweeps using string-finger-simulator's modal model."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .models.tendon import Tendon
from .models.modal_string import ModalString


@dataclass
class ModalSweepResult:
    frequency_Hz: np.ndarray
    response: np.ndarray  # actuator, frequency, sensor; complex empirical units
    natural_frequencies_Hz: np.ndarray
    guide_positions_m: np.ndarray
    metadata: dict


def run_modal_sweep(config, frequencies_Hz, *, joint_angle_rad=0.0,
                    motor_angle_rad=0.12, tension_N=None, modes=12,
                    damping_ratio=0.02):
    """Unit drive at P1-P4, sensed at P1-P4, with pose and tension held fixed.

    An explicit tension overrides the elastic law. Otherwise motor winding and
    held joint angle determine tension, regardless of axial.mode. Holding a pose
    assumes an external fixture/controller; this does not solve equilibrium.
    """
    config.validate()
    frequencies = np.asarray(frequencies_Hz, dtype=float)
    if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("Expected a nonempty vector of finite nonnegative frequencies")
    if not np.all(np.isfinite([joint_angle_rad, motor_angle_rad])):
        raise ValueError("Joint and motor angles must be finite")
    tendon = Tendon.from_config(config)
    elastic_tension = tendon.compute(joint_angle_rad, 0.0,
                                    config.motor.spool_radius_m*motor_angle_rad, 0.0)
    tension = elastic_tension if tension_N is None else float(tension_N)
    length = float(tendon.path_length(joint_angle_rad))
    pairs = config.modal_piezo
    indices = np.asarray(pairs.guide_numbers)-1
    positions = tendon.guide_path_coordinates(joint_angle_rad)[indices]
    ratios = tendon.guide_path_ratios(joint_angle_rad)[indices]
    modal = ModalString(length, tendon.mu, modes, damping_ratio)
    modal.set_tension(tension)
    shapes = np.array([modal.mode_shape(ratio) for ratio in ratios])
    shapes *= np.asarray(pairs.sensor_gains)[:, None]
    response = np.empty((4, len(frequencies), 4), dtype=complex)
    for actuator, ratio in enumerate(ratios):
        drive = pairs.actuator_amplitudes_au[actuator]*np.exp(1j*pairs.actuator_phases_rad[actuator])
        for i, frequency in enumerate(frequencies):
            response[actuator, i] = shapes @ modal.modal_response_to_drive(frequency, drive, ratio)
    return ModalSweepResult(frequencies, response, modal.natural_frequencies(), positions, {
        "model": "routed_empirical_modal", "source": "string-finger-simulator",
        "pair_ids": list(pairs.pair_ids), "guide_numbers": list(pairs.guide_numbers),
        "response_units": "arbitrary",
        "actuator_amplitudes_au": list(pairs.actuator_amplitudes_au),
        "actuator_phases_rad": list(pairs.actuator_phases_rad), "sensor_gains": list(pairs.sensor_gains),
        "joint_angle_rad": float(joint_angle_rad), "motor_angle_rad": float(motor_angle_rad),
        "tension_N": tension, "tension_source": "elastic" if tension_N is None else "prescribed",
        "path_length_m": length, "guide_positions_m": positions.tolist(),
        "modes": modes, "damping_ratio": damping_ratio,
        "contact_model": "none", "held_pose": True,
        "fixed_endpoint_pair_id": pairs.pair_ids[list(pairs.guide_numbers).index(4)],
    })


def save_modal_sweep(result, config, directory):
    """Save complex response, labeled CSV, settings and magnitude plots."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/"modal_sweep.npz", frequency_Hz=result.frequency_Hz,
                        response=result.response, natural_frequencies_Hz=result.natural_frequencies_Hz,
                        guide_positions_m=result.guide_positions_m,
                        pair_ids=result.metadata["pair_ids"])
    rows = [(a+1, f, s+1, value.real, value.imag, abs(value))
            for a in range(4) for i, f in enumerate(result.frequency_Hz)
            for s, value in enumerate(result.response[a, i])]
    np.savetxt(directory/"modal_sweep.csv", rows, delimiter=",", comments="",
               header="actuator,frequency_Hz,sensor,response_real_au,response_imag_au,magnitude_au")
    (directory/"metadata.json").write_text(json.dumps(result.metadata, indent=2, allow_nan=False)+"\n")
    config.save(directory/"config.json")
    config.save(directory/"config.yaml")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a, ax in enumerate(axes.flat):
        for s in range(4):
            ax.plot(result.frequency_Hz, abs(result.response[a, :, s]), label=result.metadata["pair_ids"][s])
        ax.set(title=f"Drive {result.metadata['pair_ids'][a]}", xlabel="Frequency [Hz]", ylabel="Response [a.u.]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory/"modal_sweep.png", dpi=150)
    plt.close(fig)
