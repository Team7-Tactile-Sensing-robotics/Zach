"""Orchestration only: mechanics -> vibration -> sensors -> common measurement."""
from dataclasses import replace
import numpy as np
from .models.tendon import Tendon
from .config import ModelConfig
from .signals import Inputs
from .data import Measurement, SimulationResult
from .models.mechanics import simulate_mechanics, forward_kinematics
from .models.string import simulate_string
from .models.piezo import simulate_piezos


def simulate(config=None, inputs=None, *, initial_joint_state=None,
                initial_string_displacement=None, initial_string_velocity=None,
                tension_law=None, contact_boundary=None,
                mechanical_solver=simulate_mechanics, vibration_solver=simulate_string,
                sensor_model=simulate_piezos):
    config = (config or ModelConfig()).validate()
    inputs = inputs or Inputs()
    n = round(config.numerics.duration_s * config.numerics.sample_rate_Hz)
    t = np.arange(n) / config.numerics.sample_rate_Hz
    sampled = inputs.sample(t)
    source_config = config
    drives = sampled["excitation_N"]
    reference = drives
    pair_metadata = {}
    if config.routed_pairs.enabled:
        rp = config.routed_pairs
        tendon = Tendon.from_config(config)
        positions = tuple(float(x) for x in tendon.guide_path_coordinates(rp.reference_angle_rad))
        length = positions[-1] + rp.anchor_extension_m
        config = replace(config,
            string=replace(config.string, length_m=length, exciter_positions_m=positions),
            piezo=replace(config.piezo, positions_m=positions, gains=rp.sensor_gains, noise_std_V=rp.noise_std_V)).validate()
        ids = ("pair1", "pair2", "pair3", "pair4")
        drives = sampled.get("actuator_forces_N")
        if drives is None:
            drives = np.zeros((len(t), 4))
            drives[:, ids.index(rp.active_actuator)] = sampled["excitation_N"]
        active_drives = np.flatnonzero(np.any(drives != 0, axis=0))
        # A single Y/U cannot identify a multi-input system. Preserve all drives
        # as channels, but disable that scalar FRF when several are active.
        reference = drives[:, active_drives[0]] if len(active_drives) == 1 else np.zeros(len(t))
        pair_metadata = {"pair_ids": list(ids), "pair_segments": [1, 1, 2, 2],
                         "pair_locations": ["segment1_base", "segment1_joint", "segment2_joint", "segment2_tip"],
                         "pair_path_positions_m": list(positions), "acoustic_length_m": length,
                         "acoustic_geometry": "frozen_reference_pose",
                         "reference_angle_rad": rp.reference_angle_rad,
                         "active_actuator_ids": [ids[i] for i in active_drives],
                         "scalar_frf_enabled": len(active_drives) == 1,
                         "scalar_frf_reference": ids[active_drives[0]] if len(active_drives) == 1 else None}
    elif "actuator_forces_N" in sampled:
        raise ValueError("Independent actuator forces require routed_pairs.enabled=true")
    mechanics = mechanical_solver(t, sampled, config, initial_joint_state, tension_law)
    active = config.contact.enabled & (sampled["contact_force_N"] > config.contact.force_threshold_N)
    string = vibration_solver(t, mechanics["tension_N"], drives, active, config,
                              initial_string_displacement, initial_string_velocity, contact_boundary)
    theta = mechanics["joint_angle_rad"] - config.joint.equilibrium_rad
    tip = forward_kinematics(theta, config.joint.distal_length_m, config.joint.origin_xy_m)
    contact_xy = forward_kinematics(theta, config.contact.finger_lever_arm_m, config.joint.origin_xy_m)
    mechanics.update({"contact_active": active.astype(float),
                      "contact_tendon_position_m": np.where(active, config.contact.tendon_position_m, np.nan),
                      "contact_lever_arm_m": np.full(n, config.contact.finger_lever_arm_m),
                      "contact_world_x_m": np.where(active, contact_xy[:, 0], np.nan),
                      "contact_world_y_m": np.where(active, contact_xy[:, 1], np.nan),
                      "tip_world_x_m": tip[:, 0], "tip_world_y_m": tip[:, 1]})
    if config.routed_pairs.enabled:
        for i, name in enumerate(pair_metadata["pair_ids"]):
            mechanics[f"actuator_{name}_force_N"] = drives[:, i]
        # World positions follow actual finger motion; acoustic coordinates are
        # frozen separately at reference_angle_rad for this linear string model.
        tendon = Tendon.from_config(config)
        for i, point in enumerate(tendon.guide_positions(mechanics["joint_angle_rad"])):
            point = np.broadcast_to(point, (len(t), 2))
            mechanics[f"pair{i+1}_world_x_m"] = point[:, 0]
            mechanics[f"pair{i+1}_world_y_m"] = point[:, 1]
    measurement = Measurement(t, sensor_model(string, config.piezo), reference, mechanics)
    return SimulationResult(measurement, string.grid_m, string.displacement_m, string.velocity_m_s,
                            source_config.to_dict(), {**string.metadata, **pair_metadata, "schema_version": "1.0",
                                                "tension_mode": config.axial.mode,
                                                "mechanics_model": "routed_tendon",
                                                "tension_law_override": tension_law is not None,
                                                "parameters_are_calibrated": False})
