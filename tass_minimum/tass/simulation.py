"""Orchestration only: mechanics -> vibration -> sensors -> common measurement."""
import numpy as np
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
    mechanics = mechanical_solver(t, sampled, config, initial_joint_state, tension_law)
    active = config.contact.enabled & (sampled["contact_force_N"] > config.contact.force_threshold_N)
    string = vibration_solver(t, mechanics["tension_N"], sampled["excitation_N"], active, config,
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
    measurement = Measurement(t, sensor_model(string, config.piezo), sampled["excitation_N"], mechanics)
    return SimulationResult(measurement, string.grid_m, string.displacement_m, string.velocity_m_s,
                            config.to_dict(), {**string.metadata, "schema_version": "1.0",
                                              "tension_mode": config.axial.mode,
                                              "mechanics_model": "routed_tendon",
                                              "tension_law_override": tension_law is not None,
                                              "parameters_are_calibrated": False})
