"""Flexion and spool winding are positive; positive external force opposes flexion."""
from typing import Protocol
import numpy as np
from scipy.integrate import solve_ivp
from ..config import ModelConfig
from .tendon import Tendon
from .spring import LinearReturnSpring


class TensionLaw(Protocol):
    def __call__(self, t: float, theta: float, omega: float,
                 displacement: float, velocity: float) -> float: ...


def motor_kinematics(time_s, angle_rad, spool_radius_m):
    s = spool_radius_m * np.asarray(angle_rad)
    v = np.gradient(s, time_s, edge_order=2)
    a = np.gradient(v, time_s, edge_order=2)
    return s, v, a


class MeasuredTension:
    def __init__(self, time_s, tension_N):
        self.time_s, self.tension_N = time_s, tension_N
    def __call__(self, t, theta, omega, displacement, velocity):
        return float(np.interp(t, self.time_s, self.tension_N))


class ElasticTension:
    def __init__(self, config: ModelConfig, tendon=None):
        self.config = config
        self.tendon = tendon if tendon is not None else Tendon.from_config(config)
    def __call__(self, t, theta, omega, displacement, velocity):
        return float(self.tendon.elastic_state(theta, omega, displacement, velocity)[2])


def joint_acceleration(theta, omega, tension, force, config, tendon=None, spring=None):
    j, contact = config.joint, config.contact
    tendon = tendon if tendon is not None else Tendon.from_config(config)
    spring = spring if spring is not None else LinearReturnSpring.from_config(config)
    return (tendon.joint_torque(theta, tension) + spring.joint_torque(theta)
            - j.damping_Nm_s_rad*omega - contact.finger_lever_arm_m*force) / j.inertia_kg_m2


def forward_kinematics(theta, distance_m, origin_xy_m):
    return np.column_stack((origin_xy_m[0] + distance_m*np.cos(theta),
                            origin_xy_m[1] + distance_m*np.sin(theta)))


def simulate_mechanics(time_s, sampled, config, initial_state=None, tension_law=None):
    tendon = Tendon.from_config(config)
    spring = LinearReturnSpring.from_config(config)
    s, v, a = motor_kinematics(time_s, sampled["motor_angle_rad"], config.motor.spool_radius_m)
    force = sampled["contact_force_N"]
    if tension_law is None:
        if config.axial.mode == "measured":
            if "tension_N" not in sampled:
                raise ValueError("Measured mode requires Inputs.tension_N")
            tension_law = MeasuredTension(time_s, sampled["tension_N"])
        else:
            tension_law = ElasticTension(config, tendon)

    def tension_at(t, theta, omega):
        value = tension_law(t, theta, omega, np.interp(t, time_s, s), np.interp(t, time_s, v))
        if not np.isfinite(value) or value < 0:
            raise ValueError("Tension law must return finite nonnegative tension")
        return value

    def rhs(t, state):
        theta, omega = state
        T = tension_at(t, theta, omega)
        return [omega, joint_acceleration(theta, omega, T, np.interp(t, time_s, force), config, tendon, spring)]

    initial = [config.joint.equilibrium_rad, 0.0] if initial_state is None else initial_state
    sol = solve_ivp(rhs, (time_s[0], time_s[-1]), initial, t_eval=time_s,
                    rtol=config.numerics.joint_rtol, atol=config.numerics.joint_atol,
                    max_step=config.numerics.joint_max_step_s, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    theta, omega = sol.y
    tension = np.array([tension_at(t, th, w) for t, th, w in zip(time_s, theta, omega)])
    segments = tendon.path_segments(theta)
    return {"motor_angle_rad": sampled["motor_angle_rad"], "tendon_displacement_m": s,
            "tendon_velocity_m_s": v, "tendon_acceleration_m_s2": a,
            "joint_angle_rad": theta, "joint_velocity_rad_s": omega,
            "joint_acceleration_rad_s2": joint_acceleration(theta, omega, tension, force, config, tendon, spring),
            "load_cell_force_N": tension.copy(),
            **{f"spring_{name}": value for name, value in spring.evaluate(theta).items()},
            "tendon_path_length_m": sum(segments.values()),
            "tendon_shortening_m": tendon.joint_displacement(theta),
            "tendon_moment_arm_m": -tendon.path_length_derivative(theta),
            "tendon_torque_Nm": tendon.joint_torque(theta, tension),
            **{f"tendon_{name}_length_m": length for name, length in segments.items()},
            "tension_N": tension, "contact_force_N": force}
