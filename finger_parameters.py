"""EDIT THIS FILE to configure finger_physical_app.py.

All values below are demonstration values, not prototype measurements.
Save changes, then rerun/refresh Streamlit. The app resets when settings change.

Units: metres, seconds, kilograms, newtons and pascals; angles marked _deg use
degrees. In the straight pose +x runs along the finger and +y is the tendon
side. Segment 1 starts at (0, 0); the joint is at (proximal_length_m, 0).
P1..P4 are the piezo/idler GUIDE CENTRES, not tendon attachment points.

Positions are independent measurements: changing a segment length does not
rescale the guide offsets, load-cell offset, or measured joint inertia. Adjust
those fields here as well when changing the hardware layout.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FingerParameters:
    # ── Finger segments and joint ──────────────────────────────────────────
    proximal_length_m: float = 0.150          # Segment 1: base to joint
    distal_length_m: float = 0.100            # Segment 2: joint to fingertip
    joint_inertia_kg_m2: float = 2e-4         # Measured/effective distal inertia
    joint_damping_Nm_s_rad: float = 0.002
    joint_limit_deg: float = 100.0           # Straight stop is 0 degrees

    # ── Piezo / idler-tube placement ────────────────────────────────────────
    # P1 and P2: (x, y) measured from the BASE, on fixed segment 1.
    tendon_p1_m: tuple[float, float] = (0.040, 0.005)
    tendon_p2_m: tuple[float, float] = (0.120, 0.005)
    # P3 and P4: (x, y) measured from the JOINT, on rotating segment 2.
    tendon_p3_offset_m: tuple[float, float] = (0.025, 0.005)
    tendon_p4_offset_m: tuple[float, float] = (0.080, 0.005)

    # ── Load-cell tendon attachment ────────────────────────────────────────
    # Joint-relative position on segment 2. The tendon continues beyond P4.
    load_cell_offset_m: tuple[float, float] = (0.100, 0.005)

    # ── Tendon material and axial mechanics ────────────────────────────────
    # Axial stiffness k = E*A/L0. These defaults preserve the previous 2500 N/m.
    tendon_young_modulus_Pa: float = 1.25e9
    tendon_area_m2: float = 5e-7             # Cross-sectional area, not diameter
    tendon_rest_length_m: float = 0.250     # Effective compliant length for EA/L0
    tendon_damping_Ns_m: float = 0.1
    tendon_slack_m: float = 0.0
    tendon_preload_N: float = 0.0
    # Optional measured stiffness. Leave None to use E, A and L0 above.
    tendon_stiffness_override_N_m: float | None = None

    # ── Linear extension spring and brackets ───────────────────────────────
    spring_stiffness_N_m: float = 500.0
    spring_free_length_m: float = 0.024
    # Joint-relative offsets in the straight pose. Negative y is opposite the
    # tendon. The first mount stays fixed; the second rotates with segment 2.
    spring_fixed_offset_m: tuple[float, float] = (-0.012, -0.020)
    spring_moving_offset_m: tuple[float, float] = (0.012, -0.020)
    # Spring preload follows from free length vs. initial bracket separation.

    # ── Motor and spool ────────────────────────────────────────────────────
    spool_radius_m: float = 0.006
    spool_position_m: tuple[float, float] = (0.0, 0.0)  # From base
    servo_time_constant_s: float = 0.10
    servo_speed_limit_deg_s: float = 300.0
    motor_limit_deg: float = 180.0          # Lower motor limit is 0 degrees

    # ── Simulation and controls (not hardware measurements) ────────────────
    max_step_s: float = 0.0005
    display_interval_s: float = 0.1
    max_frame_simulation_s: float = 0.1     # Limit catch-up after inactive tabs
    motor_button_step_deg: float = 10.0
    motor_slider_step_deg: float = 1.0
    history_samples: int = 240

    # Everything below is derived/validation code; edit the values above.
    @property
    def tendon_stiffness_N_m(self):
        if self.tendon_stiffness_override_N_m is not None:
            return self.tendon_stiffness_override_N_m
        return self.tendon_young_modulus_Pa*self.tendon_area_m2/self.tendon_rest_length_m

    def __post_init__(self):
        positive = (
            'proximal_length_m', 'distal_length_m', 'joint_inertia_kg_m2',
            'joint_limit_deg', 'tendon_young_modulus_Pa', 'tendon_area_m2',
            'tendon_rest_length_m', 'spring_free_length_m', 'spool_radius_m',
            'servo_time_constant_s', 'servo_speed_limit_deg_s', 'motor_limit_deg',
            'max_step_s', 'display_interval_s', 'max_frame_simulation_s',
            'motor_button_step_deg', 'motor_slider_step_deg', 'tendon_stiffness_N_m',
        )
        nonnegative = ('joint_damping_Nm_s_rad', 'tendon_damping_Ns_m',
                       'tendon_slack_m', 'tendon_preload_N', 'spring_stiffness_N_m')
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f'{name} must be finite and ' + ('positive' if name in positive else 'nonnegative'))
        for name in ('tendon_p1_m', 'tendon_p2_m', 'tendon_p3_offset_m',
                     'tendon_p4_offset_m', 'load_cell_offset_m', 'spring_fixed_offset_m',
                     'spring_moving_offset_m', 'spool_position_m'):
            point = getattr(self, name)
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError(f'{name} must contain two finite coordinates in metres')
        if self.motor_slider_step_deg > self.motor_limit_deg:
            raise ValueError('motor_slider_step_deg must not exceed motor_limit_deg')
        if type(self.history_samples) is not int or self.history_samples < 2:
            raise ValueError('history_samples must be an integer of at least 2')
