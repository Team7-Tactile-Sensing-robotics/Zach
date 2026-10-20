"""EDIT THIS FILE to configure finger_physical_app.py.

Geometry marked CAD comes from TASS_assy_1.STEP (millimetres converted to SI).
See tass_minimum/docs/cad_measurements.md for datums and routing assumptions.
Material, inertia, damping, spring and motor-response values remain uncalibrated.
Save changes, then rerun/refresh Streamlit. The app resets when settings change.

Units: metres, seconds, kilograms, newtons and pascals; angles marked _deg use
degrees. In the straight pose +x runs along the finger and +y is the tendon
side. Segment 1 starts at (0, 0); the joint is at (proximal_length_m, 0).
P1..P4 are centres of paired piezo/idler stations, not tendon attachment points.

Positions are independent measurements: changing a segment length does not
rescale the guide offsets, load-cell offset, or measured joint inertia. Adjust
those fields here as well when changing the hardware layout.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FingerParameters:
    # ── Finger segments and joint ──────────────────────────────────────────
    proximal_length_m: float = 0.187          # CAD: base top to hinge axis
    distal_length_m: float = 0.187            # CAD: hinge to segment end, excluding load cell
    joint_inertia_kg_m2: float = 2e-4         # Unmeasured effective distal inertia
    joint_damping_Nm_s_rad: float = 0.002
    joint_limit_deg: float = 100.0           # Straight stop is 0 degrees

    # ── Piezo / idler-tube placement ────────────────────────────────────────
    # P1 and P2: (x, y) measured from the BASE, on fixed segment 1.
    # CAD: mean of the two tube centres at each station.
    tendon_p1_m: tuple[float, float] = (0.031, 0.01553607)
    tendon_p2_m: tuple[float, float] = (0.147, 0.01553607)
    # P3 and P4: (x, y) measured from the JOINT, on rotating segment 2.
    # Inferred from CAD mounting holes, reusing the proximal bracket pairs.
    tendon_p3_offset_m: tuple[float, float] = (0.040, 0.01553607)
    tendon_p4_offset_m: tuple[float, float] = (0.164, 0.01553607)

    # ── Load-cell tendon attachment ────────────────────────────────────────
    # Joint-relative position on segment 2. The tendon continues beyond P4.
    # User-confirmed outer free-end hole, at its CAD finger-facing surface; attachment hardware
    # is absent, so the actual knot/eyelet offset still needs measurement.
    load_cell_offset_m: tuple[float, float] = (0.188, 0.01668574)

    # ── Tendon material and axial mechanics ────────────────────────────────
    # Axial stiffness k = E*A/L0. These defaults preserve the previous 2500 N/m.
    tendon_young_modulus_Pa: float = 1.25e9
    tendon_area_m2: float = 5e-7             # Cross-sectional area, not diameter
    tendon_rest_length_m: float = 0.250     # Unmeasured compliant length for EA/L0, NOT CAD route length
    tendon_damping_Ns_m: float = 0.1
    tendon_slack_m: float = 0.0
    tendon_preload_N: float = 0.0
    # Optional measured stiffness. Leave None to use E, A and L0 above.
    tendon_stiffness_override_N_m: float | None = None

    # ── Linear extension spring and brackets ───────────────────────────────
    spring_stiffness_N_m: float = 500.0
    spring_free_length_m: float = 0.046     # ASSUMED zero preload at straight; no spring in CAD
    # Joint-relative offsets in the straight pose. Negative y is opposite the
    # tendon. The first mount stays fixed; the second rotates with segment 2.
    # CAD transverse spring-mount hole axes, projected into the finger plane.
    spring_fixed_offset_m: tuple[float, float] = (-0.023, -0.02930940)
    spring_moving_offset_m: tuple[float, float] = (0.023, -0.02930940)
    # Spring preload follows from free length vs. initial bracket separation.

    # ── Motor and spool ────────────────────────────────────────────────────
    spool_radius_m: float = 0.030           # CAD winding drum, not the 33 mm flange
    spool_position_m: tuple[float, float] = (0.039110, 0.065)  # CAD centre from base
    spool_feed_branch: int = -1            # -1: lower tangent to P1; +1: upper tangent
    servo_time_constant_s: float = 0.10
    servo_speed_limit_deg_s: float = 300.0
    motor_limit_deg: float = 180.0          # Lower motor limit is 0 degrees

    # ── Acoustics: one exciter/receiver pair at each P1..P4 guide ────────────
    acoustic_linear_density_kg_m: float = 0.0005
    acoustic_damping_ratio: float = 0.02   # Modal damping, separate from joint damping
    acoustic_modes: int = 16
    acoustic_sample_rate_Hz: float = 24000.0
    acoustic_block_s: float = 0.01         # Acoustic time per screen refresh (slow playback)
    acoustic_refresh_interval_s: float = 0.5  # Plot refresh; independent of mechanical refresh
    acoustic_history_s: float = 0.20
    acoustic_min_tension_N: float = 1e-5    # Suspend taut-string model below this tension
    acoustic_drive_frequencies_Hz: tuple[float, ...] = (250.0, 250.0, 250.0, 250.0)
    acoustic_drive_amplitudes_N: tuple[float, ...] = (0.001, 0.001, 0.001, 0.001)
    acoustic_receiver_gains_V_m: tuple[float, ...] = (1000.0, 1000.0, 1000.0, 1000.0)
    acoustic_deflection_magnification: float = 1000.0
    # Held contact: demonstration tangent stiffness/damping per newton of preload.
    # This is an acoustic boundary law, not a calibrated force estimator.
    acoustic_contact_position_ratio: float = 0.50  # 0 = spool, 1 = load cell
    acoustic_contact_force_N: float = 0.50
    acoustic_contact_max_force_N: float = 5.0
    acoustic_contact_stiffness_per_N: float = 2000.0  # (N/m)/N = 1/m
    acoustic_contact_damping_per_N: float = 0.05     # (N*s/m)/N = s/m

    # ── Simulation and controls (not hardware measurements) ────────────────
    max_step_s: float = 0.0005
    display_interval_s: float = 0.1
    max_frame_simulation_s: float = 0.1     # Limit catch-up after inactive tabs
    motor_button_step_deg: float = 10.0
    motor_slider_step_deg: float = 1.0
    history_samples: int = 240

    # Everything below is derived/validation code; edit the values above.
    @property
    def spool_feed_position_m(self):
        """Tangent exit on the drum toward fixed P1 (constant during flexion).

        For the supplied layout, -1 selects the exit nearest the base. Tendon
        thickness, changing wrap radius and the finite tube shapes are omitted.
        """
        cx, cy = self.spool_position_m
        dx, dy = self.tendon_p1_m[0]-cx, self.tendon_p1_m[1]-cy
        distance_squared = dx*dx + dy*dy
        radius = self.spool_radius_m
        if distance_squared <= radius*radius:
            raise ValueError('tendon_p1_m must lie outside the spool_radius_m circle')
        radial = radius*radius/distance_squared
        tangent = self.spool_feed_branch*radius*math.sqrt(distance_squared-radius*radius)/distance_squared
        return (cx+radial*dx-tangent*dy, cy+radial*dy+tangent*dx)

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
        positive += ('acoustic_linear_density_kg_m', 'acoustic_sample_rate_Hz',
                     'acoustic_block_s', 'acoustic_refresh_interval_s',
                     'acoustic_history_s', 'acoustic_min_tension_N',
                     'acoustic_deflection_magnification', 'acoustic_contact_max_force_N')
        nonnegative = ('acoustic_damping_ratio', 'joint_damping_Nm_s_rad', 'tendon_damping_Ns_m',
                       'tendon_slack_m', 'tendon_preload_N', 'spring_stiffness_N_m',
                       'acoustic_contact_force_N', 'acoustic_contact_stiffness_per_N',
                       'acoustic_contact_damping_per_N')
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
        if type(self.spool_feed_branch) is not int or self.spool_feed_branch not in (-1, 1):
            raise ValueError('spool_feed_branch must be -1 or +1')
        self.spool_feed_position_m  # Validate the tangent geometry before constructing the model.
        if type(self.history_samples) is not int or self.history_samples < 2:
            raise ValueError('history_samples must be an integer of at least 2')
        if type(self.acoustic_modes) is not int or not 1 <= self.acoustic_modes <= 128:
            raise ValueError('acoustic_modes must be an integer from 1 to 128')
        for name in ('acoustic_drive_frequencies_Hz', 'acoustic_drive_amplitudes_N',
                     'acoustic_receiver_gains_V_m'):
            values = getattr(self, name)
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                raise ValueError(f'{name} must contain four finite values')
        if any(f < 1 or f >= 0.4*self.acoustic_sample_rate_Hz
               for f in self.acoustic_drive_frequencies_Hz):
            raise ValueError('acoustic_drive_frequencies_Hz must be >= 1 and below 0.4 times the sample rate')
        if any(a < 0 for a in self.acoustic_drive_amplitudes_N):
            raise ValueError('acoustic_drive_amplitudes_N must be nonnegative')
        if round(self.acoustic_block_s*self.acoustic_sample_rate_Hz) < 4:
            raise ValueError('acoustic_block_s must contain at least four samples')
        if self.acoustic_history_s < self.acoustic_block_s:
            raise ValueError('acoustic_history_s must be at least acoustic_block_s')
        if (not math.isfinite(self.acoustic_contact_position_ratio)
                or not 0 <= self.acoustic_contact_position_ratio <= 1):
            raise ValueError('acoustic_contact_position_ratio must be between 0 and 1')
        if self.acoustic_contact_force_N > self.acoustic_contact_max_force_N:
            raise ValueError('acoustic_contact_force_N exceeds acoustic_contact_max_force_N')
