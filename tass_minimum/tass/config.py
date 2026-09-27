"""Editable, serializable physical parameters; demonstration values, not calibration."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import math
import warnings
import yaml


@dataclass(frozen=True)
class Motor:
    spool_radius_m: float = 0.015


@dataclass(frozen=True)
class Joint:
    inertia_kg_m2: float = 2.0e-5
    stiffness_Nm_rad: float = 0.25
    damping_Nm_s_rad: float = 0.004
    equilibrium_rad: float = 0.0
    distal_length_m: float = 0.1
    origin_xy_m: tuple[float, float] = (0.15, 0.0)


@dataclass(frozen=True)
class TendonRouting:
    """World coordinates at the straight reference angle; P4 anchors the tendon."""
    spool_position_m: tuple[float, float] = (0.0, 0.0)
    p1_m: tuple[float, float] = (0.040, 0.005)
    p2_m: tuple[float, float] = (0.120, 0.005)
    p3_m: tuple[float, float] = (0.175, 0.005)
    p4_m: tuple[float, float] = (0.230, 0.005)


@dataclass(frozen=True)
class AxialTendon:
    mode: str = "measured"
    young_modulus_Pa: float = 2.0e9
    area_m2: float = 5.0e-7
    rest_length_m: float = 0.1
    damping_Ns_m: float = 0.05
    preload_N: float = 0.0
    slack_m: float = 0.0

    @property
    def stiffness_N_m(self):
        return self.young_modulus_Pa * self.area_m2 / self.rest_length_m


@dataclass(frozen=True)
class String:
    length_m: float = 0.12
    linear_density_kg_m: float = 0.002
    distributed_damping_Ns_m2: float = 0.008
    nodes: int = 81
    exciter_position_m: float = 0.024


@dataclass(frozen=True)
class Contact:
    enabled: bool = True
    finger_lever_arm_m: float = 0.04
    tendon_position_m: float = 0.048
    force_threshold_N: float = 0.0


@dataclass(frozen=True)
class Piezo:
    positions_m: tuple[float, ...] = (0.03, 0.09)
    mode: str = "displacement"
    gains: tuple[float, ...] = (1000.0, 1000.0)
    noise_std_V: tuple[float, ...] = (0.0, 0.0)
    seed: int = 7


@dataclass(frozen=True)
class Numerics:
    sample_rate_Hz: float = 48000.0
    duration_s: float = 0.25
    joint_rtol: float = 1e-8
    joint_atol: float = 1e-10
    joint_max_step_s: float = 0.001
    string_stability_factor: float = 0.8


@dataclass(frozen=True)
class ModalSweep:
    joint_angle_deg: float = 0.0
    motor_angle_rad: float = 0.12
    tension_N: float | None = None
    start_Hz: float = 20.0
    stop_Hz: float = 2000.0
    step_Hz: float = 2.0
    modes: int = 12
    damping_ratio: float = 0.02


@dataclass(frozen=True)
class Experiment:
    scenario: str = "baseline"
    motor_initial_rad: float = 0.12
    motor_final_rad: float = 0.18
    measured_tension_N: float = 4.0
    contact_force_N: float = 0.4
    excitation_amplitude_N: float = 0.002
    pulse_start_s: float = 0.002
    pulse_width_s: float = 0.0005
    chirp_start_Hz: float = 80.0
    chirp_end_Hz: float = 1800.0
    motor_ramp_start_fraction: float = 0.05
    motor_ramp_end_fraction: float = 0.4
    contact_ramp_start_fraction: float = 0.55
    contact_ramp_end_fraction: float = 0.65


@dataclass(frozen=True)
class ModalPiezo:
    """Four idealized actuator/receiver pairs on tendon guides; empirical units."""
    pair_ids: tuple[str, ...] = ("pair1", "pair2", "pair3", "pair4")
    guide_numbers: tuple[int, ...] = (1, 2, 3, 4)
    actuator_amplitudes_au: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    actuator_phases_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    sensor_gains: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Analysis:
    start_s: float | None = None
    end_s: float | None = None
    window: str = "hann"
    max_peaks: int = 5
    frf_floor_relative: float = 0.001


@dataclass(frozen=True)
class App:
    """Defaults for the independent interactive mechanics demonstrator."""
    motor_angle_deg: float = 90.0
    segment1_length_mm: float = 150.0
    segment2_length_mm: float = 100.0
    inertia_kg_m2: float = 0.0002
    joint_damping_Nm_s_rad: float = 0.002
    spring_stiffness_Nm_rad: float = 0.08
    guide_offset_mm: float = 5.0
    p1_fraction: float = 4/15
    p2_percent: int = 80
    p3_percent: int = 25
    p4_fraction: float = 0.8
    spool_radius_mm: float = 6.0
    tendon_stiffness_N_m: float = 2500.0
    tendon_rest_length_m: float = 0.25
    tendon_damping_Ns_m: float = 0.1
    slack_mm: float = 0.0
    preload_N: float = 0.0
    servo_time_constant_s: float = 0.1
    servo_speed_deg_s: float = 300.0
    contact_force_N: float = 0.0
    contact_percent: int = 70
    joint_max_deg: float = 100.0
    duration_s: float = 2.0
    time_step_s: float = 0.0005
    equilibrium_grid_points: int = 5001
    table_motor_angles_deg: tuple[float, ...] = (0, 30, 60, 90, 120, 150, 180)


@dataclass(frozen=True)
class ModelConfig:
    motor: Motor = field(default_factory=Motor)
    joint: Joint = field(default_factory=Joint)
    axial: AxialTendon = field(default_factory=AxialTendon)
    tendon_routing: TendonRouting = field(default_factory=TendonRouting)
    string: String = field(default_factory=String)
    contact: Contact = field(default_factory=Contact)
    piezo: Piezo = field(default_factory=Piezo)
    numerics: Numerics = field(default_factory=Numerics)

    modal_sweep: ModalSweep = field(default_factory=ModalSweep)
    experiment: Experiment = field(default_factory=Experiment)

    modal_piezo: ModalPiezo = field(default_factory=ModalPiezo)
    analysis: Analysis = field(default_factory=Analysis)
    app: App = field(default_factory=App)

    def validate(self):
        def positive(name, value, allow_zero=False):
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
        for name, value in [("spool radius", self.motor.spool_radius_m),
                            ("inertia", self.joint.inertia_kg_m2),
                            ("finger length", self.joint.distal_length_m),
                            ("E", self.axial.young_modulus_Pa), ("area", self.axial.area_m2),
                            ("rest length", self.axial.rest_length_m),
                            ("string length", self.string.length_m),
                            ("linear density", self.string.linear_density_kg_m),
                            ("sample rate", self.numerics.sample_rate_Hz),
                            ("duration", self.numerics.duration_s),
                            ("rtol", self.numerics.joint_rtol), ("atol", self.numerics.joint_atol),
                            ("joint max step", self.numerics.joint_max_step_s)]:
            positive(name, value)
        for name, value in [("spring stiffness", self.joint.stiffness_Nm_rad),
                            ("joint damping", self.joint.damping_Nm_s_rad),
                            ("axial damping", self.axial.damping_Ns_m),
                            ("preload", self.axial.preload_N),
                            ("initial slack", self.axial.slack_m),
                            ("string damping", self.string.distributed_damping_Ns_m2),
                            ("contact threshold", self.contact.force_threshold_N)]:
            positive(name, value, True)
        if self.axial.mode not in {"measured", "simulated"}:
            raise ValueError("axial.mode must be measured or simulated")
        if self.piezo.mode not in {"displacement", "velocity"}:
            raise ValueError("piezo.mode must be displacement or velocity")
        if not isinstance(self.string.nodes, int) or self.string.nodes < 5:
            raise ValueError("string.nodes must be an integer >= 5")
        if not 0 < self.numerics.string_stability_factor <= 1:
            raise ValueError("string_stability_factor must be in (0, 1]")
        if round(self.numerics.duration_s * self.numerics.sample_rate_Hz) < 4:
            raise ValueError("At least four output samples are required")
        if not 0 < self.string.exciter_position_m < self.string.length_m:
            raise ValueError("Exciter must be inside the string")
        if not 0 < self.contact.tendon_position_m < self.string.length_m:
            raise ValueError("Contact must be inside the string")
        if not 0 <= self.contact.finger_lever_arm_m <= self.joint.distal_length_m:
            raise ValueError("Contact lever arm must lie along distal finger")
        if len(self.joint.origin_xy_m) != 2 or not all(math.isfinite(x) for x in (*self.joint.origin_xy_m, self.joint.equilibrium_rad)):
            raise ValueError("Joint origin and equilibrium must be finite")
        from .models.tendon import Tendon
        Tendon.from_config(self)  # Validate routing coordinates and reference span.
        n = len(self.piezo.positions_m)
        if n < 1 or len(self.piezo.gains) != n or len(self.piezo.noise_std_V) != n:
            raise ValueError("Piezo positions, gains and noise arrays must have equal nonzero length")
        if any(not 0 <= x <= self.string.length_m for x in self.piezo.positions_m):
            raise ValueError("Piezo positions must lie on string")
        if not all(math.isfinite(g) for g in self.piezo.gains):
            raise ValueError("Piezo gains must be finite")
        for s in self.piezo.noise_std_V:
            positive("noise standard deviation", s, True)
        m, e = self.modal_sweep, self.experiment
        for name in ("joint_angle_deg", "motor_angle_rad"):
            if not math.isfinite(getattr(m, name)):
                raise ValueError(f"modal_sweep.{name} must be finite")
        for name in ("start_Hz", "stop_Hz"):
            positive(f"modal_sweep.{name}", getattr(m, name), True)
        for name in ("step_Hz", "damping_ratio"):
            positive(f"modal_sweep.{name}", getattr(m, name))
        if m.stop_Hz < m.start_Hz:
            raise ValueError("modal_sweep.stop_Hz must be >= start_Hz")
        if isinstance(m.modes, bool) or not isinstance(m.modes, int) or m.modes < 1:
            raise ValueError("modal_sweep.modes must be a positive integer")
        if m.tension_N is not None:
            positive("modal_sweep.tension_N", m.tension_N, True)
        if e.scenario not in {"baseline", "contact", "coupled"}:
            raise ValueError("experiment.scenario must be baseline, contact or coupled")
        for name in ("motor_initial_rad", "motor_final_rad", "excitation_amplitude_N"):
            if not math.isfinite(getattr(e, name)):
                raise ValueError(f"experiment.{name} must be finite")
        for name in ("measured_tension_N", "contact_force_N", "pulse_start_s", "chirp_start_Hz", "chirp_end_Hz"):
            positive(f"experiment.{name}", getattr(e, name), True)
        positive("experiment.pulse_width_s", e.pulse_width_s)
        for prefix in ("motor", "contact"):
            start = getattr(e, f"{prefix}_ramp_start_fraction")
            end = getattr(e, f"{prefix}_ramp_end_fraction")
            if not 0 <= start < end <= 1:
                raise ValueError(f"experiment.{prefix} ramp fractions require 0 <= start < end <= 1")
        p = self.modal_piezo
        if len(p.pair_ids) != 4 or any(not isinstance(x, str) or not x for x in p.pair_ids) or len(set(p.pair_ids)) != 4:
            raise ValueError("modal_piezo requires four unique nonempty pair IDs")
        if len(p.guide_numbers) != 4 or any(type(x) is not int for x in p.guide_numbers) or set(p.guide_numbers) != {1, 2, 3, 4}:
            raise ValueError("modal_piezo.guide_numbers must be a permutation of 1,2,3,4")
        for name in ("actuator_amplitudes_au", "actuator_phases_rad", "sensor_gains"):
            values = getattr(p, name)
            if len(values) != 4 or not all(math.isfinite(x) for x in values):
                raise ValueError(f"modal_piezo.{name} requires four finite values")
        a = self.analysis
        if a.window not in {"hann", "boxcar"} or type(a.max_peaks) is not int or a.max_peaks < 1 or not 0 < a.frf_floor_relative < 1:
            raise ValueError("Invalid analysis window, peak count or FRF threshold")
        for name in ("start_s", "end_s"):
            if getattr(a, name) is not None:
                positive(f"analysis.{name}", getattr(a, name), True)
        if a.start_s is not None and a.end_s is not None and a.end_s <= a.start_s:
            raise ValueError("analysis.end_s must exceed start_s")
        app = self.app
        ranges = {
            "motor_angle_deg": (0, 180), "segment1_length_mm": (50, 150), "segment2_length_mm": (50, 150),
            "inertia_kg_m2": (1e-6, 0.01), "joint_damping_Nm_s_rad": (0, 0.1),
            "spring_stiffness_Nm_rad": (0.001, 2), "guide_offset_mm": (1, 20),
            "p1_fraction": (0, 1), "p2_percent": (40, 95), "p3_percent": (5, 70), "p4_fraction": (0, 1),
            "spool_radius_mm": (2, 20), "tendon_stiffness_N_m": (50, 20000), "slack_mm": (0, 5),
            "servo_time_constant_s": (0.02, 0.5), "servo_speed_deg_s": (30, 1000),
            "contact_force_N": (0, 10), "contact_percent": (10, 100), "joint_max_deg": (30, 120),
            "duration_s": (0.5, 5)}
        for name, (low, high) in ranges.items():
            if not low <= getattr(app, name) <= high:
                raise ValueError(f"app.{name} must be in [{low}, {high}]")
        for name in ("p2_percent", "p3_percent", "contact_percent"):
            if type(getattr(app, name)) is not int:
                raise ValueError(f"app.{name} must be an integer")
        positive("app.tendon_rest_length_m", app.tendon_rest_length_m)
        positive("app.tendon_damping_Ns_m", app.tendon_damping_Ns_m, True)
        positive("app.preload_N", app.preload_N, True)
        positive("app.time_step_s", app.time_step_s)
        if app.time_step_s > app.duration_s:
            raise ValueError("app.time_step_s must not exceed duration_s")
        if type(app.equilibrium_grid_points) is not int or app.equilibrium_grid_points < 2:
            raise ValueError("app.equilibrium_grid_points must be an integer >= 2")
        if not app.table_motor_angles_deg or not all(math.isfinite(x) and 0 <= x <= 180 for x in app.table_motor_angles_deg):
            raise ValueError("app.table_motor_angles_deg must contain finite angles in [0, 180]")
        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        values = self.to_dict()
        if path.suffix.lower() in {".yaml", ".yml"}:
            path.write_text(yaml.safe_dump(values, sort_keys=False))
        else:
            path.write_text(json.dumps(values, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        values = (yaml.safe_load(path.read_text()) if path.suffix.lower() in {".yaml", ".yml"}
                  else json.loads(path.read_text()))
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a mapping of named sections")
        constructors = {"motor": Motor, "joint": Joint, "axial": AxialTendon,
                        "tendon_routing": TendonRouting,
                        "string": String, "contact": Contact, "piezo": Piezo, "numerics": Numerics,
                        "modal_sweep": ModalSweep, "experiment": Experiment,
                        "modal_piezo": ModalPiezo, "analysis": Analysis, "app": App}
        unknown = values.keys() - constructors.keys()
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        for key, value in values.items():
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section {key} must be a mapping")
            unknown_fields = value.keys() - constructors[key].__dataclass_fields__.keys()
            if key == "joint":
                unknown_fields -= {"moment_arm_m"}
            if unknown_fields:
                raise ValueError(f"Unknown fields in {key}: {sorted(unknown_fields)}")
        if "moment_arm_m" in values.get("joint", {}):
            values["joint"].pop("moment_arm_m")
            warnings.warn("joint.moment_arm_m is obsolete; configure tendon_routing for the new geometry model.",
                          UserWarning, stacklevel=2)
        return cls(**{key: constructors[key](**value) for key, value in values.items()}).validate()
