"""Editable, serializable physical parameters; demonstration values, not calibration."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import math
import warnings


@dataclass(frozen=True)
class Motor:
    spool_radius_m: float = 0.015


@dataclass(frozen=True)
class Joint:
    inertia_kg_m2: float = 2.0e-5
    damping_Nm_s_rad: float = 0.004
    equilibrium_rad: float = 0.0
    distal_length_m: float = 0.1
    origin_xy_m: tuple[float, float] = (0.15, 0.0)


@dataclass(frozen=True)
class TendonRouting:
    """World coordinates at the straight pose; P1..P4 guide the tendon to a load cell."""
    spool_position_m: tuple[float, float] = (0.0, 0.0)
    p1_m: tuple[float, float] = (0.040, 0.005)
    p2_m: tuple[float, float] = (0.120, 0.005)
    p3_m: tuple[float, float] = (0.175, 0.005)
    p4_m: tuple[float, float] = (0.230, 0.005)
    load_cell_m: tuple[float, float] = (0.250, 0.005)


@dataclass(frozen=True)
class ReturnSpring:
    """Linear extension spring; joint-relative attachment offsets at the straight pose."""
    stiffness_N_m: float = 500.0
    free_length_m: float = 0.024
    fixed_offset_m: tuple[float, float] = (-0.012, -0.020)
    moving_offset_m: tuple[float, float] = (0.012, -0.020)


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
class ModelConfig:
    motor: Motor = field(default_factory=Motor)
    joint: Joint = field(default_factory=Joint)
    axial: AxialTendon = field(default_factory=AxialTendon)
    tendon_routing: TendonRouting = field(default_factory=TendonRouting)
    return_spring: ReturnSpring = field(default_factory=ReturnSpring)
    string: String = field(default_factory=String)
    contact: Contact = field(default_factory=Contact)
    piezo: Piezo = field(default_factory=Piezo)
    numerics: Numerics = field(default_factory=Numerics)

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
        for name, value in [("joint damping", self.joint.damping_Nm_s_rad),
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
        from .models.spring import LinearReturnSpring
        LinearReturnSpring.from_config(self)
        n = len(self.piezo.positions_m)
        if n < 1 or len(self.piezo.gains) != n or len(self.piezo.noise_std_V) != n:
            raise ValueError("Piezo positions, gains and noise arrays must have equal nonzero length")
        if any(not 0 <= x <= self.string.length_m for x in self.piezo.positions_m):
            raise ValueError("Piezo positions must lie on string")
        if not all(math.isfinite(g) for g in self.piezo.gains):
            raise ValueError("Piezo gains must be finite")
        for s in self.piezo.noise_std_V:
            positive("noise standard deviation", s, True)
        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        constructors = {"motor": Motor, "joint": Joint, "axial": AxialTendon,
                        "tendon_routing": TendonRouting, "return_spring": ReturnSpring,
                        "string": String, "contact": Contact, "piezo": Piezo, "numerics": Numerics}
        unknown = values.keys() - constructors.keys()
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        if "moment_arm_m" in values.get("joint", {}):
            values["joint"].pop("moment_arm_m")
            warnings.warn("joint.moment_arm_m is obsolete; configure tendon_routing for the new geometry model.",
                          UserWarning, stacklevel=2)
        if "stiffness_Nm_rad" in values.get("joint", {}):
            values["joint"].pop("stiffness_Nm_rad")
            warnings.warn("joint.stiffness_Nm_rad is obsolete; configure the linear return_spring and its attachments.",
                          UserWarning, stacklevel=2)
        return cls(**{key: constructors[key](**value) for key, value in values.items()}).validate()
