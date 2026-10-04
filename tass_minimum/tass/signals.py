"""Time-varying inputs are scalar callables: signal(time_s) -> value in SI."""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

Signal = Callable[[float], float]


@dataclass(frozen=True)
class Constant:
    value: float = 0.0
    def __call__(self, t):
        return self.value


@dataclass(frozen=True)
class Sine:
    amplitude: float
    frequency_Hz: float
    phase_rad: float = 0.0
    offset: float = 0.0
    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(2 * np.pi * self.frequency_Hz * t + self.phase_rad)


@dataclass(frozen=True)
class SmoothStep:
    initial: float
    final: float
    start_s: float
    end_s: float
    def __post_init__(self):
        if self.end_s <= self.start_s:
            raise ValueError("SmoothStep end must follow start")
    def __call__(self, t):
        a = np.clip((t - self.start_s) / (self.end_s - self.start_s), 0, 1)
        return self.initial + (self.final - self.initial) * (1 - np.cos(np.pi * a)) / 2


@dataclass(frozen=True)
class Pulse:
    amplitude: float = 0.002
    start_s: float = 0.002
    width_s: float = 0.0005
    def __post_init__(self):
        if self.width_s <= 0:
            raise ValueError("Pulse width must be positive")
    def __call__(self, t):
        a = (t - self.start_s) / self.width_s
        return self.amplitude * np.sin(np.pi * a)**2 if 0 <= a <= 1 else 0.0


@dataclass(frozen=True)
class Chirp:
    amplitude: float
    start_Hz: float
    end_Hz: float
    duration_s: float
    def __post_init__(self):
        if self.duration_s <= 0 or min(self.start_Hz, self.end_Hz) < 0:
            raise ValueError("Invalid chirp duration or frequency")
    def __call__(self, t):
        if not 0 <= t < self.duration_s:
            return 0.0
        rate = (self.end_Hz - self.start_Hz) / self.duration_s
        return self.amplitude * np.sin(2 * np.pi * (self.start_Hz*t + 0.5*rate*t*t))


@dataclass(frozen=True)
class SampledSignal:
    """Linear interpolation for recorded channels; extrapolation is rejected."""
    time_s: np.ndarray
    values: np.ndarray
    def __post_init__(self):
        t, y = np.asarray(self.time_s, float), np.asarray(self.values, float)
        if t.ndim != 1 or len(t) < 2 or y.shape != t.shape or not np.all(np.diff(t) > 0):
            raise ValueError("Expected matching 1D arrays with strictly increasing time")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
            raise ValueError("Sampled signal must be finite")
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "values", y)
    def __call__(self, t):
        if t < self.time_s[0] - 1e-12 or t > self.time_s[-1] + 1e-12:
            raise ValueError("Requested time outside recorded signal")
        return float(np.interp(t, self.time_s, self.values))


@dataclass(frozen=True)
class Inputs:
    motor_angle_rad: Signal = field(default_factory=lambda: Constant(0.12))
    excitation_N: Signal = field(default_factory=Pulse)
    contact_force_N: Signal = field(default_factory=Constant)
    tension_N: Signal | None = field(default_factory=lambda: Constant(4.0))

    # Explicit forces in N; omitted pairs are off. None uses the selected demo exciter.
    actuator_forces_N: dict[str, Signal] | None = None

    def sample(self, time_s):
        result = {}
        for name in ("motor_angle_rad", "excitation_N", "contact_force_N", "tension_N"):
            fn = getattr(self, name)
            if fn is None:
                continue
            values = np.array([fn(float(t)) for t in time_s], dtype=float)
            if values.shape != time_s.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must return finite scalar values")
            if name in {"contact_force_N", "tension_N"} and np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative under the V1 sign convention")
            result[name] = values
        if self.actuator_forces_N is not None:
            names = ("pair1", "pair2", "pair3", "pair4")
            if set(self.actuator_forces_N) - set(names):
                raise ValueError("Actuator IDs must be pair1, pair2, pair3 or pair4")
            drives = np.column_stack([
                [self.actuator_forces_N.get(name, Constant(0))(float(t)) for t in time_s]
                for name in names])
            if drives.shape != (len(time_s), 4) or not np.all(np.isfinite(drives)):
                raise ValueError("Actuator forces must be finite scalar signals")
            result["actuator_forces_N"] = drives
        return result
