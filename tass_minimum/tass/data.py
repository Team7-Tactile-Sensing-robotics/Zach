"""Hardware/simulator boundary: processing consumes only Measurement."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Measurement:
    time_s: np.ndarray
    piezo_V: np.ndarray
    excitation_N: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, float)
        self.piezo_V = np.asarray(self.piezo_V, float)
        self.excitation_N = np.asarray(self.excitation_N, float)
        t = self.time_s
        if t.ndim != 1 or len(t) < 4 or not np.all(np.isfinite(t)):
            raise ValueError("Expected at least four finite timestamps")
        delta = np.diff(t)
        if np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-5, atol=1e-12):
            raise ValueError("FFT requires uniformly sampled, increasing timestamps; resample hardware data first")
        if self.piezo_V.ndim != 2 or self.piezo_V.shape[0] != len(t) or self.piezo_V.shape[1] < 1:
            raise ValueError("piezo_V shape must be (samples, channels)")
        if self.excitation_N.shape != t.shape:
            raise ValueError("excitation_N must match timestamps")
        if not np.all(np.isfinite(self.piezo_V)) or not np.all(np.isfinite(self.excitation_N)):
            raise ValueError("Signals must be finite")
        for name, values in self.channels.items():
            if np.asarray(values).shape != t.shape:
                raise ValueError(f"Channel {name} must have one value per sample")

    @property
    def sample_rate_Hz(self):
        return 1.0 / (self.time_s[1] - self.time_s[0])


@dataclass
class SimulationResult:
    measurement: Measurement
    grid_m: np.ndarray
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    config: dict
    metadata: dict
