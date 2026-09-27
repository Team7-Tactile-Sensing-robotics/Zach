"""Fixed-fixed modal approximation adapted from string-finger-simulator.

Preserves that project's empirical 1/(f-f_n+i*zeta*f_n) response. Amplitudes
are arbitrary units, not displacement per newton. No contact or transients.
"""
import numpy as np


class ModalString:
    def __init__(self, L=0.12, mu=0.0005, modes=12, damping_ratio=0.02):
        if not np.all(np.isfinite([L, mu, damping_ratio])) or min(L, mu, damping_ratio) <= 0:
            raise ValueError("Length, density and damping ratio must be positive and finite")
        if isinstance(modes, bool) or not isinstance(modes, (int, np.integer)) or modes < 1:
            raise ValueError("modes must be a positive integer")
        self.L, self.mu = float(L), float(mu)
        self.modes, self.damping_ratio = int(modes), float(damping_ratio)
        self.tension = 1.0

    def set_tension(self, T):
        if not np.isfinite(T) or T < 0:
            raise ValueError("Tension must be finite and nonnegative")
        self.tension = float(T)

    def natural_frequencies(self):
        return np.arange(1, self.modes + 1) / (2*self.L) * np.sqrt(self.tension/self.mu)

    def mode_shape(self, pos_ratio):
        if not np.isfinite(pos_ratio) or not 0 <= pos_ratio <= 1:
            raise ValueError("pos_ratio must be between 0 and 1")
        if pos_ratio in (0, 1):
            return np.zeros(self.modes)
        return np.sin(np.pi*np.arange(1, self.modes + 1)*pos_ratio)

    def modal_response_to_drive(self, drive_freq, drive_amp, actuator_pos_ratio):
        if not np.isfinite(drive_freq) or drive_freq < 0 or not np.isfinite(drive_amp):
            raise ValueError("Drive frequency must be nonnegative and drive parameters finite")
        shape = self.mode_shape(actuator_pos_ratio)
        if self.tension == 0:
            return np.zeros(self.modes, dtype=complex)
        frequencies = self.natural_frequencies()
        return complex(drive_amp)*shape / (drive_freq - frequencies + 1j*self.damping_ratio*frequencies)
