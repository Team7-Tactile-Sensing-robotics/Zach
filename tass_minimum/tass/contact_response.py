"""Small-signal contact around a preloaded equilibrium; coefficients need calibration."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContactResponseConfig:
    mode: str = 'hard_clamp'
    reference_force_N: float = 1.
    stiffness_at_reference_N_m: float = 200.
    damping_at_reference_Ns_m: float = .004
    stiffness_force_exponent: float = 1/3
    damping_force_exponent: float = 1/3
    coefficient_jitter_fraction: float = 0.
    parameter_source: str = 'assumed'

    def validate(self):
        if self.mode not in {'hard_clamp', 'compliant'}:
            raise ValueError('Contact response mode must be hard_clamp or compliant')
        values = [self.reference_force_N, self.stiffness_at_reference_N_m,
                  self.damping_at_reference_Ns_m, self.stiffness_force_exponent,
                  self.damping_force_exponent, self.coefficient_jitter_fraction]
        if not np.isfinite(values).all():
            raise ValueError('Contact response parameters must be finite')
        if self.reference_force_N <= 0 or self.stiffness_at_reference_N_m < 0 or self.damping_at_reference_Ns_m < 0:
            raise ValueError('Contact reference force must be positive; stiffness/damping nonnegative')
        if min(self.stiffness_force_exponent, self.damping_force_exponent) < 0:
            raise ValueError('Contact force exponents must be nonnegative')
        if not 0 <= self.coefficient_jitter_fraction < .5:
            raise ValueError('Contact coefficient jitter must be in [0,.5)')
        if self.parameter_source not in {'assumed', 'measured'}:
            raise ValueError('Contact parameter_source must be assumed or measured')
        return self

    def coefficients(self, force_N, rng):
        self.validate()
        if not np.isfinite(force_N) or force_N <= 0:
            raise ValueError('Contact preload must be positive and finite')
        ratio = force_N/self.reference_force_N
        variation = 1+rng.uniform(-self.coefficient_jitter_fraction, self.coefficient_jitter_fraction, 2)
        try:
            coefficients = (float(self.stiffness_at_reference_N_m*ratio**self.stiffness_force_exponent*variation[0]),
                            float(self.damping_at_reference_Ns_m*ratio**self.damping_force_exponent*variation[1]))
        except OverflowError as exc:
            raise ValueError('Contact coefficients overflow at this preload') from exc
        if not np.isfinite(coefficients).all():
            raise ValueError('Contact coefficients must remain finite at this preload')
        return coefficients

    @property
    def force_dependent(self):
        return self.mode == 'compliant' and (
            self.stiffness_at_reference_N_m*self.stiffness_force_exponent > 0 or
            self.damping_at_reference_Ns_m*self.damping_force_exponent > 0)
