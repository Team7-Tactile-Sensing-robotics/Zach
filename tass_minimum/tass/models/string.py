"""Second-order spatial finite differences, RK4 with automatic stable substeps."""
from dataclasses import dataclass
import warnings
import numpy as np
from .contact import HardContact, NoContact


def analytical_frequencies(length_m, tension_N, linear_density_kg_m, modes=3):
    return np.arange(1, modes+1) / (2*length_m) * np.sqrt(tension_N / linear_density_kg_m)


@dataclass
class StringResult:
    grid_m: np.ndarray
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    metadata: dict


def simulate_string(time_s, tension_N, excitation_N, contact_active, config,
                    initial_displacement=None, initial_velocity=None, boundary=None):
    p = config.string
    grid = np.linspace(0, p.length_m, p.nodes)
    dx = grid[1] - grid[0]
    if boundary is None:
        boundary = HardContact(grid, config.contact.tendon_position_m) if config.contact.enabled else NoContact()
    exciter = int(np.argmin(abs(grid - p.exciter_position_m)))
    if exciter in (0, p.nodes-1):
        raise ValueError("Exciter snaps to an endpoint; refine mesh or move it inward")
    if np.any(contact_active) and exciter in boundary.constrained_nodes(True):
        raise ValueError("Exciter is clamped by hard contact; choose a node on a free span")
    mu = p.linear_density_kg_m
    decay = p.distributed_damping_Ns_m2 / mu
    omega_bound = 2*np.sqrt(float(np.max(tension_N))/mu) / dx
    # A conservative RK4 bound covers both oscillation and distributed damping.
    stable_dt = config.numerics.string_stability_factor * 2.0 / max(omega_bound + decay, 1e-30)
    dt = time_s[1] - time_s[0]
    substeps = max(1, int(np.ceil(dt/stable_dt)))
    h = dt/substeps
    max_mode_Hz = omega_bound/(2*np.pi)
    if max_mode_Hz >= 0.5/dt:
        warnings.warn("Output Nyquist frequency is below the highest mesh mode. Increase sample_rate_Hz; "
                      "this V1 model has no ADC antialias filter.", RuntimeWarning)

    state = np.zeros((2, p.nodes))
    for k, initial in enumerate((initial_displacement, initial_velocity)):
        if initial is not None:
            values = np.asarray(initial, float)
            if values.shape != (p.nodes,) or not np.all(np.isfinite(values)):
                raise ValueError("Initial string state must have one finite value per node")
            state[k] = values
    displacement = np.empty((len(time_s), p.nodes))
    velocity = np.empty_like(displacement)

    def clamp(z, active):
        z[:, [0, p.nodes-1, *boundary.constrained_nodes(active)]] = 0.0

    def derivative(z, T, force, active):
        y, v = z
        out = np.zeros_like(z)
        out[0] = v
        out[1, 1:-1] = T/mu*(y[2:] - 2*y[1:-1] + y[:-2])/dx**2 - decay*v[1:-1]
        # Point force [N] / lumped nodal mass [kg], equivalent to q=F/dx [N/m].
        out[1, exciter] += force/(mu*dx)
        out[1] += boundary.acceleration(y, v, active)
        clamp(out, active)
        return out

    clamp(state, bool(contact_active[0]))
    displacement[0], velocity[0] = state
    for i in range(len(time_s)-1):
        active = bool(contact_active[i])
        clamp(state, active)
        for j in range(substeps):
            a, b, c = j/substeps, (j+0.5)/substeps, (j+1)/substeps
            T0, dT = tension_N[i], tension_N[i+1]-tension_N[i]
            F0, dF = excitation_N[i], excitation_N[i+1]-excitation_N[i]
            k1 = derivative(state, T0+a*dT, F0+a*dF, active)
            k2 = derivative(state+h*k1/2, T0+b*dT, F0+b*dF, active)
            k3 = derivative(state+h*k2/2, T0+b*dT, F0+b*dF, active)
            k4 = derivative(state+h*k3, T0+c*dT, F0+c*dF, active)
            state += h*(k1 + 2*k2 + 2*k3 + k4)/6
        # Contact switches on sample boundaries; imposing a clamp dissipates local energy.
        clamp(state, bool(contact_active[i+1]))
        displacement[i+1], velocity[i+1] = state
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(velocity)):
        raise RuntimeError("String solution diverged")
    return StringResult(grid, displacement, velocity, {
        "dx_m": dx, "substeps_per_sample": substeps, "internal_dt_s": h,
        "exciter_actual_position_m": float(grid[exciter]),
        "contact_actual_position_m": getattr(boundary, "actual_position_m", None),
        "highest_mesh_frequency_bound_Hz": max_mode_Hz,
        "contact_boundary_type": type(boundary).__name__,
    })
