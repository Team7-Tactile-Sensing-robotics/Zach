"""Static-tension finite-difference string, integrated exactly for held input samples.

Default nodal mass, Laplacian and hard constraints match models/string.py.
Optional local spring/damper contact keeps the contacted node free and retains
the coupled damping matrix. Modal coordinates integrate damped oscillators or
full-state poles using scipy.signal.lfilter.
Inputs are zero-order held over each sample interval. Output starts at rest.
"""
import numpy as np
from scipy.linalg import eig, eigh, expm, solve
from scipy.signal import lfilter


def static_response(grid, tension, mu, damping, contact_node, actuator_positions,
                    sensor_positions, forces, sample_rate, *, contact_mode='hard_clamp',
                    contact_stiffness_N_m=0., contact_damping_Ns_m=0.):
    """Point reaction is -k*u-c*u_dot for compliant contact, about static preload."""
    if contact_mode not in {'hard_clamp', 'compliant'}:
        raise ValueError('Unknown contact mode')
    if (not np.isfinite([contact_stiffness_N_m, contact_damping_Ns_m]).all()
            or min(contact_stiffness_N_m, contact_damping_Ns_m) < 0):
        raise ValueError('Contact stiffness and damping must be finite and nonnegative')
    n = len(grid)
    dx = grid[1]-grid[0]
    free = np.array([i for i in range(1, n-1) if contact_mode == 'compliant' or i != contact_node])
    lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
    stiffness = lap[np.ix_(free, free)]*tension/(mu*dx*dx)
    viscous = np.eye(len(free))*damping/mu
    if contact_mode == 'compliant' and contact_node is not None:
        if contact_node not in free:
            raise ValueError('Compliant contact must lie on an interior node')
        local = int(np.flatnonzero(free == contact_node)[0])
        stiffness[local, local] += contact_stiffness_N_m/(mu*dx)
        viscous[local, local] += contact_damping_Ns_m/(mu*dx)
    actuator_nodes = np.array([np.argmin(abs(grid-x)) for x in actuator_positions])
    loading = np.zeros((len(free), len(actuator_positions)))
    for a, node in enumerate(actuator_nodes):
        found = np.flatnonzero(free == node)
        if len(found):
            loading[found[0], a] += 1/(mu*dx)
    sensing = np.zeros((len(sensor_positions), n))
    for i, x in enumerate(sensor_positions):
        right = min(max(np.searchsorted(grid, x), 1), n-1)
        weight = (x-grid[right-1])/dx
        sensing[i, right-1], sensing[i, right] = 1-weight, weight
    if contact_mode == 'compliant' and contact_node is not None and contact_damping_Ns_m > 0:
        # Local damping couples undamped string modes. Diagonalize the full state
        # matrix instead of discarding off-diagonal damping terms. Each pole is
        # integrated exactly under the same zero-order hold as the legacy solver.
        size = len(free)
        system = np.block([[np.zeros((size,size)), np.eye(size)], [-stiffness, -viscous]])
        poles, vectors = eig(system)
        inputs = solve(vectors, np.vstack([np.zeros_like(loading), loading]))
        weights = sensing[:, free] @ vectors[:size]
        displacement = np.zeros((len(forces), len(sensor_positions)))
        for j, pole in enumerate(poles):
            if pole.imag < -1e-8:
                continue  # Conjugate contribution is accounted for below.
            factor = 2. if pole.imag > 1e-8 else 1.
            transition = np.exp(pole/sample_rate)
            held = np.expm1(pole/sample_rate)/pole if abs(pole) > 1e-15 else 1/sample_rate
            coordinate = lfilter([0., held], [1., -transition], forces @ inputs[j])
            displacement += factor*np.real(coordinate[:, None]*weights[:, j])
        return displacement, actuator_nodes
    eigenvalues, modes = eigh(stiffness)
    weights = sensing[:, free] @ modes
    coupling = modes.T @ loading
    displacement = np.zeros((len(forces), len(sensor_positions)))
    dt = 1/sample_rate
    for j, omega2 in enumerate(eigenvalues):
        # Augmented matrix computes the exact A_d and B_d for q''+c*q'+w²*q=u.
        augmented = np.array([[0., 1., 0.], [-omega2, -damping/mu, 1.], [0., 0., 0.]])
        transition = expm(augmented*dt)
        A, B = transition[:2, :2], transition[:2, 2]
        numerator = [0., B[0], A[0, 1]*B[1]-A[1, 1]*B[0]]
        denominator = [1., -np.trace(A), np.linalg.det(A)]
        drive = forces @ coupling[j]
        coordinate = lfilter(numerator, denominator, drive)
        displacement += coordinate[:, None]*weights[:, j]
    return displacement, actuator_nodes
