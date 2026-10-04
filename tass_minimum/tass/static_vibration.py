"""Static-tension finite-difference string, integrated exactly for held input samples.

Same nodal mass, Laplacian and hard constraints as models/string.py; modal
coordinates allow scipy.signal.lfilter to integrate each damped oscillator.
Inputs are zero-order held over each sample interval. Output starts at rest.
"""
import numpy as np
from scipy.linalg import eigh, expm
from scipy.signal import lfilter


def static_response(grid, tension, mu, damping, contact_node, actuator_positions,
                    sensor_positions, forces, sample_rate):
    n = len(grid)
    dx = grid[1]-grid[0]
    free = np.array([i for i in range(1, n-1) if i != contact_node])
    lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
    eigenvalues, modes = eigh(lap[np.ix_(free, free)]*tension/(mu*dx*dx))
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
