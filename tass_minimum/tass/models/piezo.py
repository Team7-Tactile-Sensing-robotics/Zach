"""Linear displacement/velocity sensing with interpolation and seeded noise."""
import numpy as np


def simulate_piezos(string_result, params):
    field = string_result.displacement_m if params.mode == "displacement" else string_result.velocity_m_s
    grid = string_result.grid_m
    channels = []
    for x, gain in zip(params.positions_m, params.gains):
        right = min(max(int(np.searchsorted(grid, x)), 1), len(grid)-1)
        left = right-1
        weight = (x-grid[left])/(grid[right]-grid[left])
        channels.append(gain*((1-weight)*field[:, left] + weight*field[:, right]))
    volts = np.column_stack(channels)
    volts += np.random.default_rng(params.seed).normal(size=volts.shape)*np.asarray(params.noise_std_V)
    return volts
