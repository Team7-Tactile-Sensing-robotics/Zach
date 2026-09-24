"""Contact boundary interface. A future compliant law can supply acceleration."""
from typing import Protocol
import numpy as np


class ContactBoundary(Protocol):
    def constrained_nodes(self, active: bool) -> tuple[int, ...]: ...
    def acceleration(self, y, velocity, active: bool): ...


class NoContact:
    def constrained_nodes(self, active):
        return ()
    def acceleration(self, y, velocity, active):
        return 0.0


class HardContact(NoContact):
    def __init__(self, grid_m, position_m):
        self.node = int(np.argmin(abs(grid_m - position_m)))
        if self.node in (0, len(grid_m)-1):
            raise ValueError("Contact snaps to an endpoint; refine the mesh or move contact inward")
        self.actual_position_m = float(grid_m[self.node])
    def constrained_nodes(self, active):
        return (self.node,) if active else ()
