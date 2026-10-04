"""Linear extension spring between a fixed bracket and a rotating bracket."""
import numpy as np


class LinearReturnSpring:
    """Axial Hooke spring, with torque from its actual line of action.

    Attachments are joint-relative offsets at q=q0, in the same reference frame
    as the tendon. The fixed attachment belongs to segment 1; the moving one
    belongs to segment 2. This is an extension spring: it goes slack below its
    free length. A shorter free length gives initial preload. Joint damping is
    handled separately; no torsional spring or spring mass is assumed.
    """

    def __init__(self, stiffness_N_m=500.0, free_length_m=0.024,
                 fixed_offset_m=(-0.012, -0.020), moving_offset_m=(0.012, -0.020),
                 joint_position=(0.15, 0.0), q0=0.0):
        if not np.isfinite(stiffness_N_m) or stiffness_N_m < 0:
            raise ValueError("Spring stiffness must be finite and nonnegative")
        if not np.isfinite(free_length_m) or free_length_m <= 0:
            raise ValueError("Spring free length must be finite and positive")
        if not np.isfinite(q0):
            raise ValueError("Spring reference angle must be finite")
        self.stiffness_N_m = float(stiffness_N_m)
        self.free_length_m = float(free_length_m)
        self.q0 = float(q0)
        for name, value in (("fixed_offset_m", fixed_offset_m), ("moving_offset_m", moving_offset_m),
                            ("joint_position", joint_position)):
            point = np.array(value, dtype=float, copy=True)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError(f"Spring {name} must contain two finite coordinates")
            setattr(self, name, point)
        self.evaluate(q0)

    @classmethod
    def from_config(cls, config):
        s = config.return_spring
        return cls(s.stiffness_N_m, s.free_length_m, s.fixed_offset_m, s.moving_offset_m,
                   config.joint.origin_xy_m, config.joint.equilibrium_rad)

    def moving_offset(self, q):
        angle = np.asarray(q) - self.q0
        x, y = self.moving_offset_m
        return np.stack((x*np.cos(angle)-y*np.sin(angle),
                         x*np.sin(angle)+y*np.cos(angle)), axis=-1)

    def attachment_positions(self, q):
        return self.joint_position+self.fixed_offset_m, self.joint_position+self.moving_offset(q)

    def evaluate(self, q):
        moving = self.moving_offset(q)
        span = moving-self.fixed_offset_m
        length = np.linalg.norm(span, axis=-1)
        if np.any(length <= 1e-12):
            raise ValueError("Spring attachments coincide; force direction is undefined")
        endpoint_derivative = np.stack((-moving[..., 1], moving[..., 0]), axis=-1)
        derivative = np.sum(span*endpoint_derivative, axis=-1)/length
        extension = np.maximum(0.0, length-self.free_length_m)
        force = self.stiffness_N_m*extension
        return {"length_m": length, "extension_m": extension, "force_N": force,
                "moment_arm_m": derivative, "torque_Nm": -force*derivative}

    def joint_torque(self, q):
        return self.evaluate(q)["torque_Nm"]
