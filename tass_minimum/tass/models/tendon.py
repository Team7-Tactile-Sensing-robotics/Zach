"""Routed, frictionless tendon: two fixed spans and a straight joint-crossing span.

P1/P2 are fixed to segment 1. P3/P4 rotate with segment 2 about the joint;
P4 is a sliding guide, followed by an attachment at the distal load cell.
Reference coordinates describe the straight finger
at q=q0, along +x, with positive flexion counter-clockwise. The spool position
is an effective fixed feed point (spool tangency and guide wrap are neglected).
"""
import numpy as np


class Tendon:
    def __init__(self, rest_length=0.25, k=500.0, c=0.1, q0=0.0,
                 spool_position=(0.0, 0.0), joint_position=(0.15, 0.0),
                 p1=(0.040, 0.005), p2=(0.120, 0.005),
                 p3=(0.175, 0.005), p4=(0.230, 0.005),
                 slack=0.0, preload=0.0, load_cell=(0.250, 0.005)):
        # rest_length is the effective unstretched compliant length. Initial
        # slack/preload are independent of the explicitly drawn path length.
        for name, value in (("rest_length", rest_length), ("k", k), ("c", c),
                            ("slack", slack), ("preload", preload)):
            if not np.isfinite(value) or value < 0 or (name in {"rest_length", "k"} and value == 0):
                raise ValueError(f"Invalid tendon {name}")
            setattr(self, name, float(value))
        if not np.isfinite(q0):
            raise ValueError("Tendon reference angle must be finite")
        self.q0 = float(q0)
        for name, value in (("spool_position", spool_position), ("joint_position", joint_position),
                            ("p1", p1), ("p2", p2),
                            ("p3_reference", p3), ("p4_reference", p4),
                            ("load_cell_reference", load_cell)):
            point = np.array(value, dtype=float, copy=True)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError(f"{name} must contain two finite coordinates")
            setattr(self, name, point)
        if np.linalg.norm(self.load_cell_reference-self.p4_reference) <= 1e-12:
            raise ValueError("Load-cell attachment must be distinct from P4")
        self.reference_path_length = float(self.path_length(self.q0))
        self.path_length_derivative(self.q0)  # Reject a collapsed joint span.
        self.compute(self.q0, 0.0, 0.0, 0.0)

    @classmethod
    def from_config(cls, config):
        a, g = config.axial, config.tendon_routing
        return cls(rest_length=a.rest_length_m, k=a.stiffness_N_m,
                   c=a.damping_Ns_m, q0=config.joint.equilibrium_rad,
                   spool_position=g.spool_position_m, joint_position=config.joint.origin_xy_m,
                   p1=g.p1_m, p2=g.p2_m, p3=g.p3_m, p4=g.p4_m,
                   slack=a.slack_m, preload=a.preload_N, load_cell=g.load_cell_m)

    def rotate_about_joint(self, point, q):
        dq = np.asarray(q) - self.q0
        x, y = np.asarray(point) - self.joint_position
        return self.joint_position + np.stack((x*np.cos(dq) - y*np.sin(dq),
                                               x*np.sin(dq) + y*np.cos(dq)), axis=-1)

    def guide_positions(self, q):
        return (self.p1.copy(), self.p2.copy(),
                self.rotate_about_joint(self.p3_reference, q),
                self.rotate_about_joint(self.p4_reference, q))

    def load_cell_position(self, q):
        return self.rotate_about_joint(self.load_cell_reference, q)

    def path_segments(self, q):
        """Lengths in m; supports scalar angles or arrays of angles."""
        p3 = self.rotate_about_joint(self.p3_reference, q)
        return {
            "spool_p1": np.full(np.shape(q), np.linalg.norm(self.p1 - self.spool_position)),
            "p1_p2": np.full(np.shape(q), np.linalg.norm(self.p2 - self.p1)),
            "p2_p3": np.linalg.norm(p3 - self.p2, axis=-1),
            "p3_p4": np.full(np.shape(q), np.linalg.norm(self.p4_reference - self.p3_reference)),
            "p4_load_cell": np.full(np.shape(q), np.linalg.norm(self.load_cell_reference - self.p4_reference)),
        }

    def path_length(self, q):
        return sum(self.path_segments(q).values())

    def joint_displacement(self, q):
        """Positive when bending shortens the route, in m."""
        return self.reference_path_length - self.path_length(q)

    def path_length_derivative(self, q):
        """Analytical dL/dq [m/rad]; the flexion moment arm is its negative."""
        p3 = self.rotate_about_joint(self.p3_reference, q)
        span = p3 - self.p2
        length = np.linalg.norm(span, axis=-1)
        if np.any(length <= 1e-12):
            raise ValueError("P2 and P3 coincide; tendon direction is undefined")
        relative = p3 - self.joint_position
        dp3_dq = np.stack((-relative[..., 1], relative[..., 0]), axis=-1)
        return np.sum(span * dp3_dq, axis=-1) / length

    def elastic_state(self, q, q_dot, wound_length, winding_velocity):
        """Pure evaluation for ODE solvers: signed extension, rate and tension.

        Initial extension = preload/k - slack. Slack suppresses damping too;
        the force is clipped during unloading because the tendon cannot push.
        """
        extension = (np.asarray(wound_length) - self.joint_displacement(q)
                     - self.slack + self.preload/self.k)
        rate = self.path_length_derivative(q)*np.asarray(q_dot) + np.asarray(winding_velocity)
        tension = np.where(extension > 0, np.maximum(0.0, self.k*extension + self.c*rate), 0.0)
        return extension, rate, tension

    def compute(self, q, q_dot, wound_length, winding_velocity):
        """Update a scalar state, as in the proposed Tendon interface."""
        extension, rate, tension = self.elastic_state(q, q_dot, wound_length, winding_velocity)
        self.free_length = self.rest_length - float(wound_length)
        self.path_length_current = float(self.path_length(q))
        self.raw_extension = float(extension)
        self.extension = max(0.0, self.raw_extension)
        self.extension_velocity = float(rate) if self.extension > 0 else 0.0
        self.tension = float(tension)
        return self.tension

    def joint_torque(self, q, tension=None):
        """Virtual work: tau = -T*dL/dq [N m]."""
        return -(self.tension if tension is None else np.asarray(tension))*self.path_length_derivative(q)

    def guide_path_coordinates(self, q):
        """Distances to P1..P4; the load-cell attachment is beyond P4."""
        spans = self.path_segments(q)
        return np.cumsum(np.stack([spans[name] for name in ("spool_p1", "p1_p2", "p2_p3", "p3_p4")], axis=-1), axis=-1)

    def guide_path_ratios(self, q):
        return self.guide_path_coordinates(q) / np.expand_dims(self.path_length(q), -1)
