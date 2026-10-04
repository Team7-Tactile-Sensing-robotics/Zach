"""Integrated string-finger-simulator tendon with analytical geometry.

Routed, frictionless tendon: two fixed spans and a straight joint-crossing span.

P1/P2 are fixed to segment 1. P3/P4 rotate with segment 2 about the joint;
P4 is a sliding guide, followed by an attachment at the distal load cell.
Reference coordinates describe the straight finger
at q=q0, along +x, with positive flexion counter-clockwise. The spool position
is an effective fixed feed point (spool tangency and guide wrap are neglected).
"""
import numpy as np


class Tendon:
    def __init__(
            self,
            rest_length=0.25,
            k=500.0,
            c=0.1,
            q0=0.0,
            spool_position=(0.0, 0.03),
            joint_position=(0.15, 0.0),
            p1=(0.040, 0.015),
            p2=(0.120, 0.015),
            p3=(0.175, 0.015),
            p4=(0.230, 0.015),
            slack=0.0,
            *,
            mu=0.0005,
        ):
        # rest_length is the physical unstretched length outside the spool
        # at zero winding. slack adds an independent free-length allowance.
        for name, value in (
            ("rest_length", rest_length),
            ("k", k),
            ("c", c),
            ("slack", slack),
        ):
            if (
                not np.isfinite(value)
                or value < 0
                or (
                    name in {"rest_length", "k"}
                    and value == 0
                )
            ):
                raise ValueError(
                    f"Invalid tendon {name}"
                )

            setattr(
                self,
                name,
                float(value),
            )
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("Tendon mu must be finite and positive")
        self.mu = float(mu)
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
        a = config.axial
        g = config.tendon_routing

        return cls(
            rest_length=a.rest_length_m,
            k=a.stiffness_N_m,
            c=a.damping_Ns_m,
            q0=config.joint.equilibrium_rad,
            spool_position=g.spool_position_m,
            joint_position=config.joint.origin_xy_m,
            p1=g.p1_m,
            p2=g.p2_m,
            p3=g.p3_m,
            p4=g.p4_m,
            slack=a.slack_m,
            mu=config.string.linear_density_kg_m,
        )


    @staticmethod
    def rotation_matrix(angle):
        """Source-compatible 2D counter-clockwise rotation for a scalar angle."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def required_length(self, q):
        """
        Physical routed tendon length required by the current geometry [m].
        
        The returned value is the actual geometric path length.
        """
        return self.path_length(q)

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

    def elastic_state(
        self,
        q,
        q_dot,
        wound_length,
        winding_velocity,
    ):
        """
        Evaluate tendon extension, extension rate, and tension.

        Physical rest-length model
        --------------------------
        free_length:
            Unstretched tendon length remaining outside the spool.

                free_length = rest_length - wound_length

        signed_extension:
            Difference between the current routed path and the available
            unstretched tendon length.

                signed_extension =
                    path_length(q)
                    - free_length
                    - slack

            signed_extension > 0:
                Tendon is stretched and can generate tension.

            signed_extension <= 0:
                Tendon is slack and tension is zero.

        extension_rate:
            Rate caused by both joint motion and spool winding.

                extension_rate =
                    dL_path/dq * q_dot
                    + winding_velocity

        tension:
            Kelvin-Voigt axial model with unilateral tension.

                tension = k*extension + c*extension_rate

            The tendon can pull but cannot push.
        """
        q = np.asarray(q)
        q_dot = np.asarray(q_dot)
        wound_length = np.asarray(wound_length)
        winding_velocity = np.asarray(
            winding_velocity
        )

        if not all(np.all(np.isfinite(value)) for value in (q, q_dot, wound_length, winding_velocity)):
            raise ValueError("Tendon state values must be finite")
        if np.any(wound_length < 0):
            raise ValueError(
                "Wound length must be nonnegative"
            )

        # Unstretched tendon length still available outside the spool.
        free_length = (
            self.rest_length
            - wound_length
        )

        if np.any(free_length < 0):
            raise ValueError(
                "Wound length cannot exceed tendon rest length"
            )

        # Physical comparison between routed distance and available
        # unstretched tendon length.
        signed_extension = (
            self.path_length(q)
            - free_length
            - self.slack
        )

        # Chain rule:
        #
        # d(path_length)/dt = dL/dq * q_dot
        #
        # Positive winding shortens free_length, so positive winding
        # velocity increases tendon extension.
        extension_rate = (
            self.path_length_derivative(q)
            * q_dot
            + winding_velocity
        )

        # The tendon generates force only while stretched.
        tension = np.where(
            signed_extension > 0.0,
            np.maximum(
                0.0,
                self.k * signed_extension
                + self.c * extension_rate,
            ),
            0.0,
        )

        return (
            signed_extension,
            extension_rate,
            tension,
        )

    def compute(
        self,
        q,
        q_dot,
        wound_length,
        winding_velocity,
    ):
        """
        Compute and store the current physical tendon state.

        Parameters
        ----------
        q : float
            Current joint angle [rad].

        q_dot : float
            Current joint angular velocity [rad/s].

        wound_length : float
            Tendon length currently wound onto the spool [m].

        winding_velocity : float
            Rate at which tendon is being wound onto the spool [m/s].
            Positive means winding in.
        """
        extension, rate, tension = self.elastic_state(
            q=q,
            q_dot=q_dot,
            wound_length=wound_length,
            winding_velocity=winding_velocity,
        )

        self.free_length = (
            self.rest_length
            - float(wound_length)
        )

        self.path_length_current = float(
            self.path_length(q)
        )

        self.required_length_current = (
            self.path_length_current
        )

        # Keep the signed value for debugging slack.
        self.raw_extension = float(extension)

        # Physical tendon extension cannot be negative.
        self.extension = max(
            0.0,
            self.raw_extension,
        )

        # Damping is disabled while the tendon is slack.
        self.extension_velocity = (
            float(rate)
            if self.extension > 0.0
            else 0.0
        )

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
