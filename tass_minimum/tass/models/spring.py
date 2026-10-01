"""Return spring shared by mechanics, the app, and static dataset balance."""
import numpy as np


class ReturnSpring:
    def __init__(self, config):
        self.params = config.spring
        self.joint = config.joint
        self.relaxed_angle = (config.joint.equilibrium_rad if self.params.relaxed_angle_rad is None
                              else self.params.relaxed_angle_rad)
        self.origin = np.asarray(config.joint.origin_xy_m, float)
        self.base = np.asarray(self.params.base_anchor_m, float)
        self.distal = np.asarray(self.params.distal_anchor_m, float)
        self.rest_length = float(self.length(self.relaxed_angle))

    def moving_anchor(self, q):
        angle = np.asarray(q)-self.joint.equilibrium_rad
        x, y = self.distal-self.origin
        return self.origin + np.stack((x*np.cos(angle)-y*np.sin(angle),
                                      x*np.sin(angle)+y*np.cos(angle)),axis=-1)

    def length(self, q):
        return np.linalg.norm(self.moving_anchor(q)-self.base,axis=-1)

    def length_derivative(self, q):
        point = self.moving_anchor(q)
        relative = point-self.origin
        velocity = np.stack((-relative[...,1],relative[...,0]),axis=-1)
        span = point-self.base
        length = np.linalg.norm(span,axis=-1)
        if np.any(length <= 1e-12):
            raise ValueError('Spring anchors coincide; force direction is undefined')
        return np.sum(span*velocity,axis=-1)/length

    def state(self,q):
        q=np.asarray(q)
        if self.params.mode == 'torsional':
            torque=self.joint.stiffness_Nm_rad*(q-self.relaxed_angle)
            return dict(spring_length_m=np.full(q.shape,np.nan),
                        spring_extension_m=np.full(q.shape,np.nan),
                        spring_force_N=np.full(q.shape,np.nan),
                        spring_resisting_torque_Nm=torque)
        extension=np.maximum(0.,self.length(q)-self.rest_length)
        force=self.params.stiffness_N_m*extension
        return dict(spring_length_m=self.length(q),spring_extension_m=extension,
                    spring_force_N=force,spring_resisting_torque_Nm=force*self.length_derivative(q))

    def resisting_torque(self,q):
        return self.state(q)['spring_resisting_torque_Nm']


def static_operating_point(config, angle_rad, force_N=0.):
    """Required motor winding for an externally specified equilibrium pose."""
    from .tendon import Tendon
    tendon=Tendon.from_config(config)
    spring=ReturnSpring(config)
    arm=-float(tendon.path_length_derivative(angle_rad))
    if arm <= 0:
        raise ValueError('Pose requires a positive tendon flexion moment arm')
    tension=(float(spring.resisting_torque(angle_rad))+config.contact.finger_lever_arm_m*force_N)/arm
    winding=tendon.rest_length+config.axial.slack_m+tension/tendon.k-float(tendon.path_length(angle_rad))
    if tension < -1e-12 or winding < -1e-12 or winding > tendon.rest_length:
        raise ValueError('Static pose is infeasible with this rest length and actuator')
    return dict(joint_angle_rad=float(angle_rad),tension_N=max(0.,tension),
                winding_m=max(0.,winding),motor_angle_rad=max(0.,winding)/config.motor.spool_radius_m,
                **{name:float(value) for name,value in spring.state(angle_rad).items()})
