import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from tass import ModelConfig, Inputs
from tass.models.spring import ReturnSpring, static_operating_point
from tass.models.tendon import Tendon
from tass.models.mechanics import simulate_mechanics
from tass.signals import Constant, SmoothStep


class SpringTests(unittest.TestCase):
    def config(self):
        return ModelConfig.load(Path(__file__).resolve().parents[1]/'config/default.json')

    def test_three_stages(self):
        c=self.config(); spring=ReturnSpring(c)
        points=[static_operating_point(c,q) for q in (spring.relaxed_angle,0,np.deg2rad(40))]
        self.assertEqual(points[0]['spring_force_N'],0)
        self.assertAlmostEqual(points[0]['tension_N'],0)
        self.assertAlmostEqual(points[0]['winding_m'],0)
        for name in ('spring_extension_m','spring_force_N','tension_N','winding_m'):
            self.assertGreater(points[1][name],points[0][name])
            self.assertGreater(points[2][name],points[1][name])
        for point in points:
            tendon=Tendon.from_config(c)
            tension=tendon.compute(point['joint_angle_rad'],0,point['winding_m'],0)
            self.assertAlmostEqual(tension,point['tension_N'],places=9)
            self.assertAlmostEqual(tendon.joint_torque(point['joint_angle_rad'],tension),point['spring_resisting_torque_Nm'],places=10)

    def test_spring_virtual_work(self):
        spring=ReturnSpring(self.config())
        for q in (0,0.2,0.6):
            eps=1e-6
            energy=lambda a: 0.5*spring.params.stiffness_N_m*max(0,float(spring.length(a))-spring.rest_length)**2
            self.assertAlmostEqual(float(spring.resisting_torque(q)),(energy(q+eps)-energy(q-eps))/(2*eps),places=8)
        self.assertEqual(spring.state(spring.relaxed_angle-0.05)['spring_force_N'],0)

    def test_dynamic_relaxation_and_actuation(self):
        c=self.config(); spring=ReturnSpring(c)
        t=np.arange(1001)/1000
        rest=Inputs(motor_angle_rad=Constant(0),contact_force_N=Constant(0)).sample(t)
        result=simulate_mechanics(t,rest,c)
        np.testing.assert_allclose(result['joint_angle_rad'],spring.relaxed_angle,atol=1e-10)
        np.testing.assert_allclose(result['tension_N'],0,atol=1e-10)
        point=static_operating_point(c,np.deg2rad(40))
        inputs=Inputs(motor_angle_rad=SmoothStep(0,point['motor_angle_rad'],0.02,0.4),contact_force_N=Constant(0)).sample(t)
        result=simulate_mechanics(t,inputs,c)
        self.assertAlmostEqual(result['joint_angle_rad'][-1],np.deg2rad(40),places=4)
        self.assertAlmostEqual(result['tension_N'][-1],point['tension_N'],places=3)

    def test_stiffer_spring_requires_more_tension(self):
        c=self.config()
        doubled=replace(c,spring=replace(c.spring,stiffness_N_m=2*c.spring.stiffness_N_m))
        first=static_operating_point(c,0.3)
        second=static_operating_point(doubled,0.3)
        self.assertAlmostEqual(second['tension_N'],2*first['tension_N'])
        self.assertGreater(second['motor_angle_rad'],first['motor_angle_rad'])

    def test_dataset_uses_extension_spring_balance(self):
        from tass.dataset import Protocol, make_trial
        c=self.config()
        protocol=replace(Protocol(),window_s=0.16)
        result=make_trial(c,protocol,0.4,2,23)
        meta=result[3]
        spring=ReturnSpring(c)
        arm=-float(Tendon.from_config(c).path_length_derivative(protocol.held_joint_angle_rad))
        self.assertAlmostEqual(meta['tension_N']*arm,
            spring.resisting_torque(protocol.held_joint_angle_rad)+meta['actual_force_N']*meta['force_lever_arm_m'])
