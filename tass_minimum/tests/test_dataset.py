import unittest
from dataclasses import replace
import numpy as np
from tass.config import ModelConfig
from tass.dataset import Protocol, make_trial
from tass.static_vibration import static_response
from tass.models.string import simulate_string
from tass.models.piezo import simulate_piezos
from tass.models.tendon import Tendon


class DatasetTests(unittest.TestCase):
    def test_static_integrator_matches_reference_rk4(self):
        c=ModelConfig()
        c=replace(c,string=replace(c.string,nodes=21),piezo=replace(c.piezo,gains=(1.,1.)))
        t=np.arange(500)/48000
        grid=np.linspace(0,c.string.length_m,c.string.nodes)
        force=np.full(len(t),0.001)
        contact=int(np.argmin(abs(grid-c.contact.tendon_position_m)))
        direct,_=static_response(grid,4,c.string.linear_density_kg_m,c.string.distributed_damping_Ns_m2,
            contact,[c.string.exciter_position_m],c.piezo.positions_m,force[:,None],48000)
        reference=simulate_string(t,np.full(len(t),4),force,np.ones(len(t),bool),c)
        expected=simulate_piezos(reference,c.piezo)
        np.testing.assert_allclose(direct,expected,rtol=0.002,atol=1e-9)

    def test_trial_reproducibility_static_balance_and_variation(self):
        c=ModelConfig();c=replace(c,string=replace(c.string,nodes=21))
        p=replace(Protocol(),window_s=0.16)
        a=make_trial(c,p,0.4,5,42)
        b=make_trial(c,p,0.4,5,42)
        other=make_trial(c,p,0.4,5,43)
        np.testing.assert_array_equal(a[1],b[1])
        self.assertFalse(np.array_equal(a[1],other[1]))
        self.assertEqual(a[1].shape,(7680,4))
        m=a[3];tendon=Tendon.from_config(c)
        torque=tendon.joint_torque(p.held_joint_angle_rad,m['tension_N'])
        self.assertAlmostEqual(torque,c.joint.stiffness_Nm_rad*p.held_joint_angle_rad+m['actual_force_N']*m['force_lever_arm_m'])
        self.assertEqual(m['force_label_N'],5)
        self.assertEqual(m['location_fraction'],0.4)
        self.assertIn(4,m['constrained_actuators'])

    def test_location_reference_is_explicit(self):
        c=ModelConfig();c=replace(c,string=replace(c.string,nodes=21))
        p=replace(Protocol(),window_s=0.16,location_reference='segment2_from_joint')
        trial=make_trial(c,p,0.6,2,1)
        self.assertAlmostEqual(trial[3]['force_lever_arm_m'],0.06)
        with self.assertRaises(ValueError):
            replace(p,location_reference='unknown').validate()
