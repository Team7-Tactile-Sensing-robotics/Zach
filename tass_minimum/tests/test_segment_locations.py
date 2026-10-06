from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
from tass.config import ModelConfig
from tass.dataset import Protocol, make_trial, recorded_channels

class SegmentLocationTests(unittest.TestCase):
    def test_straight_physical_locations_and_force_balance(self):
        c=ModelConfig.load(Path(__file__).resolve().parents[1]/'config/default.json')
        p=replace(Protocol(),location_reference='total_segment',held_joint_angle_rad=0.,
                  position_jitter_m=0.,window_s=.16)
        for fraction, distance in zip((.2,.4,.6,.8),(.05,.10,.15,.20)):
            a=make_trial(c,p,fraction,1,42)
            b=make_trial(c,p,fraction,10,42)
            self.assertAlmostEqual(a[3]['contact_segment_requested_m'],distance)
            self.assertAlmostEqual(a[3]['force_lever_arm_m'],max(0,distance-.15))
            channels=recorded_channels(*a,p)
            np.testing.assert_allclose(channels['contact_location_m'],distance)
            self.assertEqual(a[3]['joint_angle_rad'],0)
            if fraction<=.6:
                self.assertAlmostEqual(a[3]['tension_N'],b[3]['tension_N'])
                np.testing.assert_allclose(a[1],b[1],atol=1e-12)
            else:
                self.assertGreater(b[3]['tension_N'],a[3]['tension_N'])
