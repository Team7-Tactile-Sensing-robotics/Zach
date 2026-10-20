from pathlib import Path
from dataclasses import replace
import json
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

    def test_segment1_protocol_positions_force_invariance_and_selected_exports(self):
        root=Path(__file__).resolve().parents[1]
        c=ModelConfig.load(root/'config/default.json')
        p=Protocol(**json.loads((root/'config/experiments/dataset_segment1_30_70.json').read_text()))
        p=replace(p,window_s=.08,chirp_duration_s=.03,sample_rate_Hz=48000)
        for fraction,position in ((.3,.045),(.7,.105)):
            low=make_trial(c,p,fraction,1,42,include_reference=True)
            high=make_trial(c,p,fraction,5,42,include_reference=True)
            self.assertAlmostEqual(low[3]['contact_segment_requested_m'],position)
            self.assertEqual(low[3]['actual_force_N'],1)
            self.assertEqual(high[3]['actual_force_N'],5)
            self.assertEqual(low[3]['force_lever_arm_m'],0)
            self.assertFalse(low[3]['contact_force_identifiable'])
            np.testing.assert_array_equal(low[1],high[1])
            np.testing.assert_array_equal(low[4],high[4])
            np.testing.assert_array_equal(low[2][:,2:],0)
            channels=recorded_channels(*low[:4],p,reference_volts=low[4])
            self.assertIn('piezo_sensor_1_V',channels)
            self.assertIn('piezo_sensor_2_V',channels)
            self.assertIn('piezo_actuator_2_command',channels)
            self.assertFalse(any(name.startswith(('piezo_sensor_3_','piezo_sensor_4_',
                                                 'piezo_actuator_3_','piezo_actuator_4_')) for name in channels))

    def test_invalid_recorded_pairs(self):
        for ids in ((),(1,1),(0,1),(1,5),(True,2),(1.,2),(1,)):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                replace(Protocol(),active_actuator_ids=(1,2),recorded_pair_ids=ids).validate()
