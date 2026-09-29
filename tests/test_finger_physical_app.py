"""Run from the repository root: python -m unittest discover -s tests -v."""
from dataclasses import asdict, replace
import math
import unittest
import xml.etree.ElementTree as ET

from finger_physical_app import FingerModel, FingerParameters, finger_svg


class FingerModelTests(unittest.TestCase):
    def test_zero_input_stays_at_rest(self):
        model = FingerModel()
        model.advance(0.1)
        r = model.readouts()
        self.assertEqual(r['motor_angle_deg'], 0)
        self.assertEqual(r['joint_angle_deg'], 0)
        self.assertEqual(r['tension_N'], 0)

    def test_motor_is_rate_limited_and_command_does_not_reset_state(self):
        model = FingerModel()
        model.set_motor_target(180)
        model.advance(0.02)
        r = model.readouts()
        self.assertAlmostEqual(r['motor_angle_deg'], 6.0)
        self.assertGreater(r['joint_angle_deg'], 0)
        previous = asdict(model.state)
        model.set_motor_target(30)
        self.assertEqual(asdict(model.state), previous)
        model.set_motor_target(999)
        self.assertEqual(model.readouts()['motor_target_deg'], 180)
        with self.assertRaises(ValueError):
            model.set_motor_target(float('nan'))

    def test_winding_settles_and_release_returns_to_straight(self):
        model = FingerModel()
        model.set_motor_target(90)
        model.advance(2)
        r = model.readouts()
        self.assertAlmostEqual(r['joint_angle_deg'], 44.2, delta=0.05)
        self.assertAlmostEqual(r['tension_N'], 4.149, delta=0.005)
        self.assertAlmostEqual(r['tendon_torque_Nm']+r['spring_torque_Nm'], 0, delta=1e-5)
        self.assertAlmostEqual(r['span_1_m'], 0.080)
        self.assertAlmostEqual(r['span_2_m'], 0.055)
        self.assertLess(r['span_3_m'], 0.055)
        model.set_motor_target(0)
        for _ in range(20):
            model.advance(0.1)
            self.assertGreaterEqual(model.readouts()['tension_N'], 0)
            self.assertGreaterEqual(model.state.joint_angle_rad, 0)
        self.assertAlmostEqual(model.readouts()['joint_angle_deg'], 0, delta=0.01)

    def test_hold_stops_motor_but_finger_keeps_settling(self):
        model = FingerModel()
        model.set_motor_target(90)
        model.advance(0.08)
        angle = model.state.motor_angle_rad
        finger = model.state.joint_angle_rad
        model.set_motor_target(math.degrees(angle))
        model.advance(0.15)
        self.assertAlmostEqual(model.state.motor_angle_rad, angle)
        self.assertGreater(abs(model.state.joint_angle_rad-finger), 0.001)

    def test_flexion_stop_supports_torque(self):
        model = FingerModel(replace(FingerParameters(), joint_limit_deg=20))
        model.set_motor_target(180)
        model.advance(1.2)
        r = model.readouts()
        self.assertAlmostEqual(r['joint_angle_deg'], 20)
        self.assertEqual(r['joint_velocity_deg_s'], 0)
        self.assertEqual(r['joint_acceleration_deg_s2'], 0)
        self.assertLess(r['stop_reaction_Nm'], 0)
        self.assertAlmostEqual(r['tendon_torque_Nm']+r['spring_torque_Nm']+r['stop_reaction_Nm'], 0)

    def test_graphic_tracks_current_pose(self):
        model = FingerModel()
        before = finger_svg(model)
        model.set_motor_target(90)
        model.advance(0.3)
        after = finger_svg(model)
        self.assertNotEqual(before, after)
        svg = ET.fromstring(after)
        self.assertEqual(svg.attrib['viewBox'], '0 0 720 560')
        self.assertIn(f"{model.readouts()['joint_angle_deg']:.1f}°", after)


if __name__ == '__main__':
    unittest.main()
