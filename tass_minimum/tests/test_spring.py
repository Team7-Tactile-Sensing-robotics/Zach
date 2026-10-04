"""Independent geometry, energy and force checks for the bracket spring."""
import unittest
import numpy as np
from tass.models.spring import LinearReturnSpring


class SpringTests(unittest.TestCase):
    def test_reference_and_flexed_lengths(self):
        spring = LinearReturnSpring()
        self.assertEqual(spring.evaluate(0)['force_N'], 0)
        q = np.deg2rad([0, 30, 60, 90, 100])
        expected = 2*(0.012*np.cos(q/2)+0.020*np.sin(q/2))
        state = spring.evaluate(q)
        np.testing.assert_allclose(state['length_m'], expected, atol=1e-14)
        np.testing.assert_allclose(state['force_N'], 500*(expected-0.024), atol=1e-12)
        self.assertTrue(np.all(state['torque_Nm'][1:] < 0))

    def test_torque_matches_energy_gradient_and_force_cross_product(self):
        spring = LinearReturnSpring()
        q, eps = np.linspace(0.1, 1.7, 20), 1e-6
        state = spring.evaluate(q)
        energy = lambda angle: 0.5*spring.stiffness_N_m*spring.evaluate(angle)['extension_m']**2
        np.testing.assert_allclose(state['torque_Nm'], -(energy(q+eps)-energy(q-eps))/(2*eps), atol=2e-10)
        a, b = spring.attachment_positions(q)
        direction = (a-b)/np.linalg.norm(a-b, axis=-1)[:, None]
        force = state['force_N'][:, None]*direction
        arm = b-spring.joint_position
        torque = arm[:, 0]*force[:, 1] - arm[:, 1]*force[:, 0]
        np.testing.assert_allclose(state['torque_Nm'], torque, atol=1e-14)

    def test_preload_slack_and_translated_reference(self):
        self.assertAlmostEqual(LinearReturnSpring(free_length_m=0.020).evaluate(0)['force_N'], 2)
        self.assertEqual(LinearReturnSpring(free_length_m=0.030).evaluate(0)['force_N'], 0)
        first = LinearReturnSpring()
        moved = LinearReturnSpring(joint_position=(2.0, -3.0), q0=0.4)
        for key, value in first.evaluate(0.8).items():
            np.testing.assert_allclose(moved.evaluate(1.2)[key], value, atol=1e-14)

    def test_invalid_spring_parameters(self):
        for kwargs in ({'stiffness_N_m': -1}, {'free_length_m': 0},
                       {'fixed_offset_m': (np.nan, 0)},
                       {'moving_offset_m': (-0.012, -0.020)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LinearReturnSpring(**kwargs)


if __name__ == '__main__':
    unittest.main()
