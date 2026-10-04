"""Check that edited settings reach the mechanics and derived material law."""
from dataclasses import replace
import math
import unittest

import numpy as np

from finger_parameters import FingerParameters
from finger_physical_app import FingerModel, finger_svg


class FingerParameterTests(unittest.TestCase):
    def tension_at_one_mm(self, parameters):
        model = FingerModel(parameters)
        model.state.motor_angle_rad = 0.001/parameters.spool_radius_m
        model.set_motor_target(math.degrees(model.state.motor_angle_rad))
        return model.readouts()['tension_N']

    def test_material_changes_reach_tendon_force(self):
        p = FingerParameters()
        self.assertAlmostEqual(p.tendon_stiffness_N_m, 2500)
        self.assertAlmostEqual(self.tension_at_one_mm(p), 2.5)
        for modified in (
            replace(p, tendon_young_modulus_Pa=2*p.tendon_young_modulus_Pa),
            replace(p, tendon_area_m2=2*p.tendon_area_m2),
            replace(p, tendon_rest_length_m=p.tendon_rest_length_m/2),
        ):
            self.assertAlmostEqual(self.tension_at_one_mm(modified), 5.0)

    def test_measured_stiffness_override_is_explicit(self):
        p = replace(FingerParameters(), tendon_stiffness_override_N_m=1000,
                    tendon_young_modulus_Pa=2e9)
        self.assertEqual(p.tendon_stiffness_N_m, 1000)
        self.assertAlmostEqual(self.tension_at_one_mm(p), 1.0)

    def test_geometry_updates_model_and_graphic(self):
        p = replace(FingerParameters(), proximal_length_m=0.2, distal_length_m=0.15,
                    tendon_p2_m=(0.170, 0.006), tendon_p3_offset_m=(0.040, 0.006),
                    tendon_p4_offset_m=(0.110, 0.006), load_cell_offset_m=(0.150, 0.006))
        model = FingerModel(p)
        np.testing.assert_allclose(model.tendon.guide_positions(0)[2], (0.240, 0.006))
        self.assertAlmostEqual(model.readouts()['tip_x_m'], 0.350)
        self.assertAlmostEqual(model.readouts()['load_cell_tail_m'], 0.040)
        svg = finger_svg(model)
        self.assertIn('Fixed · 200 mm', svg)
        self.assertIn('Rotates · 150 mm', svg)

    def test_invalid_measurements_report_the_field(self):
        for field, value in (
            ('tendon_rest_length_m', 0), ('tendon_young_modulus_Pa', -1),
            ('joint_inertia_kg_m2', 0), ('spring_stiffness_N_m', -1),
            ('tendon_stiffness_override_N_m', -1), ('tendon_p1_m', (0, float('nan'))),
            ('history_samples', 0), ('max_step_s', 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(FingerParameters(), **{field: value})


if __name__ == '__main__':
    unittest.main()
