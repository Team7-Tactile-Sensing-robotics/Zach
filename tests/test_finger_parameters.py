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

    def test_spool_feed_is_tangent_and_updates_with_geometry(self):
        for p in (FingerParameters(),
                  replace(FingerParameters(), spool_radius_m=0.020),
                  replace(FingerParameters(), tendon_p1_m=(0.035, 0.012)),
                  replace(FingerParameters(), spool_feed_branch=1)):
            model = FingerModel(p)
            radius = model.tendon.spool_position - np.array(p.spool_position_m)
            free_span = model.tendon.p1 - model.tendon.spool_position
            self.assertAlmostEqual(np.linalg.norm(radius), p.spool_radius_m)
            self.assertAlmostEqual(float(radius @ free_span), 0)
            distance = np.linalg.norm(model.tendon.p1 - p.spool_position_m)
            self.assertAlmostEqual(np.linalg.norm(free_span),
                                   math.sqrt(distance**2-p.spool_radius_m**2))
        self.assertLess(FingerParameters().spool_feed_position_m[0],
                        FingerParameters().spool_position_m[0])

    def test_cad_route_at_straight_pose(self):
        model = FingerModel()
        r = model.readouts()
        # Independent dimensions in the STEP measurement record (metres).
        self.assertAlmostEqual(r['tip_x_m'], 0.374)
        self.assertAlmostEqual(r['span_1_m'], 0.116)
        self.assertAlmostEqual(r['span_2_m'], 0.124)
        self.assertAlmostEqual(r['span_3_m'], 0.080)
        self.assertAlmostEqual(r['spring_length_m'], 0.046)
        self.assertAlmostEqual(r['path_length_m'], 0.384182875, places=8)

    def test_invalid_measurements_report_the_field(self):
        for field, value in (
            ('tendon_rest_length_m', 0), ('tendon_young_modulus_Pa', -1),
            ('joint_inertia_kg_m2', 0), ('spring_stiffness_N_m', -1),
            ('tendon_stiffness_override_N_m', -1), ('tendon_p1_m', (0, float('nan'))),
            ('history_samples', 0), ('max_step_s', 0),
            ('spool_feed_branch', 0), ('spool_radius_m', 0.1),
            ('tendon_p1_m', FingerParameters().spool_position_m),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(FingerParameters(), **{field: value})


if __name__ == '__main__':
    unittest.main()
