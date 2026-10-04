"""Geometry, virtual-work, physical rest length and configuration regression checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from tass import ModelConfig, Inputs, simulate
from tass.models.tendon import Tendon
from tass.signals import Constant


class TendonTests(unittest.TestCase):
    def test_default_unloaded_reference_and_legacy_preload(self):
        config = ModelConfig()
        tendon = Tendon.from_config(config)
        self.assertAlmostEqual(tendon.rest_length, tendon.reference_path_length)
        self.assertAlmostEqual(tendon.tension, 0, places=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            for section in ("axial", "app"):
                values = config.to_dict()
                values[section]["preload_N"] = 0
                path.write_text(json.dumps(values))
                with self.assertWarnsRegex(UserWarning, "obsolete"):
                    loaded = ModelConfig.load(path)
                self.assertNotIn("preload_N", loaded.to_dict()[section])
                values[section]["preload_N"] = 3
                path.write_text(json.dumps(values))
                with self.assertRaisesRegex(ValueError, "physical rest length"):
                    ModelConfig.load(path)

    def test_source_state_and_density(self):
        tendon = Tendon(mu=0.0007, slack=0.001)
        self.assertEqual(tendon.mu, 0.0007)
        for q, qdot, wound, speed in [(0, 0, 0, 0), (0.4, 0.2, 0.02, 0.01),
                                       (0.7, -0.3, 0.0, 0.2)]:
            tendon.compute(q, qdot, wound, speed)
            self.assertAlmostEqual(tendon.free_length, tendon.rest_length-wound)
            self.assertAlmostEqual(tendon.required_length_current,
                                   tendon.path_length(q))
            self.assertAlmostEqual(tendon.raw_extension,
                                   tendon.required_length_current-tendon.free_length
                                   -tendon.slack)
            rotated = tendon.joint_position + tendon.rotation_matrix(q-tendon.q0) @ (
                tendon.p3_reference-tendon.joint_position)
            np.testing.assert_allclose(tendon.guide_positions(q)[2], rotated)
        for mu in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                Tendon(mu=mu)

    def test_source_preset_and_pipeline_lengths(self):
        path = Path(__file__).resolve().parents[1]/"config/string_finger_tendon.json"
        config = ModelConfig.load(path)
        tendon = Tendon.from_config(config)
        self.assertEqual(tendon.k, 500)
        self.assertEqual(tendon.c, 0.1)
        self.assertEqual(tendon.mu, 0.0005)
        self.assertEqual(tendon.rest_length, 0.25)
        config = replace(config, numerics=replace(config.numerics, duration_s=0.002))
        result = simulate(config, Inputs(excitation_N=Constant(0)))
        channels = result.measurement.channels
        np.testing.assert_allclose(channels["tendon_extension_m"],
                                   channels["tendon_required_length_m"]-channels["tendon_free_length_m"],
                                   atol=1e-15)
        np.testing.assert_allclose(channels["tendon_p4_coordinate_m"], channels["tendon_path_length_m"])

    def test_three_spans_and_shortening(self):
        # Explicit historical 5-mm guide geometry for the closed-form reference.
        tendon = Tendon(p1=(0.040, 0.005), p2=(0.120, 0.005),
                        p3=(0.175, 0.005), p4=(0.230, 0.005))
        q = np.deg2rad([0, 45, 90])
        spans = tendon.path_segments(q)
        np.testing.assert_allclose(spans["p1_p2"], 0.080, atol=1e-14)
        np.testing.assert_allclose(spans["p3_p4"], 0.055, atol=1e-14)
        np.testing.assert_allclose(spans["p4_load_cell"], 0.020, atol=1e-14)
        length_45 = np.hypot(0.030 + 0.020/np.sqrt(2), 0.030/np.sqrt(2) - 0.005)
        np.testing.assert_allclose(spans["p2_p3"], [0.055, length_45, np.hypot(0.025, 0.020)], atol=1e-14)
        self.assertTrue(np.all(np.diff(tendon.joint_displacement(q)) > 0))
        np.testing.assert_allclose(tendon.guide_path_coordinates(q)[:, -1]+0.020, tendon.path_length(q))
        self.assertEqual(tendon.guide_path_coordinates(q).shape, (3, 4))
        self.assertTrue(np.all(tendon.guide_path_ratios(q)[:, -1] < 1))

    def test_virtual_work_and_analytic_derivative(self):
        # Explicit historical 5-mm guide geometry for the closed-form reference.
        tendon = Tendon(p1=(0.040, 0.005), p2=(0.120, 0.005),
                        p3=(0.175, 0.005), p4=(0.230, 0.005))
        q = np.linspace(-0.2, 1.8, 20)
        eps = 1e-6
        length_gradient = (tendon.path_length(q+eps)-tendon.path_length(q-eps))/(2*eps)
        np.testing.assert_allclose(tendon.path_length_derivative(q), length_gradient, atol=5e-11)
        np.testing.assert_allclose(tendon.joint_torque(q, 4), -4*length_gradient, atol=2e-10)
        self.assertAlmostEqual(float(tendon.joint_torque(0, 10)), 0.05)
        # Spring energy gradient must oppose the generated joint torque.
        q, wound = 0.4, 0.04
        energy = lambda angle: 0.5*tendon.k*max(0.0, tendon.path_length(angle)-(tendon.rest_length-wound)-tendon.slack)**2
        tension = tendon.compute(q, 0, wound, 0)
        self.assertAlmostEqual(float(tendon.joint_torque(q, tension)),
                               -(energy(q+eps)-energy(q-eps))/(2*eps), places=9)

    def test_physical_slack_and_damping(self):
        reference = float(Tendon().path_length(0))
        tendon = Tendon(rest_length=reference, k=1000, c=2, slack=0.002)
        self.assertEqual(tendon.compute(0, 0, 0.001, 100), 0)
        self.assertAlmostEqual(tendon.compute(0, 0, 0.002, 0), 0, places=10)
        self.assertAlmostEqual(tendon.compute(0, 0, 0.003, 0.01), 1.02)
        self.assertEqual(tendon.compute(0, 0, 0.003, -1), 0)
        loaded = Tendon(rest_length=reference-0.002, k=1000)
        self.assertAlmostEqual(loaded.tension, 2)
        self.assertEqual(Tendon(rest_length=reference+0.002, k=1000).tension, 0)
        for wound in (-0.001, reference+0.001):
            with self.assertRaises(ValueError):
                tendon.compute(0, 0, wound, 0)
        with self.assertRaises(ValueError):
            tendon.compute(0, 0, float('nan'), 0)

    def test_extension_rate_and_rigid_transform(self):
        tendon = Tendon()
        q, speed, wound, winding_speed, dt = 0.4, 0.7, 0.01, 0.02, 1e-6
        extension, rate, _ = tendon.elastic_state(q, speed, wound, winding_speed)
        before = tendon.elastic_state(q-speed*dt, 0, wound-winding_speed*dt, 0)[0]
        after = tendon.elastic_state(q+speed*dt, 0, wound+winding_speed*dt, 0)[0]
        self.assertAlmostEqual(float(rate), float((after-before)/(2*dt)), places=9)
        offset = np.array([0.7, -0.3])
        moved = Tendon(q0=0.2, spool_position=tendon.spool_position+offset,
                       joint_position=tendon.joint_position+offset,
                       p1=tendon.p1+offset, p2=tendon.p2+offset,
                       p3=tendon.p3_reference+offset, p4=tendon.p4_reference+offset,
                       load_cell=tendon.load_cell_reference+offset)
        np.testing.assert_allclose(moved.path_length(q+0.2), tendon.path_length(q), atol=1e-14)
        np.testing.assert_allclose(moved.joint_torque(q+0.2, 4), tendon.joint_torque(q, 4), atol=1e-14)

    def test_load_cell_tail_changes_path_but_not_moment_arm(self):
        first = Tendon()
        second = Tendon(load_cell=(0.27, 0.005))
        q = np.linspace(0, 1.7, 20)
        np.testing.assert_allclose(second.path_length(q)-first.path_length(q), 0.020)
        np.testing.assert_allclose(second.path_length_derivative(q), first.path_length_derivative(q))
        np.testing.assert_allclose(second.joint_displacement(q), first.joint_displacement(q), atol=1e-14)
        distances = np.linalg.norm(first.load_cell_position(q)-first.guide_positions(q)[3], axis=-1)
        np.testing.assert_allclose(distances, 0.020)

    def test_invalid_geometry_and_slack(self):
        for args in ({"p1": (np.nan, 0)}, {"p3": tuple(Tendon().p2)}, {"slack": -1}, {"k": 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Tendon(**args)
        with self.assertRaises(ValueError):
            c = ModelConfig()
            replace(c, axial=replace(c.axial, slack_m=-0.01)).validate()

    def test_config_roundtrip_and_legacy_field(self):
        c = ModelConfig()
        c = replace(c, axial=replace(c.axial, slack_m=0.001),
                    tendon_routing=replace(c.tendon_routing, p3_m=(0.18, 0.006)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            c.save(path)
            loaded = ModelConfig.load(path)
            self.assertAlmostEqual(Tendon.from_config(loaded).path_length(0.8),
                                   Tendon.from_config(c).path_length(0.8))
            values = c.to_dict()
            values["joint"]["moment_arm_m"] = 0.008
            path.write_text(json.dumps(values))
            with self.assertWarnsRegex(UserWarning, "obsolete"):
                ModelConfig.load(path)

    def test_pipeline_reference_pose_and_channels(self):
        c = ModelConfig()
        c = replace(c, joint=replace(c.joint, equilibrium_rad=0.3),
                    numerics=replace(c.numerics, duration_s=0.002))
        result = simulate(c, Inputs(tension_N=Constant(0), excitation_N=Constant(0)))
        channels = result.measurement.channels
        np.testing.assert_allclose(channels["tip_world_x_m"], 0.25)
        np.testing.assert_allclose(channels["tip_world_y_m"], 0.0)
        np.testing.assert_allclose(channels["tendon_p2_p3_length_m"], 0.055)
        np.testing.assert_allclose(channels["tendon_moment_arm_m"], 0.005)


if __name__ == "__main__":
    unittest.main()
