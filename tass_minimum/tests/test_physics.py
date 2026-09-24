"""Analytical checks and interface checks, using only the standard unittest runner."""
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
from tass import ModelConfig, Inputs, Measurement, simulate
from tass.signals import Constant, Sine
from tass.models.mechanics import simulate_mechanics, ElasticTension
from tass.models.string import simulate_string, analytical_frequencies
from tass.models.piezo import simulate_piezos
from tass.processing import extract_frequency_features
from tass.io import save_measurement, load_measurement


class MechanicsTests(unittest.TestCase):
    def run_mechanics(self, tension, force, mode="measured"):
        config = ModelConfig()
        config = replace(config, axial=replace(config.axial, mode=mode))
        t = np.arange(0, 0.5, 0.0005)
        sampled = Inputs(tension_N=Constant(tension), contact_force_N=Constant(force)).sample(t)
        return config, simulate_mechanics(t, sampled, config)

    def test_zero_input_equilibrium(self):
        config, result = self.run_mechanics(0, 0)
        np.testing.assert_allclose(result["joint_angle_rad"], config.joint.equilibrium_rad, atol=1e-12)

    def test_static_equilibrium(self):
        config, result = self.run_mechanics(4, 0)
        expected = config.joint.equilibrium_rad + config.joint.moment_arm_m*4/config.joint.stiffness_Nm_rad
        self.assertAlmostEqual(result["joint_angle_rad"][-1], expected, places=7)

    def test_contact_force_equilibrium(self):
        config, result = self.run_mechanics(4, 0.4)
        expected = config.joint.equilibrium_rad + (config.joint.moment_arm_m*4 - config.contact.finger_lever_arm_m*0.4)/config.joint.stiffness_Nm_rad
        self.assertAlmostEqual(result["joint_angle_rad"][-1], expected, places=7)

    def test_simulated_tension_coupled_equilibrium(self):
        config, result = self.run_mechanics(0, 0, mode="simulated")
        j, k = config.joint, config.axial.stiffness_N_m
        expected = j.equilibrium_rad + j.moment_arm_m*k*config.motor.spool_radius_m*0.12/(j.stiffness_Nm_rad+k*j.moment_arm_m**2)
        self.assertAlmostEqual(result["joint_angle_rad"][-1], expected, places=7)
        self.assertTrue(np.all(result["tension_N"] >= 0))

    def test_slack_tendon_cannot_push(self):
        law = ElasticTension(ModelConfig())
        self.assertEqual(law(0, 0.2, 0, 0, 0), 0)


class StringTests(unittest.TestCase):
    def ringdown(self, tension=4, contact_fraction=None, nodes=81):
        config = ModelConfig()
        length = config.string.length_m
        contact_x = contact_fraction*length if contact_fraction else config.contact.tendon_position_m
        config = replace(config,
                         string=replace(config.string, nodes=nodes, exciter_position_m=0.009),
                         contact=replace(config.contact, tendon_position_m=contact_x),
                         piezo=replace(config.piezo, positions_m=(0.009, 0.09)))
        t = np.arange(9600)/48000.0
        x = np.linspace(0, length, nodes)
        span = contact_x if contact_fraction else length
        y = np.where(x <= span, 1e-5*np.sin(np.pi*x/span), 0.0)
        result = simulate_string(t, np.full(len(t), tension), np.zeros(len(t)),
                                 np.full(len(t), contact_fraction is not None), config, y)
        m = Measurement(t, simulate_piezos(result, config.piezo), np.zeros(len(t)))
        freq = extract_frequency_features(m)
        observed = freq.features["channels"][0]["dominant_frequency_Hz"]
        expected = analytical_frequencies(span, tension, config.string.linear_density_kg_m, 1)[0]
        return observed, expected, result

    def test_string_frequency(self):
        observed, expected, _ = self.ringdown()
        self.assertLess(abs(observed-expected), 5.0)

    def test_tension_frequency_scaling(self):
        f1, _, _ = self.ringdown(tension=2)
        f2, _, _ = self.ringdown(tension=8)
        self.assertAlmostEqual(f2/f1, 2, delta=0.05)

    def test_contact_location_frequencies_and_clamp(self):
        for fraction in (0.2, 0.4):
            with self.subTest(fraction=fraction):
                observed, expected, result = self.ringdown(contact_fraction=fraction)
                self.assertLess(abs(observed-expected), 7.5)
                node = round(fraction*(len(result.grid_m)-1))
                self.assertEqual(np.max(abs(result.displacement_m[:, node])), 0)
                self.assertEqual(np.max(abs(result.velocity_m_s[:, node])), 0)
                self.assertLess(np.max(abs(result.displacement_m[:, node+1:])), 1e-15)

    def test_point_force_impulse_is_mesh_independent(self):
        # With T=c=0, sum(mu*dx*v) must equal the applied impulse, for any mesh.
        for nodes in (41, 81):
            config = ModelConfig()
            config = replace(config, string=replace(config.string, nodes=nodes, distributed_damping_Ns_m2=0))
            t = np.arange(20)/48000.0
            force = np.full(len(t), 1e-5)
            result = simulate_string(t, np.zeros(len(t)), force, np.zeros(len(t), bool), config)
            momentum = config.string.linear_density_kg_m*(result.grid_m[1]-result.grid_m[0])*np.sum(result.velocity_m_s[-1])
            self.assertAlmostEqual(momentum, force[0]*t[-1], places=15)

    def test_right_contact_span_frequency(self):
        config = ModelConfig()
        t = np.arange(9600)/48000.0
        x = np.linspace(0, config.string.length_m, config.string.nodes)
        xc = config.contact.tendon_position_m
        span = config.string.length_m-xc
        y = np.where(x >= xc, 1e-5*np.sin(np.pi*(x-xc)/span), 0.0)
        result = simulate_string(t, np.full(len(t), 4.0), np.zeros(len(t)),
                                 np.ones(len(t), bool), config, y)
        measurement = Measurement(t, simulate_piezos(result, config.piezo), np.zeros(len(t)))
        observed = extract_frequency_features(measurement).features["channels"][1]["dominant_frequency_Hz"]
        expected = analytical_frequencies(span, 4, config.string.linear_density_kg_m, 1)[0]
        self.assertLess(abs(observed-expected), 5.0)


class PipelineTests(unittest.TestCase):
    def test_zero_state_pipeline(self):
        c = ModelConfig()
        c = replace(c, numerics=replace(c.numerics, duration_s=0.01))
        result = simulate(c, Inputs(tension_N=Constant(0), excitation_N=Constant(0)))
        self.assertEqual(np.max(abs(result.displacement_m)), 0)
        self.assertEqual(np.max(abs(result.measurement.piezo_V)), 0)

    def test_csv_roundtrip_and_shared_processing(self):
        t = np.arange(1000)/10000
        m = Measurement(t, np.column_stack([np.sin(2*np.pi*500*t), np.sin(2*np.pi*500*t)*2]), np.zeros(len(t)))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"record.csv"
            save_measurement(m, p)
            loaded = load_measurement(p)
        np.testing.assert_allclose(loaded.piezo_V, m.piezo_V, atol=1e-11)
        features = extract_frequency_features(loaded).features
        self.assertEqual(features["channels"][0]["dominant_frequency_Hz"], 500)
        self.assertAlmostEqual(features["channel_1_to_2_peak_amplitude_ratio"], 0.5)

    def test_fft_amplitude_energy_and_frf_mask(self):
        t = np.arange(1000)/10000
        u = np.sin(2*np.pi*500*t)
        m = Measurement(t, (3*u)[:, None], u)
        result = extract_frequency_features(m, window="boxcar")
        i = np.argmin(abs(result.frequency_Hz-500))
        self.assertAlmostEqual(result.amplitude_V[i, 0], 3, places=10)
        self.assertAlmostEqual(result.frf_V_N[i, 0].real, 3, places=10)
        self.assertAlmostEqual(result.features["channels"][0]["spectral_energy_V2_s"], 0.45, places=10)
        self.assertEqual(np.sum(result.frf_valid), 1)

    def test_unexcited_frf_and_silent_channel_are_undefined(self):
        m = Measurement(np.arange(100)/1000, np.zeros((100, 2)), np.zeros(100))
        result = extract_frequency_features(m)
        self.assertFalse(np.any(result.frf_valid))
        self.assertTrue(np.all(np.isnan(result.frf_V_N)))
        self.assertIsNone(result.features["channels"][0]["dominant_frequency_Hz"])
        self.assertIsNone(result.features["channel_1_to_2_peak_amplitude_ratio"])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            replace(ModelConfig(), string=replace(ModelConfig().string, linear_density_kg_m=-1)).validate()
        with self.assertRaises(ValueError):
            Measurement(np.array([0, 0.1, 0.21, 0.3]), np.zeros((4, 1)), np.zeros(4))
        with self.assertRaises(ValueError):
            Inputs(tension_N=Constant(-1)).sample(np.arange(4)/1000)


if __name__ == "__main__":
    unittest.main()
