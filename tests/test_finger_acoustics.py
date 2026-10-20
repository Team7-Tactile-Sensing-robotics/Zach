"""Analytical dynamics and app integration checks for the acoustic preview."""
from dataclasses import replace
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from streamlit.testing.v1 import AppTest

from finger_parameters import FingerParameters
from finger_physical_app import FingerModel
from finger_acoustics import FingerAcoustics, acoustic_svg
from tass_minimum.tass.static_vibration import oscillator_response


class AcousticPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.p = FingerParameters()
        self.finger = FingerModel(self.p)

    def advance(self, model, flags, tension=4., angle=0.):
        model.advance(self.finger.tendon, angle, tension, flags,
                      self.p.acoustic_drive_frequencies_Hz, self.p.acoustic_drive_amplitudes_N)

    def test_oscillator_matches_analytical_step_and_continues_across_blocks(self):
        omega, zeta, fs, count = 200., 0.05, 24000., 2400
        drive = np.full(count, 3.)
        q, v, end = oscillator_response(omega**2, 2*zeta*omega, drive, fs)
        t = np.arange(count+1)/fs
        wd = omega*math.sqrt(1-zeta*zeta)
        expected_q = 3/omega**2*(1-np.exp(-zeta*omega*t)*(
            np.cos(wd*t)+zeta*omega/wd*np.sin(wd*t)))
        expected_v = 3/wd*np.exp(-zeta*omega*t)*np.sin(wd*t)
        np.testing.assert_allclose(np.append(q, end[0]), expected_q, atol=2e-13)
        np.testing.assert_allclose(np.append(v, end[1]), expected_v, atol=2e-11)
        qa, va, mid = oscillator_response(omega**2, 2*zeta*omega, drive[:731], fs)
        qb, vb, final = oscillator_response(omega**2, 2*zeta*omega, drive[731:], fs, mid)
        np.testing.assert_allclose(np.concatenate((qa, qb)), q, atol=2e-13)
        np.testing.assert_allclose(np.concatenate((va, vb)), v, atol=2e-11)
        np.testing.assert_allclose(final, end, atol=2e-11)

    def test_independent_transmitters_superpose_and_p4_is_inside_span(self):
        first, fourth, both = [FingerAcoustics(self.p) for _ in range(3)]
        for _ in range(3):
            self.advance(first, [True, False, False, False])
            self.advance(fourth, [False, False, False, True])
            self.advance(both, [True, False, False, True])
        np.testing.assert_allclose(both.history[:, 1:5],
                                   first.history[:, 1:5]+fourth.history[:, 1:5], atol=1e-12)
        self.assertLess(fourth.guide_positions_m[3], fourth.length_m)
        self.assertGreater(np.max(abs(fourth.last_volts[:, 3])), 1e-6)
        np.testing.assert_array_equal(both.displacement([0, both.length_m]), 0.)
        self.assertFalse(both.spectrum().frf_valid.any())

    def test_switching_off_retains_vibration_and_damping_removes_energy(self):
        model = FingerAcoustics(self.p)
        for _ in range(5):
            self.advance(model, [False, True, False, False])
        initial = model.energy_J()
        energies = []
        for i in range(30):
            self.advance(model, [False]*4)
            energies.append(model.energy_J())
            if i == 0:
                self.assertGreater(np.max(abs(model.last_volts)), 1e-6)
            np.testing.assert_array_equal(model.history[-240:, 5:], 0.)
        self.assertTrue(all(b <= a+1e-17 for a, b in zip([initial]+energies, energies)))
        self.assertLess(energies[-1], initial*1e-3)

    def test_geometry_and_tension_set_frequencies_and_slack_clears_state(self):
        model = FingerAcoustics(self.p)
        self.advance(model, [True]*4)
        straight_length, straight_frequency = model.length_m, model.frequencies_Hz[0]
        self.advance(model, [True]*4, tension=16., angle=0.5)
        self.assertLess(model.length_m, straight_length)
        self.assertAlmostEqual(model.frequencies_Hz[0]/straight_frequency,
                               2*straight_length/model.length_m)
        self.advance(model, [True]*4, tension=0.)
        np.testing.assert_array_equal(model.state, 0.)
        np.testing.assert_array_equal(model.last_volts, 0.)
        self.assertIn('slack', model.status)

    def test_receiver_gain_spectrum_clock_and_history_limit(self):
        p2 = replace(self.p, acoustic_receiver_gains_V_m=(2000., 1000., 1000., 1000.))
        normal, doubled = FingerAcoustics(self.p), FingerAcoustics(p2)
        for _ in range(35):
            self.advance(normal, [True, False, False, False])
            self.advance(doubled, [True, False, False, False])
        np.testing.assert_allclose(doubled.last_volts[:, 0], 2*normal.last_volts[:, 0])
        np.testing.assert_allclose(doubled.last_volts[:, 1:], normal.last_volts[:, 1:])
        self.assertEqual(len(normal.history), round(self.p.acoustic_history_s*self.p.acoustic_sample_rate_Hz))
        np.testing.assert_allclose(np.diff(normal.history[:, 0]), 1/self.p.acoustic_sample_rate_Hz)
        self.assertAlmostEqual(normal.time_s, 0.35)
        self.assertAlmostEqual(normal.spectrum().features['channels'][0]['dominant_frequency_Hz'], 250., delta=5.)

    def test_bandwidth_and_parameter_validation(self):
        model = FingerAcoustics(self.p)
        self.advance(model, [True]*4, tension=1e6)
        self.assertFalse(np.any(model.retained))
        np.testing.assert_array_equal(model.last_volts, 0.)
        with self.assertRaises(ValueError):
            model.advance(self.finger.tendon, 0, 4, [True]*4, [self.p.acoustic_sample_rate_Hz]*4, [.001]*4)
        for name, value in [('acoustic_modes', 0), ('acoustic_linear_density_kg_m', 0),
                            ('acoustic_receiver_gains_V_m', (1, 2)),
                            ('acoustic_drive_frequencies_Hz', (12000.,)*4),
                            ('acoustic_drive_amplitudes_N', (-1.,)*4),
                            ('acoustic_history_s', 0.001)]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'acoustic'):
                replace(self.p, **{name: value})

    def test_svg_uses_live_geometry_and_marks_enabled_sources(self):
        model = FingerAcoustics(self.p)
        self.advance(model, [True, False, False, True])
        svg = acoustic_svg(model, [True, False, False, True])
        ET.fromstring(svg)
        self.assertIn('P4 TX ON', svg)
        self.assertIn('P2 TX off', svg)
        self.assertIn('Load cell · fixed', svg)

    def test_contact_single_mode_frequency_matches_point_spring_formula(self):
        p = replace(self.p, acoustic_modes=1)
        model = FingerAcoustics(p)
        model.set_contact(True, 0.37, 0.8)
        self.advance(model, [True, False, False, False])
        length = model.length_m
        mass = p.acoustic_linear_density_kg_m*length/2
        omega0_sq = (np.pi/length)**2*4/p.acoustic_linear_density_kg_m
        expected = np.sqrt(omega0_sq + 0.8*p.acoustic_contact_stiffness_per_N
                           *np.sin(np.pi*0.37)**2/mass)/(2*np.pi)
        self.assertAlmostEqual(model.frequencies_Hz[0], expected)

    def test_zero_force_disabled_and_anchor_contact_match_free_string(self):
        models = [FingerAcoustics(self.p) for _ in range(5)]
        for model, args in zip(models[1:], [(True, .4, 0), (False, .4, 3),
                                          (True, 0, 3), (True, 1, 3)]):
            model.set_contact(*args)
        for _ in range(3):
            for model in models:
                self.advance(model, [True, False, False, True])
        for model in models[1:]:
            np.testing.assert_array_equal(model.history, models[0].history)

    def test_contact_location_force_and_release_affect_response(self):
        models = [FingerAcoustics(self.p) for _ in range(3)]
        for model, position, force in zip(models, [.3, .3, .7], [.2, 2., 2.]):
            model.set_contact(True, position, force)
            for _ in range(5):
                self.advance(model, [True, False, False, False])
        self.assertGreater(models[1].frequencies_Hz[0], models[0].frequencies_Hz[0])
        for model in models[1:]:
            self.assertGreater(np.linalg.norm(model.last_volts-models[0].last_volts), 1e-4)
        self.assertGreater(np.linalg.norm(models[1].last_volts-models[2].last_volts), 1e-4)
        model = models[1]
        before = model.energy_J()
        for _ in range(20):
            self.advance(model, [False]*4)
            after = model.energy_J()
            self.assertLessEqual(after, before+1e-17)
            before = after
        state = model.state.copy()
        model.set_contact(False, .3, 2.)
        model.set_operating_point(self.finger.tendon, 0, 4.)
        np.testing.assert_array_equal(model.state, state)
        self.advance(model, [False]*4)
        self.assertTrue(np.isfinite(model.last_volts).all())

    def test_contact_bandwidth_slack_validation_and_marker(self):
        model = FingerAcoustics(self.p)
        model.set_contact(True, .47, 5.)
        self.advance(model, [True]*4)
        svg = acoustic_svg(model, [True]*4)
        ET.fromstring(svg)
        self.assertIn('Contact · 5.00 N', svg)
        self.advance(model, [True]*4, tension=1e6)
        self.assertFalse(model.retained.any())
        np.testing.assert_array_equal(model.state, 0.)
        self.advance(model, [True]*4, tension=0.)
        np.testing.assert_array_equal(model.last_volts, 0.)
        for position, force in [(-.1, 1), (1.1, 1), (.5, -1), (.5, float('nan'))]:
            with self.assertRaises(ValueError):
                model.set_contact(True, position, force)
        for name, value in [('acoustic_contact_position_ratio', 2),
                            ('acoustic_contact_force_N', 6),
                            ('acoustic_contact_stiffness_per_N', -1),
                            ('acoustic_contact_damping_per_N', -1)]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'acoustic'):
                replace(self.p, **{name: value})


class AcousticAppTests(unittest.TestCase):
    def test_controls_recording_pause_ringdown_and_reset(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'finger_physical_app.py')).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.toggle), 6)
        app.button(key='wind').click().run(timeout=20)
        self.assertEqual(app.slider(key='motor_target_deg').value, 10.)
        app.run(timeout=20)
        self.assertAlmostEqual(math.degrees(app.session_state['finger_model'].motor_target_rad), 10.)
        # Hold mechanics at a real settled pose so UI assertions are deterministic.
        finger = app.session_state['finger_model']
        finger.set_motor_target(90)
        finger.advance(1.)
        app.session_state['finger_paused'] = True
        app.toggle(key='acoustic_tx_4').set_value(True)
        app.toggle(key='acoustic_running').set_value(True).run(timeout=20)
        self.assertFalse(app.exception)
        model = app.session_state['finger_acoustics']
        self.assertGreater(np.max(abs(model.last_volts[:, 3])), 0.)
        self.assertGreater(np.max(abs(model.history[:, 8])), 0.)
        np.testing.assert_array_equal(model.history[:, 5:8], 0.)
        app.number_input(key='acoustic_frequency_4').set_value(400.).run(timeout=20)
        self.assertFalse(app.exception)
        app.slider(key='acoustic_contact_position').set_value(35.)
        app.slider(key='acoustic_contact_force').set_value(1.5)
        app.toggle(key='acoustic_contact_enabled').set_value(True).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(model.contact_active)
        self.assertAlmostEqual(model.contact_position_ratio, .35)
        self.assertEqual(model.contact_force_N, 1.5)
        self.assertGreater(model.contact_stiffness_N_m, 0)
        # Pausing holds all modal states and the acoustic clock.
        snapshot, time_before = model.state.copy(), model.time_s
        app.toggle(key='acoustic_running').set_value(False).run(timeout=20)
        np.testing.assert_array_equal(app.session_state['finger_acoustics'].state, snapshot)
        self.assertEqual(app.session_state['finger_acoustics'].time_s, time_before)
        app.toggle(key='acoustic_tx_4').set_value(False)
        app.toggle(key='acoustic_running').set_value(True).run(timeout=20)
        self.assertGreater(np.max(abs(app.session_state['finger_acoustics'].last_volts)), 0.)
        app.multiselect(key='acoustic_receivers').set_value(['RX4 (V)']).run(timeout=20)
        self.assertFalse(app.exception)
        app.button(key='acoustic_reset').click().run(timeout=20)
        self.assertEqual(app.session_state['finger_acoustics'].time_s, 0.)
        self.assertEqual(len(app.session_state['finger_acoustics'].history), 0)
        app.button(key='reset').click().run(timeout=20)
        self.assertFalse(app.exception)
        self.assertFalse(app.toggle(key='acoustic_running').value)
        self.assertTrue(all(not app.toggle(key=f'acoustic_tx_{i}').value for i in range(1, 5)))
        self.assertFalse(app.toggle(key='acoustic_contact_enabled').value)


if __name__ == '__main__':
    unittest.main()
