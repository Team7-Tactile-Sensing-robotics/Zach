"""Configuration round trips and observable parameter effects."""
from dataclasses import replace, fields
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tass import ModelConfig, run_modal_sweep
from tass.cli import scenario_inputs


class ConfigurationTests(unittest.TestCase):
    def test_presets_are_complete_and_roundtrip(self):
        root = Path(__file__).resolve().parents[1]/'config'
        for path in root.glob('*.json'):
            values = json.loads(path.read_text())
            config = ModelConfig.load(path)
            self.assertEqual(set(values), {f.name for f in fields(config)})
            for name in values:
                self.assertEqual(set(values[name]), {f.name for f in fields(getattr(config, name))})
            with tempfile.TemporaryDirectory() as directory:
                for suffix in ('json', 'yaml'):
                    target = Path(directory)/f'config.{suffix}'
                    config.save(target)
                    self.assertEqual(json.dumps(ModelConfig.load(target).to_dict()), json.dumps(config.to_dict()))

    def test_modal_channels_change_actual_response(self):
        config = ModelConfig()
        baseline = run_modal_sweep(config, [100, 200], tension_N=4)
        pairs = replace(config.modal_piezo, actuator_amplitudes_au=(2, 0, 1, 1),
                        actuator_phases_rad=(np.pi/2, 0, 0, 0), sensor_gains=(1, 3, 1, 1))
        result = run_modal_sweep(replace(config, modal_piezo=pairs), [100, 200], tension_N=4)
        expected = baseline.response*np.array([2j, 0, 1, 1])[:, None, None]*np.array([1, 3, 1, 1])[None, None, :]
        np.testing.assert_allclose(result.response, expected, atol=1e-15)
        reordered = replace(config.modal_piezo, guide_numbers=(4, 2, 3, 1))
        result = run_modal_sweep(replace(config, modal_piezo=reordered), [100], tension_N=4)
        np.testing.assert_array_equal(result.response[0], 0)
        self.assertEqual(result.metadata['fixed_endpoint_pair_id'], 'pair1')

    def test_experiment_settings_drive_signals(self):
        config = ModelConfig()
        experiment = replace(config.experiment, scenario='contact', motor_initial_rad=0.5,
                             contact_force_N=0.7, measured_tension_N=9, excitation_amplitude_N=0.004)
        inputs = scenario_inputs(replace(config, experiment=experiment))
        self.assertEqual(inputs.motor_angle_rad(0), 0.5)
        self.assertEqual(inputs.contact_force_N(0), 0.7)
        self.assertEqual(inputs.tension_N(0), 9)
        self.assertAlmostEqual(inputs.excitation_N(experiment.pulse_start_s+experiment.pulse_width_s/2), 0.004)

    def test_invalid_configuration(self):
        c = ModelConfig()
        bad = [replace(c, modal_piezo=replace(c.modal_piezo, sensor_gains=(1, 2))),
               replace(c, modal_piezo=replace(c.modal_piezo, guide_numbers=(1, 1, 3, 4))),
               replace(c, app=replace(c.app, servo_time_constant_s=0)),
               replace(c, analysis=replace(c.analysis, max_peaks=0))]
        for config in bad:
            with self.assertRaises(ValueError):
                config.validate()
