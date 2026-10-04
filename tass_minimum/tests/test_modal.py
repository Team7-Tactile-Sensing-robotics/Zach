"""Physical invariants and integration checks for routed modal sweeps."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from tass import ModelConfig, run_modal_sweep
from tass.models.modal_string import ModalString
from tass.models.tendon import Tendon
from tass.modal_sweep import save_modal_sweep


class ModalTests(unittest.TestCase):
    def test_frequency_scaling_and_source_response(self):
        model = ModalString(L=0.12, mu=0.002, modes=3)
        model.set_tension(4)
        expected = np.arange(1, 4)/(2*0.12)*np.sqrt(4/0.002)
        np.testing.assert_allclose(model.natural_frequencies(), expected)
        f, drive, position = 200, 1+2j, 0.3
        source_formula = drive*np.sin(np.pi*np.arange(1, 4)*position)/(f-expected+0.02j*expected)
        np.testing.assert_allclose(model.modal_response_to_drive(f, drive, position), source_formula)
        model.set_tension(16)
        np.testing.assert_allclose(model.natural_frequencies(), 2*expected)

    def test_routed_geometry_reciprocity_and_endpoint(self):
        config = ModelConfig()
        straight = run_modal_sweep(config, [100, 200], tension_N=4)
        bent = run_modal_sweep(config, [100, 200], joint_angle_rad=np.pi/4, tension_N=4)
        self.assertLess(bent.metadata['path_length_m'], straight.metadata['path_length_m'])
        self.assertGreater(bent.natural_frequencies_Hz[0], straight.natural_frequencies_Hz[0])
        self.assertEqual(bent.response.shape, (4, 2, 4))
        np.testing.assert_allclose(bent.response, bent.response.transpose(2, 1, 0), atol=1e-15)
        np.testing.assert_array_equal(bent.response[3], 0)
        np.testing.assert_array_equal(bent.response[:, :, 3], 0)
        self.assertAlmostEqual(bent.guide_positions_m[-1], bent.metadata['path_length_m'])

    def test_elastic_tension_physical_rest_length(self):
        config = ModelConfig()
        result = run_modal_sweep(config, [100], motor_angle_rad=0.1)
        self.assertAlmostEqual(result.metadata['tension_N'],
                               config.axial.stiffness_N_m*config.motor.spool_radius_m*0.1)
        slack = run_modal_sweep(config, [100], motor_angle_rad=0)
        np.testing.assert_array_equal(slack.response, 0)
        # Hold EA fixed: T = EA*(Lref/L0 - 1), so choose L0 for 3 N.
        ea = config.axial.young_modulus_Pa*config.axial.area_m2
        reference = float(Tendon.from_config(config).path_length(0))
        config = replace(config, axial=replace(config.axial, rest_length_m=reference/(1+3/ea)))
        self.assertAlmostEqual(run_modal_sweep(config, [100], motor_angle_rad=0).metadata['tension_N'], 3)

    def test_tendon_matches_physical_extension_law(self):
        tendon = Tendon()
        q, qdot, wound, speed = 0.3, 0.2, 0.04, 0.003
        extension = tendon.path_length(q)-(tendon.rest_length-wound)-tendon.slack
        eps = 1e-6
        derivative = (tendon.path_length(q+eps)-tendon.path_length(q-eps))/(2*eps)
        expected = max(0, tendon.k*extension+tendon.c*(derivative*qdot+speed)) if extension > 0 else 0
        self.assertAlmostEqual(tendon.compute(q, qdot, wound, speed), expected, places=8)

    def test_export_and_invalid_inputs(self):
        config = ModelConfig()
        result = run_modal_sweep(config, [100, 200], tension_N=4)
        with tempfile.TemporaryDirectory() as directory:
            save_modal_sweep(result, config, directory)
            with np.load(Path(directory)/'modal_sweep.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['response'], result.response)
            self.assertTrue((Path(directory)/'modal_sweep.png').is_file())
        for frequencies in ([], [-1], [np.nan]):
            with self.assertRaises(ValueError):
                run_modal_sweep(config, frequencies)
        with self.assertRaises(ValueError):
            run_modal_sweep(config, [100], tension_N=-1)
        with self.assertRaises(ValueError):
            ModalString(modes=0)


if __name__ == '__main__':
    unittest.main()
