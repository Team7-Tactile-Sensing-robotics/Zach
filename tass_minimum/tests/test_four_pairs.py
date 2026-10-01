import unittest
import warnings
from dataclasses import replace
import numpy as np
from tass import ModelConfig, Inputs, simulate
from tass.signals import Constant, Pulse
from tass.models.tendon import Tendon
from tass.processing import extract_frequency_features


class FourPairTests(unittest.TestCase):
    def config(self, tail=0):
        c = ModelConfig()
        return replace(c, routed_pairs=replace(c.routed_pairs, enabled=True, anchor_extension_m=tail),
                       numerics=replace(c.numerics, duration_s=0.012))

    def run_pairs(self, drives, tail=0, contact=0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return simulate(self.config(tail), Inputs(actuator_forces_N=drives, contact_force_N=Constant(contact)))

    def test_four_channels_and_segment_positions(self):
        result = self.run_pairs({'pair1': Pulse()})
        self.assertEqual(result.measurement.piezo_V.shape[1], 4)
        self.assertEqual(result.metadata['pair_segments'], [1, 1, 2, 2])
        channels = result.measurement.channels
        tendon = Tendon.from_config(self.config())
        positions = tendon.guide_positions(channels['joint_angle_rad'])
        for i, point in enumerate(positions):
            np.testing.assert_allclose(channels[f'pair{i+1}_world_x_m'], np.broadcast_to(point, (len(channels['joint_angle_rad']), 2))[:, 0])
            self.assertIn(f'actuator_pair{i+1}_force_N', channels)
        np.testing.assert_allclose(result.metadata['pair_path_positions_m'], tendon.guide_path_coordinates(0))
        self.assertEqual(result.metadata['scalar_frf_reference'], 'pair1')
        self.assertEqual(result.measurement.piezo_V[:, 3].max(), 0)

    def test_independent_drives_superpose_without_false_scalar_frf(self):
        first = self.run_pairs({'pair1': Pulse()})
        third = self.run_pairs({'pair3': Pulse()})
        both = self.run_pairs({'pair1': Pulse(), 'pair3': Pulse()})
        np.testing.assert_allclose(both.measurement.piezo_V, first.measurement.piezo_V+third.measurement.piezo_V, atol=1e-14)
        self.assertGreater(np.max(abs(third.measurement.piezo_V[:, 2])), 0)
        self.assertFalse(both.metadata['scalar_frf_enabled'])
        self.assertFalse(extract_frequency_features(both.measurement).frf_valid.any())

    def test_fourth_pair_can_drive_if_anchor_is_physically_beyond_it(self):
        at_anchor = self.run_pairs({'pair4': Pulse()})
        np.testing.assert_array_equal(at_anchor.measurement.piezo_V, 0)
        beyond = self.run_pairs({'pair4': Pulse()}, tail=0.01)
        self.assertGreater(np.max(abs(beyond.measurement.piezo_V[:, 3])), 0)
        self.assertEqual(beyond.metadata['fixed_endpoint_actuator_indices'], [])

    def test_empty_drives_and_bad_ids(self):
        empty = self.run_pairs({})
        np.testing.assert_array_equal(empty.measurement.piezo_V, 0)
        with self.assertRaises(ValueError):
            self.run_pairs({'pair5': Pulse()})
        with self.assertRaises(ValueError):
            simulate(ModelConfig(), Inputs(actuator_forces_N={'pair1': Pulse()}))

    def test_contact_and_saved_configuration_replay(self):
        original = self.run_pairs({'pair3': Pulse()}, contact=0.4)
        node = round(original.metadata['contact_actual_position_m']/original.metadata['dx_m'])
        np.testing.assert_array_equal(original.displacement_m[:, node], 0)
        # Returned config remains the user configuration, not temporary solver geometry.
        self.assertEqual(original.config['string']['length_m'], ModelConfig().string.length_m)
        self.assertEqual(original.config['routed_pairs']['enabled'], True)
