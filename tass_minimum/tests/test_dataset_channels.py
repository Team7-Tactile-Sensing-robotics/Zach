import unittest
from dataclasses import replace
import numpy as np
from tass.config import ModelConfig
from tass.dataset import Protocol, make_trial, recorded_channels


class RecordedChannelsTests(unittest.TestCase):
    def test_units_balance_and_synchronization(self):
        config = ModelConfig()
        protocol = replace(Protocol(), window_s=0.16)
        t, v, f, meta = make_trial(config, protocol, 0.4, 2, 17)
        channels = recorded_channels(t, v, f, meta, protocol)
        self.assertTrue(all(a.shape == t.shape for a in channels.values()))
        np.testing.assert_allclose(channels['servo_torque_Nm'],
                                   channels['string_tension_N']*config.motor.spool_radius_m)
        for i, gain in enumerate(protocol.actuator_force_per_volt_N_V):
            np.testing.assert_allclose(channels[f'piezo_actuator_{i+1}_V']*gain, f[:, i])
            np.testing.assert_array_equal(channels[f'piezo_sensor_{i+1}_V'], v[:, i])
        self.assertTrue(np.all(channels['contact_location_m'] == meta['contact_snapped_m']))

    def test_invalid_calibration(self):
        for gains in ((1, 2), (0, 1, 1, 1), (float('nan'), 1, 1, 1)):
            with self.assertRaises(ValueError):
                replace(Protocol(), actuator_force_per_volt_N_V=gains).validate()
