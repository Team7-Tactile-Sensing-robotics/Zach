import tempfile
import unittest
from pathlib import Path

import numpy as np

from tass.adc import ADCConfig, fit_affine_calibration, observe_adc
from tass.dataset import Protocol, recorded_channels


class ADCObservationTests(unittest.TestCase):
    def test_integer_range_clipping_and_real_schema(self):
        config = ADCConfig(enabled=True, resolution_bits=12, attenuation_db=11,
                           voltage_to_code_gain_counts_per_V=1000,
                           zero_voltage_code=2048, parameter_source="assumed")
        voltage = np.array([[-10.0], [0.0], [10.0]])
        counts = observe_adc(voltage, config, 1000, np.random.default_rng(1))
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))
        np.testing.assert_array_equal(counts[:, 0], [0, 2048, 4095])

        time = np.arange(3)/1000
        volts = np.repeat(voltage, 4, axis=1)
        drives = np.zeros_like(volts)
        metadata = {"actual_force_N": 2.0, "contact_segment_requested_m": None,
                    "contact_snapped_m": 0.1, "contact_from_joint_reference_m": 0.02,
                    "motor_angle_rad": 0.3, "servo_torque_Nm": 0.4,
                    "tension_N": 5.0, "joint_angle_rad": 0.0,
                    "spring_resisting_torque_Nm": 0.0}
        channels = recorded_channels(time, volts, drives, metadata, Protocol(),
                                     raw_adc=np.repeat(counts, 4, axis=1))
        self.assertIn("piezo_sensor_1_raw_adc", channels)
        self.assertIn("piezo_sensor_1_V", channels)
        self.assertIn("load_cell_readout_N", channels)
        self.assertEqual(channels["piezo_sensor_1_raw_adc"].dtype, counts.dtype)

    def test_affine_calibration_uses_provided_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"calibration.csv"
            path.write_text("adc_input_voltage_v,raw_adc\n0,100\n0.5,600\n1,1100\n")
            fitted = fit_affine_calibration(path)
        self.assertAlmostEqual(fitted["voltage_to_code_gain_counts_per_V"], 1000)
        self.assertAlmostEqual(fitted["zero_voltage_code"], 100)
        self.assertEqual(fitted["parameter_source"], "calibrated")

    def test_enabled_model_rejects_unknown_mapping(self):
        with self.assertRaisesRegex(ValueError, "voltage_to_code"):
            ADCConfig(enabled=True, resolution_bits=12,
                      zero_voltage_code=0, parameter_source="assumed").validate(1000)


if __name__ == "__main__":
    unittest.main()
