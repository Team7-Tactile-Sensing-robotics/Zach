import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tass.ml import (FeatureConfig, Segment, extract_features, load_bundle, predict,
                     train, write_feature_set)


def segment(name, split, voltage_name="piezo_sensor_1_voltage_v"):
    rate = 100.0
    time = np.arange(100) / rate
    force = 1.0 + {"train": 0, "validation": 1, "test": 2}[split]
    signal = force * np.sin(2 * np.pi * 10 * time)
    return Segment(name, time, {
        voltage_name: signal,
        "sample_rate_hz": np.full(len(time), rate),
        "contact_force_N": np.full(len(time), force),
        "contact_location_m": np.full(len(time), force / 10),
    }, split=split, source="simulation")


class MLTests(unittest.TestCase):
    def test_common_sensor_names_make_same_features(self):
        config = FeatureConfig(bands_Hz=((5, 15),))
        real = segment("real", "train")
        sim = segment("sim", "train", "piezo_sensor_1_V")
        real.source = "real"
        real_rows, real_names, provenance = extract_features([real], config)
        sim_rows, sim_names, _ = extract_features([sim], config)
        self.assertEqual(real_names, sim_names)
        np.testing.assert_allclose([real_rows[0][name] for name in real_names],
                                   [sim_rows[0][name] for name in sim_names])
        self.assertEqual(provenance[real_names[0]]["source_kind"], "calibrated")

    def test_raw_adc_fallback_and_leakage_guard(self):
        item = segment("a", "train", "piezo_sensor_1_raw_adc")
        # Diagnostic voltage may coexist in simulated files; auto must still
        # select the representation available from ESP32 inference.
        item.columns["piezo_sensor_1_voltage_v"] = np.zeros(len(item.time_s))
        rows, names, _ = extract_features([item], FeatureConfig())
        self.assertTrue(any("raw_adc" in name for name in names))
        self.assertFalse(any("voltage" in name for name in names))
        self.assertFalse(any("contact" in name for name in names))
        with self.assertRaises(ValueError):
            FeatureConfig(context_columns=("load_cell_readout",)).validate()

    def test_two_sensors_and_reference_change_features(self):
        item = segment("paired", "train")
        time = item.time_s
        item.columns["piezo_sensor_1_voltage_v"] = np.sin(2*np.pi*20*time)
        item.columns["piezo_sensor_2_voltage_v"] = 2*np.sin(2*np.pi*20*time)
        item.columns["piezo_sensor_1_reference_voltage_v"] = 0.5*np.sin(2*np.pi*10*time)
        item.columns["piezo_sensor_2_reference_voltage_v"] = np.sin(2*np.pi*10*time)
        rows, names, _ = extract_features([item], FeatureConfig(
            sensor_ids=(1, 2), use_reference_features=True, physics_enabled=True,
            physics_string_length_m=0.2, physics_linear_density_kg_m=0.002,
            bands_Hz=((5, 15),)))
        self.assertTrue(any(name.startswith("sensor_2_") for name in names))
        ratio = rows[0]["sensor_1_voltage__peak_amplitude_change_ratio"]
        self.assertAlmostEqual(ratio, 1.0)
        self.assertAlmostEqual(rows[0]["x_physics_m"], 0.1)
        self.assertAlmostEqual(rows[0]["tendon_delta_tension_N"], 0.0)
        self.assertAlmostEqual(rows[0]["tension_consistency"], 1.0)
        self.assertTrue(0 <= rows[0]["physics_confidence"] <= 1)

    def test_train_bundle_and_predict(self):
        segments = [segment(f"train{i}", "train") for i in range(3)]
        segments += [segment("validation0", "validation"), segment("test0", "test")]
        config = FeatureConfig(bands_Hz=((5, 15),))
        rows, names, provenance = extract_features(segments, config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_feature_set(rows, names, provenance, config, root / "features")
            report = train(root / "features", root / "model", alphas=(0.1, 1.0))
            self.assertEqual(set(report["targets"]), {"contact_force_N", "contact_location_m"})
            bundle = load_bundle(root / "model/model_bundle.pkl")
            self.assertEqual(bundle["feature_order"], names)
            capture = root / "capture.csv"
            t = segments[-1].time_s
            with capture.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["segment_id", "time_s", "sample_index", "sample_rate_hz",
                                 "piezo_sensor_1_voltage_v"])
                writer.writerows([["new", time, i, 100, value] for i, (time, value) in enumerate(zip(t, segments[-1].columns["piezo_sensor_1_voltage_v"]))])
            predictions = predict(root / "model/model_bundle.pkl", capture, root / "predictions.csv")
            self.assertIn("predicted_contact_force_N", predictions[0])

    def test_segment_cannot_cross_splits(self):
        config = FeatureConfig(bands_Hz=((5, 15),))
        rows, names, provenance = extract_features([
            segment("duplicate", "train"), segment("duplicate", "validation"), segment("test", "test")], config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_feature_set(rows, names, provenance, config, root)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                train(root, root / "model")


if __name__ == "__main__":
    unittest.main()
