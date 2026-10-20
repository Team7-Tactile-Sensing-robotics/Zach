import csv
from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tass.ml import (FeatureConfig, Segment, extract_features, load_bundle, predict,
                     train, write_feature_set, _command_voltage, load_capture_csv, _signal_features)


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
    def test_band_peaks_keep_low_mode_when_global_peak_changes_and_ignore_gain(self):
        t=np.arange(1000)/1000
        low=np.sin(2*np.pi*50*t)
        high=np.sin(2*np.pi*400*t)
        bands=((20.,200.),(200.,450.))
        a=_signal_features(low+.9*high,1000,bands,spectral_mode='band_peaks')
        b=_signal_features(low+1.1*high,1000,bands,spectral_mode='band_peaks')
        self.assertEqual(a['dominant_frequency_hz'],50.)
        self.assertEqual(b['dominant_frequency_hz'],400.)
        for result in (a,b):
            self.assertEqual(result['band_peak_frequency_20_200_hz'],50.)
            self.assertEqual(result['band_peak_frequency_200_450_hz'],400.)
        scaled=_signal_features(7*(low+.9*high),1000,bands,spectral_mode='band_peaks')
        for key in a:
            if key.startswith(('band_peak_frequency_','band_fraction_')):
                self.assertAlmostEqual(a[key],scaled[key])
        item=segment('modes','train')
        rows,names,_=extract_features([item],FeatureConfig(spectral_mode='band_peaks'))
        self.assertTrue(any('band_peak_frequency_' in name for name in names))
        self.assertFalse(any('dominant_frequency_hz' in name or 'spectral_snr_db' in name for name in names))
        legacy,legacy_names,_=extract_features([item],FeatureConfig())
        self.assertFalse(any('band_peak_frequency_' in name for name in legacy_names))
        with self.assertRaisesRegex(ValueError,'spectral_mode'):
            FeatureConfig(spectral_mode='unknown').validate()

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
        rows, names, _ = extract_features([item], FeatureConfig(sensor_representation='auto'))
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
        # Ground-truth mechanics and targets must never enter the physics calculations.
        item.columns['string_tension_N']=np.full(len(time),1e9)
        item.columns['contact_force_N']=np.full(len(time),999.)
        changed,changed_names,_=extract_features([item],FeatureConfig(
            sensor_ids=(1,2),use_reference_features=True,physics_enabled=True,
            physics_string_length_m=.2,physics_linear_density_kg_m=.002,bands_Hz=((5,15),)))
        self.assertEqual(names,changed_names)
        np.testing.assert_array_equal([rows[0][name] for name in names],
                                      [changed[0][name] for name in names])

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
            self.assertTrue((root/'model/training_loss.png').is_file())
            self.assertTrue((root/'model/training_loss.svg').is_file())
            selected=report['targets']['contact_force_N']['selected_model']
            expected=np.mean((bundle['models']['contact_force_N'].predict(
                np.array([[row[name] for name in names] for row in rows[:3]])) - 1.)**2)
            self.assertAlmostEqual(report['targets']['contact_force_N']['training_candidates'][selected]['mse'],expected)
            capture = root / "capture.csv"
            t = segments[-1].time_s
            with capture.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["segment_id", "time_s", "sample_index", "sample_rate_hz",
                                 "piezo_sensor_1_voltage_v"])
                writer.writerows([["new", time, i, 100, value] for i, (time, value) in enumerate(zip(t, segments[-1].columns["piezo_sensor_1_voltage_v"]))])
            predictions = predict(root / "model/model_bundle.pkl", capture, root / "predictions.csv")
            self.assertIn("predicted_contact_force_N", predictions[0])
            self.assertTrue((root/'predictions.png').is_file())

    def test_segment_cannot_cross_splits(self):
        config = FeatureConfig(bands_Hz=((5, 15),))
        rows, names, provenance = extract_features([
            segment("duplicate", "train"), segment("duplicate", "validation"), segment("test", "test")], config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_feature_set(rows, names, provenance, config, root)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                train(root, root / "model")

    def test_ridge_only_trains_independent_single_target_models(self):
        items=[segment(f'train{i}','train') for i in range(3)]
        items += [segment('validation','validation'),segment('test','test')]
        cfg=FeatureConfig()
        rows,names,provenance=extract_features(items,cfg)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            report=train(root/'features',root/'model',regressor='ridge',plots=False)
            bundle=load_bundle(root/'model/model_bundle.pkl')
            self.assertEqual(report['regressor_family'],'ridge')
            location=bundle['models']['contact_location_m']
            force=bundle['models']['contact_force_N']
            self.assertIsNot(location,force)
            self.assertIsNot(location.named_steps['scaler'],force.named_steps['scaler'])
            for target,model in bundle['models'].items():
                self.assertEqual(type(model.named_steps['regressor']).__name__,'Ridge')
                self.assertEqual(model.named_steps['regressor'].coef_.shape,(len(names),))
                self.assertTrue(all(name.startswith('ridge_alpha_') for name in
                                    report['targets'][target]['validation_candidates']))
            with self.assertRaisesRegex(ValueError,'only to regression'):
                train(root/'features',root/'invalid',task='classification',regressor='ridge',plots=False)

    def test_adc_voltage_and_voltage_features_match_after_identical_conversion(self):
        raw = segment("raw", "train", "piezo_sensor_1_raw_adc")
        counts = np.rint(2048 + 100*raw.columns["piezo_sensor_1_raw_adc"])
        raw.columns["piezo_sensor_1_raw_adc"] = counts
        voltage = segment("voltage", "train")
        voltage.columns["piezo_sensor_1_voltage_v"] = counts*(3.3/4095)
        # A contradictory diagnostic voltage must not override raw acquisition.
        raw.columns["piezo_sensor_1_voltage_v"] = np.zeros(len(counts))
        cfg = FeatureConfig(sensor_representation="adc_voltage",adc_volts_per_count=3.3/4095,
                            remove_sensor_mean=True)
        actual,names,provenance = extract_features([raw],cfg)
        expected,other_names,_ = extract_features([voltage],FeatureConfig(remove_sensor_mean=True))
        self.assertEqual(names,other_names)
        np.testing.assert_allclose([actual[0][name] for name in names],
                                   [expected[0][name] for name in names],rtol=1e-10,atol=1e-12)
        self.assertAlmostEqual(actual[0]["sensor_1_voltage__mean"],0,places=12)
        self.assertEqual(provenance[names[0]]["kind"],"adc_to_voltage_derived")

    def test_command_mapping_and_missing_channels(self):
        cfg = FeatureConfig(use_actuator_features=True,actuator_command_min=0,
                            actuator_command_max=255)
        np.testing.assert_allclose(_command_voltage([0,127.5,255],cfg),[-3.3,0,3.3])
        with self.assertRaisesRegex(ValueError,"command range"):
            _command_voltage([256],cfg)
        item = segment("commands","train")
        for i in (1,2):
            item.columns[f"piezo_actuator_{i}_command"] = np.full(100,127.5)
        rows,names,provenance = extract_features([item],cfg)
        key = "actuator_1_command_voltage__rms"
        self.assertEqual(rows[0][key],0)
        self.assertFalse(provenance[key]["measured_voltage"])
        with self.assertRaisesRegex(ValueError,"Missing requested"):
            extract_features([item],FeatureConfig(sensor_ids=(1,2)))
        with self.assertRaisesRegex(ValueError,"adc_volts_per_count"):
            FeatureConfig(sensor_representation="adc_voltage").validate()

    def test_adc_mapping_and_command_preprocessing_survive_model_bundle(self):
        items = [segment(f"train{i}","train","piezo_sensor_1_raw_adc") for i in range(3)]
        items += [segment("validation","validation","piezo_sensor_1_raw_adc"),
                  segment("test","test","piezo_sensor_1_raw_adc")]
        for item in items:
            item.columns["piezo_sensor_1_raw_adc"] = np.rint(2048+100*item.columns["piezo_sensor_1_raw_adc"])
            for i in (1,2):
                item.columns[f"piezo_actuator_{i}_command"] = np.sin(2*np.pi*10*item.time_s)
        cfg = FeatureConfig(sensor_representation="adc_voltage",adc_volts_per_count=3.3/4095,
                            remove_sensor_mean=True,use_actuator_features=True)
        rows,names,provenance = extract_features(items,cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/"features")
            train(root/"features",root/"model")
            bundle = load_bundle(root/"model/model_bundle.pkl")
            self.assertEqual(bundle["feature_config"]["adc_volts_per_count"],3.3/4095)
            capture = root/"capture.csv"
            with capture.open("w",newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s","piezo_sensor_1_raw_adc","piezo_actuator_1_command","piezo_actuator_2_command"])
                writer.writerows(zip(items[-1].time_s,items[-1].columns["piezo_sensor_1_raw_adc"],
                    items[-1].columns["piezo_actuator_1_command"],items[-1].columns["piezo_actuator_2_command"]))
            result = predict(root/"model/model_bundle.pkl",capture,root/"predictions.csv")
            self.assertEqual(len(result),1)
            self.assertTrue(np.isfinite(result[0]["predicted_contact_location_m"]))

    def test_indexed_capture_uses_measured_rate_and_rejects_clock_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"capture.csv"
            with path.open("w",newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["segment_id","sample_index","measured_sample_rate_hz","piezo_sensor_1_raw_adc"])
                writer.writerows(["capture",i,74970,1887+10*np.sin(2*np.pi*1000*i/74970)] for i in range(1000))
            item = load_capture_csv(path)[0]
            self.assertAlmostEqual(1/np.median(np.diff(item.time_s)),74970,places=6)
            cfg = FeatureConfig(window_s=.01,stride_s=.01,sensor_representation="adc_voltage",
                                adc_volts_per_count=3.3/4095,remove_sensor_mean=True)
            rows,_,_ = extract_features([item],cfg)
            self.assertAlmostEqual(rows[0]["sample_rate_hz"],74970,places=6)
            self.assertEqual(rows[0]["sensor_1_voltage__mean"],0)
            item.columns["sample_rate_hz"] = np.full(1000,75000)
            with self.assertRaisesRegex(ValueError,"disagrees with timestamps"):
                extract_features([item],cfg)

    def test_voltage_mode_uses_simulation_volts_and_calibrates_only_real_adc(self):
        sim=segment('sim','train')
        rng=np.random.default_rng(12)
        codes=rng.integers(1000,3000,len(sim.time_s))
        slope,offset=.002,-.4
        sim.columns['piezo_sensor_1_voltage_v']=codes*slope+offset
        # Contradictory simulated ADC must have no influence on voltage features.
        sim.columns['piezo_sensor_1_raw_adc']=np.zeros(len(codes))
        real=segment('real','test','piezo_sensor_1_raw_adc')
        real.source='real'
        real.columns['piezo_sensor_1_raw_adc']=codes.copy()
        real.columns['piezo_sensor_1_voltage_v']=np.full(len(codes),999.)
        cfg=FeatureConfig(sensor_representation='voltage',adc_volts_per_count=slope,
                          adc_voltage_offset_v=offset,remove_sensor_mean=False)
        sim_rows,names,sim_provenance=extract_features([sim],cfg)
        real_rows,real_names,real_provenance=extract_features([real],cfg)
        self.assertEqual(names,real_names)
        np.testing.assert_allclose([sim_rows[0][key] for key in names],
                                   [real_rows[0][key] for key in names],rtol=1e-12,atol=1e-12)
        self.assertEqual(sim_provenance[names[0]]['source'],'piezo_sensor_1_voltage_v')
        self.assertEqual(real_provenance[names[0]]['kind'],'adc_to_voltage_derived')
        # Offset is applied before the common DC-removal step.
        cfg=replace(cfg,remove_sensor_mean=True)
        centered,_,_=extract_features([real],cfg)
        self.assertEqual(centered[0]['sensor_1_voltage__mean'],0)
        np.testing.assert_array_equal(real.columns['piezo_sensor_1_raw_adc'],codes)
        # Simulation needs no calibration and must not fall back to ADC if volts are absent.
        extract_features([sim],FeatureConfig())
        del sim.columns['piezo_sensor_1_voltage_v']
        with self.assertRaisesRegex(ValueError,'Missing requested voltage'):
            extract_features([sim],cfg)
        with self.assertRaisesRegex(ValueError,'Real raw ADC input requires'):
            extract_features([real],FeatureConfig())

    def test_voltage_model_raw_capture_calibration_override_preserves_original(self):
        items=[segment(f'train{i}','train') for i in range(3)]
        items += [segment('validation','validation'),segment('test','test')]
        cfg=FeatureConfig(remove_sensor_mean=True)
        rows,names,provenance=extract_features(items,cfg)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            train(root/'features',root/'model',plots=False)
            capture=root/'raw.csv'
            signal=items[-1].columns['piezo_sensor_1_voltage_v']
            codes=signal/.01+1887
            with capture.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['time_s','piezo_sensor_1_raw_adc'])
                writer.writerows(zip(items[-1].time_s,codes))
            digest=hashlib.sha256(capture.read_bytes()).hexdigest()
            model=root/'model/model_bundle.pkl'
            with self.assertRaisesRegex(ValueError,'Real raw ADC input requires'):
                predict(model,capture,root/'bad.csv',plots=False)
            result=predict(model,capture,root/'predictions.csv',adc_volts_per_count=.01,
                           adc_voltage_offset_v=-18.87,plots=False)
            self.assertEqual(len(result),1)
            self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),digest)
            applied=json.loads((root/'predictions.preprocessing.json').read_text())['feature_config']
            self.assertEqual(applied['adc_volts_per_count'],.01)
            self.assertIsNone(load_bundle(model)['feature_config']['adc_volts_per_count'])
            with self.assertRaisesRegex(ValueError,'must not overwrite'):
                predict(model,capture,capture,adc_volts_per_count=.01,plots=False)
            self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),digest)

    def test_physics_model_recomputes_features_automatically_at_inference(self):
        items=[segment(f'train{i}','train') for i in range(3)]
        items += [segment('validation','validation'),segment('test','test')]
        for item in items:
            t=item.time_s
            item.columns['piezo_sensor_1_voltage_v']=np.sin(2*np.pi*20*t)
            item.columns['piezo_sensor_2_voltage_v']=2*np.sin(2*np.pi*20*t)
            item.columns['piezo_sensor_1_reference_voltage_v']=.5*np.sin(2*np.pi*10*t)
            item.columns['piezo_sensor_2_reference_voltage_v']=np.sin(2*np.pi*10*t)
        cfg=FeatureConfig(sensor_ids=(1,2),use_reference_features=True,physics_enabled=True,
            physics_string_length_m=.2,physics_linear_density_kg_m=.002,remove_sensor_mean=True)
        rows,names,provenance=extract_features(items,cfg)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            train(root/'features',root/'model',plots=False)
            model=root/'model/model_bundle.pkl'
            self.assertTrue(load_bundle(model)['feature_config']['physics_enabled'])
            capture=root/'raw.csv'
            volt_names=['piezo_sensor_1_voltage_v','piezo_sensor_2_voltage_v',
                        'piezo_sensor_1_reference_voltage_v','piezo_sensor_2_reference_voltage_v']
            raw_names=[name.replace('voltage_v','raw_adc') for name in volt_names]
            codes=np.column_stack([np.rint(2000+100*items[-1].columns[name]) for name in volt_names])
            with capture.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['time_s',*raw_names])
                writer.writerows(zip(items[-1].time_s,*codes.T))
            digest=hashlib.sha256(capture.read_bytes()).hexdigest()
            result=predict(model,capture,root/'predictions.csv',adc_volts_per_count=.01,
                           adc_voltage_offset_v=-20.,plots=False)
            self.assertEqual(len(result),1)
            applied=json.loads((root/'predictions.preprocessing.json').read_text())
            self.assertTrue(applied['feature_config']['physics_enabled'])
            self.assertEqual(applied['feature_config']['physics_sensor_pair'],[1,2])
            self.assertEqual(applied['feature_provenance']['x_physics_m']['kind'],'physics_derived')
            self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),digest)
            # A physics model needs reference signals at inference as it did during training.
            incomplete=root/'missing_reference.csv'
            with incomplete.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['time_s',*raw_names[:2]])
                writer.writerows(zip(items[-1].time_s,*codes[:,:2].T))
            with self.assertRaisesRegex(ValueError,'Reference features requested'):
                predict(model,incomplete,root/'bad.csv',adc_volts_per_count=.01,plots=False)


if __name__ == "__main__":
    unittest.main()
