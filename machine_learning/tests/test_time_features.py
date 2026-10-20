"""Known packet times, safe missing echoes and matched feature ablation checks."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from machine_learning.data import load_config
from machine_learning.features import extract_features, spectrum_features
from machine_learning.time_features import time_features, echo_features
from machine_learning.synthetic import make_samples, make_echo_samples
from machine_learning.comparison import compare_features, feature_sets, make_estimator, paired_group_interval
from machine_learning.experiments import split_groups


class TimeFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fs = 8192
        self.t = np.arange(1024)/self.fs

    def packet(self, center, amplitude=1):
        return amplitude*np.exp(-.5*((self.t-center)/.0015)**2)*np.cos(2*np.pi*1024*(self.t-center))

    def echo_config(self):
        return {**self.cfg, "echo_features_enabled": True, "echo_gate_s": [.01, .11],
                "echo_min_peak_distance_s": .01}

    def test_energy_intensity_and_segmentation_known_sine(self):
        x = 2*np.sin(2*np.pi*256*self.t)
        f = time_features(x, self.fs, self.cfg)
        self.assertAlmostEqual(f["time_energy_V2_s"], .25)
        self.assertAlmostEqual(f["time_peak_envelope_V"], 2)
        for i in range(1, 5):
            self.assertAlmostEqual(f[f"time_segment{i}_rms_V"], np.sqrt(2))
        self.assertTrue(all(v == 0 for v in time_features(np.zeros(1024), self.fs, self.cfg).values()))

    def test_known_echo_times_amplitudes_and_emission_gate(self):
        x = self.packet(.003, 5)+self.packet(.035, 1.2)+self.packet(.075, .8)
        f = echo_features(x, self.t, 0, self.echo_config())
        for i, expected in [(1, .035), (2, .075)]:
            self.assertEqual(f[f"echo_peak{i}_present"], 1)
            self.assertAlmostEqual(f[f"echo_peak{i}_time_s"], expected, delta=2/self.fs)
        self.assertAlmostEqual(f["echo_peak1_amplitude_V"], 1.2, delta=.05)
        self.assertAlmostEqual(f["echo_peak2_amplitude_V"], .8, delta=.05)

    def test_trigger_reference_not_absolute_clock_and_missing_peak_flags(self):
        x = self.packet(.04)
        original = echo_features(x, self.t, 0, self.echo_config())
        shifted_clock = echo_features(x, self.t+100, 100, self.echo_config())
        np.testing.assert_allclose(list(original.values()), list(shifted_clock.values()), atol=1e-11)
        self.assertEqual(original["echo_peak2_present"], 0)
        self.assertEqual(original["echo_peak2_time_s"], 0)
        self.assertTrue(all(v == 0 for v in echo_features(np.zeros(1024), self.t, 0, self.echo_config()).values()))
        with self.assertRaisesRegex(ValueError, "incomplete_echo_gate"):
            echo_features(x[:300], self.t[:300], 0, self.echo_config())

    def test_time_shift_changes_arrival_without_changing_unwindowed_fft_magnitude(self):
        x = self.packet(.035)
        shifted = np.roll(x, 100)
        np.testing.assert_allclose(abs(np.fft.rfft(x)), abs(np.fft.rfft(shifted)), atol=1e-12)
        a = echo_features(x, self.t, 0, self.echo_config())
        b = echo_features(shifted, self.t, 0, self.echo_config())
        self.assertAlmostEqual(b["echo_peak1_time_s"]-a["echo_peak1_time_s"], 100/self.fs, delta=1/self.fs)

    def test_feature_toggle_keeps_legacy_fft_identical_and_requires_trigger(self):
        raw = make_samples(self.cfg, sessions=1, trials_per_session=2)
        basic, _ = extract_features(raw, {**self.cfg, "time_features_enabled": False})
        expanded, _ = extract_features(raw, self.cfg)
        pd.testing.assert_frame_equal(basic, expanded[basic.columns])
        sets = feature_sets(expanded)
        self.assertEqual(len(sets["fft_baseline"]), 60)
        self.assertEqual(len(sets["fft_plus_time"]), 88)
        self.assertFalse(any("_echo_" in c for c in expanded))
        with self.assertRaisesRegex(ValueError, "trigger"):
            extract_features(raw, self.echo_config())
        pulse = make_echo_samples(self.cfg, sessions=1, trials_per_session=1)
        echo, _ = extract_features(pulse, self.echo_config())
        self.assertEqual(len(feature_sets(echo)["fft_plus_time_echo"]), 112)


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(48)
        n = 240
        a, b = rng.normal(size=(2, n))
        return pd.DataFrame(dict(session_id=np.repeat([f"s{i}" for i in range(30)], 8),
                                 trial_id=[f"t{i}" for i in range(n)], window_id=[f"w{i}" for i in range(n)],
                                 data_source="synthetic", rx1_rms_V=a, rx1_time_peak_envelope_V=b,
                                 contact_force_N=2+a+2*b+rng.normal(0, .3, n),
                                 contact_location_m=.2+.03*b+rng.normal(0, .005, n)))

    def test_paired_interval_recovers_known_error_reduction(self):
        y = np.ones(100)
        result = paired_group_interval(y, y+2, y+.5, np.repeat(np.arange(10), 10), 42)
        self.assertEqual(result["mae_change"], -1.5)
        self.assertEqual(result["bootstrap_95_low"], -1.5)
        self.assertEqual(result["bootstrap_95_high"], -1.5)

    def test_matched_rows_fold_isolation_holdout_invariance_and_reload(self):
        cfg = {**load_config(), "svr_C": [1.], "svr_gamma": ["scale"], "svr_epsilon": [.1]}
        frame = self.fixture()
        train, holdout = split_groups(frame, cfg)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            compare_features(frame, cfg, a)
            changed = frame.copy()
            changed.loc[holdout, "contact_force_N"] += 100
            compare_features(changed, cfg, b)
            pd.testing.assert_frame_equal(pd.read_csv(a/"cv_search.csv"), pd.read_csv(b/"cv_search.csv"))
            predictions = pd.read_csv(a/"holdout_predictions.csv")
            for _, rows in predictions.groupby("target"):
                memberships = [set(g.window_id) for _, g in rows.groupby(["feature_set", "family"])]
                self.assertTrue(all(g == memberships[0] for g in memberships))
            cv = pd.read_csv(a/"cv_manifest.csv").merge(frame[["window_id", "session_id"]], on="window_id")
            for _, fold in cv.groupby(["target", "fold"]):
                self.assertFalse(set(fold[fold.role == "train"].session_id) & set(fold[fold.role == "validation"].session_id))
                self.assertFalse(set(fold.session_id) & set(frame.iloc[holdout].session_id))
            model = joblib.load(a/"contact_force_N_svr_rbf_fft_plus_time.joblib")
            rows = predictions[(predictions.target == "contact_force_N") & (predictions.family == "svr_rbf") &
                               (predictions.feature_set == "fft_plus_time")]
            inputs = frame.set_index("window_id").loc[rows.window_id]
            np.testing.assert_allclose(model.predict(inputs), rows.predicted)
            scaler = model.estimator.named_steps["scale"]
            np.testing.assert_allclose(scaler.mean_, frame.iloc[train][model.features].mean())
            self.assertLess(pd.read_csv(a/"paired_mae_changes.csv").mae_change.max(), 0)

    def test_target_scaling_and_constant_features_are_fold_local(self):
        cfg = load_config()
        model = make_estimator("svr_rbf", dict(C=1., gamma="scale", epsilon=.1), cfg)
        x = np.column_stack([np.arange(50), np.ones(50)])
        y = 100+np.arange(50)*10
        model.fit(x, y)
        self.assertEqual(model.named_steps["pca"].pca_.n_components_, 1)
        self.assertAlmostEqual(model.named_steps["regressor"].transformer_.mean_[0], y.mean())
        self.assertTrue(np.isfinite(model.predict(x)).all())


if __name__ == "__main__":
    unittest.main()
