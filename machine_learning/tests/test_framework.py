"""Analytical FFT, leakage prevention, inference and end-to-end regression checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from machine_learning.data import load_config, read_samples
from machine_learning.features import spectrum_features, extract_features
from machine_learning.models import Design, Specification, TargetBins, fit_model, coefficient_table
from machine_learning.experiments import run_experiments, split_groups, adjust_pvalues, candidate_specs
from machine_learning.synthetic import make_samples


def feature_fixture():
    rng = np.random.default_rng(107)
    groups = np.repeat(np.arange(40), 10)
    a, b = rng.normal(size=(2, len(groups)))
    noise = rng.normal(scale=.5, size=len(groups))+rng.normal(scale=.15, size=40)[groups]
    return pd.DataFrame(dict(session_id=[f"s{g}" for g in groups],
                             trial_id=[f"t{i}" for i in range(len(groups))],
                             window_id=[f"w{i}" for i in range(len(groups))], data_source="synthetic",
                             rx1_rms_V=a, rx2_rms_V=b,
                             contact_force_N=4+2*a+.5*b+noise,
                             contact_location_m=.2+.03*a-.02*b+.02*noise))


class FFTTests(unittest.TestCase):
    def test_known_sine_amplitude_rms_and_frequency(self):
        fs, n = 8192, 1024
        x = 10+2*np.sin(2*np.pi*256*np.arange(n)/fs)
        f = spectrum_features(x, fs, [[200, 300], [1000, 2000]])
        self.assertAlmostEqual(f["rms_V"], np.sqrt(2))
        self.assertAlmostEqual(f["peak_amplitude_V"], 2)
        self.assertEqual(f["peak_frequency_Hz"], 256)
        self.assertAlmostEqual(f["centroid_Hz"], 256)
        self.assertAlmostEqual(f["band1_fraction"], 1)
        self.assertAlmostEqual(f["band2_fraction"], 0)

    def test_nyquist_and_zero_signals(self):
        f = spectrum_features((-1.)**np.arange(1024), 8192, [[0, 4096]])
        self.assertAlmostEqual(f["peak_amplitude_V"], 1)
        self.assertEqual(f["peak_frequency_Hz"], 4096)
        self.assertTrue(all(v == 0 for v in spectrum_features(np.zeros(1024), 8192, [[0, 1000]]).values()))

    def test_window_quality_gaps_stationarity_and_missing_labels(self):
        cfg = load_config()
        raw = make_samples(cfg, sessions=1, trials_per_session=4)
        n = cfg["window_samples"]
        raw.loc[n//2:n-1, "time_s"] += .01
        raw.loc[n:2*n-1, "quality_flag"] = "clipped"
        raw.loc[2*n:3*n-1, "contact_force_N"] = np.linspace(0, 3, n)
        raw.loc[3*n:, "contact_location_m"] = np.nan
        features, audit = extract_features(raw, cfg)
        self.assertEqual(len(features), 1)
        self.assertTrue(features.contact_location_m.isna().all())
        self.assertEqual((audit.status == "rejected").sum(), 3)
        self.assertIn("timestamp_gap_or_jitter", ";".join(audit.reason))

    def test_rejects_unsorted_time_and_above_nyquist_bands(self):
        cfg = load_config()
        raw = make_samples(cfg, sessions=1, trials_per_session=1)
        wrong = raw.copy()
        wrong.loc[3, "time_s"] = wrong.loc[2, "time_s"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extract_features(wrong, cfg)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            extract_features(raw, {**cfg, "frequency_bands_Hz": [[0, 10000]]})

    def test_csv_schema_and_trial_identity(self):
        cfg = load_config()
        raw = make_samples(cfg, sessions=2, trials_per_session=1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"raw.csv"
            raw.to_csv(path, index=False)
            self.assertEqual(len(read_samples(path, cfg)), len(raw))
            raw["trial_id"] = "reused"
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one session"):
                read_samples(path, cfg)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.frame = feature_fixture()
        self.spec = Specification("test", ("rx1_rms_V", "rx2_rms_V"), True)

    def test_scalers_fit_training_only_and_interactions_keep_parents(self):
        design = Design(self.spec)
        x = design.fit_transform(self.frame.iloc[:200])
        np.testing.assert_allclose(x[:, 1:].mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(x[:, 1:].std(axis=0), 1)
        expected_mean = design.base.mean_.copy()
        shifted = self.frame.iloc[200:].copy()
        shifted["rx1_rms_V"] += 1000
        transformed = design.transform(shifted)
        np.testing.assert_equal(design.base.mean_, expected_mean)
        self.assertGreater(transformed[:, 1].mean(), 100)
        self.assertEqual(design.terms, ["rx1_rms_V", "rx2_rms_V", "rx1_rms_V:rx2_rms_V"])

    def test_continuous_and_logistic_are_parallel_with_valid_probabilities(self):
        y = self.frame.contact_force_N
        bins = TargetBins.fit(y, self.cfg, "contact_force_N")
        for family in ("linear", "logistic"):
            fitted = fit_model(self.frame, y, self.frame.session_id, self.spec, family, bins, self.cfg, True)
            predicted, labels, probs = fitted.predict(self.frame)
            table = pd.DataFrame(coefficient_table(fitted, self.cfg))
            self.assertTrue(table.p_value.between(0, 1).all())
            self.assertTrue(table.ci_low.lt(table.ci_high).all())
            self.assertEqual(len(predicted), len(y))
            if probs is not None:
                np.testing.assert_allclose(probs.sum(axis=1), 1)
                np.testing.assert_allclose(predicted, probs @ bins.centers)
                self.assertEqual(probs.shape[1], 3)

    def test_insufficient_independent_groups_suppresses_inference(self):
        cfg = {**self.cfg, "min_inference_groups": 100}
        bins = TargetBins.fit(self.frame.contact_force_N, cfg, "contact_force_N")
        fitted = fit_model(self.frame, self.frame.contact_force_N, self.frame.session_id,
                           self.spec, "linear", bins, cfg, True)
        self.assertTrue(pd.DataFrame(coefficient_table(fitted, cfg)).p_value.isna().all())

    def test_constant_collinear_and_separated_models_are_not_reported_as_valid(self):
        bins = TargetBins.fit(self.frame.contact_force_N, self.cfg, "contact_force_N")
        for values in (np.zeros(len(self.frame)), self.frame.rx1_rms_V):
            bad = self.frame.copy()
            bad["rx2_rms_V"] = values
            with self.assertRaises(ValueError):
                fit_model(bad, bad.contact_force_N, bad.session_id, self.spec, "linear", bins, self.cfg)
        y = self.frame.rx1_rms_V
        bins = TargetBins.fit(y, self.cfg, "contact_force_N")
        with self.assertRaises(ValueError):
            fit_model(self.frame, y, self.frame.session_id, self.spec, "logistic", bins, self.cfg)

    def test_fixed_bins_and_empty_class_error(self):
        cfg = {**self.cfg, "target_bin_edges": {"contact_force_N": [2., 4.]}}
        bins = TargetBins.fit(self.frame.contact_force_N, cfg, "contact_force_N")
        np.testing.assert_array_equal(bins.classify([1, 2, 3, 4, 5]), [0, 1, 1, 2, 2])
        with self.assertRaisesRegex(ValueError, "empty_target_bins"):
            TargetBins.fit(np.ones(20), self.cfg, "contact_force_N")

    def test_multiple_testing_and_hierarchical_specification_sweep(self):
        table = pd.DataFrame(dict(term=["intercept", "a", "b", "c"], p_value=[.001, .01, .04, np.nan]))
        adjusted = adjust_pvalues(table, .05)
        self.assertTrue(adjusted.loc[[1, 2], "q_by"].ge(adjusted.loc[[1, 2], "q_bh"]).all())
        self.assertTrue(np.isnan(adjusted.loc[0, "q_by"]))
        specs, count = candidate_specs(["rx1_rms_V", "rx2_rms_V"], self.cfg)
        self.assertTrue(any(s.interactions and len(s.features) == 2 for s in specs))


class IntegrationTests(unittest.TestCase):
    def test_holdout_isolation_saved_predictions_and_manifests(self):
        cfg = {**load_config(), "max_experiments": 5}
        frame = feature_fixture()
        train_idx, test_idx = split_groups(frame, cfg)
        self.assertFalse(set(frame.iloc[train_idx].session_id) & set(frame.iloc[test_idx].session_id))
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(temp)/"first", Path(temp)/"second"
            summary = run_experiments(frame.copy(), cfg, first)
            changed = frame.copy()
            changed.loc[test_idx, "contact_force_N"] += 1000
            other = run_experiments(changed, cfg, second)
            self.assertEqual(summary["targets"][0]["selected"], other["targets"][0]["selected"])
            a, b = [pd.read_csv(p/"coefficients.csv") for p in (first, second)]
            np.testing.assert_allclose(a.coefficient, b.coefficient)
            cv = pd.read_csv(first/"cv_manifest.csv").merge(frame[["window_id", "session_id"]], on="window_id")
            for _, fold in cv.groupby(["target", "fold"]):
                self.assertFalse(set(fold[fold.role == "train"].session_id) & set(fold[fold.role == "validation"].session_id))
                self.assertFalse(set(fold.session_id) & set(frame.iloc[test_idx].session_id))
            predictions = pd.read_csv(first/"holdout_predictions.csv")
            for target in ("contact_force_N", "contact_location_m"):
                for family in ("linear", "logistic"):
                    model = joblib.load(first/f"{target}_{family}.joblib")
                    records = predictions[(predictions.target == target) & (predictions.family == family)]
                    inputs = frame.set_index("window_id").loc[records.window_id]
                    np.testing.assert_allclose(model.predict(inputs)[0], records.predicted)
            with self.assertRaisesRegex(ValueError, "already contains"):
                run_experiments(frame, cfg, first)

    def test_targets_cannot_be_predictors_and_sources_cannot_be_mixed(self):
        frame = feature_fixture()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "never targets"):
                run_experiments(frame, {**load_config(), "feature_columns": ["contact_force_N"]}, temp)
            frame.loc[0, "data_source"] = "hardware"
            with self.assertRaisesRegex(ValueError, "separately"):
                run_experiments(frame, load_config(), temp)


if __name__ == "__main__":
    unittest.main()
