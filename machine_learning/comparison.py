"""Matched feature ablation: same rows/splits, fold-local PCA, OLS and RBF SVR.

This predictive benchmark is separate from the exploratory OLS/logit significance
search. It neither implements the paper's supervised PB nor its multioutput SVR.
"""
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from .data import STATES, TARGETS
from .experiments import split_groups


class LimitedPCA(TransformerMixin, BaseEstimator):
    def __init__(self, components=20):
        self.components = components

    def fit(self, x, y=None):
        self.pca_ = PCA(n_components=min(self.components, x.shape[1], x.shape[0]-1), svd_solver="full").fit(x)
        return self

    def transform(self, x):
        return self.pca_.transform(x)


class ComparisonModel:
    """Saved benchmark model accepting the same extracted feature tables."""
    def __init__(self, features, estimator):
        self.features = features
        self.estimator = estimator

    def predict(self, frame):
        x = frame[self.features].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Prediction requires finite benchmark features")
        return self.estimator.predict(x)


def feature_sets(frame):
    channel = [c for c in frame if c.startswith(tuple(f"{kind}{i}_" for kind in ("rx", "tx") for i in range(1, 5)))]
    baseline = [c for c in channel if "_time_" not in c and "_echo_" not in c]+[c for c in STATES if c in frame]
    time = [c for c in channel if "_time_" in c and "_echo_" not in c]
    echo = [c for c in channel if "_echo_" in c]
    if not time:
        raise ValueError("Comparison requires time features; re-extract with time_features_enabled=true")
    sets = {"fft_baseline": baseline, "fft_plus_time": baseline+time}
    if echo:
        sets["fft_plus_time_echo"] = baseline+time+echo
    return sets


def make_estimator(family, params, cfg):
    regressor = LinearRegression() if family == "linear_pca" else SVR(kernel="rbf", **params)
    # Target scaling is also fit only on the training fold; C/epsilon are in standardized target units.
    return Pipeline([("nonconstant", VarianceThreshold(threshold=1e-20)),
                     ("scale", StandardScaler()),
                     ("pca", LimitedPCA(cfg["comparison_pca_components"])),
                     ("regressor", TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler()))])


def metrics(y, prediction):
    return dict(mae=float(mean_absolute_error(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                r2=float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else np.nan)


def paired_group_interval(truth, baseline, expanded, groups, seed, repetitions=2000):
    """Bootstrap whole holdout groups. Negative difference means expanded is better."""
    differences = np.abs(truth-expanded)-np.abs(truth-baseline)
    grouped = pd.DataFrame({"group": np.asarray(groups), "difference": differences}).groupby("group").difference.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(grouped), (repetitions, len(grouped)))
    changes = grouped["sum"].to_numpy()[selected].sum(axis=1)/grouped["count"].to_numpy()[selected].sum(axis=1)
    return dict(mae_change=float(differences.mean()),
                bootstrap_95_low=float(np.quantile(changes, .025)),
                bootstrap_95_high=float(np.quantile(changes, .975)), n_holdout_groups=len(grouped))


def compare_features(frame, cfg, output):
    with threadpool_limits(limits=1):
        return _compare_features(frame.copy(), cfg, output)


def _compare_features(frame, cfg, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose a fresh comparison directory")
    if frame.empty or frame.window_id.duplicated().any():
        raise ValueError("Need nonempty, unique feature windows")
    if frame[["session_id", "trial_id", "data_source", "window_id"]].isna().any().any():
        raise ValueError("Comparison requires complete recording identifiers")
    if len(set(frame.data_source)) != 1 or not set(frame.data_source) <= {"hardware", "synthetic"}:
        raise ValueError("Compare hardware and synthetic datasets separately")
    if (frame.groupby("trial_id").session_id.nunique() > 1).any():
        raise ValueError("Trial IDs must identify one session")
    sets = feature_sets(frame)
    train_idx, test_idx = split_groups(frame, cfg)
    development, holdout = frame.iloc[train_idx], frame.iloc[test_idx]
    union = list(dict.fromkeys(c for columns in sets.values() for c in columns))
    if np.isinf(frame[union+TARGETS].to_numpy(float)).any():
        raise ValueError("Infinite features/targets")
    missing = [c for c in union if development[c].isna().all()]
    sets = {name: [c for c in columns if c not in missing] for name, columns in sets.items()}
    union = [c for c in union if c not in missing]
    manifest = frame[["window_id", "session_id", "trial_id"]].copy()
    manifest["split"] = "holdout"
    manifest.iloc[train_idx, manifest.columns.get_loc("split")] = "development"
    manifest.to_csv(out/"split_manifest.csv", index=False)
    (out/"feature_sets.json").write_text(json.dumps(sets, indent=2)+"\n")
    (out/"config_used.json").write_text(json.dumps(cfg, indent=2)+"\n")
    scores, predictions, searches, fold_manifest, deltas, eligibility = [], [], [], [], [], []
    grids = {"linear_pca": [{}], "svr_rbf": [dict(C=c, gamma=g, epsilon=e) for c, g, e in
              product(cfg["svr_C"], cfg["svr_gamma"], cfg["svr_epsilon"])]}
    for target in TARGETS:
        # Use the same complete-case intersection for every arm of an ablation.
        train = development.dropna(subset=union+[target]).reset_index(drop=True)
        test = holdout.dropna(subset=union+[target]).reset_index(drop=True)
        eligibility.append(dict(target=target, development_rows=len(train), holdout_rows=len(test),
                                dropped_development=len(development)-len(train), dropped_holdout=len(holdout)-len(test)))
        group = cfg["group_column"]
        if train[group].nunique() < cfg["cv_folds"] or test[group].nunique() < 2 or train[target].nunique() < 2:
            raise ValueError(f"{target}: not enough labeled independent groups/target variation for comparison")
        splits = list(GroupKFold(cfg["cv_folds"]).split(train, groups=train[group]))
        for fold, (a, b) in enumerate(splits):
            for role, indices in (("train", a), ("validation", b)):
                fold_manifest.extend(dict(target=target, fold=fold, role=role, window_id=w)
                                     for w in train.iloc[indices].window_id)
        fitted_predictions = {}
        for name, columns in sets.items():
            for family, grid in grids.items():
                best = None
                for params in grid:
                    row = dict(target=target, feature_set=name, family=family, parameters=json.dumps(params))
                    try:
                        fold_scores = []
                        for a, b in splits:
                            estimator = make_estimator(family, params, cfg)
                            estimator.fit(train.iloc[a][columns].to_numpy(), train.iloc[a][target].to_numpy())
                            fold_scores.append(mean_absolute_error(train.iloc[b][target], estimator.predict(train.iloc[b][columns].to_numpy())))
                        row.update(status="ok", cv_mae=float(np.mean(fold_scores)), cv_mae_std=float(np.std(fold_scores)))
                        if best is None or row["cv_mae"] < best[0]:
                            best = (row["cv_mae"], params)
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        row.update(status="failed", reason=str(exc))
                    searches.append(row)
                if best is None:
                    raise ValueError(f"All candidates failed: {target}, {name}, {family}")
                estimator = make_estimator(family, best[1], cfg).fit(train[columns].to_numpy(), train[target].to_numpy())
                model = ComparisonModel(columns, estimator)
                predicted = model.predict(test)
                fitted_predictions[(family, name)] = predicted
                scores.append(dict(target=target, feature_set=name, family=family,
                                   n_features=len(columns), pca_components=estimator.named_steps["pca"].pca_.n_components_,
                                   cv_mae=best[0], parameters=json.dumps(best[1]), **metrics(test[target], predicted)))
                joblib.dump(model, out/f"{target}_{family}_{name}.joblib")
                predictions.extend(dict(target=target, feature_set=name, family=family, window_id=w,
                                        observed=y, predicted=p) for w, y, p in zip(test.window_id, test[target], predicted))
                print(f"{target} / {family} / {name}: holdout MAE {scores[-1]['mae']:.6g}", flush=True)
        for family in grids:
            for name in sets:
                if name != "fft_baseline":
                    deltas.append(dict(target=target, family=family, feature_set=name,
                                       **paired_group_interval(test[target].to_numpy(), fitted_predictions[(family, "fft_baseline")],
                                           fitted_predictions[(family, name)], test[group], cfg["seed"])))
        scores.append(dict(target=target, feature_set="training_mean", family="baseline",
                           **metrics(test[target], np.full(len(test), train[target].mean()))))
    pd.DataFrame(scores).to_csv(out/"holdout_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out/"holdout_predictions.csv", index=False)
    pd.DataFrame(searches).to_csv(out/"cv_search.csv", index=False)
    pd.DataFrame(fold_manifest).to_csv(out/"cv_manifest.csv", index=False)
    pd.DataFrame(deltas).to_csv(out/"paired_mae_changes.csv", index=False)
    summary = dict(data_source=frame.data_source.iloc[0], targets=eligibility,
                   excluded_missing_predictors=missing, feature_counts={name: len(c) for name, c in sets.items()},
                   protocol="Same rows and group splits; training-only variance filter, standardization, PCA and target scaling; CV-tuned RBF SVR and OLS reference.",
                   limitation="Synthetic results validate software only. Echo timing requires triggered captures. No WPD/PB/PSO or multipoint reproduction.")
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (out/"REPORT.md").write_text("\n".join([
        "# Time-feature comparison", "", f"Data source: **{summary['data_source']}**.",
        summary["protocol"], "", "These are separate continuous force/location regressors, not sequential predictions.",
        "PCA is applied to both baseline and expanded inputs; this is not a comparison with the earlier 237-specification winner.",
        "", "```text", pd.DataFrame(scores).to_string(index=False), "```", "",
        "Paired MAE change = expanded minus baseline; negative means improvement.",
        "95% intervals resample whole holdout groups and are descriptive, not multiplicity-adjusted significance tests.",
        "An interval spanning zero is inconclusive. They do not include training/tuning uncertainty.",
        "", "```text", pd.DataFrame(deltas).to_string(index=False), "```", "", summary["limitation"],
        "SVR does not provide ordinary coefficient p-values; use the existing OLS/logit experiment runner for exploratory coefficient tables.",
    ])+"\n")
    return summary
