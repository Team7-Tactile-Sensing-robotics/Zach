"""Grouped model comparison, exploratory inference and untouched final holdout."""
from itertools import combinations
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from statsmodels.stats.multitest import multipletests

from .data import STATES, TARGETS
from .models import Specification, TargetBins, fit_model, coefficient_table, evaluate


def candidate_specs(features, cfg):
    available, seen, specs = set(features), set(), []

    def add(names, interaction=False):
        names = tuple(sorted(set(names) & available))
        if not names or (interaction and len(names) < 2):
            return
        key = (names, interaction)
        if key not in seen:
            seen.add(key)
            specs.append(Specification(f"m{len(specs):03d}", names, interaction))

    rx = [f"rx{i}_rms_V" for i in range(1, 5)]
    tx = [f"tx{i}_rms_V" for i in range(1, 5)]
    for names in (rx, rx+STATES, rx+tx, rx+tx+STATES,
                  [n for n in features if n.startswith("rx")], features):
        add(names)
        if len(names) <= 8:
            add(names, True)
    # Every combination of the four receiver RMS channels, plus all individual features.
    for size in range(1, 5):
        for names in combinations(rx, size):
            add(names)
            add(names, True)
    for name in features:
        add([name])
    core = [name for name in cfg["core_features"] if name in available]
    for size in range(2, min(cfg["max_subset_size"], len(core))+1):
        for names in combinations(core, size):
            add(names)
            add(names, True)
    return specs[:cfg["max_experiments"]], len(specs)


def split_groups(frame, cfg):
    groups = frame[cfg["group_column"]]
    if groups.nunique() < cfg["cv_folds"]+2:
        raise ValueError("Collect more independent groups for development CV and a separate holdout")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["holdout_fraction"], random_state=cfg["seed"])
    train, test = next(splitter.split(frame, groups=groups))
    return train, test


def adjust_pvalues(frame, alpha):
    """One family across both targets, both estimators and ALL tested non-intercept coefficients."""
    frame = frame.copy()
    frame["q_bh"] = np.nan
    frame["q_by"] = np.nan
    mask = frame.p_value.notna() & frame.term.ne("intercept")
    if mask.any():
        for method, name in [("fdr_bh", "q_bh"), ("fdr_by", "q_by")]:
            frame.loc[mask, name] = multipletests(frame.loc[mask, "p_value"], method=method)[1]
    frame["significant_exploratory"] = frame.q_by.lt(alpha)
    return frame


def run_experiments(frame, cfg, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"run_summary.json").exists():
        raise ValueError("Output already contains a run; choose a fresh directory to preserve the holdout record")
    required = {"session_id", "trial_id", "data_source", "window_id", *TARGETS}
    if required-set(frame):
        raise ValueError(f"Feature table missing {sorted(required-set(frame))}")
    if frame[list(required- set(TARGETS))].isna().any().any() or frame.window_id.duplicated().any():
        raise ValueError("Missing IDs or duplicate windows in feature table")
    if len(set(frame.data_source)) != 1 or not set(frame.data_source) <= {"hardware", "synthetic"}:
        raise ValueError("Run hardware and synthetic experiments separately; do not mix their holdouts")
    if (frame.groupby("trial_id").session_id.nunique() > 1).any():
        raise ValueError("trial_id must identify one session")
    # IDs and source are explicitly excluded; target values can never be predictors.
    allowed = [c for c in frame if c.startswith(tuple(f"{kind}{i}_" for kind in ("rx", "tx") for i in range(1, 5))) or c in STATES]
    features = cfg["feature_columns"] if cfg["feature_columns"] is not None else allowed
    if not features or len(set(features)) != len(features) or set(features)-set(allowed):
        raise ValueError("feature_columns must contain unique FFT/state predictors, never targets or identifiers")
    for name in features+TARGETS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if np.isinf(frame[features+TARGETS].to_numpy()).any():
        raise ValueError("Infinite features or targets")
    train_idx, test_idx = split_groups(frame, cfg)
    development, holdout = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    unavailable = [c for c in features if development[c].isna().all()]
    features = [c for c in features if c not in unavailable]
    specs, candidate_count = candidate_specs(features, cfg)
    if not specs:
        raise ValueError("No available predictors")
    manifest = frame[["window_id", "session_id", "trial_id", "data_source"]].copy()
    manifest["split"] = "holdout"
    manifest.iloc[train_idx, manifest.columns.get_loc("split")] = "development"
    manifest.to_csv(out/"split_manifest.csv", index=False)
    (out/"config_used.json").write_text(json.dumps(cfg, indent=2)+"\n")
    (out/"versions.json").write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        scikit_learn=sklearn.__version__, statsmodels=statsmodels.__version__), indent=2)+"\n")
    (out/"specifications.json").write_text(json.dumps([dict(name=s.name, features=s.features, interactions=s.interactions) for s in specs], indent=2)+"\n")
    experiments, coefficients, folds, scores, predictions, summaries = [], [], [], [], [], []
    for target in TARGETS:
        # All candidates/families use identical complete-case rows for fair comparisons.
        train = development.dropna(subset=features+[target]).reset_index(drop=True)
        test = holdout.dropna(subset=features+[target]).reset_index(drop=True)
        group = cfg["group_column"]
        eligibility = {"target": target, "development_rows": len(train), "holdout_rows": len(test),
                       "excluded_development_rows": len(development)-len(train),
                       "excluded_holdout_rows": len(holdout)-len(test)}
        if len(test) < 2 or train[group].nunique() < cfg["cv_folds"] or train[target].nunique() < 2:
            summaries.append({**eligibility, "status": "insufficient_labeled_rows_or_groups"})
            continue
        try:
            bins = TargetBins.fit(train[target], cfg, target)
            splits = list(GroupKFold(cfg["cv_folds"]).split(train, groups=train[group]))
            fold_bins = [TargetBins.fit(train.iloc[a][target], cfg, target) for a, b in splits]
        except ValueError as exc:
            summaries.append({**eligibility, "status": str(exc)})
            continue
        for fold, (a, b) in enumerate(splits):
            for role, indices in (("train", a), ("validation", b)):
                folds.extend(dict(target=target, fold=fold, role=role, window_id=w)
                             for w in train.iloc[indices].window_id)
        best = {}
        for index, spec in enumerate(specs):
            for family in ("linear", "logistic"):
                key = dict(target=target, family=family, model=spec.name)
                row = {**key, "features": ";".join(spec.features), "interactions": spec.interactions}
                try:
                    fitted = fit_model(train, train[target], train[group], spec, family, bins, cfg, inference=True)
                    row.update(fitted.diagnostics)
                    coefficients.extend({**key, **c} for c in coefficient_table(fitted, cfg))
                    fold_scores = []
                    for fold, ((a, b), local_bins) in enumerate(zip(splits, fold_bins)):
                        local = fit_model(train.iloc[a], train.iloc[a][target], train.iloc[a][group],
                                          spec, family, local_bins, cfg)
                        prediction = local.predict(train.iloc[b])
                        metrics = evaluate(train.iloc[b][target], *prediction, local_bins)
                        fold_scores.append(metrics)
                    row.update({f"cv_{metric}": float(np.mean([s[metric] for s in fold_scores]))
                                for metric in fold_scores[0]})
                    row["cv_mae_std"] = float(np.std([s["mae"] for s in fold_scores]))
                    row["status"] = "ok"
                    if family not in best or row["cv_mae"] < best[family][0]:
                        best[family] = (row["cv_mae"], fitted, spec.name)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    row["status"] = "failed_or_skipped"
                    row["reason"] = str(exc)
                experiments.append(row)
            if index % 25 == 0:
                print(f"{target}: {index+1}/{len(specs)} specifications", flush=True)
        # Only the CV winner from each family sees the held-out groups.
        for family, (_, fitted, name) in best.items():
            pred, labels, probabilities = fitted.predict(test)
            scores.append(dict(target=target, family=family, model=name, split="holdout",
                               **evaluate(test[target], pred, labels, probabilities, bins)))
            joblib.dump(fitted, out/f"{target}_{family}.joblib")
            for i, (_, record) in enumerate(test.iterrows()):
                result = dict(target=target, family=family, model=name, window_id=record.window_id,
                              observed=record[target], predicted=pred[i],
                              observed_class=bins.classify([record[target]])[0], predicted_class=labels[i])
                if probabilities is not None:
                    result.update({f"probability_class_{j}": value for j, value in enumerate(probabilities[i])})
                predictions.append(result)
        # Training-only mean and majority baseline; useful even when all candidate fits fail.
        class_counts = np.bincount(bins.classify(train[target]), minlength=len(bins.centers))
        baseline_prob = np.tile(class_counts/class_counts.sum(), (len(test), 1))
        scores.append(dict(target=target, family="baseline", model="training_mean_and_majority", split="holdout",
                           **evaluate(test[target], np.full(len(test), train[target].mean()),
                                      baseline_prob.argmax(axis=1), baseline_prob, bins)))
        summaries.append({**eligibility, "status": "evaluated" if best else "no_valid_candidates",
                          "bin_edges": bins.edges.tolist(), "bin_centers": bins.centers.tolist(),
                          "selected": {family: name for family, (_, fitted, name) in best.items()}})
    experiment_df = pd.DataFrame(experiments)
    experiment_df.to_csv(out/"experiments.csv", index=False)
    pd.DataFrame(folds).to_csv(out/"cv_manifest.csv", index=False)
    score_df = pd.DataFrame(scores)
    score_df.to_csv(out/"holdout_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out/"holdout_predictions.csv", index=False)
    if coefficients:
        coef = adjust_pvalues(pd.DataFrame(coefficients), cfg["alpha"])
        coef.to_csv(out/"coefficients.csv", index=False)
        summary = coef[coef.term != "intercept"].groupby(["target", "family", "term"]).agg(
            appearances=("p_value", "size"), valid_tests=("p_value", "count"),
            smallest_raw_p=("p_value", "min"), smallest_q_by=("q_by", "min"),
            significant_tests=("significant_exploratory", "sum"))
        summary.to_csv(out/"variable_significance.csv")
    else:
        pd.DataFrame(columns=["target", "family", "term", "p_value", "q_by"]).to_csv(out/"coefficients.csv", index=False)
        pd.DataFrame(columns=["target", "family", "term", "valid_tests"]).to_csv(out/"variable_significance.csv", index=False)
    run_summary = dict(data_source=frame.data_source.iloc[0], n_windows=len(frame),
                       candidate_specifications=candidate_count, evaluated_specifications=len(specs),
                       missing_predictors_excluded=unavailable, targets=summaries,
                       comparison="Parallel OLS continuous outputs vs multinomial-logistic bins with probability-weighted training-bin means",
                       inference="Exploratory only. Cluster covariance; BY correction across all non-intercept tests; not post-selection confirmation.")
    (out/"run_summary.json").write_text(json.dumps(run_summary, indent=2)+"\n")
    lines = ["# Regression comparison", "", f"Data source: **{run_summary['data_source']}**.",
             f"{len(specs)} predictor specifications; {len(experiments)} target/family experiments attempted.", "",
             "Linear and logistic models are independent comparisons. Logistic continuous estimates are probability-weighted bin means.",
             "Selection uses grouped development CV MAE only. Final holdout evaluates one winner per family/target plus a baseline.", "",
             "```text", score_df.to_string(index=False), "```", "",
             "See experiments.csv for failed fits, coefficients.csv for confidence intervals/raw p/BH/BY q-values,",
             "and variable_significance.csv for exploratory term summaries. Repeated appearances are not independent evidence.",
             "Unavailable inference means too few independent groups or invalid covariance; more waveform samples alone do not solve this.",
             "No hardware validation is implied by synthetic results."]
    if not experiment_df.empty:
        lines += ["", "Fit outcomes (a failed/skipped candidate is not included in selection):", "", "```text",
                  experiment_df.groupby(["target", "family", "status"]).size().to_string(), "```"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    return run_summary
