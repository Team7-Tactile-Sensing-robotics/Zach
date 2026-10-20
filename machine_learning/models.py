"""Standardized OLS and multinomial logistic fits with explicit failure handling."""
from dataclasses import dataclass
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t as student_t
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, log_loss)


@dataclass(frozen=True)
class Specification:
    name: str
    features: tuple
    interactions: bool = False


class Design:
    """Training-only base scaling, then hierarchical pair products and term scaling."""
    def __init__(self, spec):
        self.spec = spec

    def _expand(self, z):
        pieces = [z]
        if self.spec.interactions:
            pieces.extend((z[:, i]*z[:, j])[:, None]
                          for i, j in combinations(range(z.shape[1]), 2))
        return np.column_stack(pieces)

    def fit_transform(self, frame):
        x = frame[list(self.spec.features)].to_numpy(float)
        self.base = StandardScaler().fit(x)
        if np.any(self.base.var_ <= 1e-20):
            raise ValueError("constant_predictor")
        z = self.base.transform(x)
        expanded = self._expand(z)
        self.terms = list(self.spec.features)
        if self.spec.interactions:
            self.terms += [f"{a}:{b}" for a, b in combinations(self.spec.features, 2)]
        self.term_scaler = StandardScaler().fit(expanded)
        if np.any(self.term_scaler.var_ <= 1e-20):
            raise ValueError("constant_interaction")
        return self.transform(frame)

    def transform(self, frame):
        if set(self.spec.features)-set(frame):
            raise ValueError("Missing predictors required by saved model")
        x = frame[list(self.spec.features)].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Predictions require complete, finite feature rows")
        z = self.term_scaler.transform(self._expand(self.base.transform(x)))
        return np.column_stack([np.ones(len(frame)), z])


@dataclass
class TargetBins:
    edges: np.ndarray
    centers: np.ndarray

    @classmethod
    def fit(cls, y, cfg, target):
        y = np.asarray(y, dtype=float)
        fixed = cfg["target_bin_edges"].get(target)
        edges = (np.asarray(fixed, dtype=float) if fixed is not None else
                 np.unique(np.quantile(y, np.arange(1, cfg["logistic_classes"])/cfg["logistic_classes"])))
        if fixed is None:
            edges = edges[(edges > y.min()) & (edges < y.max())]
        labels = np.digitize(y, edges)
        if len(edges) < 1 or not np.array_equal(np.unique(labels), np.arange(len(edges)+1)):
            raise ValueError("empty_target_bins: use suitable fixed boundaries or collect more target variation")
        return cls(edges, np.array([y[labels == i].mean() for i in range(len(edges)+1)]))

    def classify(self, y):
        return np.digitize(y, self.edges)


@dataclass
class FittedModel:
    family: str
    design: Design
    result: object
    bins: TargetBins
    diagnostics: dict

    def predict(self, frame):
        output = np.asarray(self.result.predict(self.design.transform(frame)))
        if self.family == "linear":
            return output, self.bins.classify(output), None
        return output @ self.bins.centers, output.argmax(axis=1), output


def fit_model(frame, y, groups, spec, family, bins, cfg, inference=False):
    design = Design(spec)
    x = design.fit_transform(frame)
    n, p = x.shape
    condition = float(np.linalg.cond(x))
    classes = len(bins.centers)
    parameters = p if family == "linear" else p*(classes-1)
    if np.linalg.matrix_rank(x) < p or condition > cfg["max_condition_number"]:
        raise ValueError("rank_deficient_or_ill_conditioned")
    if n < cfg["min_rows_per_parameter"]*parameters:
        raise ValueError("too_few_rows_per_parameter")
    group_codes, unique_groups = pd.factorize(groups)
    ngroups = len(unique_groups)
    robust = inference and ngroups >= max(cfg["min_inference_groups"], parameters+1)
    covariance = {"cov_type": "cluster", "cov_kwds": {"groups": group_codes, "use_correction": True}} if robust else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "linear":
            result = sm.OLS(np.asarray(y), x).fit(**covariance)
        else:
            labels = bins.classify(y)
            if np.bincount(labels, minlength=classes).min() < 5:
                raise ValueError("too_few_examples_in_class")
            result = sm.MNLogit(labels, x).fit(method="bfgs", maxiter=300, disp=False, **covariance)
            if not result.mle_retvals.get("converged", False):
                raise ValueError("logistic_nonconvergence")
            if np.max(np.abs(result.params)) > 30:
                raise ValueError("possible_separation_large_coefficients")
            probabilities = np.asarray(result.predict(x))
            if np.all(probabilities.argmax(axis=1) == labels):
                raise ValueError("possible_perfect_separation")
            hessian_condition = np.linalg.cond(-result.model.hessian(result.params.ravel(order="F")))
            if not np.isfinite(hessian_condition) or hessian_condition > cfg["max_condition_number"]:
                raise ValueError("logistic_hessian_ill_conditioned")
        if any("separation" in str(w.message).lower() for w in caught):
            raise ValueError("logistic_separation")
    if not np.isfinite(result.params).all():
        raise ValueError("nonfinite_coefficients")
    if robust and (not np.isfinite(result.bse).all() or np.any(result.bse <= 0)):
        robust = False
    diag = dict(n_rows=n, n_groups=ngroups, n_parameters=parameters,
                condition_number=condition, aic=float(result.aic), bic=float(result.bic),
                inference_status="cluster_robust_exploratory" if robust else "unavailable_insufficient_groups_or_covariance",
                fit_warnings="; ".join(sorted({str(w.message) for w in caught})))
    return FittedModel(family, design, result, bins, diag)


def coefficient_table(fitted, cfg):
    params = np.asarray(fitted.result.params).reshape(len(fitted.design.terms)+1, -1)
    available = fitted.diagnostics["inference_status"] == "cluster_robust_exploratory"
    errors = np.asarray(fitted.result.bse).reshape(params.shape) if available else np.full_like(params, np.nan)
    rows = []
    for k in range(params.shape[1]):
        for j, term in enumerate(["intercept"]+fitted.design.terms):
            estimate, se = params[j, k], errors[j, k]
            # OLS cluster t reference uses G-1; MNLogit Wald inference is asymptotic normal.
            distribution = student_t(fitted.diagnostics["n_groups"]-1) if fitted.family == "linear" else norm
            p = 2*distribution.sf(abs(estimate/se)) if available else np.nan
            critical = distribution.ppf(1-cfg["alpha"]/2)
            rows.append(dict(term=term, class_contrast="continuous" if fitted.family == "linear" else f"{k+1} vs 0",
                             coefficient=estimate, standard_error=se, p_value=p,
                             ci_low=estimate-critical*se, ci_high=estimate+critical*se,
                             odds_ratio=np.exp(estimate) if fitted.family == "logistic" else np.nan,
                             inference_status=fitted.diagnostics["inference_status"]))
    return rows


def evaluate(y, predicted, predicted_class, probabilities, bins):
    truth_class = bins.classify(y)
    scores = dict(mae=mean_absolute_error(y, predicted), rmse=np.sqrt(mean_squared_error(y, predicted)),
                  r2=r2_score(y, predicted) if len(y) > 1 and np.var(y) > 0 else np.nan,
                  accuracy=accuracy_score(truth_class, predicted_class),
                  macro_f1=f1_score(truth_class, predicted_class, labels=np.arange(len(bins.centers)),
                                    average="macro", zero_division=0))
    scores["log_loss"] = log_loss(truth_class, probabilities, labels=np.arange(len(bins.centers))) if probabilities is not None else np.nan
    return scores
