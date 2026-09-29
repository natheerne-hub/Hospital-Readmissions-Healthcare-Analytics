"""Tune, calibrate and stress-test the HistGradientBoosting readmission model.

Steps (the locked patient-group test set is only touched in step 5):
1. Patient-history features: earlier encounters of the same patient in the
   dataset, ordered by encounter_id (assumed chronological, as UCI assigns ids
   in admission order). Only strictly earlier encounters are counted.
2. Randomized hyperparameter search scored by patient-grouped cross-validation
   on the training split, for the engineered feature set with and without the
   history features. Selection: mean CV PR-AUC, then ROC-AUC.
3. Probability calibration (none / sigmoid / isotonic) chosen by
   patient-grouped cross-fitted Brier score on the validation split.
4. Operating thresholds chosen on the cross-fitted calibrated validation
   probabilities.
5. One locked-test evaluation of the tuned model and of the registered model,
   with paired bootstrap confidence intervals for the difference.

Research only; not clinically validated.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, ParameterSampler

from improve_readmission_model import (
    build_model,
    engineer_features,
    filter_eligible_cohort,
    metrics,
)
from repair_model_selection import CURRENT_PARAMS, operating_points
from train_patient_readmission import (
    load_data,
    make_patient_group_splits,
    prepare_xy,
    threshold_metrics,
)

OUT = Path("modeling/artifacts/tuned_calibrated_model.json")
HISTORY_FEATURES = ["prior_dataset_encounters", "prior_dataset_early_readmissions"]
SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "max_iter": [150, 250, 400],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0, 1.0, 3.0, 10.0],
    "max_features": [0.5, 0.8, 1.0],
}
SEARCH_ITERATIONS = int(os.environ.get("TUNE_SEARCH_ITERATIONS", "25"))
CV_FOLDS = int(os.environ.get("TUNE_CV_FOLDS", "4"))
BOOTSTRAP_RESAMPLES = int(os.environ.get("TUNE_BOOTSTRAP_RESAMPLES", "1000"))
EPSILON = 1e-6


def add_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Count each patient's strictly earlier encounters and early readmissions.

    An earlier encounter's outcome is known by the time of a later admission,
    because the later admission is itself (or follows) that readmission.
    """
    out = df.copy()
    ordered = out.sort_values(["patient_nbr", "encounter_id"])
    early = (ordered["readmitted"] == "<30").astype(int)
    patient = ordered.groupby("patient_nbr", sort=False)
    out.loc[ordered.index, "prior_dataset_encounters"] = patient.cumcount()
    out.loc[ordered.index, "prior_dataset_early_readmissions"] = (
        early.groupby(ordered["patient_nbr"], sort=False).cumsum() - early
    )
    return out.astype({name: int for name in HISTORY_FEATURES})


def logit(probability):
    probability = np.clip(probability, EPSILON, 1 - EPSILON)
    return np.log(probability / (1 - probability))


class Calibrator:
    def __init__(self, method: str):
        self.method = method

    def fit(self, probability, y):
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=1e6).fit(
                logit(probability).reshape(-1, 1), y
            )
        elif self.method == "isotonic":
            self.model = IsotonicRegression(
                out_of_bounds="clip", y_min=0, y_max=1
            ).fit(probability, y)
        return self

    def predict(self, probability):
        if self.method == "sigmoid":
            return self.model.predict_proba(logit(probability).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(probability)
        return probability


def calibration_report(y, probability, bins: int = 10):
    y = np.asarray(y)
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, bins + 1)))
    bucket = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, len(edges) - 2)
    table, ece = [], 0.0
    for index in range(len(edges) - 1):
        mask = bucket == index
        if not mask.any():
            continue
        predicted, observed = float(probability[mask].mean()), float(y[mask].mean())
        ece += mask.mean() * abs(predicted - observed)
        table.append({
            "encounters": int(mask.sum()),
            "mean_predicted": round(predicted, 4),
            "observed_rate": round(observed, 4),
        })
    fit = LogisticRegression(C=1e6).fit(logit(probability).reshape(-1, 1), y)
    return {
        "expected_calibration_error": round(float(ece), 4),
        "calibration_intercept": round(float(fit.intercept_[0]), 4),
        "calibration_slope": round(float(fit.coef_[0][0]), 4),
        "mean_predicted_probability": round(float(np.mean(probability)), 4),
        "observed_rate": round(float(y.mean()), 4),
        "reliability_deciles": table,
    }


def fold_scores(estimator, X, y, train_index, test_index):
    model = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
    probability = model.predict_proba(X.iloc[test_index])[:, 1]
    target = y.iloc[test_index]
    return (
        average_precision_score(target, probability),
        roc_auc_score(target, probability),
        brier_score_loss(target, probability),
    )


def grouped_cv(X, y, groups, parameters):
    folds = GroupKFold(n_splits=CV_FOLDS).split(X, y, groups)
    scores = np.array(Parallel(n_jobs=-1)(
        delayed(fold_scores)(build_model(X, parameters), X, y, train, test)
        for train, test in folds
    ))
    return {
        "cv_pr_auc": round(float(scores[:, 0].mean()), 4),
        "cv_pr_auc_sd": round(float(scores[:, 0].std()), 4),
        "cv_roc_auc": round(float(scores[:, 1].mean()), 4),
        "cv_brier_score": round(float(scores[:, 2].mean()), 4),
    }


def cross_fitted_calibration(y, probability, groups, method):
    """Out-of-fold calibrated probabilities so calibration is never scored in-sample."""
    y = np.asarray(y)
    calibrated = np.empty_like(probability)
    for train, test in GroupKFold(n_splits=5).split(probability, y, groups):
        calibrated[test] = Calibrator(method).fit(probability[train], y[train]).predict(probability[test])
    return calibrated


def score_triplet(y, probability):
    return np.array([
        roc_auc_score(y, probability),
        average_precision_score(y, probability),
        brier_score_loss(y, probability),
    ])


def paired_bootstrap(y, candidate, baseline):
    rng = np.random.default_rng(2026)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue
        deltas.append(score_triplet(y[index], candidate[index]) - score_triplet(y[index], baseline[index]))
    deltas = np.asarray(deltas)
    observed = score_triplet(y, candidate) - score_triplet(y, baseline)
    return {
        name: {
            "observed": round(float(observed[i]), 5),
            "ci_95_low": round(float(np.quantile(deltas[:, i], 0.025)), 5),
            "ci_95_high": round(float(np.quantile(deltas[:, i], 0.975)), 5),
        }
        for i, name in enumerate(["roc_auc", "pr_auc", "brier_score"])
    }


def main():
    raw = add_patient_history(load_data())
    train, validation, test = [
        filter_eligible_cohort(frame)[0] for frame in make_patient_group_splits(raw)
    ]
    X_train_raw, y_train, dropped = prepare_xy(train)
    X_val_raw, y_val, _ = prepare_xy(validation, dropped)
    X_test_raw, y_test, _ = prepare_xy(test, dropped)

    # Registered model: raw feature set, no history columns, current parameters.
    registered_columns = [c for c in X_train_raw.columns if c not in HISTORY_FEATURES]
    registered = build_model(X_train_raw[registered_columns], CURRENT_PARAMS)
    registered.fit(X_train_raw[registered_columns], y_train)

    engineered = engineer_features(X_train_raw)
    feature_sets = {
        "engineered": [c for c in engineered.columns if c not in HISTORY_FEATURES],
        "engineered_plus_history": list(engineered.columns),
    }
    X_train = engineered
    X_val = engineer_features(X_val_raw).reindex(columns=X_train.columns)
    X_test = engineer_features(X_test_raw).reindex(columns=X_train.columns)
    groups = train["patient_nbr"].to_numpy()

    sampled = list(ParameterSampler(SEARCH_SPACE, SEARCH_ITERATIONS, random_state=2026))
    candidates = [dict(CURRENT_PARAMS)] + sampled
    search = []
    for feature_set, columns in feature_sets.items():
        for index, parameters in enumerate(candidates):
            row = {
                "feature_set": feature_set,
                "candidate": "registered_parameters" if index == 0 else index,
                "parameters": parameters,
                **grouped_cv(X_train[columns], y_train, groups, parameters),
            }
            search.append(row)
            print(json.dumps({k: row[k] for k in ["feature_set", "candidate", "cv_pr_auc", "cv_roc_auc"]}))
    best = max(search, key=lambda row: (row["cv_pr_auc"], row["cv_roc_auc"]))
    columns = feature_sets[best["feature_set"]]
    tuned = build_model(X_train[columns], best["parameters"]).fit(X_train[columns], y_train)

    val_raw_probability = tuned.predict_proba(X_val[columns])[:, 1]
    val_groups = validation["patient_nbr"].to_numpy()
    calibration_rows, cross_fitted = [], {}
    for method in ["uncalibrated", "sigmoid", "isotonic"]:
        probability = cross_fitted_calibration(y_val, val_raw_probability, val_groups, method)
        cross_fitted[method] = probability
        report = calibration_report(y_val, probability)
        calibration_rows.append({
            "method": method,
            **metrics(y_val, probability),
            "expected_calibration_error": report["expected_calibration_error"],
            "calibration_slope": report["calibration_slope"],
        })
    chosen = min(calibration_rows, key=lambda row: (row["brier_score"], row["expected_calibration_error"]))
    method = chosen["method"]
    calibrator = Calibrator(method).fit(val_raw_probability, y_val.to_numpy())
    thresholds = operating_points(y_val, cross_fitted[method])

    # Locked test: scored once, after every choice above is frozen.
    y = y_test.to_numpy()
    tuned_probability = calibrator.predict(tuned.predict_proba(X_test[columns])[:, 1])
    registered_probability = registered.predict_proba(X_test_raw[registered_columns])[:, 1]
    registered_thresholds = operating_points(
        y_val, registered.predict_proba(X_val_raw[registered_columns])[:, 1]
    )

    def evaluate(probability, points):
        return {
            **metrics(y, probability),
            "calibration": calibration_report(y, probability),
            "operating_points": {
                name: threshold_metrics(y, probability, point["threshold"])
                for name, point in points.items()
            },
        }

    comparison = paired_bootstrap(y, tuned_probability, registered_probability)
    robust = (
        comparison["roc_auc"]["ci_95_low"] > 0
        and comparison["pr_auc"]["ci_95_low"] > 0
        and comparison["brier_score"]["ci_95_high"] <= 0
    )
    artifact = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "outcome": "readmission within 30 days among eligible discharges",
        "split": {
            "policy": "patient-group train/validation/test; eligible discharges only",
            "train_encounters": int(len(train)),
            "validation_encounters": int(len(validation)),
            "locked_test_encounters": int(len(test)),
            "test_prevalence": round(float(y.mean()), 4),
        },
        "history_features": {
            "columns": HISTORY_FEATURES,
            "assumption": "encounter_id increases with admission time within a patient; only strictly earlier encounters are counted",
        },
        "hyperparameter_search": {
            "policy": f"{SEARCH_ITERATIONS} random candidates plus the registered parameters, {CV_FOLDS}-fold patient-grouped CV on the training split, selected by mean PR-AUC then ROC-AUC",
            "search_space": SEARCH_SPACE,
            "candidates": sorted(search, key=lambda row: -row["cv_pr_auc"]),
            "selected": best,
        },
        "calibration": {
            "policy": "5-fold patient-grouped cross-fitting on validation; lowest Brier then ECE; final calibrator refit on all validation",
            "candidates": calibration_rows,
            "selected": method,
        },
        "thresholds_from_validation": {
            name: point["threshold"] for name, point in thresholds.items()
        },
        "locked_test": {
            "tuned_calibrated": evaluate(tuned_probability, thresholds),
            "registered_model": evaluate(registered_probability, registered_thresholds),
        },
        "paired_bootstrap_tuned_minus_registered": {
            "resamples": BOOTSTRAP_RESAMPLES,
            **comparison,
        },
        "robust_improvement": robust,
        "decision_rule": "Promote only if ROC-AUC and PR-AUC intervals are above zero and the Brier interval is not above zero.",
        "clinical_status": "research evidence only; not clinically validated; patient probability stays locked",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: v for k, v in artifact.items() if k != "hyperparameter_search"}
        | {"selected": best},
        indent=2,
    ))


if __name__ == "__main__":
    main()
