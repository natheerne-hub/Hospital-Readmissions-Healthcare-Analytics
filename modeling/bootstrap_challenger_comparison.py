"""Paired bootstrap uncertainty for registered baseline vs XGBoost challenger."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from benchmark_xgboost_challenger import build as build_xgboost
from improve_readmission_model import engineer_features, filter_eligible_cohort
from repair_model_selection import CURRENT_PARAMS
from improve_readmission_model import build_model as build_histgb
from train_patient_readmission import load_data, make_patient_group_splits, prepare_xy

OUT = Path("modeling/artifacts/challenger_bootstrap_uncertainty.json")
XGB_PARAMS = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.035,
    "min_child_weight": 12,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 6.0,
}


def eligible(frame):
    return filter_eligible_cohort(frame)[0]


def score_triplet(y, probability):
    return np.array([
        roc_auc_score(y, probability),
        average_precision_score(y, probability),
        brier_score_loss(y, probability),
    ])


def summarize(values):
    return {
        "mean": round(float(np.mean(values)), 5),
        "ci_95_low": round(float(np.quantile(values, 0.025)), 5),
        "ci_95_high": round(float(np.quantile(values, 0.975)), 5),
    }


def main():
    train, validation, test = [
        eligible(frame) for frame in make_patient_group_splits(load_data())
    ]
    X_train, y_train, dropped = prepare_xy(train)
    X_test, y_test, _ = prepare_xy(test, dropped)
    X_test = X_test.reindex(columns=X_train.columns)

    baseline = build_histgb(X_train, CURRENT_PARAMS)
    baseline.fit(X_train, y_train)
    baseline_probability = baseline.predict_proba(X_test)[:, 1]

    augmented_train = engineer_features(
        X_train, drop_raw_diagnoses=False
    )
    augmented_test = engineer_features(
        X_test, drop_raw_diagnoses=False
    ).reindex(columns=augmented_train.columns)
    challenger = build_xgboost(augmented_train, XGB_PARAMS)
    challenger.fit(augmented_train, y_train)
    challenger_probability = challenger.predict_proba(augmented_test)[:, 1]

    y = y_test.to_numpy()
    observed_baseline = score_triplet(y, baseline_probability)
    observed_challenger = score_triplet(y, challenger_probability)
    observed_delta = observed_challenger - observed_baseline

    rng = np.random.default_rng(2026)
    bootstrap = []
    for _ in range(1000):
        indices = rng.integers(0, len(y), len(y))
        sampled_y = y[indices]
        if sampled_y.min() == sampled_y.max():
            continue
        bootstrap.append(
            score_triplet(sampled_y, challenger_probability[indices])
            - score_triplet(sampled_y, baseline_probability[indices])
        )
    bootstrap = np.asarray(bootstrap)
    names = ["roc_auc", "pr_auc", "brier_score"]
    uncertainty = {
        name: summarize(bootstrap[:, index])
        for index, name in enumerate(names)
    }
    robust_improvement = (
        uncertainty["roc_auc"]["ci_95_low"] > 0
        and uncertainty["pr_auc"]["ci_95_low"] > 0
        and uncertainty["brier_score"]["ci_95_high"] <= 0
    )
    artifact = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "1000 paired nonparametric bootstrap resamples of the same locked test encounters",
        "observed": {
            "baseline": dict(zip(names, observed_baseline.round(5))),
            "challenger": dict(zip(names, observed_challenger.round(5))),
            "challenger_minus_baseline": dict(
                zip(names, observed_delta.round(5))
            ),
        },
        "delta_uncertainty": uncertainty,
        "robust_improvement": robust_improvement,
        "decision": (
            "Promote only if discrimination confidence intervals are above zero "
            "and the Brier interval is not above zero."
        ),
        "clinical_status": "research uncertainty analysis; not clinical validation",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

