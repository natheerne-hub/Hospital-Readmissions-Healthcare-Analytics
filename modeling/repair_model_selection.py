"""Repair model selection with matched-cohort, same-test comparison.

This script does not promote a candidate unless it beats the registered model on
the same eligible patient-group test encounters.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from improve_readmission_model import (
    EXCLUDED_DISPOSITIONS,
    build_model,
    engineer_features,
    filter_eligible_cohort,
    metrics,
)
from train_patient_readmission import (
    choose_threshold_on_validation,
    load_data,
    make_patient_group_splits,
    prepare_xy,
    threshold_metrics,
)

OUT = Path("modeling/artifacts/model_repair_comparison.json")
CURRENT_PARAMS = {
    "max_iter": 250,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
}
SEARCH_PARAMS = [
    {"max_iter": 220, "learning_rate": 0.04, "max_leaf_nodes": 15,
     "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_iter": 260, "learning_rate": 0.05, "max_leaf_nodes": 31,
     "min_samples_leaf": 30, "l2_regularization": 2.0},
    {"max_iter": 300, "learning_rate": 0.04, "max_leaf_nodes": 31,
     "min_samples_leaf": 60, "l2_regularization": 3.0},
    {"max_iter": 260, "learning_rate": 0.05, "max_leaf_nodes": 63,
     "min_samples_leaf": 80, "l2_regularization": 5.0},
]


def eligible(frame):
    return filter_eligible_cohort(frame)[0]


def make_xy(frame, dropped=None):
    X, y, sparse = prepare_xy(frame, dropped)
    return X, y, sparse


def operating_points(y_true, probability):
    f1_point = choose_threshold_on_validation(y_true, probability)
    constrained = []
    for threshold in [round(x / 100, 2) for x in range(5, 51)]:
        row = threshold_metrics(y_true, probability, threshold)
        if row["sensitivity_recall"] >= 0.40:
            constrained.append(row)
    precision_point = max(
        constrained,
        key=lambda row: (
            row["precision"],
            -row["alerts_per_100_encounters"],
        ),
    )
    return {
        "f1_optimized": f1_point,
        "minimum_40pct_sensitivity": precision_point,
    }


def main():
    raw = load_data()
    train_all, validation_all, test_all = make_patient_group_splits(raw)
    train = eligible(train_all)
    validation = eligible(validation_all)
    test = eligible(test_all)

    base_train_X, train_y, dropped = make_xy(train)
    base_validation_X, validation_y, _ = make_xy(validation, dropped)
    base_test_X, test_y, _ = make_xy(test, dropped)
    base_validation_X = base_validation_X.reindex(columns=base_train_X.columns)
    base_test_X = base_test_X.reindex(columns=base_train_X.columns)

    variants = {
        "raw_registered_features": (
            base_train_X,
            base_validation_X,
            base_test_X,
        ),
        "augmented_keep_raw_diagnoses": (
            engineer_features(base_train_X, drop_raw_diagnoses=False),
            engineer_features(
                base_validation_X, drop_raw_diagnoses=False
            ),
            engineer_features(base_test_X, drop_raw_diagnoses=False),
        ),
        "grouped_diagnoses": (
            engineer_features(base_train_X, drop_raw_diagnoses=True),
            engineer_features(
                base_validation_X, drop_raw_diagnoses=True
            ),
            engineer_features(base_test_X, drop_raw_diagnoses=True),
        ),
    }

    baseline_model = build_model(base_train_X, CURRENT_PARAMS)
    baseline_model.fit(base_train_X, train_y)
    baseline_validation_probability = baseline_model.predict_proba(
        base_validation_X
    )[:, 1]
    baseline_thresholds = operating_points(
        validation_y, baseline_validation_probability
    )

    candidates = []
    fitted = {}
    for variant, (X_train, X_validation, X_test) in variants.items():
        X_validation = X_validation.reindex(columns=X_train.columns)
        X_test = X_test.reindex(columns=X_train.columns)
        for index, parameters in enumerate(SEARCH_PARAMS, start=1):
            model = build_model(X_train, parameters)
            model.fit(X_train, train_y)
            probability = model.predict_proba(X_validation)[:, 1]
            row = {
                "variant": variant,
                "candidate": index,
                "parameters": parameters,
                **metrics(validation_y, probability),
                "operating_points": operating_points(
                    validation_y, probability
                ),
            }
            candidates.append(row)
            fitted[(variant, index)] = (model, X_test)

    winner = max(
        candidates,
        key=lambda row: (
            row["roc_auc"],
            row["pr_auc"],
            -row["brier_score"],
        ),
    )
    winner_model, winner_test_X = fitted[
        (winner["variant"], winner["candidate"])
    ]

    baseline_test_probability = baseline_model.predict_proba(base_test_X)[:, 1]
    winner_test_probability = winner_model.predict_proba(winner_test_X)[:, 1]

    baseline_test = {
        **metrics(test_y, baseline_test_probability),
        "f1_optimized": threshold_metrics(
            test_y,
            baseline_test_probability,
            baseline_thresholds["f1_optimized"]["threshold"],
        ),
        "minimum_40pct_sensitivity": threshold_metrics(
            test_y,
            baseline_test_probability,
            baseline_thresholds["minimum_40pct_sensitivity"]["threshold"],
        ),
    }
    winner_test = {
        **metrics(test_y, winner_test_probability),
        "f1_optimized": threshold_metrics(
            test_y,
            winner_test_probability,
            winner["operating_points"]["f1_optimized"]["threshold"],
        ),
        "minimum_40pct_sensitivity": threshold_metrics(
            test_y,
            winner_test_probability,
            winner["operating_points"]["minimum_40pct_sensitivity"]["threshold"],
        ),
    }

    deltas = {
        "roc_auc": round(
            winner_test["roc_auc"] - baseline_test["roc_auc"], 4
        ),
        "pr_auc": round(
            winner_test["pr_auc"] - baseline_test["pr_auc"], 4
        ),
        "brier_score": round(
            winner_test["brier_score"] - baseline_test["brier_score"], 4
        ),
    }
    promote = (
        deltas["roc_auc"] > 0
        and deltas["pr_auc"] > 0
        and deltas["brier_score"] <= 0
    )

    artifact = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_policy": (
            "Registered baseline and candidates use the same patient assignment, "
            "eligible-discharge test encounters, and untouched test labels."
        ),
        "excluded_discharge_dispositions": sorted(EXCLUDED_DISPOSITIONS),
        "encounters": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "registered_baseline_test": baseline_test,
        "validation_candidates": candidates,
        "selected_candidate": winner,
        "selected_candidate_test": winner_test,
        "test_delta_candidate_minus_baseline": deltas,
        "promotion_decision": {
            "promote": promote,
            "rule": (
                "Require higher test ROC-AUC and PR-AUC with no worse Brier score."
            ),
        },
        "clinical_status": "research comparison; not clinical validation",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

