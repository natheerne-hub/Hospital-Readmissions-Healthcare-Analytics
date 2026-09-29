"""Compare class-imbalance strategies for 30-day readmission research.

Strategies (each applied only inside the training fold):
- none: fit on the natural outcome prevalence;
- class_weight: reweight the minority class ("balanced");
- smote: synthetic minority oversampling after preprocessing;
- undersample: random majority-class undersampling.

Selection uses validation PR-AUC (ties: ROC-AUC, then Brier). The locked
patient-group test set is scored once, for the selected candidate only.
Resampling and reweighting change the probability scale, so Brier score is
reported for every candidate and a probability-scale warning is attached.

Research only; not clinically validated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from improve_readmission_model import (
    build_preprocessor,
    engineer_features,
    filter_eligible_cohort,
    metrics,
)
from repair_model_selection import operating_points
from train_patient_readmission import (
    load_data,
    make_patient_group_splits,
    prepare_xy,
    threshold_metrics,
)

OUT = Path("modeling/artifacts/imbalance_strategy_comparison.json")
STRATEGIES = ["none", "class_weight", "smote", "undersample"]
HGB_PARAMS = {
    "max_iter": 260, "learning_rate": 0.05, "max_leaf_nodes": 31,
    "min_samples_leaf": 30, "l2_regularization": 2.0,
}


def estimator(family: str, strategy: str):
    class_weight = "balanced" if strategy == "class_weight" else None
    if family == "logistic_regression":
        return LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=42
        )
    return HistGradientBoostingClassifier(
        random_state=42, class_weight=class_weight, **HGB_PARAMS
    )


def build(X, family: str, strategy: str):
    steps = [("preprocess", build_preprocessor(X))]
    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=42)))
    elif strategy == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=42)))
    steps.append(("model", estimator(family, strategy)))
    return ImbPipeline(steps)


def eligible(frame):
    return filter_eligible_cohort(frame)[0]


def main():
    raw = load_data()
    train, validation, test = [
        eligible(frame) for frame in make_patient_group_splits(raw)
    ]
    X_train, y_train, dropped = prepare_xy(train)
    X_validation, y_validation, _ = prepare_xy(validation, dropped)
    X_test, y_test, _ = prepare_xy(test, dropped)
    X_train = engineer_features(X_train)
    X_validation = engineer_features(X_validation).reindex(columns=X_train.columns)
    X_test = engineer_features(X_test).reindex(columns=X_train.columns)

    rows, fitted = [], {}
    for family in ["logistic_regression", "hist_gradient_boosting"]:
        for strategy in STRATEGIES:
            model = build(X_train, family, strategy)
            model.fit(X_train, y_train)
            probability = model.predict_proba(X_validation)[:, 1]
            rows.append({
                "model": family,
                "strategy": strategy,
                **metrics(y_validation, probability),
                "mean_predicted_probability": round(float(probability.mean()), 4),
                "operating_points": operating_points(y_validation, probability),
            })
            fitted[(family, strategy)] = model

    winner = max(
        rows,
        key=lambda row: (row["pr_auc"], row["roc_auc"], -row["brier_score"]),
    )
    model = fitted[(winner["model"], winner["strategy"])]
    test_probability = model.predict_proba(X_test)[:, 1]
    threshold = winner["operating_points"]["f1_optimized"]["threshold"]

    artifact = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "outcome": "readmission within 30 days among eligible discharges",
        "selection_policy": (
            "validation PR-AUC, then ROC-AUC, then Brier; "
            "locked test scored once for the selected candidate"
        ),
        "training_prevalence": round(float(y_train.mean()), 4),
        "validation_prevalence": round(float(y_validation.mean()), 4),
        "validation_candidates": rows,
        "selected": {"model": winner["model"], "strategy": winner["strategy"]},
        "locked_test": {
            **metrics(y_test, test_probability),
            "mean_predicted_probability": round(float(test_probability.mean()), 4),
            "prevalence": round(float(y_test.mean()), 4),
            "f1_optimized": threshold_metrics(y_test, test_probability, threshold),
        },
        "probability_scale_warning": (
            "class_weight, smote and undersample shift predicted probabilities "
            "above the observed prevalence; recalibrate before any probability "
            "is interpreted as risk."
        ),
        "smote_note": (
            "SMOTE interpolates in the preprocessed space, so one-hot columns "
            "of synthetic encounters can be fractional; treat it as a benchmark, "
            "not a clinically meaningful synthetic patient."
        ),
        "clinical_status": "research benchmark; not clinically validated",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: artifact[k] for k in ["selected", "locked_test"]}, indent=2
    ))


if __name__ == "__main__":
    main()
