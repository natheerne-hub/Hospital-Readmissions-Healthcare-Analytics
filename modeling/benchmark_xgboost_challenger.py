"""XGBoost challenger benchmark on the matched eligible-discharge cohort."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from improve_readmission_model import (
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

OUT = Path("modeling/artifacts/xgboost_challenger.json")


def preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore", min_frequency=20
            )),
        ]), categorical),
    ])


def build(X, parameters):
    return Pipeline([
        ("preprocess", preprocessor(X)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            **parameters,
        )),
    ])


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
    X_validation = X_validation.reindex(columns=X_train.columns)
    X_test = X_test.reindex(columns=X_train.columns)

    variants = {
        "raw": (X_train, X_validation, X_test),
        "augmented": (
            engineer_features(X_train, drop_raw_diagnoses=False),
            engineer_features(
                X_validation, drop_raw_diagnoses=False
            ),
            engineer_features(X_test, drop_raw_diagnoses=False),
        ),
    }
    parameters = [
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.04,
         "min_child_weight": 8, "subsample": 0.85,
         "colsample_bytree": 0.85, "reg_lambda": 3.0},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
         "min_child_weight": 10, "subsample": 0.85,
         "colsample_bytree": 0.85, "reg_lambda": 5.0},
        {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.035,
         "min_child_weight": 12, "subsample": 0.8,
         "colsample_bytree": 0.8, "reg_lambda": 6.0},
    ]

    rows, fitted = [], {}
    for variant, (train_X, validation_X, test_X) in variants.items():
        validation_X = validation_X.reindex(columns=train_X.columns)
        test_X = test_X.reindex(columns=train_X.columns)
        for index, params in enumerate(parameters, start=1):
            model = build(train_X, params)
            model.fit(train_X, y_train)
            probability = model.predict_proba(validation_X)[:, 1]
            row = {
                "variant": variant,
                "candidate": index,
                "parameters": params,
                **metrics(y_validation, probability),
                "operating_points": operating_points(
                    y_validation, probability
                ),
            }
            rows.append(row)
            fitted[(variant, index)] = (model, test_X)

    winner = max(
        rows,
        key=lambda row: (
            row["roc_auc"], row["pr_auc"], -row["brier_score"]
        ),
    )
    model, winner_test_X = fitted[
        (winner["variant"], winner["candidate"])
    ]
    test_probability = model.predict_proba(winner_test_X)[:, 1]
    test_result = {
        **metrics(y_test, test_probability),
        "f1_optimized": threshold_metrics(
            y_test,
            test_probability,
            winner["operating_points"]["f1_optimized"]["threshold"],
        ),
        "minimum_40pct_sensitivity": threshold_metrics(
            y_test,
            test_probability,
            winner["operating_points"]["minimum_40pct_sensitivity"]["threshold"],
        ),
    }
    artifact = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "validation selection; same locked eligible test cohort",
        "validation_candidates": rows,
        "selected": winner,
        "locked_test": test_result,
        "clinical_status": "research challenger; not clinically validated",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

