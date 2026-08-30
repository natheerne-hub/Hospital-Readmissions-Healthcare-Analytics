"""Strict improvement benchmark for 30-day readmission research.

Research only; not clinically validated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_patient_readmission import (
    choose_threshold_on_validation,
    load_data,
    make_patient_group_splits,
    prepare_xy,
    threshold_metrics,
)

OUT = Path("modeling/artifacts/improved_model_benchmark.json")
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def filter_eligible_cohort(df: pd.DataFrame):
    disposition = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    excluded = disposition.isin(EXCLUDED_DISPOSITIONS)
    return df.loc[~excluded].copy(), {
        "input_encounters": int(len(df)),
        "excluded_expired_or_hospice": int(excluded.sum()),
        "eligible_encounters": int((~excluded).sum()),
    }


def diagnosis_group(value) -> str:
    try:
        code = float(value)
    except (TypeError, ValueError):
        return "other_or_missing"
    if not np.isfinite(code):
        return "other_or_missing"
    integer = int(code)
    if 390 <= code <= 459 or integer == 785:
        return "circulatory"
    if 460 <= code <= 519 or integer == 786:
        return "respiratory"
    if 520 <= code <= 579 or integer == 787:
        return "digestive"
    if 249 <= code < 251:
        return "diabetes"
    if 580 <= code <= 629 or integer == 788:
        return "genitourinary"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 800 <= code <= 999:
        return "injury"
    if 140 <= code <= 239:
        return "neoplasm"
    return "other_or_missing"


def engineer_features(
    X: pd.DataFrame, *, drop_raw_diagnoses: bool = True
) -> pd.DataFrame:
    out = X.copy()
    if "age" in out:
        age_map = {f"[{x}-{x + 10})": x + 5 for x in range(0, 100, 10)}
        out["age_midpoint"] = out["age"].map(age_map).astype(float)

    prior_names = ["number_inpatient", "number_emergency", "number_outpatient"]
    prior = out.reindex(columns=prior_names).apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    out["prior_total_visits"] = prior.sum(axis=1)
    out["prior_acute_visits"] = prior["number_inpatient"] + prior["number_emergency"]
    out["any_prior_inpatient"] = (prior["number_inpatient"] > 0).astype(int)
    out["any_prior_emergency"] = (prior["number_emergency"] > 0).astype(int)
    out["log_prior_total_visits"] = np.log1p(out["prior_total_visits"])

    stay = pd.to_numeric(out.get("time_in_hospital"), errors="coerce").clip(lower=1)
    for source, target in [
        ("num_lab_procedures", "labs_per_day"),
        ("num_procedures", "procedures_per_day"),
        ("num_medications", "medications_per_day"),
    ]:
        if source in out:
            out[target] = pd.to_numeric(out[source], errors="coerce") / stay

    medication_columns = [x for x in MEDICATION_COLUMNS if x in out]
    if medication_columns:
        medications = out[medication_columns].astype("string")
        out["medication_change_count"] = medications.isin(["Up", "Down"]).sum(axis=1)
        out["active_medication_count"] = (~medications.isin(["No", "<NA>"])).sum(axis=1)

    grouped = []
    for column in DIAGNOSIS_COLUMNS:
        if column in out:
            grouped_name = f"{column}_group"
            out[grouped_name] = out[column].map(diagnosis_group)
            grouped.append(grouped_name)
    if grouped:
        out["diagnosis_group_count"] = out[grouped].nunique(axis=1)
    if drop_raw_diagnoses:
        return out.drop(columns=DIAGNOSIS_COLUMNS, errors="ignore")
    return out


def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                sparse_output=False,
            )),
        ]), categorical),
    ])


def build_model(X: pd.DataFrame, parameters: dict):
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", HistGradientBoostingClassifier(
            random_state=42, **parameters
        )),
    ])


def metrics(y_true, probability):
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probability)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 4),
    }


def split_validation(validation_df: pd.DataFrame):
    patients = validation_df["patient_nbr"].dropna().unique()
    calibration, threshold = train_test_split(
        patients, test_size=0.5, random_state=2026
    )
    return (
        validation_df[validation_df["patient_nbr"].isin(calibration)].copy(),
        validation_df[validation_df["patient_nbr"].isin(threshold)].copy(),
    )


def fitted_calibrators(y_cal, p_cal):
    epsilon = 1e-6
    logit = np.log(
        np.clip(p_cal, epsilon, 1 - epsilon)
        / np.clip(1 - p_cal, epsilon, 1)
    )
    sigmoid = LogisticRegression(random_state=42).fit(
        logit.reshape(-1, 1), y_cal
    )
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    return {"uncalibrated": None, "sigmoid": sigmoid, "isotonic": isotonic}


def calibrate(name, calibrator, probability):
    if name == "uncalibrated":
        return probability
    if name == "isotonic":
        return calibrator.predict(probability)
    epsilon = 1e-6
    logit = np.log(
        np.clip(probability, epsilon, 1 - epsilon)
        / np.clip(1 - probability, epsilon, 1)
    )
    return calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]


def main():
    raw = load_data()
    eligible, cohort = filter_eligible_cohort(raw)
    train_df, validation_df, test_df = make_patient_group_splits(eligible)
    calibration_df, threshold_df = split_validation(validation_df)

    X_train, y_train, dropped = prepare_xy(train_df)
    X_cal, y_cal, _ = prepare_xy(calibration_df, dropped)
    X_threshold, y_threshold, _ = prepare_xy(threshold_df, dropped)
    X_test, y_test, _ = prepare_xy(test_df, dropped)

    X_train = engineer_features(X_train)
    X_cal = engineer_features(X_cal).reindex(columns=X_train.columns)
    X_threshold = engineer_features(X_threshold).reindex(columns=X_train.columns)
    X_test = engineer_features(X_test).reindex(columns=X_train.columns)

    candidates = [
        {"max_iter": 220, "learning_rate": 0.04, "max_leaf_nodes": 15,
         "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"max_iter": 260, "learning_rate": 0.05, "max_leaf_nodes": 31,
         "min_samples_leaf": 30, "l2_regularization": 2.0},
        {"max_iter": 300, "learning_rate": 0.04, "max_leaf_nodes": 31,
         "min_samples_leaf": 60, "l2_regularization": 3.0},
        {"max_iter": 260, "learning_rate": 0.05, "max_leaf_nodes": 63,
         "min_samples_leaf": 80, "l2_regularization": 5.0},
    ]

    rows, models = [], []
    for index, parameters in enumerate(candidates, start=1):
        model = build_model(X_train, parameters)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_cal)[:, 1]
        rows.append({
            "candidate": index,
            "parameters": parameters,
            **metrics(y_cal, probability),
        })
        models.append(model)

    winner_index = max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["roc_auc"],
            rows[index]["pr_auc"],
            -rows[index]["brier_score"],
        ),
    )
    winner = models[winner_index]
    cal_probability = winner.predict_proba(X_cal)[:, 1]
    threshold_raw = winner.predict_proba(X_threshold)[:, 1]
    calibrators = fitted_calibrators(y_cal.to_numpy(), cal_probability)

    calibration_rows = []
    threshold_probabilities = {}
    for name, calibrator in calibrators.items():
        probability = calibrate(name, calibrator, threshold_raw)
        threshold_probabilities[name] = probability
        calibration_rows.append({"method": name, **metrics(y_threshold, probability)})

    calibration_winner = min(
        calibration_rows,
        key=lambda row: (
            row["brier_score"], -row["roc_auc"], -row["pr_auc"]
        ),
    )
    method = calibration_winner["method"]
    operating_probability = threshold_probabilities[method]
    threshold = choose_threshold_on_validation(
        y_threshold, operating_probability
    )["threshold"]

    test_raw = winner.predict_proba(X_test)[:, 1]
    test_probability = calibrate(method, calibrators[method], test_raw)

    artifact = {
        "schema_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "outcome": "readmission within 30 days among eligible discharges",
        "cohort": cohort,
        "split": {
            "policy": "patient-group train/calibration/threshold/test",
            "train_encounters": int(len(train_df)),
            "calibration_encounters": int(len(calibration_df)),
            "threshold_encounters": int(len(threshold_df)),
            "locked_test_encounters": int(len(test_df)),
        },
        "candidate_selection": {
            "policy": "selected before locked test evaluation",
            "candidates": rows,
            "selected": rows[winner_index],
        },
        "calibration": {
            "candidates": calibration_rows,
            "selected": calibration_winner,
        },
        "threshold_selection": {
            "threshold": threshold,
            "metrics": threshold_metrics(
                y_threshold, operating_probability, threshold
            ),
        },
        "locked_test": {
            **metrics(y_test, test_probability),
            "threshold_metrics": threshold_metrics(
                y_test, test_probability, threshold
            ),
        },
        "feature_engineering": [
            "ICD-9 clinical diagnosis groups",
            "prior utilization totals and flags",
            "care intensity per hospital day",
            "medication activity and change counts",
            "age midpoint and diagnosis diversity",
        ],
        "dropped_sparse_columns": dropped,
        "clinical_status": "research only; not externally validated",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
