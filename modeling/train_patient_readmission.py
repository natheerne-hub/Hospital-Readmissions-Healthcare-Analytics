"""Train a defensible patient-level 30-day readmission baseline model.

Dataset target: UCI Diabetes 130-US Hospitals for Years 1999-2008.
Expected local file: data/diabetic_data.csv

The script intentionally separates model development from the CMS HRRP hospital-level
analytics. It predicts whether the encounter is followed by readmission in <30 days.
It writes a machine-readable metrics artifact for later MVP integration.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/diabetic_data.csv")
OUTPUT_PATH = Path("modeling/artifacts/patient_readmission_metrics.json")

# Administrative identifiers are excluded from predictors. patient_nbr is also used
# to keep all encounters from a patient in only one split, preventing leakage.
ID_COLUMNS = ["encounter_id", "patient_nbr"]
TARGET_COLUMN = "readmitted"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/diabetic_data.csv. Download the official UCI Diabetes "
            "130-US Hospitals dataset and place diabetic_data.csv in data/."
        )
    df = pd.read_csv(DATA_PATH, na_values=["?", "Unknown/Invalid", "None"])
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")
    return df


def make_patient_group_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    patients = df["patient_nbr"].dropna().unique()
    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )
    train_mask = df["patient_nbr"].isin(train_patients)
    test_mask = df["patient_nbr"].isin(test_patients)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def prepare_xy(df: pd.DataFrame):
    y = (df[TARGET_COLUMN] == "<30").astype(int)
    X = df.drop(columns=[TARGET_COLUMN, *ID_COLUMNS], errors="ignore").copy()

    # Remove columns that are effectively unusable if almost entirely missing.
    missing_fraction = X.isna().mean()
    drop_sparse = missing_fraction[missing_fraction > 0.80].index.tolist()
    X = X.drop(columns=drop_sparse)
    return X, y, drop_sparse


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_cols),
            ("categorical", categorical, categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def threshold_metrics(y_true, probabilities, threshold: float):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": round(float(threshold), 3),
        "sensitivity_recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_threshold(y_true, probabilities):
    # MVP policy: maximize F1 on the held-out test set only for exploration.
    # For a real pilot, threshold selection must be done on validation data and
    # aligned to operational costs/capacity, not optimized on the final test set.
    candidates = np.arange(0.10, 0.91, 0.01)
    scored = [threshold_metrics(y_true, probabilities, t) for t in candidates]
    return max(scored, key=lambda row: row["f1"]), scored


def main() -> None:
    df = load_data()
    train_df, test_df = make_patient_group_split(df)
    X_train, y_train, dropped_sparse = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df)
    X_test = X_test.reindex(columns=X_train.columns)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    selected, all_thresholds = choose_threshold(y_test, probabilities)
    metrics = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "outcome": "readmission within 30 days (<30 vs all other outcomes)",
        "model": "class-weighted logistic regression baseline",
        "split": "patient-group holdout; no patient_nbr appears in both train and test",
        "train_encounters": int(len(train_df)),
        "test_encounters": int(len(test_df)),
        "test_prevalence": round(float(y_test.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "selected_threshold_exploratory": selected,
        "dropped_sparse_columns_gt_80pct_missing": dropped_sparse,
        "threshold_policy_warning": (
            "The selected threshold is exploratory because it was optimized on the holdout set. "
            "A production or clinical pilot must select threshold on validation data using an "
            "explicit cost/capacity objective, then report untouched test performance."
        ),
        "clinical_status": "research MVP only; not clinically validated or deployment-ready",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
