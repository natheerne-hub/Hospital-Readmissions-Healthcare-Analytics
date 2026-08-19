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
ID_COLUMNS = ["encounter_id", "patient_nbr"]
TARGET_COLUMN = "readmitted"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/diabetic_data.csv. Download the official UCI Diabetes "
            "130-US Hospitals dataset and place diabetic_data.csv in data/."
        )
    df = pd.read_csv(DATA_PATH, na_values=["?", "Unknown/Invalid", "None"])
    if TARGET_COLUMN not in df.columns or "patient_nbr" not in df.columns:
        raise ValueError("Expected 'readmitted' and 'patient_nbr' columns were not found.")
    return df


def make_patient_group_splits(df: pd.DataFrame, random_state: int = 42):
    """Create 64/16/20 train/validation/test splits by patient identity."""
    patients = df["patient_nbr"].dropna().unique()
    train_val_patients, test_patients = train_test_split(
        patients, test_size=0.20, random_state=random_state
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.20, random_state=random_state
    )

    train_df = df[df["patient_nbr"].isin(train_patients)].copy()
    val_df = df[df["patient_nbr"].isin(val_patients)].copy()
    test_df = df[df["patient_nbr"].isin(test_patients)].copy()
    return train_df, val_df, test_df


def prepare_xy(df: pd.DataFrame, sparse_columns: list[str] | None = None):
    y = (df[TARGET_COLUMN] == "<30").astype(int)
    X = df.drop(columns=[TARGET_COLUMN, *ID_COLUMNS], errors="ignore").copy()

    if sparse_columns is None:
        missing_fraction = X.isna().mean()
        sparse_columns = missing_fraction[missing_fraction > 0.80].index.tolist()
    X = X.drop(columns=sparse_columns, errors="ignore")
    return X, y, sparse_columns


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )),
    ])


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
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def choose_threshold_on_validation(y_true, probabilities):
    candidates = np.arange(0.10, 0.91, 0.01)
    scored = [threshold_metrics(y_true, probabilities, t) for t in candidates]
    return max(scored, key=lambda row: row["f1"])


def discrimination_metrics(y_true, probabilities):
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "prevalence": round(float(y_true.mean()), 4),
    }


def main() -> None:
    df = load_data()
    train_df, val_df, test_df = make_patient_group_splits(df)

    X_train, y_train, dropped_sparse = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df, dropped_sparse)
    X_test, y_test, _ = prepare_xy(test_df, dropped_sparse)
    X_val = X_val.reindex(columns=X_train.columns)
    X_test = X_test.reindex(columns=X_train.columns)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    selected = choose_threshold_on_validation(y_val, val_prob)
    threshold = selected["threshold"]

    test_prob = pipeline.predict_proba(X_test)[:, 1]
    final_threshold_metrics = threshold_metrics(y_test, test_prob, threshold)

    metrics = {
        "schema_version": "1.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "outcome": "readmission within 30 days (<30 vs all other outcomes)",
        "model": "class-weighted logistic regression baseline",
        "split": "patient-group train/validation/test split; no patient_nbr crosses splits",
        "encounters": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "validation": {
            **discrimination_metrics(y_val, val_prob),
            "threshold_selected_by": "maximum F1 on validation set",
            "selected_threshold": selected,
        },
        "test": {
            **discrimination_metrics(y_test, test_prob),
            "threshold_metrics": final_threshold_metrics,
        },
        "dropped_sparse_columns_gt_80pct_missing_train": dropped_sparse,
        "threshold_policy": (
            "Threshold is selected only on validation data. Final performance is reported "
            "on an untouched patient-group test set. A Dubai pilot should replace F1-only "
            "selection with an explicit clinical/operational cost and capacity objective."
        ),
        "clinical_status": "research MVP only; not clinically validated or deployment-ready",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
