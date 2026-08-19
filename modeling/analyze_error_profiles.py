"""Profile false positives and false negatives for the patient readmission baseline.

The goal is to identify systematic differences between TP/FP/TN/FN groups without
making causal claims. Outputs summary statistics for clinically interpretable features.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from train_patient_readmission import (
    load_data,
    make_patient_group_splits,
    prepare_xy,
    build_pipeline,
    choose_threshold_on_validation,
)

OUT = Path('modeling/artifacts/readmission_error_profiles.json')

FOCUS_NUMERIC = [
    'number_inpatient', 'number_emergency', 'number_outpatient', 'time_in_hospital',
    'num_medications', 'num_lab_procedures', 'num_procedures', 'number_diagnoses',
]
FOCUS_CATEGORICAL = ['age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']


def group_label(y, pred):
    if y == 1 and pred == 1: return 'TP'
    if y == 0 and pred == 1: return 'FP'
    if y == 0 and pred == 0: return 'TN'
    return 'FN'


def summarize_numeric(frame: pd.DataFrame, col: str):
    if col not in frame.columns:
        return None
    s = pd.to_numeric(frame[col], errors='coerce')
    return {
        'n_non_null': int(s.notna().sum()),
        'mean': round(float(s.mean()), 4) if s.notna().any() else None,
        'median': round(float(s.median()), 4) if s.notna().any() else None,
        'p25': round(float(s.quantile(0.25)), 4) if s.notna().any() else None,
        'p75': round(float(s.quantile(0.75)), 4) if s.notna().any() else None,
    }


def summarize_categorical(frame: pd.DataFrame, col: str):
    if col not in frame.columns:
        return None
    vc = frame[col].astype('string').fillna('MISSING').value_counts(normalize=True).head(10)
    return [{'value': str(k), 'share': round(float(v), 4)} for k, v in vc.items()]


def main():
    df = load_data()
    train_df, val_df, test_df = make_patient_group_splits(df)
    X_train, y_train, dropped = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df, dropped)
    X_test, y_test, _ = prepare_xy(test_df, dropped)
    X_val = X_val.reindex(columns=X_train.columns)
    X_test = X_test.reindex(columns=X_train.columns)

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold_on_validation(y_val, val_prob)['threshold']
    test_prob = model.predict_proba(X_test)[:, 1]
    pred = (test_prob >= threshold).astype(int)

    profile = test_df.copy()
    profile['_y'] = y_test.to_numpy()
    profile['_pred'] = pred
    profile['_prob'] = test_prob
    profile['_group'] = [group_label(y, p) for y, p in zip(profile['_y'], profile['_pred'])]

    groups = {}
    for g in ['TP', 'FP', 'TN', 'FN']:
        part = profile[profile['_group'] == g]
        groups[g] = {
            'count': int(len(part)),
            'mean_predicted_probability': round(float(part['_prob'].mean()), 4) if len(part) else None,
            'numeric': {c: summarize_numeric(part, c) for c in FOCUS_NUMERIC if c in part.columns},
            'categorical': {c: summarize_categorical(part, c) for c in FOCUS_CATEGORICAL if c in part.columns},
        }

    artifact = {
        'schema_version': '1.0.0',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'threshold': threshold,
        'focus': 'Compare FP/FN profiles against correctly classified groups to guide feature engineering and model selection.',
        'interpretation_warning': 'These are descriptive error profiles, not causal explanations of readmission.',
        'groups': groups,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
