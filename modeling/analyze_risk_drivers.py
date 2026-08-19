"""Analyze which feature groups drive the readmission baseline.

Outputs global permutation importance on untouched test encounters and logistic
coefficient summaries. Importance is predictive, not causal.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.inspection import permutation_importance
from train_patient_readmission import load_data, make_patient_group_splits, prepare_xy, build_pipeline

OUT = Path('modeling/artifacts/readmission_risk_drivers.json')


def main():
    df = load_data()
    train_df, val_df, test_df = make_patient_group_splits(df)
    X_train, y_train, dropped = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df, dropped)
    X_test = X_test.reindex(columns=X_train.columns)
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    # Permutation importance measures loss of ROC-AUC when each original input
    # column is disrupted on the untouched test set. Positive values indicate
    # useful predictive information; it does not prove causality.
    perm = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1)
    ranked = sorted([
        {'feature': c, 'roc_auc_drop_mean': round(float(m), 6), 'roc_auc_drop_std': round(float(s), 6)}
        for c, m, s in zip(X_test.columns, perm.importances_mean, perm.importances_std)
    ], key=lambda x: x['roc_auc_drop_mean'], reverse=True)

    preprocess = model.named_steps['preprocess']
    feature_names = preprocess.get_feature_names_out()
    coef = model.named_steps['model'].coef_[0]
    coef_rows = sorted([
        {'encoded_feature': str(f), 'coefficient': round(float(c), 6), 'odds_ratio': round(float(np.exp(c)), 6)}
        for f, c in zip(feature_names, coef)
    ], key=lambda x: abs(x['coefficient']), reverse=True)

    artifact = {
        'schema_version': '1.0.0',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'permutation importance on untouched patient-group test set plus fitted logistic coefficients',
        'interpretation_warning': 'Predictive importance/association only. Do not interpret as a causal effect of treatment, hospital quality, medication, or diagnosis.',
        'top_original_features_by_test_roc_auc_drop': ranked[:25],
        'top_encoded_logistic_terms_by_absolute_coefficient': coef_rows[:40],
        'dropped_sparse_columns': dropped,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))

if __name__ == '__main__':
    main()
