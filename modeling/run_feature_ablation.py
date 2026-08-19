"""Run feature-group ablation experiments on the patient readmission baseline.

Each experiment retrains the same baseline after removing a predefined feature group,
then reports ROC-AUC/PR-AUC/Brier on the untouched patient-group test set. The goal
is to learn which groups materially contribute predictive signal and which add noise.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from train_patient_readmission import load_data, make_patient_group_splits, prepare_xy, build_pipeline

OUT = Path('modeling/artifacts/feature_ablation_results.json')

GROUPS = {
    'prior_utilization': ['number_inpatient', 'number_emergency', 'number_outpatient'],
    'diagnoses': ['diag_1', 'diag_2', 'diag_3', 'number_diagnoses'],
    'medication_burden': ['num_medications'],
    'hospital_course': ['time_in_hospital', 'num_lab_procedures', 'num_procedures'],
    'discharge_and_admission': ['admission_type_id', 'discharge_disposition_id', 'admission_source_id'],
    'demographics': ['age', 'gender', 'race'],
}


def evaluate(X_train, y_train, X_test, y_test):
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        'roc_auc': round(float(roc_auc_score(y_test, prob)), 4),
        'pr_auc': round(float(average_precision_score(y_test, prob)), 4),
        'brier_score': round(float(brier_score_loss(y_test, prob)), 4),
    }


def main():
    df = load_data()
    train_df, _, test_df = make_patient_group_splits(df)
    X_train, y_train, dropped = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df, dropped)
    X_test = X_test.reindex(columns=X_train.columns)

    baseline = evaluate(X_train, y_train, X_test, y_test)
    experiments = []
    for name, cols in GROUPS.items():
        present = [c for c in cols if c in X_train.columns]
        if not present:
            continue
        Xt = X_train.drop(columns=present)
        Xs = X_test.drop(columns=present).reindex(columns=Xt.columns)
        metrics = evaluate(Xt, y_train, Xs, y_test)
        experiments.append({
            'removed_group': name,
            'removed_features': present,
            **metrics,
            'delta_roc_auc_vs_baseline': round(metrics['roc_auc'] - baseline['roc_auc'], 4),
            'delta_pr_auc_vs_baseline': round(metrics['pr_auc'] - baseline['pr_auc'], 4),
            'delta_brier_vs_baseline': round(metrics['brier_score'] - baseline['brier_score'], 4),
        })

    artifact = {
        'schema_version': '1.0.0',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline': baseline,
        'experiments': sorted(experiments, key=lambda r: r['delta_roc_auc_vs_baseline']),
        'interpretation': 'A negative delta after removing a group suggests the group adds predictive signal; a positive delta suggests the removed group may be noisy or redundant for this baseline. This is model-specific, not causal.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
