"""Benchmark nonlinear patient readmission candidates without touching the final test set for model selection.

Selection is based on validation ROC-AUC/PR-AUC/Brier and operational threshold metrics.
The test set is evaluated only once for the validation-selected winner.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from train_patient_readmission import load_data, make_patient_group_splits, prepare_xy, threshold_metrics, choose_threshold_on_validation

OUT=Path('modeling/artifacts/next_model_benchmark.json')
# Ablation showed prior utilization is essential; diagnoses improved discrimination when removed
# but worsened Brier, so we benchmark both full and diagnosis-reduced feature sets on validation.
DROP_DIAG=['diag_1','diag_2','diag_3','number_diagnoses']

def dense_preprocess(X):
    num=X.select_dtypes(include=[np.number]).columns.tolist(); cat=[c for c in X.columns if c not in num]
    return ColumnTransformer([
      ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
      ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=20,sparse_output=False))]),cat)
    ])

def sparse_preprocess(X):
    num=X.select_dtypes(include=[np.number]).columns.tolist(); cat=[c for c in X.columns if c not in num]
    return ColumnTransformer([
      ('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),num),
      ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=20))]),cat)
    ])

def score(y,p): return {'roc_auc':round(float(roc_auc_score(y,p)),4),'pr_auc':round(float(average_precision_score(y,p)),4),'brier_score':round(float(brier_score_loss(y,p)),4)}

def main():
    df=load_data(); tr,va,te=make_patient_group_splits(df)
    Xtr,ytr,dropped=prepare_xy(tr); Xva,yva,_=prepare_xy(va,dropped); Xte,yte,_=prepare_xy(te,dropped)
    Xva=Xva.reindex(columns=Xtr.columns); Xte=Xte.reindex(columns=Xtr.columns)
    variants={'full':[], 'diagnosis_reduced':DROP_DIAG}
    results=[]; fitted={}
    for variant,drop in variants.items():
      a=Xtr.drop(columns=drop,errors='ignore'); b=Xva.drop(columns=drop,errors='ignore')
      candidates={
       'logistic':Pipeline([('prep',dense_preprocess(a)),('model',LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',random_state=42))]),
       'hist_gradient_boosting':Pipeline([('prep',dense_preprocess(a)),('model',HistGradientBoostingClassifier(max_iter=250,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1.0,random_state=42))]),
       'random_forest':Pipeline([('prep',sparse_preprocess(a)),('model',RandomForestClassifier(n_estimators=350,min_samples_leaf=10,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=42))])}
      for name,m in candidates.items():
        m.fit(a,ytr); p=m.predict_proba(b)[:,1]; th=choose_threshold_on_validation(yva,p)
        row={'model':name,'feature_variant':variant,**score(yva,p),'validation_threshold':th}
        results.append(row); fitted[(name,variant)]=(m,drop,th['threshold'])
    # Winner uses discrimination first, then PR-AUC, then lower Brier. Test is untouched until here.
    winner=max(results,key=lambda r:(r['roc_auc'],r['pr_auc'],-r['brier_score']))
    m,drop,th=fitted[(winner['model'],winner['feature_variant'])]
    testX=Xte.drop(columns=drop,errors='ignore'); pt=m.predict_proba(testX)[:,1]
    artifact={'schema_version':'1.0.0','generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'selection_policy':'Select on validation only. Final test is evaluated once for selected winner.',
      'ablation_context':'Prior utilization must be retained. Diagnosis-reduced variant is tested because prior ablation improved ROC-AUC/PR-AUC when diagnoses were removed, while worsening Brier; calibration remains part of selection evidence.',
      'validation_candidates':results,'selected_winner':winner,
      'winner_test':{**score(yte,pt),'threshold_metrics':threshold_metrics(yte,pt,th)},
      'clinical_status':'research benchmark only; not deployment-ready'}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(artifact,indent=2)); print(json.dumps(artifact,indent=2))
if __name__=='__main__': main()
