"""Train and package the validation-selected research model.

Creates a reproducible sklearn pipeline artifact plus a typed input manifest.
Research/demo use only; not clinically validated.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
from benchmark_next_models import dense_preprocess
from train_patient_readmission import load_data, make_patient_group_splits, prepare_xy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

MODEL_OUT=Path('modeling/artifacts/readmission_research_model.joblib')
MANIFEST_OUT=Path('modeling/artifacts/readmission_research_model_manifest.json')

def main():
    df=load_data(); tr,va,_=make_patient_group_splits(df)
    Xtr,ytr,dropped=prepare_xy(tr)
    Xva,yva,_=prepare_xy(va,dropped); Xva=Xva.reindex(columns=Xtr.columns)
    Xfit=pd.concat([Xtr,Xva],axis=0); yfit=pd.concat([ytr,yva],axis=0)
    model=Pipeline([('prep',dense_preprocess(Xfit)),('model',HistGradientBoostingClassifier(max_iter=250,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1.0,random_state=42))])
    model.fit(Xfit,yfit)
    numeric=Xfit.select_dtypes(include='number').columns.tolist()
    feature_types={c:('numeric' if c in numeric else 'categorical') for c in Xfit.columns}
    MODEL_OUT.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,MODEL_OUT)
    manifest={
      'schema_version':'1.1.0','generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'model_id':'uci-diabetes-histgb-research-v2','architecture':'HistGradientBoostingClassifier',
      'feature_variant':'full','input_features':Xfit.columns.tolist(),'feature_types':feature_types,
      'dropped_sparse_columns':dropped,'outcome':'readmission within 30 days (<30 vs other)',
      'fit_data':'training + validation patient groups only; final test excluded',
      'clinical_status':'research/demo only; not clinically validated',
      'public_patient_probability_allowed':False,
      'intended_use':'technical demonstration, reproducible inference testing, and external-validation preparation only'
    }
    MANIFEST_OUT.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'model_path':str(MODEL_OUT),'manifest':manifest},indent=2))
if __name__=='__main__': main()
