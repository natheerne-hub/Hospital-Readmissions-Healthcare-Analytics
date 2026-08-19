"""Guarded inference interface for the packaged research model."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).parent
MODEL=ROOT/'artifacts/readmission_research_model.joblib'
MANIFEST=ROOT/'artifacts/readmission_research_model_manifest.json'

def load_bundle():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    return joblib.load(MODEL),manifest

def predict_research_encounter(payload: dict) -> dict:
    model,manifest=load_bundle()
    expected=manifest['input_features']
    unknown=sorted(set(payload)-set(expected))
    if unknown: raise ValueError(f'Unknown fields: {unknown}')
    # Missing fields are represented as NA and handled by the fitted preprocessing pipeline.
    row=pd.DataFrame([{c:payload.get(c,pd.NA) for c in expected}])
    probability=float(model.predict_proba(row)[0,1])
    return {'model_id':manifest['model_id'],'research_probability':round(probability,4),'outcome':'30-day readmission','clinical_status':manifest['clinical_status'],'public_patient_probability_allowed':False,'warning':'Research estimate only. Do not use for diagnosis, discharge, treatment, or autonomous clinical decisions.'}
