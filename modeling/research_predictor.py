"""Guarded inference interface for the packaged research model.

This module permits reproducible technical inference tests only. Its output is
explicitly marked research-only and must not be exposed as a clinical decision.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).parent
MODEL=ROOT/'artifacts/readmission_research_model.joblib'
MANIFEST=ROOT/'artifacts/readmission_research_model_manifest.json'

def load_bundle():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    return joblib.load(MODEL),manifest

def _coerce_row(payload: dict, manifest: dict) -> pd.DataFrame:
    expected=manifest['input_features']; types=manifest['feature_types']
    unknown=sorted(set(payload)-set(expected))
    if unknown: raise ValueError(f'Unknown fields: {unknown}')
    row={}
    for col in expected:
        value=payload.get(col,None)
        if types[col]=='numeric':
            row[col]=np.nan if value in (None,'') else float(value)
        else:
            row[col]=np.nan if value in (None,'') else str(value)
    return pd.DataFrame([row],columns=expected)

def predict_research_encounter(payload: dict) -> dict:
    model,manifest=load_bundle(); row=_coerce_row(payload,manifest)
    probability=float(model.predict_proba(row)[0,1])
    return {
      'model_id':manifest['model_id'],
      'research_probability':round(probability,4),
      'outcome':'30-day readmission',
      'clinical_status':manifest['clinical_status'],
      'public_patient_probability_allowed':False,
      'warning':'Research estimate only. Do not use for diagnosis, discharge, treatment, triage, or autonomous clinical decisions.'
    }
