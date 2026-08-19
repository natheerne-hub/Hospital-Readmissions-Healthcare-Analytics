from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'runtime' / 'model' / 'readmission_research_model.joblib'
MANIFEST_PATH = ROOT / 'runtime' / 'model' / 'readmission_research_model_manifest.json'
RESEARCH_THRESHOLD = 0.13

REQUIRED_RESEARCH_FIELDS = [
    'age', 'gender', 'number_inpatient', 'number_emergency',
    'time_in_hospital', 'num_medications', 'number_diagnoses', 'diag_1'
]

NUMERIC_FIELDS = {
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
}

GLOBAL_DRIVER_FIELDS = [
    'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
    'discharge_disposition_id', 'medical_specialty', 'number_emergency', 'age'
]


@lru_cache(maxsize=1)
def load_bundle():
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise RuntimeError('Research model runtime bundle is not available yet.')
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    model = joblib.load(MODEL_PATH)
    return model, manifest


def normalize_payload(payload: dict, manifest: dict) -> pd.DataFrame:
    expected = manifest['input_features']
    unknown = sorted(set(payload) - set(expected) - {'research_acknowledged'})
    if unknown:
        raise HTTPException(status_code=400, detail=f'Unknown fields: {unknown}')

    missing_required = [
        f for f in REQUIRED_RESEARCH_FIELDS
        if payload.get(f) in (None, '')
    ]
    if missing_required:
        raise HTTPException(
            status_code=400,
            detail=f'Missing required research fields: {missing_required}'
        )

    row = {}
    for feature in expected:
        value = payload.get(feature, np.nan)
        if value in (None, ''):
            value = np.nan
        elif feature in NUMERIC_FIELDS:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f'Field {feature} must be numeric.'
                )
        row[feature] = value
    return pd.DataFrame([row])


@app.get('/')
def health():
    try:
        _, manifest = load_bundle()
        return {
            'ok': True,
            'model_id': manifest['model_id'],
            'clinical_status': manifest['clinical_status'],
            'mode': 'research_only'
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post('/')
def predict(payload: dict):
    if payload.get('research_acknowledged') is not True:
        raise HTTPException(
            status_code=400,
            detail='Research-only acknowledgement is required.'
        )

    try:
        model, manifest = load_bundle()
        row = normalize_payload(payload, manifest)
        probability = float(model.predict_proba(row)[0, 1])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Inference failed: {exc}')

    entered_global_drivers = [
        {'feature': f, 'value': payload.get(f)}
        for f in GLOBAL_DRIVER_FIELDS
        if payload.get(f) not in (None, '')
    ]

    return {
        'model_id': manifest['model_id'],
        'research_probability': round(probability, 4),
        'research_threshold': RESEARCH_THRESHOLD,
        'threshold_signal': 'above' if probability >= RESEARCH_THRESHOLD else 'below',
        'clinical_status': manifest['clinical_status'],
        'public_patient_probability_allowed': False,
        'entered_global_driver_fields': entered_global_drivers,
        'interpretation': (
            'Research estimate only. The threshold signal is for technical demonstration and '
            'must not be used for diagnosis, treatment, discharge, or autonomous clinical decisions.'
        )
    }
