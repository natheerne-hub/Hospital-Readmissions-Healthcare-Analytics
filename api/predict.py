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
REGISTRY_PATH = ROOT / 'modeling' / 'model_registry.json'

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

SMOKE_TEST_PAYLOAD = {
    'research_acknowledged': True,
    'age': '[60-70)',
    'gender': 'Male',
    'number_inpatient': 2,
    'number_emergency': 1,
    'number_outpatient': 0,
    'time_in_hospital': 4,
    'num_medications': 15,
    'number_diagnoses': 7,
    'diag_1': '428',
    'diag_2': '250.4',
    'diag_3': '401',
    'discharge_disposition_id': 1,
    'admission_type_id': 1,
    'admission_source_id': 7,
    'diabetesMed': 'Yes',
    'insulin': 'Steady',
    'change': 'No'
}


@lru_cache(maxsize=1)
def load_bundle():
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise RuntimeError('Research model runtime bundle is not available yet.')
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    model = joblib.load(MODEL_PATH)
    return model, manifest


@lru_cache(maxsize=1)
def load_registry_entry():
    if not REGISTRY_PATH.exists():
        raise RuntimeError('Model registry is unavailable.')
    registry = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
    return next(
        m for m in registry['models']
        if m['task'] == 'patient_30_day_readmission_probability'
    )


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


def health_payload():
    try:
        _, manifest = load_bundle()
        registry_entry = load_registry_entry()
        return {
            'ok': True,
            'model_id': manifest['model_id'],
            'clinical_status': manifest['clinical_status'],
            'mode': 'research_only',
            'public_patient_probability_allowed': bool(
                registry_entry.get('patient_probability_allowed', False)
            )
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def prediction_payload(payload: dict):
    if payload.get('research_acknowledged') is not True:
        raise HTTPException(
            status_code=400,
            detail='Research-only acknowledgement is required.'
        )

    try:
        model, manifest = load_bundle()
        registry_entry = load_registry_entry()
        row = normalize_payload(payload, manifest)

        # Run the packaged model so deployment health and feature compatibility are
        # exercised, but never expose patient-level output while the registry lock is on.
        _ = float(model.predict_proba(row)[0, 1])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Inference check failed: {exc}')

    entered_global_drivers = [
        {'feature': f, 'value': payload.get(f)}
        for f in GLOBAL_DRIVER_FIELDS
        if payload.get(f) not in (None, '')
    ]

    probability_allowed = bool(registry_entry.get('patient_probability_allowed', False))

    if not probability_allowed:
        return {
            'model_id': manifest['model_id'],
            'prediction_locked': True,
            'clinical_status': manifest['clinical_status'],
            'public_patient_probability_allowed': False,
            'entered_global_driver_fields': entered_global_drivers,
            'interpretation': (
                'The packaged research model executed successfully, but patient-level '
                'probability and threshold outputs are intentionally withheld. External '
                'validation, calibration review, subgroup assessment, workflow evaluation, '
                'and governance approval are required before public probability output can be enabled.'
            )
        }

    raise HTTPException(
        status_code=503,
        detail='Probability output requires a separately reviewed release path.'
    )


@app.get('/')
@app.get('/api/predict')
def health(demo: int = 0):
    if demo == 1:
        result = prediction_payload(SMOKE_TEST_PAYLOAD)
        result['smoke_test'] = True
        return result
    return health_payload()


@app.post('/')
@app.post('/api/predict')
def predict(payload: dict):
    return prediction_payload(payload)
