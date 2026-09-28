"""Checks for SHAP explanations and the imbalance-strategy benchmark wiring."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))

pytest.importorskip("shap")
pytest.importorskip("imblearn")

from explain_with_shap import ReadmissionExplainer, load_bundle, output_to_input_map  # noqa: E402


def sample_encounters(features):
    rows = [
        {"age": "[70-80)", "time_in_hospital": 5, "number_inpatient": 3,
         "number_emergency": 1, "diag_1": "428", "insulin": "Up"},
        {"age": "[40-50)", "time_in_hospital": 1, "number_inpatient": 0,
         "diag_1": "786", "metformin": "Steady"},
    ]
    return pd.DataFrame(rows).reindex(columns=features)


def test_one_hot_columns_map_to_longest_feature_name():
    class Stub:
        def get_feature_names_out(self):
            return np.array(["cat__glyburide_No", "cat__glyburide-metformin_No", "num__number_inpatient"])

    mapping = output_to_input_map(Stub(), ["glyburide", "glyburide-metformin", "number_inpatient"])
    assert mapping == ["glyburide", "glyburide-metformin", "number_inpatient"]


def test_shap_contributions_reproduce_model_probability():
    model, manifest = load_bundle()
    explainer = ReadmissionExplainer(model, manifest)
    X = sample_encounters(manifest["input_features"])

    contributions = explainer.shap_by_input(X)
    assert list(contributions.columns) == manifest["input_features"]
    log_odds = explainer.base_value + contributions.sum(axis=1).to_numpy()
    expected = model.predict_proba(X)[:, 1]
    assert np.allclose(1 / (1 + np.exp(-log_odds)), expected, atol=1e-3)


def test_encounter_explanation_is_signed_and_bounded():
    explainer = ReadmissionExplainer()
    row = sample_encounters(explainer.features).iloc[0]
    explanation = explainer.explain_encounter(row, top_n=3)

    assert 0 <= explanation["model_probability"] <= 1
    assert len(explanation["top_risk_increasing"]) <= 3
    assert all(item["shap_log_odds"] > 0 for item in explanation["top_risk_increasing"])
    assert all(item["shap_log_odds"] < 0 for item in explanation["top_risk_decreasing"])


def test_imbalance_strategies_apply_only_inside_pipeline():
    from compare_imbalance_strategies import STRATEGIES, build

    X = sample_encounters(load_bundle()[1]["input_features"])
    for strategy in STRATEGIES:
        pipeline = build(X, "logistic_regression", strategy)
        assert pipeline.steps[0][0] == "preprocess"
        assert ("resample" in pipeline.named_steps) == (strategy in {"smote", "undersample"})
        weight = pipeline.named_steps["model"].class_weight
        assert weight == ("balanced" if strategy == "class_weight" else None)
