"""Lightweight integrity tests for public portfolio claims and safety guardrails."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_hrrp_snapshot_core_claims():
    summary = load_json("data/hrrp_summary.json")
    kpis = summary["kpis"]

    assert summary["source"]["unit_of_analysis"] == "hospital-condition record"
    assert kpis["records"] == 18_330
    assert kpis["hospitals"] == 3_055
    assert kpis["conditions"] == 6
    assert kpis["valid_err_records"] == 11_720
    assert kpis["duplicate_rows"] == 0
    assert kpis["persistent_signal_min_conditions"] >= 5
    assert 0 <= kpis["err_above_1_pct"] <= 100


def test_dashboard_matches_core_hrrp_claims():
    summary = load_json("data/hrrp_summary.json")
    dashboard = load_json("data/python_dashboard.json")
    kpis = summary["kpis"]
    headline = {item["label"]: item["value"] for item in dashboard["headline_kpis"]}

    assert headline["Hospitals"] == kpis["hospitals"]
    assert headline["Valid ERR records"] == kpis["valid_err_records"]
    assert headline["Mean ERR"] == kpis["mean_err"]
    assert headline["ERR above 1"] == kpis["err_above_1_pct"]
    assert dashboard["persistent_signals"]["high"] == kpis["persistent_high_err_hospitals"]
    assert dashboard["persistent_signals"]["low"] == kpis["persistent_low_err_hospitals"]


def test_hrrp_guardrails_are_explicit():
    summary = load_json("data/hrrp_summary.json")
    guardrails = " ".join(summary["guardrails"]).lower()

    assert "hospital-level" in guardrails
    assert "individual patient" in guardrails
    assert "not converted to zero" in guardrails


def test_patient_probability_is_publicly_locked():
    evidence = load_json("data/model_evidence.json")
    registry = load_json("modeling/model_registry.json")
    patient_model = next(
        model for model in registry["models"]
        if model["task"] == "patient_30_day_readmission_probability"
    )

    assert evidence["public_probability_allowed"] is False
    assert patient_model["patient_probability_allowed"] is False
    assert "locked" in evidence["status"]
    assert "not clinically validated" in evidence["clinical_status"].lower()


def test_public_api_does_not_expose_locked_probability():
    api_source = (ROOT / "api/predict.py").read_text(encoding="utf-8")
    simulator_source = (ROOT / "simulator.js").read_text(encoding="utf-8")

    assert "prediction_locked" in api_source
    assert "public_patient_probability_allowed" in api_source
    assert "research_probability':" not in api_source
    assert "data.research_probability" not in simulator_source
    assert "Patient-level probability output locked" in simulator_source


def test_readme_keeps_evidence_layers_separate():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "hospital-level" in readme
    assert "patient-level" in readme
    assert "never merged" in readme or "never combines" in readme
    assert "not a medical device" in readme
