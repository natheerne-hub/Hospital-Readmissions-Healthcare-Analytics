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


def test_hrrp_guardrails_are_explicit():
    summary = load_json("data/hrrp_summary.json")
    guardrails = " ".join(summary["guardrails"]).lower()

    assert "hospital-level" in guardrails
    assert "individual patient" in guardrails
    assert "not converted to zero" in guardrails


def test_patient_probability_is_publicly_locked():
    evidence = load_json("data/model_evidence.json")
    serialized = json.dumps(evidence).lower()

    # The public evidence artifact must not imply clinical deployment readiness.
    assert "research" in serialized
    assert "clinical" in serialized


def test_readme_keeps_evidence_layers_separate():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "hospital-level" in readme
    assert "patient-level" in readme
    assert "never merged" in readme or "never combines" in readme
    assert "not a medical device" in readme
