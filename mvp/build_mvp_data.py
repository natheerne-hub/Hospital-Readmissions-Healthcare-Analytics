"""Build traceable JSON assets for the Readmission Intelligence MVP.

This script converts the official CMS HRRP hospital-level CSV into compact JSON
used by the static MVP. Every displayed KPI is recomputed from source columns.

Run from repository root after placing the source CSV in the root or data/ folder:
    python mvp/build_mvp_data.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

INPUT_CANDIDATES = [
    Path("FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv"),
    Path("data/FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv"),
]
OUTPUT = Path("mvp/data/hrrp_summary.json")

MEASURE_MAP = {
    "READM-30-AMI-HRRP": "Acute Myocardial Infarction (AMI)",
    "READM-30-HF-HRRP": "Heart Failure (HF)",
    "READM-30-COPD-HRRP": "COPD",
    "READM-30-PN-HRRP": "Pneumonia",
    "READM-30-CABG-HRRP": "CABG",
    "READM-30-HIP-KNEE-HRRP": "Hip/Knee Replacement",
}


def locate_source() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "CMS HRRP CSV not found. Put the official FY 2026 CSV in the repository "
        "root or data/ before running this builder."
    )


def safe_float(value: float | int | None, digits: int = 4):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def build_payload(df: pd.DataFrame, source_path: Path) -> dict:
    df = df.copy()
    df["Condition"] = df["Measure Name"].map(MEASURE_MAP)
    readmissions_numeric = pd.to_numeric(df["Number of Readmissions"], errors="coerce")
    valid_err = df["Excess Readmission Ratio"].dropna()

    condition_summary = []
    for condition, group in df.groupby("Condition", dropna=True):
        err = group["Excess Readmission Ratio"].dropna()
        condition_summary.append(
            {
                "condition": condition,
                "records": int(len(group)),
                "valid_err": int(err.notna().sum()),
                "mean_err": safe_float(err.mean(), 4),
                "median_err": safe_float(err.median(), 4),
                "mean_predicted_rate": safe_float(group["Predicted Readmission Rate"].mean(), 4),
                "mean_expected_rate": safe_float(group["Expected Readmission Rate"].mean(), 4),
                "err_above_1_pct": safe_float((err > 1).mean() * 100, 1) if len(err) else None,
            }
        )
    condition_summary.sort(
        key=lambda row: row["mean_predicted_rate"] if row["mean_predicted_rate"] is not None else -1,
        reverse=True,
    )

    valid = df[df["Excess Readmission Ratio"].notna()].copy()
    hospital = (
        valid.groupby(["Facility ID", "Facility Name", "State"])
        .agg(
            conditions_reported=("Condition", "nunique"),
            mean_err=("Excess Readmission Ratio", "mean"),
            conditions_err_above_1=("Excess Readmission Ratio", lambda x: int((x > 1).sum())),
        )
        .reset_index()
    )
    robust = hospital[hospital["conditions_reported"] >= 5].copy()
    robust["high_err_pct"] = robust["conditions_err_above_1"] / robust["conditions_reported"] * 100
    persistent_high = int((robust["high_err_pct"] == 100).sum())

    strict_low = (
        valid.groupby(["Facility ID", "Facility Name", "State"])
        .agg(
            conditions_reported=("Condition", "nunique"),
            all_err_below_1=("Excess Readmission Ratio", lambda x: bool((x < 1).all())),
        )
        .reset_index()
    )
    persistent_low = int(
        ((strict_low["conditions_reported"] >= 5) & strict_low["all_err_below_1"]).sum()
    )

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "publisher": "Centers for Medicare & Medicaid Services (CMS)",
            "dataset": "Hospital Readmissions Reduction Program (HRRP), FY 2026",
            "local_source_file": source_path.name,
            "reporting_period": {
                "start": str(pd.to_datetime(df["Start Date"]).min().date()),
                "end": str(pd.to_datetime(df["End Date"]).max().date()),
            },
            "unit_of_analysis": "hospital-condition record",
        },
        "kpis": {
            "records": int(len(df)),
            "hospitals": int(df["Facility ID"].nunique()),
            "states_territories": int(df["State"].nunique()),
            "conditions": int(df["Condition"].nunique()),
            "valid_err_records": int(len(valid_err)),
            "mean_err": safe_float(valid_err.mean(), 3),
            "median_err": safe_float(valid_err.median(), 3),
            "err_above_1_pct": safe_float((valid_err > 1).mean() * 100, 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_readmission_records": int(readmissions_numeric.notna().sum()),
            "suppressed_or_non_numeric_readmission_records": int(readmissions_numeric.isna().sum()),
            "persistent_high_err_hospitals": persistent_high,
            "persistent_low_err_hospitals": persistent_low,
            "persistent_signal_min_conditions": 5,
        },
        "conditions": condition_summary,
        "definitions": {
            "err": "Predicted readmission rate divided by expected readmission rate under CMS HRRP methodology.",
            "err_above_1_pct": "Share of records with a reportable ERR strictly greater than 1.",
            "persistent_high_err": "Hospital with at least five reportable conditions and ERR > 1 for every reported condition.",
            "persistent_low_err": "Hospital with at least five reportable conditions and ERR < 1 for every reported condition.",
        },
        "guardrails": [
            "ERR is a hospital-level risk-adjusted performance signal, not a complete quality score.",
            "Missing or suppressed values are not converted to zero.",
            "This dataset cannot support individual patient readmission probabilities.",
        ],
    }


def main() -> None:
    source = locate_source()
    df = pd.read_csv(source)
    payload = build_payload(df, source)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote traceable MVP asset: {OUTPUT}")


if __name__ == "__main__":
    main()
