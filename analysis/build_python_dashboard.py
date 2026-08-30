"""Build the public Python dashboard asset from the verified HRRP summary.

This keeps the website presentation separate from the analysis logic: Python
derives the dashboard values, while the browser only renders the resulting JSON.
Run from the repository root:

    python analysis/build_python_dashboard.py
"""
from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("data/hrrp_summary.json")
STATE_SOURCE = Path("data/state_readmission_summary.csv")
OUTPUTS = [
    Path("data/python_dashboard.json"),
    Path("mvp/data/python_dashboard.json"),
]


def load_states() -> list[dict]:
    with STATE_SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "state": row["State"],
            "mean_err": float(row["Avg_Ratio"]),
            "median_err": float(row["Median_Ratio"]),
            "min_err": float(row["Min_Ratio"]),
            "max_err": float(row["Max_Ratio"]),
            "hospitals": int(row["Hospital_Count"]),
            "valid_records": int(row["Record_Count"]),
        }
        for row in rows
    ]


def build_dashboard(summary: dict) -> dict:
    kpis = summary["kpis"]
    records = int(kpis["records"])
    valid_err = int(kpis["valid_err_records"])
    reportable_pct = round(valid_err / records * 100, 1)
    not_reportable_pct = round(100 - reportable_pct, 1)
    above_pct = float(kpis["err_above_1_pct"])

    states = load_states()
    state_ranked = sorted(states, key=lambda row: row["mean_err"], reverse=True)

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/build_python_dashboard.py",
        "source": summary["source"],
        "headline_kpis": [
            {"label": "Hospitals", "value": int(kpis["hospitals"]), "display": f'{int(kpis["hospitals"]):,}'},
            {"label": "Valid ERR records", "value": valid_err, "display": f"{valid_err:,}"},
            {"label": "Mean ERR", "value": float(kpis["mean_err"]), "display": f'{float(kpis["mean_err"]):.3f}'},
            {"label": "ERR above 1", "value": above_pct, "display": f"{above_pct:.1f}%"},
        ],
        "reporting_coverage": {
            "reportable_pct": reportable_pct,
            "not_reportable_pct": not_reportable_pct,
            "valid_records": valid_err,
            "total_records": records,
        },
        "err_distribution": {
            "above_1_pct": above_pct,
            "at_or_below_1_pct": round(100 - above_pct, 1),
        },
        "persistent_signals": {
            "high": int(kpis["persistent_high_err_hospitals"]),
            "low": int(kpis["persistent_low_err_hospitals"]),
            "minimum_conditions": int(kpis["persistent_signal_min_conditions"]),
        },
        "state_map": {
            "metric": "Mean Excess Readmission Ratio (ERR)",
            "national_mean_err": float(kpis["mean_err"]),
            "states": states,
            "highest": state_ranked[:5],
            "lowest": list(reversed(state_ranked[-5:])),
        },
        "condition_insights": summary.get("condition_findings", []),
        "interpretation": {
            "primary": summary["definitions"]["err"],
            "guardrails": summary["guardrails"],
        },
    }


def main() -> None:
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload = build_dashboard(summary)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
