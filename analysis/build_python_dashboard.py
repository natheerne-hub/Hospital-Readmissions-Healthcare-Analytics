"""Build the public Python dashboard asset from the verified HRRP summary.

This keeps the website presentation separate from the analysis logic: Python
derives the dashboard values, while the browser only renders the resulting JSON.
Run from the repository root:

    python analysis/build_python_dashboard.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("data/hrrp_summary.json")
OUTPUTS = [
    Path("data/python_dashboard.json"),
    Path("mvp/data/python_dashboard.json"),
]


def build_dashboard(summary: dict) -> dict:
    kpis = summary["kpis"]
    records = int(kpis["records"])
    valid_err = int(kpis["valid_err_records"])
    reportable_pct = round(valid_err / records * 100, 1)
    not_reportable_pct = round(100 - reportable_pct, 1)
    above_pct = float(kpis["err_above_1_pct"])

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

