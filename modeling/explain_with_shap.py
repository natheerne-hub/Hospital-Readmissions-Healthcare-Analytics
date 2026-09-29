"""SHAP explanations for the packaged research readmission model.

Explains the exact runtime bundle (runtime/model/) on locked patient-group test
encounters, which the packaged model never saw during fitting.

- Global: mean |SHAP| per original input feature (one-hot columns summed back).
- Local: top drivers raising and lowering risk for individual encounters.

SHAP values are in the model's log-odds space and describe model behavior,
not clinical causation. Research only; not clinically validated.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))

from train_patient_readmission import load_data, make_patient_group_splits  # noqa: E402

MODEL_PATH = ROOT / "runtime/model/readmission_research_model.joblib"
MANIFEST_PATH = ROOT / "runtime/model/readmission_research_model_manifest.json"
OUT = Path("modeling/artifacts/shap_explanations.json")
SAMPLE_SIZE = 2000
EXAMPLE_ENCOUNTERS = 5


def load_bundle():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return joblib.load(MODEL_PATH), manifest


def output_to_input_map(preprocess, input_features: list[str]) -> list[str]:
    """Map every transformed column back to the original input feature."""
    mapping = []
    by_length = sorted(input_features, key=len, reverse=True)
    for name in preprocess.get_feature_names_out():
        _, _, column = name.partition("__")
        # One-hot outputs are "<feature>_<category>"; pick the longest match so
        # "glyburide-metformin_No" is not attributed to "glyburide".
        match = next(
            (f for f in by_length if column == f or column.startswith(f + "_")),
            column,
        )
        mapping.append(match)
    return mapping


class ReadmissionExplainer:
    def __init__(self, model=None, manifest=None):
        if model is None:
            model, manifest = load_bundle()
        self.manifest = manifest
        self.features = manifest["input_features"]
        self.preprocess = model.named_steps["prep"]
        self.explainer = shap.TreeExplainer(model.named_steps["model"])
        self.mapping = output_to_input_map(self.preprocess, self.features)
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    def shap_by_input(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.preprocess.transform(X.reindex(columns=self.features))
        values = np.asarray(self.explainer.shap_values(transformed))
        if values.ndim == 3:  # some shap versions return one slice per class
            values = values[..., -1]
        frame = pd.DataFrame(values, columns=self.mapping, index=X.index)
        return frame.T.groupby(level=0, sort=False).sum().T.reindex(columns=self.features)

    def explain_encounter(self, row: pd.Series, top_n: int = 5) -> dict:
        contributions = self.shap_by_input(row.to_frame().T).iloc[0]
        log_odds = self.base_value + float(contributions.sum())

        def describe(items):
            return [
                {
                    "feature": feature,
                    "value": None if pd.isna(row[feature]) else str(row[feature]),
                    "shap_log_odds": round(float(value), 4),
                }
                for feature, value in items.items()
            ]

        ordered = contributions.sort_values()
        return {
            "model_probability": round(float(1 / (1 + np.exp(-log_odds))), 4),
            "base_log_odds": round(self.base_value, 4),
            "top_risk_increasing": describe(ordered[ordered > 0][::-1].head(top_n)),
            "top_risk_decreasing": describe(ordered[ordered < 0].head(top_n)),
        }


def main():
    explainer = ReadmissionExplainer()
    _, _, test = make_patient_group_splits(load_data())
    X_test = test.reindex(columns=explainer.features)
    sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)

    contributions = explainer.shap_by_input(sample)
    importance = contributions.abs().mean().sort_values(ascending=False)
    share_positive = (contributions > 0).mean()
    global_rows = [
        {
            "feature": feature,
            "mean_abs_shap_log_odds": round(float(value), 4),
            "share_of_encounters_pushing_risk_up": round(float(share_positive[feature]), 4),
        }
        for feature, value in importance.items()
    ]

    examples = [
        {"row_index": int(index), **explainer.explain_encounter(sample.loc[index])}
        for index in sample.index[:EXAMPLE_ENCOUNTERS]
    ]

    artifact = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": explainer.manifest["model_id"],
        "method": "shap.TreeExplainer on the packaged HistGradientBoosting model",
        "explained_encounters": "random sample of locked patient-group test encounters",
        "sample_size": int(len(sample)),
        "value_scale": "log-odds; one-hot contributions summed per original feature",
        "base_log_odds": round(explainer.base_value, 4),
        "global_importance": global_rows,
        "example_encounters": examples,
        "interpretation_warning": (
            "SHAP describes how the model uses inputs, not clinical causes. "
            "Encounter-level explanations are research output and must not be "
            "shown to clinicians while patient probability output is locked."
        ),
        "clinical_status": "research only; not clinically validated",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"top_10": global_rows[:10], "example": examples[0]}, indent=2))


if __name__ == "__main__":
    main()
