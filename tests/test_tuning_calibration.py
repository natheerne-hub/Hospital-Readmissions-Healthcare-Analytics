"""Checks for patient-history features and calibration helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))

from tune_and_calibrate_model import Calibrator, add_patient_history, calibration_report  # noqa: E402


def test_history_counts_only_strictly_earlier_encounters():
    df = pd.DataFrame({
        "encounter_id": [30, 10, 20, 5, 40],
        "patient_nbr": [1, 1, 1, 2, 2],
        "readmitted": ["NO", "<30", "<30", ">30", "<30"],
    })
    out = add_patient_history(df).set_index("encounter_id")

    assert out.loc[[10, 20, 30], "prior_dataset_encounters"].tolist() == [0, 1, 2]
    assert out.loc[[10, 20, 30], "prior_dataset_early_readmissions"].tolist() == [0, 1, 2]
    # The last encounter's own outcome never feeds its own features.
    assert out.loc[40, "prior_dataset_early_readmissions"] == 0
    assert out.loc[40, "prior_dataset_encounters"] == 1


def test_calibrators_stay_in_unit_interval_and_fix_scale():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.02, 0.4, 5000)
    y = (rng.random(5000) < truth).astype(int)
    inflated = np.clip(truth * 2, 0, 0.99)

    for method in ["sigmoid", "isotonic"]:
        calibrated = Calibrator(method).fit(inflated, y).predict(inflated)
        assert calibrated.min() >= 0 and calibrated.max() <= 1
        assert abs(calibrated.mean() - y.mean()) < 0.01
    assert np.array_equal(Calibrator("uncalibrated").fit(inflated, y).predict(inflated), inflated)


def test_calibration_report_flags_miscalibration():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.02, 0.4, 5000)
    y = (rng.random(5000) < truth).astype(int)

    good = calibration_report(y, truth)
    bad = calibration_report(y, np.clip(truth * 2, 0, 0.99))
    assert good["expected_calibration_error"] < bad["expected_calibration_error"]
    assert 0.8 < good["calibration_slope"] < 1.2
    assert sum(row["encounters"] for row in good["reliability_deciles"]) == 5000
