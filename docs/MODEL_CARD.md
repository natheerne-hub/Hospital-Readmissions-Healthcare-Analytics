# Model Card — Readmission Intelligence Research Model

## Status

| Item | Status |
|---|---|
| Model ID | `uci-diabetes-readmission-hgb-v2` |
| Architecture | `HistGradientBoostingClassifier` |
| Patient-level candidate | Trained and evaluated on a held-out UCI test set |
| Product mode | Research/demo only |
| Probability display | Clinically locked |
| External validation | Not completed |
| Clinical deployment | Not approved or claimed |

## Model purpose

The model estimates the probability of early readmission (`<30 days`) for an inpatient encounter represented by the approved fields in the UCI Diabetes 130-US Hospitals dataset. It supports reproducible research, portfolio demonstration, and evaluation of a guarded inference workflow.

It must not be used for diagnosis, treatment selection, discharge decisions, denial of care, or autonomous patient prioritization.

## Data scope

- **Source:** UCI Diabetes 130-US Hospitals, 1999–2008.
- **Unit of analysis:** one inpatient encounter.
- **Target:** early readmission recorded as `<30 days`.
- **Transportability:** not established for another hospital, country, population, or current clinical workflow.

This model is separate from the CMS HRRP hospital-level analytics layer. CMS HRRP rows represent hospital-condition aggregates and are not model inputs.

## Development and evaluation

Median imputation and all learned preprocessing are fitted on training data only. The selected candidate is a histogram-based gradient boosting model using the full approved feature set. Selection and reporting use a held-out test set documented in the reproducible evidence artifacts.

| Hold-out metric | Value |
|---|---:|
| ROC-AUC | 0.6815 |
| PR-AUC | 0.2286 |
| Brier score | 0.0905 |
| Research threshold | 0.13 |
| Sensitivity | 0.5113 |
| Specificity | 0.7372 |
| Precision | 0.1889 |
| F1 | 0.2759 |
| False-positive rate | 0.2628 |
| False-discovery rate | 0.8111 |

Confusion matrix at threshold 0.13: TP 1,109; FP 4,762; TN 13,358; FN 1,060.

## Threshold policy

The 0.13 threshold is a research operating point, not a clinical recommendation. A production threshold would need to reflect intervention capacity, harm from false negatives, burden from false positives, calibration in the target population, and formal governance approval.

## Explainability policy

Feature contributions or importance values describe the fitted model. They do not establish clinical causation, and they must only use variables available at the intended prediction time.

## Known limitations

- The source data are historical and specific to US hospitals treating patients with diabetes.
- External validity and current-population transportability have not been demonstrated.
- ROC-AUC is moderate, and precision is low at the selected threshold.
- The 81.1% false-discovery rate could create substantial review burden.
- Calibration has not been approved for clinical probability communication.
- Performance may differ across demographic, operational, and clinical subgroups.
- Dataset coding, missingness, and retrospective labels may introduce bias.
- No prospective workflow, impact, safety, or local health-system study has been completed.

## Conditions for unlocking clinical probability

Probability display must remain locked until the project has:

1. External validation on authorized target-population data.
2. Calibration assessment and, if needed, recalibration.
3. Subgroup performance and fairness review.
4. Prospective workflow and clinical-utility evaluation.
5. Monitoring, rollback, and model-change controls.
6. Privacy, security, clinical, and governance approval.

## Safety statement

This model and its simulator are for research, education, and product demonstration only. They are not medical devices, do not provide medical advice, and must not be the sole basis for patient-care decisions.
