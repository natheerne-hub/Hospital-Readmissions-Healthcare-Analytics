# Model Card — Readmission Intelligence MVP

## Status

**Hospital-level analytics:** implemented foundation  
**Patient-level prediction model:** not yet trained  
**Clinical deployment:** not validated / not claimed

## Intended use

The current MVP supports exploration of hospital-level CMS HRRP readmission performance signals, data quality, reporting completeness, and multi-condition patterns.

## Current data unit

One row represents a hospital-condition HRRP record. It is not a patient encounter.

## Current outcome/performance measures

The product uses source HRRP measures including Excess Readmission Ratio, predicted readmission rate, expected readmission rate, discharges, and reportable readmission counts where available.

## Patient-level target outcome — future phase

If a patient-level model is developed, the target must be explicitly defined before training, for example: all-cause unplanned readmission within 30 days after eligible index discharge. Inclusion/exclusion rules and the exact source definition must be documented with the dataset.

## Features — future phase

No patient-level features are approved yet. Candidate features may only be used if they exist legitimately in the selected dataset and are available at the intended prediction time. Features that leak post-outcome information must be excluded.

## Validation requirements

Before any patient-level risk percentage is shown as a model output, the project must include:

1. Reproducible train/validation/test methodology.
2. Class-balance report.
3. Baseline comparator.
4. ROC-AUC and PR-AUC.
5. Sensitivity, specificity, precision and F1 at the selected threshold.
6. Confusion matrix.
7. Calibration curve/assessment and Brier score.
8. Confidence intervals where feasible.
9. Subgroup performance checks where variables and sample sizes permit.
10. External validation before claiming transportability to a new health system.

## Threshold policy

A default probability threshold of 0.50 is not automatically accepted. The operating threshold must be chosen according to the intended intervention capacity and the relative cost of false negatives versus false positives.

## Explainability policy

Any feature-attribution method must be described as model explanation/association, not proof of clinical causation. Explanations must use features available to the model at inference time.

## Known limitations

- Current data are aggregate hospital-level HRRP data.
- Current analytics are descriptive and signal-oriented.
- Missing/suppressed reporting reduces analyzable coverage for some measures.
- ERR does not measure every dimension of hospital quality.
- No patient-level model performance exists yet, so no patient-level accuracy/AUC claim is allowed.
- No Dubai-specific clinical validity is claimed until external validation is performed on authorized local data.

## Safety statement

This MVP is for analytics, research, product demonstration, and pilot evaluation. It is not a diagnostic tool and must not be used as the sole basis for patient care decisions.
