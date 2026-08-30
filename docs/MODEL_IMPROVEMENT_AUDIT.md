# Model Improvement Audit — August 2026

## Objective

Reduce false-positive burden and test whether feature engineering or a stronger boosting model provides a reliable improvement over the registered HistGradientBoosting research model.

## Corrections applied

- Kept patient-group separation so no patient crosses development and test subsets.
- Defined an eligible-discharge analysis excluding expired or hospice dispositions where ordinary readmission is structurally inappropriate.
- Compared every candidate against the registered baseline on the same eligible test encounters.
- Added clinically interpretable feature engineering for prior utilization, care intensity, medication activity, age, and ICD-9 diagnosis groups.
- Added a capacity-oriented operating point requiring at least 40% sensitivity.
- Added paired bootstrap uncertainty rather than promoting a model from point estimates alone.

## Matched test results

| Model | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Registered baseline | 0.6770 | 0.2280 | 0.0922 |
| Engineered HistGradientBoosting | 0.6801 | 0.2294 | 0.0921 |
| XGBoost challenger | 0.6822 | 0.2310 | 0.0920 |

At its validation-selected threshold of 0.15, XGBoost produced sensitivity 0.4354, specificity 0.8087, precision 0.2177, F1 0.2903, and 17.04 false-positive alerts per 100 encounters. This is a research tradeoff, not a clinical recommendation.

## Paired bootstrap result

One thousand paired bootstrap resamples compared XGBoost with the registered baseline on the same test encounters.

| Delta: challenger − baseline | Point estimate | 95% CI |
|---|---:|---:|
| ROC-AUC | +0.00516 | +0.00177 to +0.00936 |
| PR-AUC | +0.00302 | -0.00319 to +0.00927 |
| Brier score | -0.00023 | -0.00048 to +0.00002 |

ROC-AUC improved consistently, but the PR-AUC interval crossed zero and the Brier interval narrowly crossed zero.

## Decision

**Do not promote the challenger.**

The registered HistGradientBoosting model remains the public research artifact. XGBoost is retained as a research challenger because the evidence does not yet show a robust improvement across discrimination and calibration.

## Next evidence needed

1. Repeated or nested patient-group cross-validation.
2. Temporal or hospital-level external validation.
3. Confidence intervals for operational threshold metrics.
4. Subgroup performance and calibration.
5. A prespecified alert-capacity and intervention-benefit policy.
6. New clinical predictors for patients with little prior utilization.

No result in this audit establishes clinical validity.

