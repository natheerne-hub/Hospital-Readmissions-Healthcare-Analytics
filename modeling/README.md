# Patient-Level 30-Day Readmission Modeling

## Purpose

This module is the research-model layer for the Dr. Nather Yunis Suliaman Readmission Intelligence MVP. It is deliberately separate from the CMS HRRP hospital-level analytics because the two datasets answer different questions.

- **CMS HRRP:** hospital-condition performance signals.
- **UCI Diabetes 130-US Hospitals:** patient-encounter readmission prediction research.

The current patient model must not be described as clinically validated or Dubai-ready.

## Outcome

Binary target:

- Positive: `readmitted == "<30"`
- Negative: all other recorded readmission outcomes

This produces a probability of early readmission for research evaluation only.

## Leakage control

`encounter_id` and `patient_nbr` are excluded from predictors. `patient_nbr` is used to split the data into train, validation, and test patient groups so the same patient cannot appear in more than one split.

This is important because encounter-level random splitting could let information from repeat patients leak across evaluation sets and inflate apparent performance.

## Split policy

Approximate patient-level proportions:

- Train: 64%
- Validation: 16%
- Test: 20%

The validation set is used for threshold selection. The test set remains untouched until final evaluation.

## Baseline model

The first model is a class-weighted logistic regression with:

- median imputation for numeric features;
- most-frequent imputation for categorical features;
- standardization of numeric variables;
- one-hot encoding of categorical variables;
- class weighting to reduce bias toward the majority class.

A transparent baseline is intentionally used before moving to tree ensembles or more complex models. If a more complex model does not deliver meaningful validated improvement, it should not replace the baseline merely because it is more sophisticated.

## Metrics

The modeling artifact reports:

- ROC-AUC
- PR-AUC
- Brier score
- Sensitivity / Recall
- Specificity
- Precision
- F1
- Confusion matrix counts
- Outcome prevalence
- Selected decision threshold

Accuracy is not used as the primary success metric because readmission outcomes can be imbalanced.

## Threshold policy

For the research MVP, the threshold is selected on validation data by maximizing F1. This is a reproducible baseline rule, not the final clinical policy.

For a real hospital or Dubai pilot, threshold selection should instead reflect an explicit operational objective such as:

- available case-management capacity;
- cost of missed high-risk patients;
- acceptable false-positive workload;
- required minimum sensitivity;
- expected intervention benefit.

The chosen threshold must then be frozen before final test or external validation reporting.

## External validation

A model developed on historical US diabetes encounters cannot be assumed to generalize to Dubai. A Dubai Health Data Sandbox pilot should therefore evaluate:

1. feature availability and semantic mapping;
2. outcome definition compatibility;
3. missingness and coding differences;
4. discrimination;
5. calibration;
6. subgroup performance;
7. threshold performance under local capacity constraints;
8. model recalibration or retraining if justified.

## Clinical status

**Research MVP only. Not for diagnosis, treatment decisions, or unsupervised clinical deployment.**

## Improvement audit

The August 2026 repair cycle added matched-cohort comparison, eligible-discharge filtering, clinical feature engineering, an XGBoost challenger, operational threshold reporting, and paired bootstrap uncertainty.

The challenger was **not promoted** because improvement was not robust across PR-AUC and Brier score confidence intervals. See [`docs/MODEL_IMPROVEMENT_AUDIT.md`](../docs/MODEL_IMPROVEMENT_AUDIT.md).

```bash
pip install -r modeling/requirements-research.txt
python modeling/improve_readmission_model.py
python modeling/repair_model_selection.py
python modeling/benchmark_xgboost_challenger.py
python modeling/bootstrap_challenger_comparison.py
```

## Class-imbalance strategies

`compare_imbalance_strategies.py` benchmarks four ways of handling the ~11% early-readmission prevalence for both logistic regression and HistGradientBoosting: no adjustment, `class_weight="balanced"`, SMOTE, and random undersampling. Resampling runs inside the pipeline on training data only, selection uses validation PR-AUC, and the locked test set is scored once for the selected candidate.

Reweighting and resampling push predicted probabilities above the observed prevalence, so the Brier score and mean predicted probability are reported for every candidate. Any such model must be recalibrated before its output is read as risk.

## SHAP explanations

`explain_with_shap.py` explains the exact packaged runtime model (`runtime/model/`) on locked test encounters with `shap.TreeExplainer`. It writes global importance (mean |SHAP| per original feature, one-hot columns summed back) and, for sample encounters, the top factors raising and lowering the model's output. `ReadmissionExplainer.explain_encounter` gives the same breakdown for a single encounter.

SHAP describes how the model uses its inputs, not clinical causes, and encounter-level explanations stay research output while patient probability display is locked.

```bash
pip install shap==0.51.0 imbalanced-learn==0.14.2
python modeling/compare_imbalance_strategies.py
python modeling/explain_with_shap.py
```
