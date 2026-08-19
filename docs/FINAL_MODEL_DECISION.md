# Final Research Model Decision

## Decision

The current selected research candidate for patient-level 30-day readmission is **HistGradientBoostingClassifier with the full feature set**.

This selection was made on the validation set only. The held-out patient-group test set was evaluated once after model selection.

## Test performance

- ROC-AUC: **0.6815**
- PR-AUC: **0.2286**
- Brier score: **0.0905**
- Validation-selected threshold: **0.13**
- Sensitivity: **0.5113**
- Specificity: **0.7372**
- Precision: **0.1889**
- F1: **0.2759**
- False-positive rate: **0.2628**
- False-discovery rate: **0.8111**
- Alerts per 100 encounters: **28.94**
- False-positive alerts per 100 encounters: **23.47**
- Missed readmissions per 100 encounters: **5.22**
- TP: **1109**
- FP: **4762**
- TN: **13358**
- FN: **1060**

## Comparison with baseline logistic regression

Baseline test ROC-AUC was 0.6382 and PR-AUC was 0.1913. The selected model improves both discrimination metrics and substantially improves Brier score, but the operational false-positive burden remains high.

## Key evidence

Prior utilization is the most important feature group. Removing `number_inpatient`, `number_emergency`, and `number_outpatient` reduced ROC-AUC from 0.6382 to 0.5997 and PR-AUC from 0.1913 to 0.1526 in the baseline ablation study.

The strongest individual predictive driver in the baseline permutation analysis was `number_inpatient`.

False negatives often have little or no prior inpatient utilization, showing a limitation of the current data: some true future readmissions are difficult to identify from historical utilization signals alone.

## Why probability remains locked

The model is **not deployment-ready**. Public or clinical patient risk probability remains disabled because:

1. False-positive burden is still high.
2. Precision remains low due to outcome prevalence and limited discrimination.
3. External validation has not been performed.
4. Subgroup performance and fairness have not yet been approved.
5. Calibration must be assessed for the intended target population and workflow.
6. The operational intervention threshold must be defined before deployment.

## Product use now

The model may be used as a **research MVP / technical demonstration** with transparent performance reporting. It must not be presented as a clinically validated decision-support system.

## Next scientific priorities

1. Improve features for patients with no prior utilization, since these drive many false negatives.
2. Add richer comorbidity and diagnosis representations instead of raw diagnosis codes where scientifically appropriate.
3. Evaluate temporal and subgroup stability.
4. Test explicit operational thresholds based on intervention capacity, not maximum F1 alone.
5. Perform external validation and local recalibration on an authorized target population such as a Dubai sandbox dataset before any clinical claim.
