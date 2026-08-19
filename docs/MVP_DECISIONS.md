# MVP Decision Log — Suliaman HealthData AI Readmission Intelligence

## Purpose

This document records why the MVP is built the way it is. The goal is traceability: every visible claim, percentage, threshold, and feature should have a defensible reason and a known evidence source.

## 1. Product scope

### Decision
Build the first MVP as a **hospital-level readmission intelligence product**, not an individual patient risk predictor.

### Why
The current repository uses the CMS Hospital Readmissions Reduction Program (HRRP) hospital dataset. Its unit of analysis is a hospital-condition record, not an individual patient encounter. A patient-level probability such as “72% readmission risk for this patient” cannot be validly trained from aggregate hospital-condition rows.

### Consequence
The MVP may show hospital performance signals, condition comparisons, reporting completeness, data-quality indicators, and persistent ERR patterns. It must not claim validated patient-level prediction until a suitable patient-level dataset is introduced and validated.

## 2. Primary performance signal

### Decision
Use the **Excess Readmission Ratio (ERR)** as the primary hospital-level signal because it is native to HRRP and already supported by the reproducible analysis.

### Interpretation
- ERR < 1: fewer readmissions than model-expected for that HRRP measure.
- ERR near 1: near model-expected.
- ERR > 1: more readmissions than model-expected.

### Guardrail
ERR is not displayed as a complete hospital-quality ranking and is not treated as causal evidence.

## 3. Persistent signal rule

### Decision
For persistent high/low hospital patterns, require at least **five reported clinical conditions**.

### Why
A hospital with one or two reportable measures should not be presented as having a broad multi-condition pattern. The existing analysis already applies a >=5-condition coverage rule before labeling persistent signals.

### Current evidence
The repository analysis identifies 75 hospitals with ERR > 1 across all reported conditions and 98 hospitals with ERR < 1 across all reported conditions under the >=5-condition rule.

## 4. Missing and suppressed data

### Decision
Never convert “Too Few to Report” to zero.

### Why
Zero would mean no readmissions, while the source state means the value was suppressed/not reportable. Conflating those states would bias rates and denominator logic.

### Implementation rule
Create numeric analysis fields with safe coercion while preserving the original source value and use measure-specific valid subsets.

## 5. Demo data

### Decision
Use clearly labeled synthetic hospital examples for interactive UI behavior until a cleaned export is directly wired into the app.

### Why
This allows the interface and product logic to be demonstrated without misrepresenting a real hospital or exposing data that have not yet been integrated into the application layer.

### Guardrail
Synthetic examples must be labeled in the interface. Placeholder chart lengths must never be presented as measured values.

## 6. Patient-level AI roadmap

### Decision
Do not manufacture model-performance percentages before a patient-level model exists.

### Required evaluation set
When a patient-level model is developed, report at minimum:
- ROC-AUC
- PR-AUC
- sensitivity/recall
- specificity
- precision
- F1
- Brier score
- calibration assessment
- selected operating threshold and rationale

### Why not accuracy alone
Readmission outcomes may be imbalanced. A high accuracy can be achieved by over-predicting the majority class, so it is insufficient as the primary evidence of clinical utility.

## 7. Threshold selection

### Decision
Do not default to 0.50 or choose risk bands for visual appeal.

### Future rule
Select operating thresholds using validation data and the intended operational use case, explicitly balancing false negatives and false positives. Any Low/Moderate/High bands must be documented with the exact derivation.

## 8. Dubai validation concept

### Decision
Treat Dubai Health Data Sandbox testing as a future **external-validation and local-calibration phase**, not as proof already obtained.

### Why
Performance on one source population does not prove transportability to a different health system or patient population. A Dubai pilot should test discrimination, calibration, subgroup behavior, data mapping, and operational usefulness on authorized de-identified data.

## 9. Public safety statement

The public MVP is a healthcare analytics and decision-support demonstration. It is not a diagnostic device, is not a substitute for clinician judgment, and must not claim clinical deployment readiness before appropriate technical, clinical, regulatory, privacy, and external-validation work is complete.

## 10. Definition of “professional” for this project

Professional means:
1. Numbers are traceable.
2. Unknowns are labeled as unknowns.
3. Demo values are labeled as demo values.
4. Data limitations are visible.
5. Model metrics are not invented.
6. Every threshold has a rationale.
7. The product can explain its data flow and validation status.
8. The UI does not overstate clinical meaning.
