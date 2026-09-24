# Data Intake Review Template

Use this review before any new dataset is allowed into Suliaman HealthData AI Readmission Intelligence.

## 1. Identity and provenance
- Dataset name:
- Source organization:
- Source URL/reference:
- Access method:
- License / data-use agreement:
- Public, restricted, or sandbox-only:
- Version / extraction date:

## 2. Unit of analysis
- Patient, encounter, admission, hospital, hospital-condition, claim, or other:
- One row represents:
- Can multiple rows belong to the same patient/entity?:
- Stable grouping identifier available?:

## 3. Population and setting
- Country / health system:
- Care setting:
- Inclusion criteria:
- Exclusion criteria:
- Date range:
- Sample size:

## 4. Outcome
- Exact outcome definition:
- Prediction horizon:
- Index time:
- Outcome ascertainment method:
- Positive-class prevalence:
- Is the outcome compatible with an existing model target?:

## 5. Features
- Feature list / data dictionary available?:
- Features available at prediction time:
- Post-outcome or leakage-prone features:
- Coding systems used:
- Missingness summary:

## 6. Data quality
- Duplicate policy:
- Suppression policy:
- Invalid-value policy:
- Missing-value semantics:
- Date consistency:
- Outlier review:

## 7. Privacy and governance
- Contains direct identifiers?:
- De-identification status:
- Permitted processing environment:
- Export restrictions:
- Retention requirements:
- Human-subject / ethics requirements if applicable:

## 8. Intended role in the platform
Choose one:
- Hospital intelligence
- Patient model development
- Internal validation
- External validation
- Calibration
- Benchmarking
- Data-quality research
- Other

## 9. Harmonization decision
- Proposed to combine with another dataset?:
- If yes, why is the combination scientifically valid?:
- Common outcome definition confirmed?:
- Common prediction time confirmed?:
- Feature semantics mapped?:
- Population shift assessed?:
- Duplicate/overlap risk assessed?:

**Rule:** increasing row count alone is never sufficient justification to merge datasets.

## 10. Decision
- Approved role:
- Rejected / deferred reasons:
- Required transformations:
- Required validation:
- Responsible reviewer:
- Review date:
