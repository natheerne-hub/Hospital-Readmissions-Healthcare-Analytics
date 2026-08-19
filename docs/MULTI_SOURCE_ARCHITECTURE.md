# Multi-Source Architecture

## Product rule

Suliaman HealthData AI Readmission Intelligence is one product with multiple evidence layers. Datasets remain separate unless a documented harmonization study proves that combining them is valid.

## Layers

### 1. Hospital Intelligence
Source: CMS HRRP FY 2026.
Purpose: hospital-condition benchmarking, ERR interpretation, reporting completeness, persistent multi-condition signals, and data-quality analytics.
It cannot generate an individual patient's readmission probability.

### 2. Patient Risk AI
Initial research source: UCI Diabetes 130-US Hospitals 1999-2008.
Purpose: patient-level 30-day readmission probability research.
A public probability is allowed only after reproducible training, validation-based threshold selection, untouched test evaluation, calibration review, and publication of the model artifact.

### 3. Dubai Validation Layer
Future source: authorized Dubai/NABIDH sandbox data.
Purpose: semantic mapping, external validation, local calibration, subgroup checks, and operational pilot evaluation.
No Dubai-specific validity is claimed before this phase is completed.

## Request routing

Every risk request passes through a router before a model can run.

1. Validate the requested task.
2. Identify the unit of analysis: hospital or patient encounter.
3. Check that a registered model matches the intended population and outcome.
4. Check required features and prediction-time availability.
5. Check that the model status permits probability output.
6. If any condition fails, return an explicit unavailable/insufficient-data result instead of a fabricated percentage.
7. If all conditions pass, run the selected model and attach model ID, version, dataset, limitations, and interpretation metadata to the response.

## Data separation

CMS HRRP records and UCI patient encounters are not concatenated into one training table. Their units of analysis, populations, and purposes differ. They coexist in the product as independent evidence layers.

## Future datasets

A new dataset enters the platform only after a Data Intake Review documenting:

- provenance and license/access conditions;
- unit of analysis;
- population and setting;
- outcome definition;
- observation and prediction windows;
- available features and coding systems;
- missingness and data quality;
- leakage risks;
- overlap with existing datasets;
- intended model or analytics role;
- whether harmonization is scientifically defensible.

More rows alone are not a justification for combining datasets.

## Public response contract

A patient risk response must eventually contain:

- estimated probability;
- risk band only if thresholds are validated and documented;
- model ID and version;
- intended population;
- source dataset used for model development;
- key model-attribution factors when technically valid;
- missing required fields;
- validation status;
- limitations and safety statement.

If the model is not unlocked, the product must state that a validated patient probability is not currently available.
