# Readmission Intelligence

**Hospital Quality Analytics + Patient-Level Readmission Research**

A healthcare data portfolio project that combines **CMS hospital-performance analytics**, a reproducible **patient-level readmission research pipeline**, and an interactive web experience—while deliberately keeping hospital-level and patient-level evidence separate.

[**Live Dashboard**](https://hospital-readmissions-healthcare-an.vercel.app) · [**Research Simulator**](https://hospital-readmissions-healthcare-an.vercel.app/simulator.html) · [**Kaggle Notebook**](https://www.kaggle.com/code/nateer/cms-hrrp-hospital-readmissions-analytics) · [**Model Card**](docs/MODEL_CARD.md)

> **Portfolio and research demonstration only.** This project is not a medical device, has not undergone external clinical validation, and must not be used as the sole basis for patient-care decisions.

## Why this project matters

Hospital readmission is both a healthcare-quality problem and a difficult prediction problem. This project demonstrates how a clinically informed analyst can move from public healthcare data to reproducible analytics, communicate uncertainty, and build a guarded research workflow without overstating what the data can support.

The project addresses two different questions:

1. **Hospital level:** Where do risk-adjusted readmission signals vary across hospitals, conditions, and US states?
2. **Patient research level:** How well can routinely available encounter variables discriminate early readmission in a historical public dataset?

These questions use different datasets and different units of analysis. Their results are **never merged into a single clinical inference**.

## Executive snapshot

| Area | Key result |
|---|---|
| CMS HRRP coverage | **18,330** hospital-condition records across **3,055 hospitals** |
| Clinical scope | **6 HRRP conditions**, 51 states/territories |
| Valid ERR observations | **11,720** |
| Mean / median ERR | **1.002 / 0.997** |
| Valid observations with ERR > 1 | **48.1%** |
| Patient research model | HistGradientBoostingClassifier |
| Hold-out ROC-AUC | **0.6815** |
| Hold-out PR-AUC | **0.2286** |
| Research status | **Not clinically validated — probability display remains locked** |

## Evidence layers

| Evidence layer | Dataset & unit | Purpose |
|---|---|---|
| **Hospital analytics** | CMS HRRP FY 2026 — one hospital-condition record | Risk-adjusted performance signals, reporting completeness, geographic variation, and multi-condition patterns |
| **Patient-model research** | UCI Diabetes 130-US Hospitals — one inpatient encounter | Research evaluation of early-readmission discrimination and a guarded inference workflow |

Hospital-level HRRP signals are not patient-level risk estimates. The UCI model is not assumed to transfer to another hospital system or population.

## Live analytics product

The responsive web application demonstrates an end-to-end analytics workflow:

- Python-generated hospital analytics with traceable JSON outputs
- Interactive US state map using mean Excess Readmission Ratio (ERR)
- Executive KPIs and condition-level comparisons
- State and hospital exploration
- Explicit data-quality and interpretation notes
- Patient-model evidence and model-status reporting
- Guarded research simulator backed by a Python prediction API
- Clear separation between descriptive analytics and predictive research

## Hospital analytics — CMS HRRP FY 2026

| Metric | Result |
|---|---:|
| Hospital-condition records | 18,330 |
| Unique hospitals | 3,055 |
| States / territories | 51 |
| Clinical conditions | 6 |
| Valid ERR records | 11,720 |
| Mean ERR | 1.002 |
| Median ERR | 0.997 |
| Valid records with ERR > 1 | 48.1% |
| Persistent high-ERR hospitals* | 75 |
| Persistent low-ERR hospitals* | 98 |

\*Among hospitals reporting ERR for at least five conditions. These are **investigation signals**, not universal hospital-quality rankings.

The six conditions are acute myocardial infarction, heart failure, COPD, pneumonia, CABG, and hip/knee replacement. The reporting period is July 2021 through June 2024.

### Interpretation

ERR is treated as a **risk-adjusted performance signal**, not as a complete measure of hospital quality. Suppressed observations such as `Too Few to Report` remain missing/not reported rather than being converted to zero. Geographic and condition-level differences are descriptive and do not establish causation.

## Patient-level readmission research

The selected research candidate is a `HistGradientBoostingClassifier` trained on the approved feature set from the UCI Diabetes 130-US Hospitals dataset. The target is early readmission (`<30 days`).

| Hold-out metric | Result |
|---|---:|
| ROC-AUC | 0.6815 |
| PR-AUC | 0.2286 |
| Brier score | 0.0905 |
| Research threshold | 0.13 |
| Sensitivity | 0.5113 |
| Specificity | 0.7372 |
| Precision | 0.1889 |
| F1 score | 0.2759 |

At the selected research threshold, the hold-out confusion matrix is **TP 1,109 · FP 4,762 · TN 13,358 · FN 1,060**.

The false-discovery rate is **81.1%**. For that reason, the project does not present the model as clinically ready and keeps patient probability display locked pending external validation, calibration review, subgroup assessment, workflow evaluation, and governance approval.

See the [Model Card](docs/MODEL_CARD.md), [Final Model Decision](docs/FINAL_MODEL_DECISION.md), and [Model Improvement Audit](docs/MODEL_IMPROVEMENT_AUDIT.md) for the detailed evidence and decision trail.

## What I demonstrated

This repository is designed to show more than notebook execution. It demonstrates:

- **Healthcare data reasoning:** preserving the unit of analysis and separating hospital-quality metrics from patient risk
- **Data quality:** explicit treatment of suppressed and missing healthcare observations
- **Analytics:** hospital, condition, and geographic comparisons using CMS HRRP data
- **Machine learning:** reproducible classification, hold-out evaluation, threshold analysis, and challenger-model auditing
- **Clinical caution:** interpreting sensitivity, specificity, precision, calibration, and false-discovery risk in context
- **Health-data communication:** translating technical results into an interactive dashboard without hiding limitations
- **Software delivery:** web UI, Python analytics, prediction API, versioned model artifacts, and integrity tests

## Architecture

```text
CMS HRRP FY 2026
       │
       └──> Python descriptive analytics ──> versioned JSON ──> web dashboard

UCI Diabetes encounters
       │
       └──> preprocessing + modeling ──> evaluation ──> versioned research artifact
                                                         │
                                                         └──> guarded API ──> research simulator
```

The browser never combines the two datasets into one inference. Each layer retains its own provenance, unit of analysis, limitations, and permitted claims.

## Repository structure

```text
├── index.html, app.js, styles.css      # Interactive hospital dashboard
├── simulator.html, simulator.js        # Research simulator UI
├── api/predict.py                       # Guarded prediction endpoint
├── analysis/                            # CMS HRRP analytics workflow
├── notebooks/                           # Portfolio / Kaggle notebooks
├── dashboard.js                         # Dashboard browser renderer
├── modeling/                            # UCI modeling and evaluation pipeline
├── runtime/model/                       # Versioned research artifact + manifest
├── data/                                # Documentation and derived web assets
├── docs/                                # Model card, decisions, architecture, audits
├── dashboard/                           # Dashboard implementation notes
└── tests/                               # API and integrity checks
```

## Reproduce the hospital dashboard

```bash
python analysis/build_python_dashboard.py
python -m http.server 8000
```

Then open `http://localhost:8000`.

The build script regenerates `data/python_dashboard.json` and its MVP mirror from the verified HRRP summary. JavaScript renders the published output rather than silently recalculating the healthcare metrics in the browser.

## Data provenance

- **CMS HRRP FY 2026:** [CMS Hospital Readmissions Reduction Program dataset](https://data.cms.gov/provider-data/dataset/9n3s-kdb3)
- **UCI patient research data:** [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J), CC BY 4.0
- **Portfolio notebook:** [CMS HRRP Hospital Readmissions Analytics on Kaggle](https://www.kaggle.com/code/nateer/cms-hrrp-hospital-readmissions-analytics)

Raw public source data are not presented as original work. Dataset provenance, licensing, missingness, and unit of analysis are treated as part of the analysis itself.

## Limitations & responsible use

- This is a portfolio/research project, not a medical device.
- CMS HRRP results are hospital-level performance signals and cannot estimate an individual patient's risk.
- The UCI dataset is historical and may not represent current clinical practice or another health system.
- Predictive performance requires external validation, calibration assessment, subgroup/fairness analysis, prospective workflow evaluation, and clinical governance before real-world use.
- Model explanations describe statistical/model behavior, not clinical causation.

## Author

**Dr. Natheer Soliman, MD**  
Medical Doctor · Healthcare & Clinical Data Analytics · Health Informatics · Medical AI

[Portfolio](https://natheerne-hub.github.io/natheersoliman.github.io/) · [GitHub Profile](https://github.com/natheerne-hub) · [Kaggle](https://www.kaggle.com/nateer)

**Open to:** healthcare analytics opportunities · research collaboration · health informatics projects · remote/international work
