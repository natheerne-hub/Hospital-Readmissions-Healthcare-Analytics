# Readmission Intelligence

Healthcare analytics and research modeling for understanding hospital readmission patterns.

[**Open the live web app**](https://hospital-readmissions-healthcare-an.vercel.app) · [**View the patient-risk simulator**](https://hospital-readmissions-healthcare-an.vercel.app/simulator.html)

> Research and portfolio project only. It is not a diagnostic tool and must not be used as the sole basis for patient-care decisions.

## What this project does

Readmission Intelligence brings two complementary evidence layers into one web experience while keeping their data and conclusions strictly separate:

| Evidence layer | Dataset and unit | Purpose |
|---|---|---|
| Hospital analytics | CMS HRRP FY 2026; one hospital-condition record | Explore risk-adjusted performance signals, reporting completeness, geographic variation, and multi-condition patterns |
| Patient-model research | UCI Diabetes 130-US Hospitals; one inpatient encounter | Evaluate a research model for early readmission risk and demonstrate a guarded inference workflow |

Hospital-level HRRP signals are never presented as patient-level risk. The UCI research model is not claimed to be clinically validated or transferable to a new health system.

## Live MVP

The responsive web app includes:

- A Python-generated hospital analytics dashboard with traceable JSON output
- An interactive US state map colored by mean Excess Readmission Ratio
- Executive hospital-level KPIs and condition comparisons
- State and hospital exploration
- Transparent data-quality and interpretation notes
- Patient-model evidence, metrics, and model status
- A guarded research simulator backed by a Python prediction API
- Explicit separation of descriptive analytics from predictive modeling

## Hospital analytics — CMS HRRP

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

\*Among hospitals reporting ERR for at least five conditions. These are investigation signals, not universal rankings of quality.

The six conditions are acute myocardial infarction, heart failure, COPD, pneumonia, CABG, and hip/knee replacement. The reporting period is July 2021 through June 2024.

## Patient-model research — UCI Diabetes

The selected candidate is a `HistGradientBoostingClassifier` trained on the full approved feature set. The target is early readmission (`<30 days`) in the UCI Diabetes 130-US Hospitals dataset.

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

At the selected research threshold, the hold-out confusion matrix was TP 1,109, FP 4,762, TN 13,358, and FN 1,060. The false-discovery rate was 81.1%, so probability display remains clinically locked pending external validation, calibration review, subgroup assessment, workflow evaluation, and governance approval.

See [the model card](docs/MODEL_CARD.md) and [final model decision](docs/FINAL_MODEL_DECISION.md) for the full interpretation.

The latest [model improvement audit](docs/MODEL_IMPROVEMENT_AUDIT.md) compares engineered HistGradientBoosting and XGBoost challengers using matched cohorts and paired bootstrap uncertainty. Neither challenger replaces the registered model because improvement was not robust across all required metrics.

## Architecture

```text
CMS HRRP data ──> descriptive hospital analytics ──> web dashboard

UCI encounters ──> reproducible modeling pipeline ──> versioned artifact
                                                        │
                                                        └──> guarded research API ──> simulator
```

The browser never combines the two datasets into a single inference. Each layer preserves its own unit of analysis, provenance, limitations, and permitted claims.

## Repository guide

```text
├── index.html, app.js, styles.css     # Web dashboard
├── simulator.html, simulator.js       # Research simulator
├── api/predict.py                      # Guarded prediction endpoint
├── analysis/                           # CMS HRRP analysis workflow
├── notebooks/                          # Recruiter- and Kaggle-ready notebooks
├── dashboard.js                        # Python-dashboard browser renderer
├── modeling/                           # UCI training and evaluation pipeline
├── runtime/model/                      # Versioned research artifact and manifest
├── data/                               # Data documentation and derived web assets
├── docs/                               # Model card, decisions, and architecture
├── dashboard/                          # Dashboard implementation notes
└── tests/                              # API and integrity checks
```

### Portfolio notebook

- [CMS HRRP Hospital Readmissions — Kaggle-ready notebook](notebooks/hrrp_hospital_readmissions_kaggle.ipynb)

The notebook keeps the hospital-level HRRP analysis self-contained and does not mix it with the separate patient-level research model.

## Run locally

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. To run the prediction API locally, install `requirements.txt` and start the Python API using a Vercel-compatible local workflow.

## Data provenance and permitted reuse

- **Hospital analytics:** [CMS Hospital Readmissions Reduction Program FY 2026](https://data.cms.gov/provider-data/dataset/9n3s-kdb3). The repository stores derived summaries, while the analysis workflow expects the official source CSV.
- **Patient-model research:** [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J), licensed **CC BY 4.0**.
- The UCI data are de-identified public research data but include demographic variables that require careful bias, fairness, and subgroup evaluation.
- Raw source files are not presented as original work. Any Kaggle notebook should cite the official source, preserve attribution, and avoid implying that public availability equals clinical suitability.

## Data and interpretation safeguards

- Suppressed values such as `Too Few to Report` are preserved as missing/not reported, never converted to zero.
- ERR is treated as a risk-adjusted performance signal, not a complete measure of hospital quality.
- Descriptive associations do not establish causation.
- Patient-model explanations describe model behavior, not clinical causes.
- The patient model is restricted to research/demo use and has no local-health-system validation.

## Rebuild the Python dashboard

The public dashboard separates computation from presentation. Python derives the published metrics, and JavaScript only renders the versioned JSON output.

```bash
python analysis/build_python_dashboard.py
```

This regenerates `data/python_dashboard.json` and its MVP mirror from the verified HRRP summary.

The state layer is generated from `data/state_readmission_summary.csv`, an aggregation of the official [CMS Hospital Readmissions Reduction Program FY 2026 dataset](https://data.cms.gov/provider-data/dataset/9n3s-kdb3).

## Author

**Dr. Natheer Soliman, MD**  
Medical Doctor · Clinical & Healthcare Data Analytics · Medical AI

[Portfolio](https://natheerne-hub.github.io/natheersoliman.github.io/) · [GitHub](https://github.com/natheerne-hub)
