# Dashboard

The browser-based Readmission Intelligence MVP is implemented and available at:

https://hospital-readmissions-healthcare-an.vercel.app

It currently provides hospital-level CMS HRRP KPIs, condition comparisons, state and hospital exploration, data-quality context, patient-model evidence, and access to the guarded research simulator.

## Current dashboard views

- Executive overview
- Clinical-condition comparison
- State-level signals
- Hospital-level exploration
- Data quality and interpretation notes
- Patient-model evidence and research status

## Power BI extension

A Power BI version remains a planned portfolio extension. The analysis workflow already prepares a clean import-ready CSV and supports these visuals:

- KPI cards for hospitals, valid ERR records, mean/median ERR, and ERR above 1
- Predicted versus expected readmission rate by condition
- State-level mean ERR with minimum-record filters
- Persistent high/low multi-condition hospital signals
- Hospital detail table with state, condition, and facility filters

ERR must remain labeled as a descriptive HRRP performance signal, not a universal hospital ranking.
