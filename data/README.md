# Data

This project analyzes the FY 2026 Hospital Readmissions Reduction Program (HRRP) hospital dataset used in the notebook workflow.

## Analysis-ready export

The analysis generates:

`Hospital_Readmissions_PowerBI.csv`

Expected shape from the current workflow:

- **18,330 rows**
- **11 columns**
- **3,055 unique hospitals**
- **51 states/territories**
- **6 clinical conditions**

The export intentionally preserves missing values where the source does not report a measure. Suppressed values such as `Too Few to Report` are not treated as zero.

## Power BI fields

- `Facility_ID`
- `Facility_Name`
- `State`
- `Condition`
- `Number_of_Discharges`
- `Number_of_Readmissions`
- `Excess_Readmission_Ratio`
- `Predicted_Readmission_Rate`
- `Expected_Readmission_Rate`
- `Start_Date`
- `End_Date`

The cleaned CSV can be reproduced from the analysis workflow before importing it into Power BI Desktop.
