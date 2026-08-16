"""Hospital Readmissions Healthcare Analytics

Reproducible Python workflow for the FY 2026 HRRP hospital dataset.

Expected input filename:
    FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv

The script preserves the original source fields, creates analysis-ready variables,
prints the core portfolio KPIs, and exports a Power BI-ready CSV.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

INPUT_CANDIDATES = [
    Path("FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv"),
    Path("data/FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv"),
    Path("/content/FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv"),
]
OUTPUT_PATH = Path("Hospital_Readmissions_PowerBI.csv")

MEASURE_MAP = {
    "READM-30-AMI-HRRP": "Acute Myocardial Infarction (AMI)",
    "READM-30-HF-HRRP": "Heart Failure (HF)",
    "READM-30-COPD-HRRP": "COPD",
    "READM-30-PN-HRRP": "Pneumonia",
    "READM-30-CABG-HRRP": "CABG",
    "READM-30-HIP-KNEE-HRRP": "Hip/Knee Replacement",
}


def locate_input() -> Path:
    path = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "HRRP CSV not found. Place the source CSV in the repository root, "
            "data/ folder, or /content when using Google Colab."
        )
    return path


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Preserve source values while creating an analysis-ready numeric field.
    df["Number of Readmissions Numeric"] = pd.to_numeric(
        df["Number of Readmissions"], errors="coerce"
    )

    df["Condition"] = df["Measure Name"].map(MEASURE_MAP)
    df["Start Date"] = pd.to_datetime(df["Start Date"])
    df["End Date"] = pd.to_datetime(df["End Date"])
    return df


def print_data_audit(df: pd.DataFrame) -> None:
    print("=== DATA AUDIT ===")
    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nMeasure counts:")
    print(df["Measure Name"].value_counts(dropna=False))
    print("\nFootnotes:")
    print(df["Footnote"].value_counts(dropna=False))


def print_footnote_analysis(df: pd.DataFrame) -> None:
    summary = df.groupby("Footnote", dropna=False).agg(
        Total_Rows=("Facility ID", "size"),
        Missing_Discharges=("Number of Discharges", lambda x: x.isna().sum()),
        Missing_Readmissions=("Number of Readmissions Numeric", lambda x: x.isna().sum()),
        Missing_Excess_Ratio=("Excess Readmission Ratio", lambda x: x.isna().sum()),
    )
    print("\n=== FOOTNOTE / MISSINGNESS ANALYSIS ===")
    print(summary)


def print_overall_kpis(df: pd.DataFrame) -> None:
    valid_err = df["Excess Readmission Ratio"].dropna()
    print("\n=== HOSPITAL READMISSIONS KPI SUMMARY ===")
    print(f"Unique Hospitals: {df['Facility ID'].nunique():,}")
    print(f"States/Territories: {df['State'].nunique()}")
    print(f"Clinical Conditions: {df['Condition'].nunique()}")
    print(f"Valid ERR Records: {len(valid_err):,}")
    print(f"Mean ERR: {valid_err.mean():.3f}")
    print(f"Median ERR: {valid_err.median():.3f}")
    print(f"ERR > 1: {(valid_err > 1).sum():,}")
    print(f"ERR < 1: {(valid_err < 1).sum():,}")
    print(f"ERR = 1: {(valid_err == 1).sum():,}")
    print(f"Percentage with ERR > 1: {(valid_err > 1).mean() * 100:.1f}%")


def condition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Condition")
        .agg(
            Valid_ERR=("Excess Readmission Ratio", "count"),
            Mean_ERR=("Excess Readmission Ratio", "mean"),
            Median_ERR=("Excess Readmission Ratio", "median"),
            Mean_Predicted_Rate=("Predicted Readmission Rate", "mean"),
            Mean_Expected_Rate=("Expected Readmission Rate", "mean"),
        )
        .sort_values("Mean_ERR", ascending=False)
    )

    summary["ERR_Above_1_Pct"] = (
        df[df["Excess Readmission Ratio"].notna()]
        .groupby("Condition")["Excess Readmission Ratio"]
        .apply(lambda x: (x > 1).mean() * 100)
    )

    print("\n=== CONDITION SUMMARY ===")
    print(summary.round(3))
    return summary


def state_analysis(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("State").agg(
        Hospitals=("Facility ID", "nunique"),
        Valid_ERR=("Excess Readmission Ratio", "count"),
        Mean_ERR=("Excess Readmission Ratio", "mean"),
        Median_ERR=("Excess Readmission Ratio", "median"),
        Mean_Predicted_Rate=("Predicted Readmission Rate", "mean"),
        Mean_Expected_Rate=("Expected Readmission Rate", "mean"),
    )

    summary["ERR_Above_1_Pct"] = (
        df[df["Excess Readmission Ratio"].notna()]
        .groupby("State")["Excess Readmission Ratio"]
        .apply(lambda x: (x > 1).mean() * 100)
    )
    return summary.sort_values("Mean_ERR", ascending=False)


def hospital_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[df["Excess Readmission Ratio"].notna()].copy()

    summary = (
        valid.groupby(["Facility ID", "Facility Name", "State"])
        .agg(
            Conditions_Reported=("Condition", "nunique"),
            Mean_ERR=("Excess Readmission Ratio", "mean"),
            Median_ERR=("Excess Readmission Ratio", "median"),
            Conditions_ERR_Above_1=(
                "Excess Readmission Ratio", lambda x: (x > 1).sum()
            ),
            Mean_Predicted_Rate=("Predicted Readmission Rate", "mean"),
            Mean_Expected_Rate=("Expected Readmission Rate", "mean"),
        )
        .reset_index()
    )

    robust = summary[summary["Conditions_Reported"] >= 5].copy()
    robust["High_ERR_Pct"] = (
        robust["Conditions_ERR_Above_1"] / robust["Conditions_Reported"] * 100
    )

    persistent_high = robust[robust["High_ERR_Pct"] == 100].copy()

    strict_low = (
        valid.groupby(["Facility ID", "Facility Name", "State"])
        .agg(
            Conditions_Reported=("Condition", "nunique"),
            Mean_ERR=("Excess Readmission Ratio", "mean"),
            Median_ERR=("Excess Readmission Ratio", "median"),
            All_ERR_Below_1=("Excess Readmission Ratio", lambda x: (x < 1).all()),
        )
        .reset_index()
    )

    persistent_low = strict_low[
        (strict_low["Conditions_Reported"] >= 5) & strict_low["All_ERR_Below_1"]
    ].copy()

    print("\n=== HOSPITAL SIGNALS ===")
    print("Hospitals with valid ERR:", len(summary))
    print("Hospitals with >=5 reported conditions:", len(robust))
    print("Persistent high-ERR hospitals:", len(persistent_high))
    print("Persistent low-ERR hospitals:", len(persistent_low))
    return persistent_high, persistent_low


def export_powerbi(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    out = df[
        [
            "Facility ID",
            "Facility Name",
            "State",
            "Condition",
            "Number of Discharges",
            "Number of Readmissions Numeric",
            "Excess Readmission Ratio",
            "Predicted Readmission Rate",
            "Expected Readmission Rate",
            "Start Date",
            "End Date",
        ]
    ].copy()

    out.columns = [
        "Facility_ID",
        "Facility_Name",
        "State",
        "Condition",
        "Number_of_Discharges",
        "Number_of_Readmissions",
        "Excess_Readmission_Ratio",
        "Predicted_Readmission_Rate",
        "Expected_Readmission_Rate",
        "Start_Date",
        "End_Date",
    ]

    out.to_csv(output_path, index=False)
    print(f"\nExported {len(out):,} rows to {output_path}")
    return out


def plot_condition_rates(condition_summary: pd.DataFrame) -> None:
    plot_data = condition_summary.sort_values("Mean_Predicted_Rate")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_data.index, plot_data["Mean_Predicted_Rate"])
    plt.xlabel("Mean Predicted Readmission Rate (%)")
    plt.ylabel("Clinical Condition")
    plt.title("Average Predicted 30-Day Readmission Rate by Condition")
    plt.tight_layout()
    plt.show()


def main() -> None:
    input_path = locate_input()
    print("Loading:", input_path)
    df = load_and_prepare(input_path)

    print_data_audit(df)
    print_footnote_analysis(df)
    print_overall_kpis(df)

    conditions = condition_analysis(df)
    states = state_analysis(df)
    high, low = hospital_analysis(df)

    print("\nHighest state-level Mean ERR (>=100 valid records):")
    print(states[states["Valid_ERR"] >= 100].head(10).round(3))

    print("\nHighest persistent high-ERR signals:")
    print(high.sort_values("Mean_ERR", ascending=False).head(10).round(3))

    print("\nLowest persistent low-ERR signals:")
    print(low.sort_values("Mean_ERR").head(10).round(3))

    plot_condition_rates(conditions)
    export_powerbi(df)


if __name__ == "__main__":
    main()
