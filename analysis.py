"""
Analysis ...

Input: ...
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Enable when the rainfall data is available
#import statsmodels.formula.api as smf

DATA_FILE = Path("datasets/04610_cleaned.csv")
OUTPUT_DIR = Path("datasets/analysis_results")

START_YEAR = 2001
END_YEAR = 2024

CROPS = ["Hvete", "Bygg", "Havre"]

# Data loading - load and validate cleaned crop yield datasets
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {DATA_FILE}. Run clean_merge.ipynb first."
        )
    
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    required = [
        "region_code",
        "region_name",
        "year",
        "crop",
        "yield_kg_per_decare",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df["region_code"] = df["region_code"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["yield_kg_per_decare"] = pd.to_numeric(df["yield_kg_per_decare"], errors="coerce")

    df = df[df["year"].between(START_YEAR, END_YEAR) & df["crop"].isin(CROPS)].copy()

    if df.duplicated(["region_code", "year", "crop"]).any():
        raise ValueError("Duplicate region/year/crop rows found.")
    
    return df


# Data quality - reporting missing values and avalable years by  region
def data_quality(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.isna().sum().rename("missing_values").to_csv(OUTPUT_DIR / "missing_values.csv")

    coverage = (df.groupby(["region_name", "crop"]).agg(available_years=("yield_kg_per_decare", lambda values: values.notna().sum()),
                                                                         first_year=("year", "min"), last_year=("year", "max"),)
            .reset_index()                           
            )
    
    coverage.to_csv(OUTPUT_DIR / "regional_coverage.csv", index=False)

    print("Available yield observations: ", df["yield_kg_per_decare"].notna().sum())
    print("\nObservations by crop: ")
    print(df.groupby("crop")["yield_kg_per_decare"].count().to_string())



# Prepare data - calculating typical yields and deviations
def prepare_data(df):
    

    df = df.dropna(
        subset=["region_code", "year", "crop", "yield_kg_per_decare"]
    ).copy()

    # Check zero or negative yield values before filtering
    invalid_yields = df[df["yield_kg_per_decare"] <= 0]

    print("\nZero or negative yield observations:")
    print(
        invalid_yields[
            ["region_name", "year", "crop", "yield_kg_per_decare"]
        ].to_string(index=False)
    )

    # Exclude zero or negative yields from the analysis
    df = df[df["yield_kg_per_decare"] > 0].copy()

    groups = df.groupby(["region_code", "crop"])


    df["typical_yield"] = groups[
        "yield_kg_per_decare"
    ].transform("mean")

    df["available_years"] = groups[
        "year"
    ].transform("nunique")

    # Deviation requires at least two observations
    df["yield_deviation"] = np.where(
        df["available_years"] >= 2,
        100 * (
            df["yield_kg_per_decare"] / df["typical_yield"] - 1
        ),
        np.nan
    )
    
    return df
""" 
TO DO: Enable after merging rainfall data by region and year!!
    
     weather = df.drop_duplicates(["region_code", "year"])
    
     rainfall_mean = weather.groupby("region_code")[
         "rainfall_sum"
     ].mean()
    
     regional_mean = df["region_code"].map(rainfall_mean)
    
     df["rainfall_deviation"] = np.where(
         regional_mean > 0,
         100 * (df["rainfall_sum"] / regional_mean - 1),
         np.nan
     )
    
     values = (
         df.drop_duplicates(["region_code", "year"])
         ["rainfall_deviation"]
         .dropna()
     )
    
     if values.nunique() >= 3:
        low, high = values.quantile([1/3, 2/3])
    
         if low < high:
            df["rainfall_group"] = pd.cut(
                 df["rainfall_deviation"],
                bins=[-np.inf, low, high, np.inf],
                 labels=["Dry", "Normal", "Wet"]
             )

"""


# Compare average yields between crop types
def compare_crops(df):

    # Calculate average yield and number of observations
    summary = (
        df.groupby("crop")
        .agg(
            mean_yield=("yield_kg_per_decare", "mean"),
            observations=("yield_kg_per_decare", "count")
        )
        .reset_index()
    )

    # Save the results to CSV
    summary.to_csv(
        OUTPUT_DIR / "crop_summary.csv",
        index=False
    )

    print("\nAverage yield by crop:")
    print(summary.to_string(index=False))

    # Create a bar chart
    ax = summary.plot.bar(
        x="crop",
        y="mean_yield",
        figsize=(8, 5),
        rot=0,
        legend=False
    )

    ax.set_title("Average crop yield (2001-2024)")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Yield (kg/decare)")

    plt.tight_layout()

    # Save the figure
    plt.savefig(
        OUTPUT_DIR / "average_crop_yields.png",
        dpi=150
    )

    plt.close()

"""
     TO DO: enable when rainfall data has been merged.
    
     comparison = (
         df.dropna(subset=["rainfall_group"])
         .groupby(["crop", "rainfall_group"], observed=True)
         .agg(
             mean_yield=("yield_kg_per_decare", "mean"),
             mean_deviation=("yield_deviation", "mean"),
             observations=("yield_kg_per_decare", "count")
         )
         .reset_index()
     )
    
     comparison.to_csv(
         OUTPUT_DIR / "crop_rainfall_comparison.csv",
         index=False
     )
"""


# Regression analysis - relationship between yield deviations and rainfall
"""
# Enable after merging rainfall data
def regression_analysis(df):

    results = []

    for crop, group in df.groupby("crop"):
        data = group.dropna(subset=[
            "rainfall_deviation",
            "yield_deviation"
        ]).copy()

        # Require enough observations for regression
        if (
            len(data) < 10
            or data["year"].nunique() < 3
            or data["rainfall_deviation"].nunique() < 3
        ):
            continue

        # Control for differences between regions
        model = smf.ols(
            "yield_deviation ~ rainfall_deviation + C(region_code)",
            data=data
        )

        fitted = model.fit()

        results.append({
            "crop": crop,
            "observations": len(data),
            "rainfall_coefficient": fitted.params["rainfall_deviation"],
            "p_value": fitted.pvalues["rainfall_deviation"],
            "r_squared": fitted.rsquared
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "regression_results.csv",
        index=False
    )
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    data_quality(df)

    df = prepare_data(df)

    df.to_csv(
        OUTPUT_DIR / "prepared_data.csv",
        index=False
    )
 
    compare_crops(df)

    # Enable when rainfall data is available
    #regression_analysis(df)
   

    print(f"Loaded {len(df)} rows.")
    print("Analysis completed.")


if __name__ == "__main__":
    main()
