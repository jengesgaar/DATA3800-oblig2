"""
Analysis ...

Input: ...
"""

from pathlib import Path
import pandas as pd

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



def main():
    df = load_data()
    data_quality(df)
    print(f"Loaded {len(df)} rows.")

if __name__ == "__main__":
    main()