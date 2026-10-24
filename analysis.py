"""
Analysis ...

Input: ...
"""

from pathlib import Path
import pandas as pd

DATA_FILE = Path("datasets/04610_cleaned.csv")
#OUTPUT_DIR = Path("datasets/analysis_results")

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

def main():
    df = load_data()
    print(f"Loaded {len(df)} rows.")

if __name__ == "__main__":
    main()