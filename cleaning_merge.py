import pandas as pd


def clean_crop_data(input_file, output_file):
    # Read the original CSV file
    crop_df = pd.read_csv(
        input_file,
        sep=";",
        skiprows=1,
        encoding="cp1252"
    )

    # Replace special values with missing values
    crop_clean_df = crop_df.replace([".", "..", ":", "-"], pd.NA)

    # Convert crop yield columns from text to numbers
    crop_clean_df[crop_clean_df.columns[1:]] = crop_clean_df[
        crop_clean_df.columns[1:]
    ].apply(pd.to_numeric, errors="coerce")

    # Reshape the data from wide to long format
    crop_long_df = crop_clean_df.melt(
        id_vars="region",
        var_name="year_crop",
        value_name="yield_kg_per_decare"
    )

    # Split year and crop type
    crop_long_df[["year", "crop"]] = crop_long_df["year_crop"].str.split(
        " ", n=1, expand=True
    )

    # Convert year values from text to integer
    crop_long_df["year"] = crop_long_df["year"].astype(int)

    # Split region code and region name
    crop_long_df[["region_code", "region_name"]] = crop_long_df["region"].str.split(
        " ", n=1, expand=True
    )

    # Select the columns for the cleaned dataset
    crop_final_df = crop_long_df[
        ["region_code", "region_name", "year", "crop", "yield_kg_per_decare"]
    ].copy()

    # Check that each region-year-crop combination is unique
    duplicates = crop_final_df.duplicated(
        ["region_code", "year", "crop"]
    ).sum()

    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate region-year-crop combinations")

    # Save the cleaned dataset
    crop_final_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("Rows:", crop_final_df.shape[0])
    print("Missing yield values:", crop_final_df["yield_kg_per_decare"].isna().sum())
    print("Saved cleaned dataset to:", output_file)

    return crop_final_df


if __name__ == "__main__":
    clean_crop_data(
        "datasets/04610.csv",
        "datasets/04610_cleaned.csv"
    )