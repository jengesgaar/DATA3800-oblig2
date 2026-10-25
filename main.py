import weather_variables
import cleaning_merge
import correlation
import analysis
import pandas as pd
from tools.dataset import standardized_metrics, overview

def main():

    weather_variables.main(True)
    cleaning_merge.clean_crop_data("datasets/aggricultural/04610.csv", 
                                   "datasets/aggricultural/04610_cleaned.csv")

    weather = pd.read_csv("datasets/weather/weather_yearly.csv", dtype={"region_kode":str})
    crop = pd.read_csv("datasets/aggricultural/04610_cleaned.csv", dtype={"region_code":str})

    # Region and year combinations that have at least one crop yield
    region_years = crop.dropna(subset=["yield_kg_per_decare"])[["region_code", "year"]].drop_duplicates()

    # Keep only the weather rows for the regions and year combinations that are active to make sure that standardization is correct
    weather = weather.merge(
        region_years,
        left_on=["region_kode", "harvest_year"],
        right_on=["region_code", "year"],
        how="inner",
    )

    weather = weather.drop(columns=["region_code", "year"])

    # standardize
    metric_columns = [c for c in weather.columns if c not in ["region_kode", "harvest_year"]]
    weather = standardized_metrics(weather, metric_columns)
    crop = standardized_metrics(crop, ["yield_kg_per_decare"], group_by="crop")

    merged = crop.merge(weather, left_on=["region_code", "year"], right_on=["region_kode", "harvest_year"])
    merged.to_csv("datasets/merged.csv", index=False)
    overview(merged)

    correlation.run_all(merged)
    analysis.main()


main()