import time
import pandas as pd
import requests
from tools.dataset import overview

path_daily = "datasets/weather/weather_daily.csv"
path_yearly = "datasets/weather/weather_yearly.csv"

path_station_region = "datasets/weather/station_region.csv"

# Personal user-id to be able to retreive data from the api
client_id = "49ec570f-0c96-4a78-86aa-8aa882f6a37a"

api = "https://frost.met.no/observations/v0.jsonld"

# The variables that are retreived from the api
variables = "min(air_temperature P1D),mean(air_temperature P1D),sum(precipitation_amount P1D)"


# The weather stations who are source. They are chosen with the help of LLM. 
stations = [
    # 01 Østfold
    "SN3810", "SN1130", "SN3290", "SN17150", "SN3190", "SN17000",
    # 02-03 Oslo og Akershus
    "SN19710", "SN4780", "SN18700", "SN17850",
    # 04 Hedmark
    "SN6020", "SN12320", "SN12290", "SN12550", "SN2950", "SN7010", "SN5650",
    # 05 Oppland
    "SN23420", "SN13150", "SN12680", "SN21680", "SN23160", "SN11500",
    # 06 Buskerud
    "SN26900", "SN20301", "SN28380", "SN28800", "SN24880", "SN24890",
    # 07 Vestfold
    "SN27500", "SN27800", "SN27450", "SN27435", "SN26990", "SN27045", "SN27330",
    # 08 Telemark
    "SN32060", "SN32890", "SN32850", "SN30255", "SN37230",
    # 09 Aust-Agder
    "SN39690", "SN38140", "SN36560", "SN36200", "SN40250",
    # 10 Vest-Agder
    "SN39040", "SN41670", "SN41770", "SN42160", "SN41110", "SN41090",
    # 11 Rogaland
    "SN43010", "SN46610", "SN44560", "SN44300",
    # 12 Hordaland
    "SN50540", "SN50300", "SN52290", "SN49490", "SN51530",
    # 14 Sogn og Fjordane
    "SN56420", "SN58070", "SN52860",
    # 15 Møre og Romsdal
    "SN59610", "SN61420", "SN63420", "SN60500", "SN60990",
    # 16 Sør-Trøndelag
    "SN71850", "SN66150", "SN68340", "SN68860", "SN71550",
    # 17 Nord-Trøndelag
    "SN69150", "SN70850", "SN70150", "SN69100",
    # 18 Nordland
    "SN87110", "SN86520", "SN76530", "SN76450",
]

# Time periods for gathering data
# Yield years in dataset 04610 are yields planted from august the year prior until september the year of the yield when the 
# crops are harvested
time_periods = [(year, f"{year - 1}-08-01", f"{year}-10-01") for year in range(2001, 2025)]


def main(skip_gather=False):
    data_retreival(skip_gather)
    df = pd.read_csv(path_daily, dtype={"region_kode": str})
    df_daily = clean(df)
    df_yearly = yearly_metrics(df_daily)
    df_yearly.to_csv(path_yearly, index=False)





def gather_api(station, start_date, end_date):
    """
    Gathers data from met.no api using chosen variables 

    Variables:
        station (string): 
            -weather station id

        start_date (string | date format):
            -start date for the retreived data

        end_date (string | date format)
            -end date for the retreived data

    Returns:
        NOT NAMED (list of dictionaries)
            - the data gathered from api 
            - empty list if there is no data

    """
    parameters = {
        "sources": station,
        "elements": variables,
        "referencetime": f"{start_date}/{end_date}",
        "timeresolutions": "P1D" # Only daily data
    }
    for n in range(10):
        r = requests.get(api, parameters, auth=(client_id, ""), timeout=180)
        if r.status_code == 200:
            return r.json()["data"]

        # No available data
        if r.status_code in (404, 412):
            return []

        print(f"Error  {r.status_code}")
        time.sleep(5*(n+1))
    raise RuntimeError(f"Gather API timed out. Could not gather {station} {start_date}-{end_date}")

def data_retreival(skip_gather=False):
    """
    Pipeline for gathering, cleaning and merging weather data
    The retreival process takes some time- because of this, if the data has 
    been retreived previously, the user has the option to skip the entire process.

    Variables:
        skip_gather (bool)
            -preset False
            -Chooses whether to skip the process

    """
    if skip_gather:
        return

    start_time = time.time()

    df = pd.DataFrame()
    # Gathers all data neccesary from all timelines across all weather stations
    for station in stations:
        for harvest_year, start_date, end_date in time_periods:
            data = gather_api(station, start_date, end_date)
            if not data:
                continue

            # one row per timestamp, then one row per observation at that timestamp
            del_df = pd.DataFrame(data)[["sourceId", "referenceTime", "observations"]]
            del_df = del_df.explode("observations", ignore_index=True)
            obs = pd.DataFrame(del_df["observations"].tolist())
            del_df = pd.concat([del_df[["sourceId", "referenceTime"]], obs], axis=1)
            del_df["harvest_year"] = harvest_year
            df = pd.concat([df, del_df], ignore_index=True)

        print(f"finished with station {station}")


    # Creates shortened names instead of the names frost API uses and adds the timeframe used for variables
    names = {"min(air_temperature P1D)": "min_temp", "mean(air_temperature P1D)": "mean_temp", "sum(precipitation_amount P1D)": "nedbor"}
    df["series"] = df["elementId"].map(names) + "_" + df["timeOffset"] # For example min_temp_PT18H for the temperature from 18.00 one day to 18.00 the next day.

    # One row per station per day. 
    df = df.pivot(
        index=["sourceId", "harvest_year", "referenceTime"], 
        columns="series", 
        values=["value", "qualityCode", "exposureCategory", "performanceCategory"])

    # adds the quality codes for the metric
    df.columns = [s if v == "value" else f"{s}_{v}" for v, s in df.columns]
    df = df[sorted(df.columns)].reset_index()
    


    # Region code from dataset 04610
    regions = pd.read_csv(path_station_region, sep=";", dtype={"region_kode": str}, encoding="utf-8-sig")

    # to make it compatible with regions dataset
    df["station"] = df["sourceId"].str.split(":").str[0]

    # merges datasets by station and stasjons_id
    df = df.merge(regions[["stasjon_id", "region_kode", "fra_aar", "til_aar"]],
        left_on="station", right_on="stasjon_id", how="left")

    # keeps only the rows where harvest year is within the correct year range. Makes sure that the dataset doesnt have duplicate data across 
    # the same region under different names
    df = df[(df["harvest_year"] >= df["fra_aar"]) & (df["harvest_year"] <= df["til_aar"])]

    # No longer neccessary columns as these columns were only to connect the correct data to the correct region inside 04610
    df = df.drop(columns=["station", "stasjon_id", "fra_aar", "til_aar"]).reset_index(drop=True)

    # Dryspell bool
    # df["temp_over"] = ((df["mean_temp"]-5))

    
    df.to_csv(path_daily, index=False)
    
    end_time = time.time()

    print(f"Finished gathering data. It took {(end_time-start_time)/60} minutes")


def clean(df):

    # According to met.no websites, the preffered readings are min_temp_PT18H, mnedbor_PT6H and mean_temp_PT0H. These will be used unless empty, and if they are empty
    # the backup is the other dataset of the same metric. The same metric type will be used for quality exposure and performance

    prefered_metrics = ["min_temp_PT18H", "nedbor_PT6H", "mean_temp_PT0H"]
    secondary_metrics = ["min_temp_PT0H", "nedbor_PT18H", "mean_temp_PT6H"]

    prefered_quality = ["min_temp_PT18H_qualityCode", "nedbor_PT6H_qualityCode", "mean_temp_PT0H_qualityCode"]
    secondary_quality = ["min_temp_PT0H_qualityCode", "nedbor_PT18H_qualityCode", "mean_temp_PT6H_qualityCode"]

    prefered_exposure = ["min_temp_PT18H_exposureCategory", "nedbor_PT6H_exposureCategory", "mean_temp_PT0H_exposureCategory"]
    secondary_exposure = ["min_temp_PT0H_exposureCategory", "nedbor_PT18H_exposureCategory", "mean_temp_PT6H_exposureCategory"]

    prefered_performance = ["min_temp_PT18H_performanceCategory", "nedbor_PT6H_performanceCategory", "mean_temp_PT0H_performanceCategory"]
    secondary_performance = ["min_temp_PT0H_performanceCategory", "nedbor_PT18H_performanceCategory", "mean_temp_PT6H_performanceCategory"]

    new_metrics = ["min_temp", "nedbor", "mean_temp"]
    new_quality = ["min_temp_quality", "nedbor_quality", "mean_temp_quality"]
    new_exposure = ["min_temp_exposure", "nedbor_exposure", "mean_temp_exposure"]
    new_performance = ["min_temp_performance", "nedbor_performance", "mean_temp_performance"]

    for n in range(len(prefered_metrics)):
        use_primary = df[prefered_metrics[n]].notna()
        df[new_metrics[n]] =  df[prefered_metrics[n]].where(use_primary, df[secondary_metrics[n]])
        df[new_quality[n]] =  df[prefered_quality[n]].where(use_primary, df[secondary_quality[n]])
        df[new_exposure[n]] =  df[prefered_exposure[n]].where(use_primary, df[secondary_exposure[n]])
        df[new_performance[n]] =  df[prefered_performance[n]].where(use_primary, df[secondary_performance[n]])

        
        # Remove the redundant columns
        df = df.drop(columns=[
            prefered_metrics[n], secondary_metrics[n],
            prefered_quality[n], secondary_quality[n],
            prefered_exposure[n], secondary_exposure[n],
            prefered_performance[n], secondary_performance[n]
        ])

    # remove bad exposure data. Places where met.no itself says that the location is not represental of the region. 
    for m, q, e, p in zip(new_metrics, new_quality, new_exposure, new_performance):
        bad_exposure = df[e].isin([4,5])
        df[[m, q, e, p]] = df[[m, q, e, p]].mask(bad_exposure, axis=0)

    # Combining every observation for each region at each day into an average across that region. 
    grouped = df.groupby(["region_kode", "harvest_year", "referenceTime"], as_index=False)

    region_daily = grouped.agg(
        min_temp=("min_temp", "mean"),
        nedbor=("nedbor", "mean"),
        mean_temp=("mean_temp", "mean"),
        min_temp_stations=("min_temp", "count"),
        nedbor_stations=("nedbor", "count"),
        mean_temp_stations=("mean_temp", "count"),
    )

    # calculation for the amount of water a plant needs in x temperature for it to be able to draw in the water
    region_daily["dry_threshold"] = ((region_daily["mean_temp"] - 5) / 3).clip(lower=0)

    # Day counted as dry if the plant is not able to draw in the water
    region_daily["dry_day"] = region_daily["nedbor"] <= region_daily["dry_threshold"]

    # frost day: temperature below 0 
    region_daily["frost_day"] = region_daily["min_temp"] < 0

    # yearly, monthly
    region_daily["date"] = pd.to_datetime(region_daily["referenceTime"])
    region_daily["year"] = region_daily["date"].dt.year
    region_daily["month"] = region_daily["date"].dt.month

    # adds the crop period into the dataset
    region_daily["period"] = region_daily.apply(find_period, axis=1)

    return region_daily


def yearly_metrics(df_daily):
    # Total downpour per region, harvest year and period. Will not be used for data analysis because of overlapping data
    downpour = df_daily.groupby(["region_kode", "harvest_year", "period"], as_index=False)["nedbor"].sum()

    # One row per region  and harvest year, one column per period
    df_yearly = downpour.pivot(index=["region_kode", "harvest_year"], columns="period", values="nedbor")
    # Names all columns without names
    df_yearly.columns = ["downpour_" + period for period in df_yearly.columns]
    df_yearly = df_yearly.reset_index()

    

    # Standard deviation of downpour per period in each region and harvest year
    std = df_daily.groupby(["region_kode", "harvest_year"] + ["period"], as_index=False)["nedbor"].std()
    std = std.pivot(index=["region_kode", "harvest_year"], columns="period", values="nedbor")
    std.columns = ["downpour_std_" + period for period in std.columns]

    # dry days per period in percentage
    dry = df_daily.groupby(["region_kode", "harvest_year"] + ["period"], as_index=False)["dry_day"].mean()
    dry = dry.pivot(index=["region_kode", "harvest_year"], columns="period", values="dry_day") * 100
    dry.columns = ["dry_pct_"+period for period in dry.columns]

    # Percentage of frost days in relevant periods
    frost_periods = df_daily[df_daily["period"].isin(["former_8_9", "10_3", "latter_8_9"])]
    frost = frost_periods.groupby(["region_kode", "harvest_year"] + ["period"], as_index=False)["frost_day"].mean()
    frost = frost.pivot(index=["region_kode", "harvest_year"], columns="period", values="frost_day") * 100
    frost.columns = ["frost_pct_"+ period for period in frost.columns]

    # longest dry spell
    df_daily = df_daily.sort_values(["region_kode", "harvest_year", "date"])
    longest_dry_spells = df_daily.groupby(["region_kode", "harvest_year"])["dry_day"].apply(longest_dry_spell).rename("longest_dry_spell")

    # earliest day of frost prior autumn
    former_autumn = df_daily[df_daily["year"] == df_daily["harvest_year"] -1]
    first_frost = former_autumn.groupby(["region_kode", "harvest_year"]).apply(first_frost_day).rename("first_frost_day")

    # last day of frost the harvesting year
    harvest_year = df_daily[(df_daily["year"] == df_daily["harvest_year"]) & (df_daily["month"] <= 5)]
    last_frost = harvest_year.groupby(["region_kode", "harvest_year"]).apply(first_frost_day, True).rename("last_frost_day")
    

    # Put everything together
    df_yearly = df_yearly.set_index(["region_kode", "harvest_year"])
    df_yearly = df_yearly.join([std, dry, frost, longest_dry_spells, first_frost, last_frost]).reset_index()

    return df_yearly







def find_period(row):
    """
    """

    month = row["month"]
    before_harvest_year = row["year"] == row["harvest_year"] -1

    # autumn crops planting
    if before_harvest_year and month in (8, 9):
        return "former_8_9"
    # autumn crops growing period
    if month in (10, 11, 12, 1, 2, 3):
        return "10_3"
    # spring crops planting
    if month in (4, 5):
        return "4_5"
    if month in (6,7):
        return "6_7"
    if month in (8,9):
        return "latter_8_9"


def longest_dry_spell(dry_days):
    longest = 0
    current = 0
    for is_dry in dry_days:
        if is_dry:
            current += 1
            longest = max(current, longest)
        else:
            current = 0

    return longest


def first_frost_day(df, last_day=False):
    frost = df[df["frost_day"]]
    if len(frost) == 0:
        return None
    if last_day:
        return frost["date"].max().dayofyear
    return frost["date"].min().dayofyear

# Når vi har fått fram hele datasettet: Må vi vise fram hvor mange rader som ikke har quality code, men at vi bruker Exposure category fordi den er dekkende: 
# Vis faktiske talll



