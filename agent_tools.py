"""
Data tools for the GenAI agent.

Plain functions that look things up in our own data and return numbers.
No LLM code here on purpose: the agent layer decides which of these to call,
but the functions themselves do not care who is calling them. That keeps the
tools testable on their own, and lets us swap LLM provider without touching
anything in this file.

Run with:  python tools.py
to print a few example calls and check that the data loads.
"""
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "datasets"
MERGED = DATA_DIR / "merged.csv"
BY_CROP = DATA_DIR / "correlation" / "by_crop.csv"

# The agent will be asked about "wheat" or "Hvete" or "hvete". Map all of it
# onto the names actually used in the data.
CROP_ALIASES = {
    "wheat": "Hvete", "hvete": "Hvete",
    "barley": "Bygg", "bygg": "Bygg",
    "oats": "Havre", "oat": "Havre", "havre": "Havre",
    "rye": "Rug og rughvete", "rug": "Rug og rughvete",
}

# Human readable names for the weather columns, used in the answers.
VARIABLE_LABELS = {
    "downpour_10_3_z": "rainfall October-March",
    "downpour_4_5_z": "rainfall April-May",
    "downpour_6_7_z": "rainfall June-July",
    "downpour_former_8_9_z": "rainfall early August-September",
    "downpour_latter_8_9_z": "rainfall late August-September",
    "first_frost_day_z": "first day of frost",
    "last_frost_day_z": "last day of frost",
    "longest_dry_spell_z": "longest dry spell",
}

def _fold(text):
    """
    Lowercases and strips the accents, so "Trondelag" matches "Trøndelag"
    and "Nordlandda" matches "Nordland - Nordlánnda". Without this the
    agent only finds a region if the user types the Sami spelling.
    """
    text = text.strip().lower()
    for a, b in (("ø", "o"), ("æ", "ae"), ("å", "a")):
        text = text.replace(a, b)
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")

_merged = None
_by_crop = None


def _load():
    """Reads the csv files once and keeps them in memory."""
    global _merged, _by_crop
    if _merged is None:
        _merged = pd.read_csv(MERGED, dtype={"region_code": str})
        _by_crop = pd.read_csv(BY_CROP)
    return _merged, _by_crop


def _resolve_crop(crop):
    """
    Turns whatever the user called a crop into the name used in the data.

    crop (string):
        -crop name in English or Norwegian, any casing
    """
    if crop is None:
        return None
    key = _fold(crop)
    if key in CROP_ALIASES:
        return CROP_ALIASES[key]
    df, _ = _load()
    for name in df.crop.unique():
        if _fold(name) == key:
            return name
    return None


def _resolve_region(region):
    """
    Matches a region name against the data, which uses names like
    "Hedmark (-2019)" and "Nordland - Nordlanndda". A user asking about
    "Hedmark" should still get an answer, so we match on the start of
    the name and fall back to a substring search.

    region (string):
        -region name, full or partial

    Returns the matching name, or a list of candidates when it is ambiguous.
    """
    if region is None:
        return None
    df, _ = _load()
    names = sorted(df.region_name.unique())
    key = _fold(region)

    exact = [n for n in names if _fold(n) == key]
    if exact:
        return exact[0]

    starts = [n for n in names if _fold(n).startswith(key)]
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:
        return starts

    contains = [n for n in names if key in _fold(n)]
    if len(contains) == 1:
        return contains[0]
    return contains or None


def list_options() -> dict:
    """
    Lists the regions, crops and years that exist in the data.

    The agent should call this before guessing a name, so it asks about
    things we actually have rather than inventing a region.
    """
    df, _ = _load()
    return {
        "regions": sorted(df.region_name.unique()),
        "crops": sorted(df.crop.unique()),
        "first_year": int(df.year.min()),
        "last_year": int(df.year.max()),
    }


def get_yield(region: str, crop: str, year: int) -> dict:
    """
    Looks up the yield for one region, crop and year, and compares it with
    what that region normally gets for that crop.

    region (string):
        -region name, full or partial
    crop (string):
        -crop name in English or Norwegian
    year (int):
        -harvest year between 2001 and 2024
    """
    df, _ = _load()
    region_name = _resolve_region(region)
    crop_name = _resolve_crop(crop)

    if crop_name is None:
        return {"error": f"Unknown crop '{crop}'", "available": list_options()["crops"]}
    if region_name is None:
        return {"error": f"Unknown region '{region}'", "available": list_options()["regions"]}
    if isinstance(region_name, list):
        return {"error": f"'{region}' matches several regions", "candidates": region_name}

    same = df[(df.region_name == region_name) & (df.crop == crop_name)]
    if same.empty:
        return {"error": f"No data for {crop_name} in {region_name}"}

    row = same[same.year == int(year)]
    if row.empty:
        return {"error": f"No data for {crop_name} in {region_name} in {year}",
                "years_available": sorted(int(y) for y in same.year.unique())}

    value = float(row.iloc[0].yield_kg_per_decare)
    years = int(same.yield_kg_per_decare.notna().sum())
    out = {
        "region": region_name,
        "crop": crop_name,
        "year": int(year),
        "yield_kg_per_decare": round(value, 1),
        "years_in_average": years,
    }

    # With only a handful of years the "normal" is mostly this year itself,
    # so the deviation would read as 0 % and mean nothing. Say so instead.
    if years >= 5:
        typical = float(same.yield_kg_per_decare.mean())
        out["typical_for_this_region"] = round(typical, 1)
        out["deviation_pct"] = round((value / typical - 1) * 100, 1)
    else:
        out["note"] = (f"Only {years} year(s) of data for this region and "
                       "crop, too few to say what is normal.")
    return out


def get_crop_correlations(crop: str) -> dict:
    """
    Returns how strongly each weather variable correlates with yield for one
    crop, based on the correlation analysis in correlation.py.

    The p values are Bonferroni adjusted for the eight variables tested, so
    "significant" here already accounts for multiple testing.

    crop (string):
        -crop name in English or Norwegian
    """
    _, corr = _load()
    crop_name = _resolve_crop(crop)
    if crop_name is None:
        return {"error": f"Unknown crop '{crop}'", "available": list_options()["crops"]}

    rows = corr[corr.crop == crop_name].copy()
    if rows.empty:
        return {"error": f"No correlation results for {crop_name}"}

    rows["strength"] = rows.correlation.abs()
    rows = rows.sort_values("strength", ascending=False)

    return {
        "crop": crop_name,
        "observations": int(rows.iloc[0].n),
        "variables": [
            {
                "variable": VARIABLE_LABELS.get(r.variable, r.variable),
                "correlation": round(float(r.correlation), 3),
                "p_adjusted": round(float(r.p_adjusted), 4),
                "significant": bool(r.significant_5pct),
            }
            for r in rows.itertuples()
        ],
    "note": "Correlations are modest. The strongest are around 0.3 and "
                "most are not significant. Weather is one factor among many, "
                "alongside soil, fertiliser, variety and management.",
    }


def rank_crops_by_weather(variable: str) -> dict:    
    """
    Compares all crops on one weather variable, so the agent can answer
    questions like "which crop handles a dry spell best".

    variable (string):
        -a weather variable, either the column name or its label, for
         example "longest dry spell" or "rainfall April-May"
    """
    _, corr = _load()
    key = _fold(variable)

    matches = [col for col, label in VARIABLE_LABELS.items()
               if key == _fold(col) or key in _fold(label)]
    if not matches:
        return {"error": f"Unknown variable '{variable}'",
                "available": list(VARIABLE_LABELS.values())}
    if len(matches) > 1:
        return {"error": f"'{variable}' matches several variables",
                "candidates": [VARIABLE_LABELS[m] for m in matches]}
    column = matches[0]

    rows = corr[corr.variable == column].sort_values("correlation")
    return {
        "variable": VARIABLE_LABELS[column],
        "crops": [
            {
                "crop": r.crop,
                "correlation": round(float(r.correlation), 3),
                "significant": bool(r.significant_5pct),
            }
            for r in rows.itertuples()
        ],
        "reading": "A negative correlation means yields tend to fall as this "
                   "variable rises. The crop closest to zero is the least affected.",
    }


def predict_yield(region, crop, **weather):
    """
    Placeholder for the model. Not wired up yet.

    When the model exists it will be loaded from a joblib file and called
    here, so the agent code does not have to change. Agreed contract:
    takes a region, a crop and weather values, returns a predicted yield
    in kg per decare.

    region (string):
        -region name
    crop (string):
        -crop name
    weather (keyword arguments):
        -weather values for the season, names to be agreed with the model
    """
    return {
        "error": "The prediction model is not finished yet.",
        "status": "Use get_yield for historical numbers in the meantime.",
    }


if __name__ == "__main__":
    opts = list_options()
    print(f"{len(opts['regions'])} regions, {len(opts['crops'])} crops, "
          f"{opts['first_year']}-{opts['last_year']}\n")

    print("get_yield('Hedmark', 'barley', 2018):")
    print(" ", get_yield("Hedmark", "barley", 2018), "\n")

    print("get_crop_correlations('wheat'), strongest three:")
    res = get_crop_correlations("wheat")
    for v in res["variables"][:3]:
        print("  ", v)
    print()

    print("rank_crops_by_weather('longest dry spell'):")
    for c in rank_crops_by_weather("longest dry spell")["crops"]:
        print("  ", c)
