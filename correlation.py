"""
Correlation between rainfall and crop yield.

Run with:  python correlation.py

If the dataset from cleaning_merge.py is missing, test data is used instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- setup ---

DATAFILE = Path("datasets/analysedata.csv")
OUTDIR = Path("datasets/correlation")

# These must match what cleaning_merge.py produces.
# TODO: double-check the names
TARGET = "yield_deviation"

EXPLANATORY = [
    "rainfall_sum",          # total rainfall during the growing season
    "rainfall_std",          # how unevenly the rain fell
    "rainfall_planting",     # rainfall around planting
    "rainfall_harvesting",   # rainfall around harvest
    "percentage_wet",        # share of wet days
    "percentage_dry",        # share of dry days
    "longest_dryspell",      # longest dry spell in days
    "days_with_frost",
]

COL_COUNTY, COL_CROP, COL_YEAR = "county", "crop", "year"
GROUPS = [COL_COUNTY, COL_CROP, COL_YEAR]

# Counties with fewer years than this are skipped. With 8 years the
# correlation has to reach 0.71 to be significant.
MIN_YEARS_PER_COUNTY = 8


def check_columns(df, required=None):
    """
    Checks that the columns we need are present in the dataframe.

    df (DataFrame):
        -dataset to check
    required (list of strings):
        -column names to look for, defaults to the ones used by this module
    """
    required = required or (EXPLANATORY + [TARGET] + GROUPS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing {missing}. Found: {sorted(df.columns)}")


def _correlate(x, y, method):
    f = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = f(x, y)
    return float(r), float(p)

# --- analysis ---

def correlation_matrix(df, method="spearman"):
    """
    Correlates every numeric column against every other one.

    Spearman is the default. It only cares about the ranking of the values,
    not the distance between them, so an extreme year like 2018 does not
    pull as hard.

    df (DataFrame):
        -dataset containing the explanatory columns and the target
    method (string):
        -"spearman" or "pearson"
    """
    check_columns(df)
    return df[EXPLANATORY + [TARGET]].corr(method=method).round(3)


def correlation_with_pvalue(df, method="spearman"):
    """
    Same numbers, but with p-value and n so we can see how much to trust them.

    df (DataFrame):
        -dataset containing the explanatory columns and the target
    method (string):
        -"spearman" or "pearson"
    """
    check_columns(df)
    rows = []
    for variable in EXPLANATORY:
        pair = df[[variable, TARGET]].dropna()
        if len(pair) < 4:
            rows.append({"variable": variable, "n": len(pair),
                         "correlation": np.nan, "p_value": np.nan})
            continue
        r, p = _correlate(pair[variable], pair[TARGET], method)
        rows.append({"variable": variable, "n": len(pair),
                     "correlation": round(r, 3), "p_value": round(p, 4)})

    out = pd.DataFrame(rows)
    # We test 8 variables at once, so we multiply the p-value by the number
    # of tests to avoid being fooled by random results.
    out["p_adjusted"] = (out.p_value * len(EXPLANATORY)).clip(upper=1.0).round(4)
    out["significant_5pct"] = out.p_adjusted < 0.05
    return out


def by_crop(df, method="spearman"):
    """
    One table per crop.

    The whole point of the assignment is to see whether the crops react
    differently, and that disappears if we pool them.

    df (DataFrame):
        -dataset containing the explanatory columns and the target
    method (string):
        -"spearman" or "pearson"
    """
    check_columns(df)
    parts = []
    for crop, group in df.groupby(COL_CROP):
        part = correlation_with_pvalue(group, method)
        part.insert(0, COL_CROP, crop)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def by_county(df, variable="rainfall_sum", method="spearman"):
    """
    One correlation per county and crop, to see whether it varies regionally.

    Counties with too few years are dropped. A correlation based on five
    points looks just as convincing in a table as one based on twenty.

    df (DataFrame):
        -dataset containing the target and the given variable
    variable (string):
        -explanatory column to correlate against the target
    method (string):
        -"spearman" or "pearson"
    """
    check_columns(df, [variable, TARGET, COL_COUNTY, COL_CROP])
    rows = []
    for (county, crop), group in df.groupby([COL_COUNTY, COL_CROP]):
        pair = group[[variable, TARGET]].dropna()
        if len(pair) < MIN_YEARS_PER_COUNTY:
            continue
        r, p = _correlate(pair[variable], pair[TARGET], method)
        rows.append({COL_COUNTY: county, COL_CROP: crop, "n": len(pair),
                     "correlation": round(r, 3), "p_value": round(p, 4)})
    return (pd.DataFrame(rows)
              .sort_values([COL_CROP, "correlation"], ascending=[True, False]))

# --- figures ---

def heatmap(matrix, title, file):
    """
    Draws a heatmap of a correlation matrix and saves it as a png.

    matrix (DataFrame):
        -table of correlation values between -1 and 1
    title (string):
        -title printed above the figure
    file (Path):
        -where to save the png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(matrix.columns) + 3,
                                    0.9 * len(matrix.index) + 2.5))
    image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    # Print the value inside each cell, white text where the background is dark
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.55 else "black")

    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=.8, label="Correlation")
    fig.tight_layout()
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file, dpi=150)
    plt.close(fig)


def run_all(df, outdir=OUTDIR):
    """
    Runs every analysis and saves the tables and figures.

    df (DataFrame):
        -dataset containing the explanatory columns and the target
    outdir (Path):
        -folder to write the csv files and figures to
    """
    outdir.mkdir(parents=True, exist_ok=True)
    results = {
        "matrix_spearman": correlation_matrix(df, "spearman"),
        "matrix_pearson":  correlation_matrix(df, "pearson"),
        "overall":         correlation_with_pvalue(df),
        "by_crop":         by_crop(df),
        "by_county":       by_county(df),
    }
    for name, table in results.items():
        table.to_csv(outdir / f"{name}.csv", index=name.startswith("matrix"))

    heatmap(results["matrix_spearman"], "Correlations (Spearman)",
            outdir / "heatmap_overall.png")
    heatmap(results["by_crop"].pivot(index=COL_CROP, columns="variable",
                                     values="correlation"),
            "Correlation against yield deviation, by crop",
            outdir / "heatmap_by_crop.png")
    return results

# --- test data ---

def test_data(seed=42):
    """
    Random data with the same columns as the real dataset.

    Wheat and barley are set up to lose out on heavy rainfall, oats are not.
    That lets me check whether the code picks it up.

    seed (int):
        -seed for the random generator, so the data is reproducible
    """
    rng = np.random.default_rng(seed)
    effect = {"Wheat": -0.45, "Barley": -0.20, "Oats": 0.05}
    rows = []
    for county in ["Ostlandet", "Innlandet", "Vestfold", "Agder",
                   "Rogaland", "Trondelag", "Nordland"]:
        normal = rng.uniform(250, 550)
        for year in range(2001, 2025):
            total_mm = normal * rng.normal(1.0, 0.22)
            deviation = total_mm / normal - 1
            wet = float(np.clip(rng.normal(0.45 + 0.30 * deviation, 0.06), 0, 1))
            for crop, e in effect.items():
                rows.append({
                    COL_COUNTY: county, COL_CROP: crop, COL_YEAR: year,
                    TARGET: e * deviation + rng.normal(0, 0.10),
                    "rainfall_sum": round(total_mm, 1),
                    "rainfall_std": round(abs(rng.normal(4, 1)), 2),
                    "rainfall_planting": round(total_mm * rng.uniform(.15, .3), 1),
                    "rainfall_harvesting": round(total_mm * rng.uniform(.2, .35), 1),
                    "percentage_wet": round(wet, 3),
                    "percentage_dry": round(1 - wet, 3),
                    "longest_dryspell": int(np.clip(rng.normal(10 - 8 * deviation, 3), 0, 40)),
                    "days_with_frost": int(np.clip(rng.normal(4 - 2 * deviation, 2), 0, 30)),
                })
    return pd.DataFrame(rows)


def main():
    if DATAFILE.exists():
        df = pd.read_csv(DATAFILE)
        print(f"Read {len(df)} rows from {DATAFILE}")
    else:
        df = test_data()
        print(f"Could not find {DATAFILE}, using test data ({len(df)} rows).")
        print("NB: the numbers below mean nothing, they are random.\n")

    results = run_all(df)
    print(results["by_crop"].to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
