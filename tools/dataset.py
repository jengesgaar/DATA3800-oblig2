
import pandas as pd

def overview(df):
    rows, cols = df.shape
    print(f"Rows:   {rows}")
    print(f"Columns:   {cols}")
    print("Column Names:  ")
    print(df.columns.tolist())

    # print(df.head().to_string())

    print(f"Duplicate rows:   {df.duplicated().sum()}")
    print(f"Empty cells:      {df.isna().sum().sum()}")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        print("Empty cells per column:")
        print(missing.to_string())


def standardized_metrics(df, columns, group_by=None):
    """
    standardizes columns using z-values
    Group by is for metrics that needs to be grouped.

    z = (column value - column mean) / column std

    """

    for c in columns:
        if group_by is None:
            mean = df[c].mean()
            std = df[c].std()

        else:
            mean = df.groupby(group_by)[c].transform("mean")
            std = df.groupby(group_by)[c].transform("std")
        df[c + "_z"] = (df[c] - mean) / std

    return df