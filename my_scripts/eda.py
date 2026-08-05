# EDA 

"""""
EDA utilities for the TED CAN dataset.

Includes:
    - Target distribution visualisation (pie chart)
    - Correlation matrix heatmap
    - Top‑N country frequency chart
    - Histogram and boxplot helpers for numeric variables
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 1. Load a single year file (all columns)
# ---------------------------------------------------------

def load_single_year(data_dir, year):
    """
    Loads one TED dataset file with ALL columns.
    Adds a YEAR column.
    """
    file = Path(data_dir) / f"Export_OpenDataCAN_year{year}.csv"
    df = pd.read_csv(file, sep=",", low_memory=False)
    df["YEAR"] = year
    return df


# ---------------------------------------------------------
# 2. Load multiple years and combine
# ---------------------------------------------------------

def load_years(data_dir, years):
    """
    Loads multiple years of TED data and concatenates them.
    """
    frames = []
    for y in years:
        df_y = load_single_year(data_dir, y)
        frames.append(df_y)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# 3. Basic preprocessing
# ---------------------------------------------------------

def basic_preprocessing(df):
    """
    Minimal preprocessing:
    - replace empty strings with NaN
    - convert NUMBER_OFFERS to numeric
    """
    df = df.replace({"": pd.NA, " ": pd.NA})
    if "NUMBER_OFFERS" in df.columns:
        df["NUMBER_OFFERS"] = pd.to_numeric(df["NUMBER_OFFERS"], errors="coerce")
    return df


# ---------------------------------------------------------
# 4. Filter Germany only
# ---------------------------------------------------------

def filter_germany(df):
    """
    Filters rows where ISO_COUNTRY_CODE == 'DE'.
    """
    if "ISO_COUNTRY_CODE" not in df.columns:
        raise KeyError("Column ISO_COUNTRY_CODE not found in dataset.")
    return df[df["ISO_COUNTRY_CODE"] == "DE"]


# ---------------------------------------------------------
# 5. Overview of dataset
# ---------------------------------------------------------

def overview(df):
    """
    Displays an overview of key column properties:
    - dtype
    - total non-null
    - missing count
    - missing %
    - number of unique values
    - list of unique values
    """
    df_copy = df.copy()
    summary = pd.DataFrame({
        "dtype": df_copy.dtypes,
        "total": df_copy.count(),
        "missing_n": df_copy.isna().sum(),
        "missing_%": df_copy.isna().mean() * 100,
        "uniques_n": df_copy.nunique(),
        "uniques": [df_copy[col].unique() for col in df_copy.columns]
    })
    display(summary)













