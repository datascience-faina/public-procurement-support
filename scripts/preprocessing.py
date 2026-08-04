import pandas as pd


# ---------------------------------------------------------
# 1. Load single year
# ---------------------------------------------------------

def load_single_year(path, year, usecols=None):
    """
    Loads a single TED file in the format Export_OpenDataCAN_yearXXXX.csv
    with selected columns.
    """
    file = f"{path}/Export_OpenDataCAN_year{year}.csv"
    df = pd.read_csv(file, sep=",", low_memory=False, usecols=usecols)
    df["YEAR"] = year
    return df


# ---------------------------------------------------------
# 2. Load all year
# ---------------------------------------------------------

def load_years(path, years, usecols=None):
    """
    Loads data for several years and combines it into a single DataFrame.
    """
    frames = []
    for y in years:
        df_y = load_single_year(path, y, usecols)
        frames.append(df_y)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# 3. Filter by Germany only
# ---------------------------------------------------------

def filter_germany(df):
    """
    Returns only tenders from Germany.
    """
    return df[df["ISO_COUNTRY_CODE"] == "DE"]


# ---------------------------------------------------------
# 4. Preprocessing steps
# ---------------------------------------------------------

def basic_preprocessing(df):
    """
    Minimal preprocessing:
    - converting NUMBER_OFFERS to numeric format
    - replacing empty values ​​with NaN
    """
    df = df.replace({"": pd.NA, " ": pd.NA})
    df["NUMBER_OFFERS"] = pd.to_numeric(df["NUMBER_OFFERS"], errors="coerce")
    return df
