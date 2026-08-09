"""
Preprocessing for TED CAN dataset.

    - Clean column names
    - Convert numeric fields
    - Convert date fields
    - Clean categorical fields
    - Replace NaN in categorical columns with 'Unknown'
    - Logical consistency checks
    - Remove unrealistic values
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------

def convert_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------
# Memory optimisation
# ---------------------------------------------------------

def optimize_int_columns(df):
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= 0 and col_max < 256:
            df[col] = df[col].astype("int8")
        elif col_min >= -32768 and col_max < 32768:
            df[col] = df[col].astype("int16")
        else:
            df[col] = df[col].astype("int32")
    return df


# ---------------------------------------------------------
# Categorical & text conversion
# ---------------------------------------------------------

def convert_to_string(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_categorical(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def fill_categorical_unknown(df):
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


# ---------------------------------------------------------
# Date conversion
# ---------------------------------------------------------

def convert_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


# ---------------------------------------------------------
# Drop irrelevante rows  
# ---------------------------------------------------------

def drop_cancelled(df):
    return df[df["CANCELLED"] == 0].reset_index(drop=True)


def drop_missing_offers(df):
    return df[df["NUMBER_OFFERS"].notna()].reset_index(drop=True)


def drop_unrealistic_offers(df, max_offers=100):
    return df[df["NUMBER_OFFERS"] <= max_offers].reset_index(drop=True)


def drop_unrealistic_value(df, max_realistic=10000000000):

    # convert all relevant columns to numeric
    cols = ["AWARD_VALUE_EURO", "VALUE_EURO", "VALUE_EURO_FIN_1",
        "VALUE_EURO_FIN_2", "AWARD_EST_VALUE_EURO", "AWARD_VALUE_EURO_FIN_1"]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # candidate columns for filling award value
    candidates = [ "VALUE_EURO", "VALUE_EURO_FIN_1", "VALUE_EURO_FIN_2",
        "AWARD_EST_VALUE_EURO", "AWARD_VALUE_EURO_FIN_1"]

    # fill AWARD_VALUE_EURO from any realistic candidate
    for c in candidates:
        mask = ( df["AWARD_VALUE_EURO"].isna() & df[c].notna() & (df[c] <= max_realistic))
        df.loc[mask, "AWARD_VALUE_EURO"] = df.loc[mask, c]

    # drop unrealistic award values
    df = df[df["AWARD_VALUE_EURO"].isna() | (df["AWARD_VALUE_EURO"] <= max_realistic)]

    # drop remaining missing award values
    df = df[df["AWARD_VALUE_EURO"].notna()]
    
    return df.reset_index(drop=True)


# ---------------------------------------------------------
# Clean column names
# ---------------------------------------------------------

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


# ---------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------

def preprocess(df):

    # --- Clean column names first ---
    df = clean_columns(df)

    # --- Convert numeric ---
    df = convert_numeric(df, ["AWARD_VALUE_EURO", "NUMBER_OFFERS", "CRIT_PRICE_WEIGHT"])
    df = optimize_int_columns(df)

    # --- Convert categorical ---
    df = convert_to_string(df, ["ISO_COUNTRY_CODE", "CAE_TYPE", "TYPE_OF_CONTRACT",
                                "MAIN_CPV_CODE_GPA", "TOP_TYPE", "TITLE", "CPV", "CRIT_WEIGHTS"])
    df = clean_categorical(df)
    df = fill_categorical_unknown(df)

    # --- Dates ---
    df = convert_dates(df, ["DT_DISPATCH", "DT_AWARD"])

    # --- Drop cancelled ---
    df = drop_cancelled(df)

    # --- Drop missing offers ---
    df = drop_missing_offers(df)

    # --- Drop unrealistic offers ---
    df = drop_unrealistic_offers(df, max_offers=500)

    # --- Drop unrealistic award values ---
    df = drop_unrealistic_value(df, max_realistic=10000000000)

    # --- Remove zero award values ---
    df = df[df["AWARD_VALUE_EURO"] > 0]

    # ---------------------------------------------------------
    # Keep relevant columns
    # ---------------------------------------------------------
    keep_cols = [
        "YEAR", "ISO_COUNTRY_CODE", "CAE_TYPE", "TYPE_OF_CONTRACT", "TOP_TYPE",
        "CPV", "MAIN_CPV_CODE_GPA", "AWARD_VALUE_EURO", "CRIT_PRICE_WEIGHT",
        "NUMBER_OFFERS", "DT_DISPATCH", "DT_AWARD", "TITLE", "CRIT_WEIGHTS"
    ]

    df = df[keep_cols].reset_index(drop=True)

    return df
