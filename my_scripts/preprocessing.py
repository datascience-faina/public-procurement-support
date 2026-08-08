# Preprocessing

"""
Preprocessing for TED CAN dataset.

    - Drop low-quality columns (missing >40%)
    - Drop columns irrelevant for analysis
    - Clean column names
    - Convert numeric fields
    - Convert date fields
    - Clean categorical fields
    - Replace NaN in categorical columns with 'Unknown'
    - Logical consistency checks
    - Derived features
    - Log-transformations
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Drop irrelevante rows  
# ---------------------------------------------------------
# Drop irrelevant tenders
def drop_tenders(df):

    # cancelled tenders
    df = df[df["CANCELLED"] == 0].reset_index(drop=True)

    # missing information on the number of offers
    df = df[df["NUMBER_OFFERS"].notna()].reset_index(drop=True)

    # unrealistic number of proposals
    df = df[(df["NUMBER_OFFERS"] <= 100) |
    (df["AWARD_VALUE_EURO"].isna())].copy().reset_index(drop=True)

    return df

# ---------------------------------------------------------
# Choice relevante columns 
# ---------------------------------------------------------
# Keep columns relevant for analysis
def keep_columns(df):

    # Column for keeping
    keep_cols = ["YEAR", "ISO_COUNTRY_CODE", "CAE_TYPE", "TYPE_OF_CONTRACT", "TOP_TYPE", "MAIN_CPV_CODE_GPA", 
                 "VALUE_EURO", "CRIT_PRICE_WEIGHT", "NUMBER_OFFERS", "DT_DISPATCH", "DT_AWARD", "TITLE"]

    df = df[keep_cols].reset_index(drop=True)

    return df


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
# Date conversion
# ---------------------------------------------------------

def convert_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df

# ---------------------------------------------------------
# Categorical & text conversion
# ---------------------------------------------------------

def convert_to_string(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

# ---------------------------------------------------------
# Clean categorical fields
# ---------------------------------------------------------

def clean_categorical(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df

# ---------------------------------------------------------
# Replace NaN in categorical columns with "Unknown"
# ---------------------------------------------------------

def fill_categorical_unknown(df):
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


# ---------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------

def preprocess(df):

    df = drop_tenders(df)

    df = keep_columns(df)

    df = clean_columns(df)

    df = convert_numeric(df, ["VALUE_EURO", "NUMBER_OFFERS", "CRIT_PRICE_WEIGHT"])

    df = optimize_int_columns(df)

    df = convert_dates(df, ["DT_DISPATCH", "DT_AWARD"])

    df = convert_to_string(df, ["ISO_COUNTRY_CODE", "CAE_TYPE", "TYPE_OF_CONTRACT", "MAIN_CPV_CODE_GPA",
        "TOP_TYPE", "TITLE"])
    
    df = clean_categorical(df)

    df = fill_categorical_unknown(df)

    return df


