# Preprocessing

"""
Preprocessing utilities for TED CAN dataset.

Includes:
    - Column name cleaning
    - Removal of low-quality / high-missingness columns
    - Numeric conversion
    - Date conversion
    - Categorical standardisation
    - Missing value handling
    - Logical consistency checks
    - Derived feature creation
    - Log-transformations
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Column name cleaning
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
# Remove columns with excessive missingness or low value
# ---------------------------------------------------------

def drop_low_quality_columns(df):
    cols_to_drop = [
        # Winner info (missing >90%)
        "WIN_NAME", "WIN_ADDRESS", "WIN_TOWN", "WIN_POSTAL_CODE",

        # Criteria weights (missing >95%)
        "CRIT_PRICE_WEIGHT", "CRIT_QUALITY_WEIGHT",

        # Secondary financial fields (redundant + missing)
        "VALUE_EURO_FIN_1", "VALUE_EURO_FIN_2", "AWARD_VALUE_EURO_FIN_1",

        # Subcontracting indicator (missing >50%)
        "B_SUBCONTRACTED",

        # Additional CPVs (sparse)
        "ADDITIONAL_CPVS",

        # GPA fields (missing >90%)
        "GPA_COVERAGE", "B_GPA",

        # High-cardinality text fields
        "TED_NOTICE_URL",
        "CAE_ADDRESS", "CAE_TOWN", "CAE_NAME",
        "WIN_ADDRESS", "WIN_NAME"
    ]

    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------

def convert_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------
# Date conversion
# ---------------------------------------------------------

def convert_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ---------------------------------------------------------
# Categorical cleaning
# ---------------------------------------------------------

def clean_categorical(df):
    if "ISO_COUNTRY_CODE" in df.columns:
        df["ISO_COUNTRY_CODE"] = df["ISO_COUNTRY_CODE"].astype(str).str.strip().str.upper()

    if "WIN_COUNTRY_CODE" in df.columns:
        df["WIN_COUNTRY_CODE"] = df["WIN_COUNTRY_CODE"].astype(str).str.strip().str.upper()

    if "CPV" in df.columns:
        df["CPV"] = df["CPV"].astype(str).str.strip()

    if "MAIN_CPV_CODE_GPA" in df.columns:
        df["MAIN_CPV_CODE_GPA"] = df["MAIN_CPV_CODE_GPA"].astype(str).str.strip()

    return df


# ---------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------

def handle_missing(df, essential_cols):
    df = df.replace({"": np.nan, " ": np.nan})
    df = df.dropna(subset=essential_cols)
    return df


# ---------------------------------------------------------
# Logical consistency checks
# ---------------------------------------------------------

def logical_consistency(df):
    # Award value cannot exceed contract value
    if "AWARD_VALUE_EURO" in df.columns and "VALUE_EURO" in df.columns:
        df = df[df["AWARD_VALUE_EURO"] <= df["VALUE_EURO"]]

    # No negative financial values
    for col in ["VALUE_EURO", "AWARD_VALUE_EURO", "AWARD_EST_VALUE_EURO"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    return df


# ---------------------------------------------------------
# Derived features
# ---------------------------------------------------------

def create_features(df):
    # Failed tender: 0–1 offers
    if "NUMBER_OFFERS" in df.columns:
        df["IS_FAILED_TENDER"] = (df["NUMBER_OFFERS"] <= 1).astype(int)
        df["IS_LOW_COMPETITION"] = (df["NUMBER_OFFERS"] == 2).astype(int)

    # CPV section (first 2 digits)
    if "CPV" in df.columns:
        df["CPV_SECTION"] = df["CPV"].str[:2]

    return df


# ---------------------------------------------------------
# Log-transformations
# ---------------------------------------------------------

def log_transform(df):
    for col in ["VALUE_EURO", "AWARD_VALUE_EURO", "AWARD_EST_VALUE_EURO"]:
        if col in df.columns:
            df[f"LOG_{col}"] = np.log1p(df[col])
    return df


# ---------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------

def preprocess(df):
    df = clean_columns(df)
    df = drop_low_quality_columns(df)

    df = convert_numeric(df, [
        "VALUE_EURO", "AWARD_VALUE_EURO", "AWARD_EST_VALUE_EURO",
        "NUMBER_OFFERS", "LOTS_NUMBER"
    ])

    df = convert_dates(df, [
        "DT_DISPATCH", "DT_AWARD", "DT_RECEIPT"
    ])

    df = clean_categorical(df)

    df = handle_missing(df, [
        "ISO_COUNTRY_CODE", "VALUE_EURO", "NUMBER_OFFERS"
    ])

    df = logical_consistency(df)
    df = create_features(df)
    df = log_transform(df)

    return df
