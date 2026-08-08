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
# Drop columns with >40% missing OR irrelevant for analysis
# ---------------------------------------------------------

def drop_columns(df):

    # A. Missing >40%
    missing_cols = [
        "WIN_NAME", "WIN_ADDRESS", "WIN_TOWN", "WIN_POSTAL_CODE", "WIN_NATIONALID",
        "CAE_NAME", "CAE_ADDRESS", "CAE_TOWN", "CAE_POSTAL_CODE", "CAE_NATIONALID",
        "CAE_GPA_ANNEX",
        "GPA_COVERAGE", "ISO_COUNTRY_CODE_GPA", "ISO_COUNTRY_CODE_ALL", "B_GPA",
        "VALUE_EURO_FIN_1", "VALUE_EURO_FIN_2", "AWARD_VALUE_EURO_FIN_1",
        "ADDITIONAL_CPVS",
        "B_MULTIPLE_CAE", "B_MULTIPLE_COUNTRY", "B_ON_BEHALF",
        "B_INVOLVES_JOINT_PROCUREMENT",
        "B_FRA_AGREEMENT", "B_FRA_CONTRACT", "FRA_ESTIMATED",
        "TED_NOTICE_URL", "NUMBER_TENDERS_SME", "NUMBER_TENDERS_OTHER_EU", 
        "NUMBER_TENDERS_NON_EU", "NUMBER_OFFERS_ELECTR", "AWARD_EST_VALUE_EURO"
    ]

    # B. Irrelevant for analysis or too match unique categorical values
    irrelevant_cols = [
        "ID_NOTICE_CAN", "ID_AWARD", "ID_LOT_AWARDED",
        "CONTRACT_NUMBER",
        "INFO_ON_NON_AWARD", "INFO_UNPUBLISHED",
        "MAIN_ACTIVITY", "EU_INST_CODE",
        "XSD_VERSION", "TAL_LOCATION_NUTS", "ID_LOT", "WIN_COUNTRY_CODE", 
        "CPV_GROUP", "CPV_CLASS", "CPV"
    ]


    cols_to_drop = missing_cols + irrelevant_cols 
    existing = [c for c in cols_to_drop if c in df.columns]

    return df.drop(columns=existing).reset_index(drop=True)


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
# Build one TEXT_ALL for NLP
# ---------------------------------------------------------

def build_text_all(df, cols):
    df[cols] = df[cols].fillna(" ").astype(str)
    df["TEXT_ALL"] = df[cols].agg(" ".join, axis=1)
    df = df.drop(columns=cols).reset_index(drop=True)
    return df


# ---------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------

def preprocess(df):

    # 1. Drop columns
    df = drop_columns(df)

    # 2. Clean column names
    df = clean_columns(df)

    # 3. Numeric conversion
    df = convert_numeric(df, [
        "VALUE_EURO", "AWARD_VALUE_EURO", "AWARD_EST_VALUE_EURO",
        "NUMBER_OFFERS", "LOTS_NUMBER", "CRIT_PRICE_WEIGHT"])

    # 4. Optimize memory
    df = optimize_int_columns(df)

    # 5. Date conversion
    df = convert_dates(df, ["DT_DISPATCH", "DT_AWARD"])

    # 6. Categorical conversion
    df = convert_to_string(df, [
        "CAE_TYPE", "B_AWARDED_BY_CENTRAL_BODY",
        "TYPE_OF_CONTRACT", "B_DYN_PURCH_SYST", "MAIN_CPV_CODE_GPA",
        "B_EU_FUNDS", "TOP_TYPE", "B_ACCELERATED", "CRIT_CODE",
        "B_ELECTRONIC_AUCTION", "B_AWARDED_TO_A_GROUP",
        "B_CONTRACTOR_SME", "B_SUBCONTRACTED", "CRIT_CRITERIA", "CRIT_WEIGHTS", "TITLE"])

    # 7. Clean categorical
    df = clean_categorical(df)

    # 8. Replace NaN in categorical with "Unknown"
    df = fill_categorical_unknown(df)

    # 9. Build "TEXT_ALL" for NLP
    df = build_text_all(df, ["TITLE", "CRIT_CRITERIA", "CRIT_WEIGHTS"])


    return df


