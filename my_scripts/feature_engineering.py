# Feature engineering

"""
Feature Engineering for TED CAN dataset.

Creates:
    - Date-based features
    - CPV hierarchy features
    - Competition features
    - Value-based bins
    - Tender complexity features
    - Boolean indicators
    - Missing-information flags

Removes:
    - Raw columns replaced by engineered features
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Date features
# ---------------------------------------------------------

def create_date_features(df):
    if "DT_AWARD" in df.columns:
        df["AWARD_YEAR"] = df["DT_AWARD"].dt.year
        df["AWARD_MONTH"] = df["DT_AWARD"].dt.month
        df["AWARD_QUARTER"] = df["DT_AWARD"].dt.quarter

    if "DT_DISPATCH" in df.columns and "DT_AWARD" in df.columns:
        df["DAYS_TO_AWARD"] = (df["DT_AWARD"] - df["DT_DISPATCH"]).dt.days

    return df


# ---------------------------------------------------------
# CPV hierarchy features
# ---------------------------------------------------------

def create_cpv_features(df):
    if "CPV" in df.columns:
        df["CPV_DIVISION"] = df["CPV"].str[:2]
        df["CPV_GROUP"] = df["CPV"].str[:3]
        df["CPV_CLASS"] = df["CPV"].str[:4]

    if "MAIN_CPV_CODE_GPA" in df.columns:
        df["MAIN_CPV_DIVISION"] = df["MAIN_CPV_CODE_GPA"].astype(str).str[:2]

    return df


# ---------------------------------------------------------
# Competition features
# ---------------------------------------------------------

def create_competition_features(df):
    if "NUMBER_OFFERS" in df.columns:
        df["IS_FAILED_TENDER"] = (df["NUMBER_OFFERS"] <= 1).astype(int)
        df["IS_LOW_COMPETITION"] = (df["NUMBER_OFFERS"] == 2).astype(int)

        df["OFFERS_BIN"] = pd.cut(
            df["NUMBER_OFFERS"],
            bins=[-1, 1, 2, 5, 1000],
            labels=["failed", "low", "medium", "high"]
        )

    return df


# ---------------------------------------------------------
# Value bins (contract size categories)
# ---------------------------------------------------------

def create_value_bins(df):
    if "VALUE_EURO" in df.columns:
        df["VALUE_BIN"] = pd.cut(
            df["VALUE_EURO"],
            bins=[-1, 10000, 100000, 1000000, 10000000, np.inf],
            labels=["micro", "small", "medium", "large", "mega"]
        )

    return df


# ---------------------------------------------------------
# Tender complexity features
# ---------------------------------------------------------

def create_complexity_features(df):
    if "LOTS_NUMBER" in df.columns:
        df["HAS_MULTIPLE_LOTS"] = (df["LOTS_NUMBER"] > 1).astype(int)

        df["LOTS_BIN"] = pd.cut(
            df["LOTS_NUMBER"],
            bins=[-1, 1, 5, 20, np.inf],
            labels=["single", "few", "many", "mega"]
        )

    return df


# ---------------------------------------------------------
# Boolean indicators
# ---------------------------------------------------------

def create_boolean_indicators(df):
    # mapping for typical Y/N flags
    yn_map = {
        "Y": 1,
        "N": 0,
        "nan": 0,   
        "None": 0,
        "UNKNOWN": 0
    }

    for col in [
        "B_ELECTRONIC_AUCTION",
        "B_DYN_PURCH_SYST",
        "B_ACCELERATED",
        "B_AWARDED_TO_A_GROUP",
        "B_CONTRACTOR_SME"
    ]:
        if col in df.columns:
            # format into string
            df[col] = df[col].astype(str).str.strip()

            # transformation Y/N into 1/0
            df[col] = df[col].map(yn_map).fillna(0).astype(int)

    return df


# ---------------------------------------------------------
# Missing-information flags
# ---------------------------------------------------------

def create_missing_flags(df):
    for col in ["VALUE_EURO", "AWARD_VALUE_EURO", "NUMBER_OFFERS"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

    return df


# ---------------------------------------------------------
# Remove raw columns replaced by engineered features
# ---------------------------------------------------------

def drop_redundant_columns(df):
    cols_to_drop = [
        "CPV",
        "DT_AWARD",
        "DT_DISPATCH",
        "VALUE_EURO",
        "AWARD_VALUE_EURO",
        "AWARD_EST_VALUE_EURO",
        "LOTS_NUMBER",
        "NUMBER_OFFERS"
    ]

    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------
# Full feature engineering pipeline
# ---------------------------------------------------------

def feature_engineering(df):

    df = create_date_features(df)
    df = create_cpv_features(df)
    df = create_competition_features(df)
    df = create_value_bins(df)
    df = create_complexity_features(df)
    df = create_boolean_indicators(df)
    df = create_missing_flags(df)

    df = drop_redundant_columns(df)

    return df





