# Feature engineering

"""
Feature Engineering for TED CAN dataset.

Creates:
    - Date-based features
    - CPV maping with 12 group
    - Boolean indicators
    - Missing-information flags
Replace:
    - NaN numeric with mediane
Removes:
    - Raw columns replaced by engineered features
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Target features
# ---------------------------------------------------------

def create_target(df):
    if "NUMBER_OFFERS" in df.columns:
        df["IS_FAILED_TENDER"] = (df["NUMBER_OFFERS"] <= 1).astype(int)

    return df

# ---------------------------------------------------------
# Date features
# ---------------------------------------------------------

def create_date_features(df):
    if "DT_AWARD" in df.columns:
        df["AWARD_MONTH"] = df["DT_AWARD"].dt.month
        df["AWARD_QUARTER"] = df["DT_AWARD"].dt.quarter

    if "DT_DISPATCH" in df.columns and "DT_AWARD" in df.columns:
        df["DAYS_TO_AWARD"] = (df["DT_AWARD"] - df["DT_DISPATCH"]).dt.days

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
    for col in ["VALUE_EURO", "AWARD_VALUE_EURO"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

    return df

# ---------------------------------------------------------
# Rebuild CVP into 12 group
# ---------------------------------------------------------

def map_cpv_division_to_category(df):
    CPV_MAP = {
        "Construction & Building Works": list(range(45, 47)),   # 45–46
        "Machinery & Industrial Equipment": list(range(30, 33)),  # 30–32
        "Electrical, Optical & Precision Equipment": list(range(33, 35)),  # 33–34
        "IT, Software & Telecommunications": list(range(48, 51)),  # 48–50
        "Transport Services & Logistics": list(range(60, 64)),  # 60–63
        "Business, Consulting & Administrative Services": list(range(79, 81)),  # 79–80
        "Cleaning, Facility & Maintenance Services": list(range(90, 92)),  # 90–91
        "Energy, Utilities & Environmental Services": list(range(92, 100)),  # 92–99
    }

    def _map_single_value(cpv_div):
        try:
            cpv = int(cpv_div)
        except:
            return "Other"   # Unknown or invalid CPV

        for category, values in CPV_MAP.items():
            if cpv in values:
                return category

        return "Other"       # CPV not in any defined range

    df["CPV_CATEGORY"] = df["MAIN_CPV_CODE_GPA"].apply(_map_single_value)
    return df


# ---------------------------------------------------------
# NaN in numerical replace with median
# ---------------------------------------------------------

def nan_numeric_median(df):
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns

    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df


# ---------------------------------------------------------
# Remove raw columns replaced by engineered features
# ---------------------------------------------------------

def drop_redundant_columns(df):
    cols_to_drop = [
        "DT_AWARD",
        "DT_DISPATCH",
        "MAIN_CPV_CODE_GPA"
    ]

    existing = [c for c in cols_to_drop if c in df.columns]

    # Drop redundant columns 
    df = df.drop(columns=existing).reset_index(drop=True)

    return df

# ---------------------------------------------------------
# Full feature engineering pipeline
# ---------------------------------------------------------

def feature_engineering(df):
    df = create_target(df)
    df = create_date_features(df)
    df = create_boolean_indicators(df)
    df = create_missing_flags(df)
    df = map_cpv_division_to_category(df)
    df = nan_numeric_median(df)
    df = drop_redundant_columns(df)

    return df
    


