# Feature engineering

"""
Feature Engineering for TED CAN dataset.

Creates:
    - Date-based features
    - CPV_CATEGORY (from CPV)
    - Reconstructed CRIT_PRICE_WEIGHT (from CRIT_WEIGHTS)
    - CPV 12-group mapping
    - Boolean indicators
    - Missing-information flags
    - CAE_TYPE_CATEGORY
    - Target features

Replace:
    - NaN numeric with median

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

    # Award quarter
    if "DT_AWARD" in df.columns:
        df["AWARD_QUARTER"] = df["DT_AWARD"].dt.quarter

    # Days to award (with logical constraints)
    if "DT_DISPATCH" in df.columns and "DT_AWARD" in df.columns:

        # raw difference
        df["DAYS_TO_AWARD"] = (df["DT_DISPATCH"] - df["DT_AWARD"]).dt.days

        # indicator for missing/unrealistic values
        df["DAYS_TO_AWARD_MISSING"] = 0

        # unrealistic negative values
        neg_mask = df["DAYS_TO_AWARD"] < 0

        # unrealistic long durations (> 5 years)
        long_mask = df["DAYS_TO_AWARD"] > 1825   # 5 years = 1825 days

        # mark missing
        df.loc[neg_mask | long_mask, "DAYS_TO_AWARD_MISSING"] = 1

        # set unrealistic values to NaN
        df.loc[neg_mask | long_mask, "DAYS_TO_AWARD"] = np.nan

    return df


# ---------------------------------------------------------
# Missing-information flags
# ---------------------------------------------------------

def create_missing_flags(df):
    df["CRIT_PRICE_WEIGHT_MISSING"] = df["CRIT_PRICE_WEIGHT"].isna().astype(int)
    return df


# ---------------------------------------------------------
# CPV CATEGORY reconstruction (from CPV)
# ---------------------------------------------------------

def create_cpv_category(df):
    df = df[df["CPV"].notna()]
    df["CPV_CATEGORY"] = df["CPV"].str[:2]
    return df


# ---------------------------------------------------------
# CRIT_PRICE_WEIGHT reconstruction
# ---------------------------------------------------------

def reconstruct_crit_price_weight(df):

    # Convert to numeric
    df["CRIT_PRICE_WEIGHT"] = pd.to_numeric(df["CRIT_PRICE_WEIGHT"], errors="coerce")

    # Reconstruct from CRIT_WEIGHTS
    if "CRIT_WEIGHTS" in df.columns:
        extracted = df["CRIT_WEIGHTS"].str.extract(r"^(\d{1,3})")
        extracted = pd.to_numeric(extracted[0], errors="coerce")
        df.loc[df["CRIT_PRICE_WEIGHT"].isna(), "CRIT_PRICE_WEIGHT"] = extracted

    # Remove unrealistic values
    df = df[df["CRIT_PRICE_WEIGHT"].isna() | (df["CRIT_PRICE_WEIGHT"] <= 100)]

    return df


# ---------------------------------------------------------
# CPV 12-group mapping (based on CPV_CATEGORY)
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
            return "Other"

        for category, values in CPV_MAP.items():
            if cpv in values:
                return category

        return "Other"

    df["CPV_GROUP"] = df["CPV_CATEGORY"].apply(_map_single_value)
    return df


# ---------------------------------------------------------
# CAE_TYPE mapping
# ---------------------------------------------------------

def map_cae_type(df):
    mapping = {
        "1": "Ministry",
        "3": "National Agency",
        "4": "Regional/Local Authority",
        "5": "Regional/Local Agency",
        "5A": "Local Agency",
        "6": "Public Body",
        "8": "Public Undertaking",
        "N": "Not Specified",
        "R": "Other Public Authority",
        "Z": "Unknown"
    }

    df["CAE_CATEGORY"] = (
        df["CAE_TYPE"]
        .astype(str)
        .str.strip()
        .map(mapping)
        .fillna("Unknown")
    )

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
        "DT_AWARD", "DT_DISPATCH",
        "MAIN_CPV_CODE_GPA", "CAE_TYPE",
        "CRIT_WEIGHTS", "CPV", "CPV_CATEGORY"
    ]

    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing).reset_index(drop=True)

    return df

# ---------------------------------------------------------
# Memory optimisation
# ---------------------------------------------------------

# round and convert float into int
def optimize_float_columns(df, cols_to_int):
    for col in cols_to_int:
        df[col] = df[col].round().astype("int32")

    return df


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
# Full feature engineering pipeline
# ---------------------------------------------------------

def feature_engineering(df):

    df = create_target(df)
    df = create_date_features(df)
    df = create_missing_flags(df)

    df = create_cpv_category(df)
    df = reconstruct_crit_price_weight(df)
    df = map_cpv_division_to_category(df)
    df = map_cae_type(df)

    df = nan_numeric_median(df)

    cols_to_int = ["CRIT_PRICE_WEIGHT", "NUMBER_OFFERS", "DAYS_TO_AWARD", "AWARD_QUARTER"]
    df = optimize_float_columns(df, cols_to_int)

    df = optimize_int_columns(df)


    df = drop_redundant_columns(df)

    return df



# ---------------------------------------------------------
# Maping for Topics (NLP)
# ---------------------------------------------------------

def create_topic_id(df):
    topic_cols = [f"NLP_TOPIC_{i}" for i in range(15)]
    df["TOPIC_ID"] = df[topic_cols].idxmax(axis=1).str.replace("NLP_TOPIC_", "").astype(int)
    return df

def map_topic_name(df):
    topic_mapping = {
        0: "Lot / Position Structure",
        1: "Pharmaceuticals (General, Oncology, Psychotropic)",
        2: "Works, Services and Supplies",
        3: "Tasks and Lots",
        4: "Procurement Packages",
        5: "Product / Service Groups",
        6: "Medical Devices and Pharmaceuticals",
        7: "Lot Selection",
        8: "Public Utility / Municipal Services",
        9: "Cardiovascular & Neurological Medicines",
        10: "Procurement Parts / Segments",
        11: "Equipment, Services, Framework Agreements",
        12: "Laboratory Materials and Reagents",
        13: "Maintenance, Risks, Technical Services",
        14: "Technical / IT / System Services"
    }
    
    df["TOPIC_NAME"] = df["TOPIC_ID"].map(topic_mapping).fillna("Unknown")
    return df
