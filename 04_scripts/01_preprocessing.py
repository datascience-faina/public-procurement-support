# Preprocessinggit 
### Datenbereinigung, Normalisierung, Typkonvertierung


"""
1_preprocessing.py
-----------------------------------------
Funktionen zur Bereinigung und Normalisierung
der TED-Vergabedaten.
"""

import pandas as pd
import numpy as np


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Spaltennamen vereinheitlichen."""
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )
    return df


def convert_dates(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """Datumsfelder in datetime konvertieren."""
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fehlende Werte sinnvoll behandeln."""
    df = df.replace({"": np.nan, " ": np.nan})
    return df


def preprocess_ted(df: pd.DataFrame) -> pd.DataFrame:
    """Komplette Preprocessing-Pipeline."""
    df = clean_column_names(df)
    df = handle_missing_values(df)

    date_cols = ["DT_DISPATCH", "DT_AWARD"]
    df = convert_dates(df, date_cols)

    return df