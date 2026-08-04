# 2. Feature engineering
### Neue Merkmale: Wettbewerb, Effizienz, Digitalisierung


"""
2_feature_engineering.py
-----------------------------------------
Erzeugt neue Merkmale für Wettbewerb, Effizienz
und Digitalisierungsanalyse.
"""

import pandas as pd


def competition_index(df: pd.DataFrame) -> pd.Series:
    """Einfacher Wettbewerbsindex."""
    return (
        df["NUMBER_OFFERS"].fillna(0)
        + df["NUMBER_TENDERS_SME"].fillna(0)
        + df["NUMBER_TENDERS_OTHER_EU"].fillna(0)
    )


def efficiency_gap(df: pd.DataFrame) -> pd.Series:
    """Differenz zwischen geschätztem und tatsächlichem Wert."""
    return (
        df["AWARD_EST_VALUE_EURO"].fillna(0)
        - df["AWARD_VALUE_EURO"].fillna(0)
    )


def sme_participation_rate(df: pd.DataFrame) -> pd.Series:
    """Anteil der KMU-Angebote."""
    return (
        df["NUMBER_TENDERS_SME"].fillna(0)
        / df["NUMBER_OFFERS"].replace(0, pd.NA)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alle Feature-Engineering-Schritte."""
    df["COMPETITION_INDEX"] = competition_index(df)
    df["EFFICIENCY_GAP"] = efficiency_gap(df)
    df["SME_PARTICIPATION_RATE"] = sme_participation_rate(df)

    return df