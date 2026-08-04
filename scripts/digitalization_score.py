# 6. Digitalization Score
### AI‑Readiness Score für Vergabeverfahren


"""
6_digitalization_score.py
-----------------------------------------
Berechnung eines Digitalisierungs- und KI-Reifegrades.
"""

import pandas as pd


def digitalization_score(df: pd.DataFrame) -> pd.Series:
    """Einfacher Digitalisierungsindex."""
    return (
        df["B_ELECTRONIC_AUCTION"].fillna(0)
        + df["NUMBER_OFFERS_ELECTR"].fillna(0)
        + df["B_DYN_PURCH_SYST"].fillna(0)
    )