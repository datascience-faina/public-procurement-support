# Outliers

"""
Outlier analysis utilities.

Includes:
    - Log-transform
    - IQR outliers
    - MAD outliers
    - Histogram and boxplot helpers (with saving)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Global save path (set once)
# ---------------------------------------------------------

visual_eu = Path("../visuals")
visual_de = Path("../germany_deep_dive")


# ---------------------------------------------------------
# Helper: save figure if requested
# ---------------------------------------------------------

def _save_fig(save, name, path = visual_eu):
    """
    Saves the current matplotlib figure if save=True.
    Default save location: ../visuals.
    Custom save path can be provided (e.g., for Germany deep dive).
    """
    if save:
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{name}.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------
# 1. Log-transform
# ---------------------------------------------------------

def log_transform(df, cols):
    """Apply log1p transform to selected columns."""
    return df[cols].apply(lambda x: np.log1p(x))


# ---------------------------------------------------------
# 2. IQR outliers
# ---------------------------------------------------------

def iqr_bounds(series, factor=1.5):
    """Compute lower and upper bounds using IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - factor * IQR, Q3 + factor * IQR


def iqr_outliers(series, factor=1.5):
    """Return outlier values using IQR rule."""
    lower, upper = iqr_bounds(series, factor)
    return series[(series < lower) | (series > upper)]


# ---------------------------------------------------------
# 3. MAD outliers
# ---------------------------------------------------------

def mad_outliers(series, threshold=3.5):
    """Return outlier values using MAD rule."""
    series = series.dropna()
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    modified_z = 0.6745 * (series - median) / mad
    return series[np.abs(modified_z) > threshold]


# ---------------------------------------------------------
# 4. Visualization helpers (histogram + boxplot)
# ---------------------------------------------------------

def plot_hist(series, title, save=False, path = visual_eu):
    """Histogram with KDE."""
    plt.figure(figsize=(8, 4))
    sns.histplot(series, bins=50, kde=True)
    plt.title(title)
    plt.tight_layout()

    filename = f"hist_{title.replace(' ', '_')}"

    _save_fig(save, filename, path)
    plt.show()


def plot_box(series, title, save=False, path = visual_eu):
    """Boxplot for numeric series."""
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=series)
    plt.title(title)
    plt.tight_layout()

    filename = f"box_{title.replace(' ', '_')}"

    _save_fig(save, filename, path)
    plt.show()
