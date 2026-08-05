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
# Global save path
# ---------------------------------------------------------

VISUALS_DIR = Path("../visuals")


def _save_fig(save, name):
    """Save figure to ../visuals if save=True."""
    if save:
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(VISUALS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")


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

def plot_hist(series, title, save=False, suffix=""):
    """Histogram with KDE."""
    plt.figure(figsize=(8, 4))
    sns.histplot(series, bins=50, kde=True)
    plt.title(title)
    plt.tight_layout()

    filename = f"hist_{title.replace(' ', '_')}"
    if suffix:
        filename += f"_{suffix}"

    _save_fig(save, filename)
    plt.show()


def plot_box(series, title, save=False, suffix=""):
    """Boxplot for numeric series."""
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=series)
    plt.title(title)
    plt.tight_layout()

    filename = f"box_{title.replace(' ', '_')}"
    if suffix:
        filename += f"_{suffix}"

    _save_fig(save, filename)
    plt.show()
