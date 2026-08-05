"""
Outlier detection utilities for procurement dataset.

This module provides:
    - IQR-based outlier detection (1.5 IQR and 3 IQR)
    - MAD-based outlier detection (robust for financial variables)
    - Log-transform helper for skewed financial data
    - Simple plotting helpers for histograms and boxplots

All functions are designed to be imported and used inside Jupyter notebooks.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Log-transform
# ---------------------------------------------------------

def log_transform(df, cols):
    """
    Apply natural logarithm transform to selected columns.

    VARs:
        df: pandas DataFrame
        cols: list of column names to transform

    RETURNS:
        DataFrame with log-transformed columns using log1p (safe for zeros)

    NOTES:
        Financial variables are typically right-skewed.
        Log-transform reduces skewness and makes distributions more normal-like.
    """
    return df[cols].apply(lambda x: np.log1p(x))


# ---------------------------------------------------------
# IQR outliers
# ---------------------------------------------------------

def iqr_bounds(series, factor=1.5):
    """
    Compute lower and upper bounds for outlier detection using IQR.

    VARs:
        series: pandas Series (numeric)
        factor: multiplier for IQR (1.5 = strict, 3 = relaxed)

    RETURNS:
        (lower_bound, upper_bound)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return lower, upper


def iqr_outliers(series, factor=1.5):
    """
    Identify outliers using the IQR rule.

    VARs:
        series: pandas Series
        factor: IQR multiplier (1.5 or 3)

    RETURNS:
        Series containing only outlier values
    """
    lower, upper = iqr_bounds(series, factor)
    return series[(series < lower) | (series > upper)]


# ---------------------------------------------------------
# MAD outliers
# ---------------------------------------------------------

def mad_outliers(series, threshold=3.5):
    """
    Identify outliers using Median Absolute Deviation (MAD).

    VARs:
        series: pandas Series (numeric)
        threshold: modified Z-score threshold (default 3.5)

    RETURNS:
        Series containing outlier values

    NOTES:
        MAD is robust for financial variables with heavy tails.
        Recommended for VALUE_EURO and AWARD_VALUE_EURO.
    """
    series = series.dropna()
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    modified_z = 0.6745 * (series - median) / mad
    return series[np.abs(modified_z) > threshold]


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------

def plot_hist(series, title):
    """
    Plot histogram with KDE for a numeric Series.

    VARs:
        series: pandas Series
        title: plot title (string)

    RETURNS:
        None (displays plot)
    """
    plt.figure(figsize=(8,4))
    sns.histplot(series, bins=50, kde=True)
    plt.title(title)
    plt.show()


def plot_box(series, title):
    """
    Plot boxplot for a numeric Series.

    VARs:
        series: pandas Series
        title: plot title

    RETURNS:
        None (displays plot)
    """
    plt.figure(figsize=(8,4))
    sns.boxplot(x=series)
    plt.title(title)
    plt.show()


def plot_box_by(df, x, y, title):
    """
    Plot boxplot of y grouped by categorical variable x.

    VARs:
        df: pandas DataFrame
        x: categorical column name
        y: numeric column name
        title: plot title

    RETURNS:
        None (displays plot)
    """
    plt.figure(figsize=(12,5))
    sns.boxplot(data=df, x=x, y=y)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()
