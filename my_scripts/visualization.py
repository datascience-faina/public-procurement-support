# Visualization
### Visualisierungen


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# Global save path (set once)
# ---------------------------------------------------------

VISUALS_DIR = Path("../visuals")


# ---------------------------------------------------------
# Helper: save figure if requested
# ---------------------------------------------------------

def _save_fig(save, name):
    """
    Saves the current matplotlib figure if save=True.
    Saves into the global VISUALS_DIR folder.
    """
    if save:
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(VISUALS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------
# 1. Histogram plots for numeric columns
# ---------------------------------------------------------

def plot_histograms(df, cols, bins=50, save=False):
    """
    Plots histograms for selected numeric columns.
    Useful for understanding distributions and skewness.
    """
    for col in cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()

        _save_fig(save, f"hist_{col}")
        plt.show()


# ---------------------------------------------------------
# 2. Bar chart for top categories
# ---------------------------------------------------------

def plot_bar_top_categories(df, col, top_n=20, save=False):
    """
    Plots a bar chart of the top N most frequent categories.
    Useful for categorical EDA.
    """
    counts = df[col].value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Top {top_n} categories in {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()

    _save_fig(save, f"bar_top_{col}")
    plt.show()


# ---------------------------------------------------------
# 3. Correlation matrix
# ---------------------------------------------------------

def plot_correlation_matrix(df, cols=None, save=False):
    """
    Plots a correlation matrix for numeric columns.
    If cols is provided, only those columns are used.
    """
    if cols:
        df_corr = df[cols].corr()
    else:
        df_corr = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()

    _save_fig(save, "correlation_matrix")
    plt.show()


# ---------------------------------------------------------
# 4. Target distribution: failed / risk / safe tenders
# ---------------------------------------------------------

def plot_target_distribution(df, save=False):
    """
    Plots a pie chart showing the distribution of tenders:
    - failed (0–1 offers)   - red
    - risk zone (2 offers)  - yellow/orange
    - safe (>=3 offers)     - green
    Labels include both percentage and absolute counts.
    """

    # classify tenders
    failed = (df["NUMBER_OFFERS"].fillna(0) <= 1).sum()
    risk = (df["NUMBER_OFFERS"] == 2).sum()
    safe = (df["NUMBER_OFFERS"] >= 3).sum()

    labels = ["Failed (0–1)", "Risk (2)", "Safe (>=3)"]
    values = [failed, risk, safe]
    colors = ["#D62728", "#FFBF00", "#2CA02C"]   # red, yellow, green

    total = sum(values)
    percentages = [v / total * 100 for v in values]

    # labels with % and counts
    pie_labels = [
        f"{labels[i]}: {percentages[i]:.1f}%\n({values[i]:,} tenders)"
        for i in range(3)
    ]

    plt.figure(figsize=(8, 8))
    plt.pie(
        values,
        labels=pie_labels,
        colors=colors,
        autopct=None,
        startangle=90,
        textprops={"fontsize": 12}
    )
    plt.title("Tender Outcome Distribution", fontsize=14)
    plt.tight_layout()

    _save_fig(save, "target_distribution_pie")
    plt.show()
