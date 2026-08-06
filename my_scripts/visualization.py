# Visualization
### Visualisierungen


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# seaborn style
sns.set_theme(style="whitegrid")

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
# Target distribution: failed / risk / safe tenders
# ---------------------------------------------------------

def plot_target_distribution(df, save=False, path=visual_eu):
    """
    Plots a pie chart showing the distribution of tenders:
    - failed (0–1 offers)
    - risk zone (2 offers)
    - safe (>=3 offers)
    Labels include both percentage and absolute counts.
    """

    # classify tenders
    failed = (df["NUMBER_OFFERS"].fillna(0) <= 1).sum()
    risk = (df["NUMBER_OFFERS"] == 2).sum()
    safe = (df["NUMBER_OFFERS"] >= 3).sum()

    labels = ["Failed (0–1)", "Risk (2)", "Safe (>=3)"]
    values = [failed, risk, safe]
    colors = ["#D62728", "#FFBF00", "#2CA02C"]

    total = sum(values)
    percentages = [v / total * 100 for v in values]

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

    filename = "target_distribution_pie"

    _save_fig(save, filename, path)
    plt.show()


# ---------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------

def plot_correlation_matrix(df, cols=None, save=False, path=visual_eu):
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

    _save_fig(save, "correlation_matrix", path)
    plt.show()



# ---------------------------------------------------------
# Top N countries
# ---------------------------------------------------------

def plot_bar_top_categories(df, col, top_n=20, save=False, path=visual_eu):
    """Bar chart of top N categories."""

    counts = df[col].value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Top {top_n} categories in {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()

    _save_fig(save, f"bar_top_{col}", path)
    plt.show()

# ---------------------------------------------------------
# Competition EDA plots
# ---------------------------------------------------------

def plot_offers_bin_distribution(df, col="OFFERS_BIN", save=False, path=visual_eu):
    """
    Plot distribution of competition levels (OFFERS_BIN) as percentages.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], stat="percent")
    plt.title("Distribution of Competition Levels")
    plt.xlabel("Competition Category")
    plt.ylabel("Percentage")
    plt.tight_layout()

    _save_fig(save, "competition_offers_bin", path)
    plt.show()


def plot_failed_rate_by_country(df, country_col="ISO_COUNTRY_CODE",
                                failed_col="IS_FAILED_TENDER",
                                top_n=15, save=False, path=visual_eu):
    """
    Plot failed tender rate by country (top_n countries).
    """
    country_comp = (
        df.groupby(country_col)[failed_col]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    country_comp.head(top_n).plot(kind="bar")
    plt.title(f"Failed Tender Rate by Country (Top {top_n})")
    plt.ylabel("Share of Failed Tenders")
    plt.xlabel("Country")
    plt.tight_layout()

    _save_fig(save, "competition_failed_by_country", path)
    plt.show()


def plot_failed_rate_by_cpv_division(df, cpv_col="CPV_DIVISION",
                                     failed_col="IS_FAILED_TENDER",
                                     top_n=20, save=False, path=visual_eu):
    """
    Plot failed tender rate by CPV division (top_n divisions).
    """
    cpv_comp = (
        df.groupby(cpv_col)[failed_col]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    cpv_comp.head(top_n).plot(kind="bar")
    plt.title(f"Failed Tender Rate by CPV Division (Top {top_n})")
    plt.ylabel("Share of Failed Tenders")
    plt.xlabel("CPV Division")
    plt.tight_layout()

    _save_fig(save, "competition_failed_by_cpv", path)
    plt.show()


def plot_failed_rate_by_procedure(df, proc_col="TYPE_OF_CONTRACT",
                                  failed_col="IS_FAILED_TENDER",
                                  save=False, path=visual_eu):
    """
    Plot failed tender rate by procedure type.
    """
    proc_comp = (
        df.groupby(proc_col)[failed_col]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    proc_comp.plot(kind="bar")
    plt.title("Failed Tender Rate by Procedure Type")
    plt.ylabel("Share of Failed Tenders")
    plt.xlabel("Procedure Type")
    plt.tight_layout()

    _save_fig(save, "competition_failed_by_procedure", path)
    plt.show()


def plot_failed_rate_over_time(df, year_col="AWARD_YEAR",
                               failed_col="IS_FAILED_TENDER",
                               save=False, path=visual_eu):
    """
    Plot failed tender rate over time (by award year).
    """
    year_comp = (
        df.groupby(year_col)[failed_col]
        .mean()
    )

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=year_comp.index, y=year_comp.values)
    plt.title("Failed Tender Rate Over Time")
    plt.ylabel("Share of Failed Tenders")
    plt.xlabel("Year")
    plt.tight_layout()

    _save_fig(save, "competition_failed_over_time", path)
    plt.show()
