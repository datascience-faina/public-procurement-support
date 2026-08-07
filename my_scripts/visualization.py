# Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix

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
# EDA polts
# ---------------------------------------------------------

# Target distribution: failed / risk / safe tenders

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


# Correlation matrix

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


# Top N countries

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

# plot for offers distributing

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

# plot for country rating

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

# plot for cvp division

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

# plot for prcedure rating

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

# plot for time overrating
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



# ---------------------------------------------------------
# Topics for Germany
# ---------------------------------------------------------

# PCA for German topics

def plot_pca_topics(df, save=False, name="pca_topics_de"):
    topic_cols = [c for c in df.columns if c.startswith("NLP_TOPIC_")]
    X_topics = df[topic_cols]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_topics)

    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "dominant_topic": X_topics.idxmax(axis=1)
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="dominant_topic",
        palette="tab10",
        alpha=0.6
    )
    plt.title("PCA of NMF Topics — Germany")

    if save:
        _save_fig(save, name, path = visual_de)

    plt.show()



# KMeans clustering of topics

def plot_kmeans_topics(df, n_clusters=6, save=False, name="kmeans_topics_de"):
    """
    KMeans clustering of German NMF topics.
    """
    topic_cols = [c for c in df.columns if c.startswith("NLP_TOPIC_")]
    X_topics = df[topic_cols]

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_topics)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_topics)

    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": clusters
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        alpha=0.6
    )
    plt.title(f"KMeans Clusters ({n_clusters}) — Germany")

    if save:
        _save_fig(save, name, path = visual_de)

    plt.show()


# Heatmap of topic correlations

def plot_topic_heatmap(df, save=False, name="topic_heatmap_de"):
    """
    Correlation heatmap of NMF topics for Germany.
    """
    topic_cols = [c for c in df.columns if c.startswith("NLP_TOPIC_")]
    X_topics = df[topic_cols]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        X_topics.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Correlation Heatmap of NMF Topics — Germany")

    if save:
        _save_fig(save, name, path = visual_de)

    plt.show()




# ---------------------------------------------------------
# Model Evaluation Plots
# ---------------------------------------------------------

def plot_roc_curve(model, features_test, target_test, save=False):
    """
    ROC curve for binary classifier.
    """
    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, features_test, target_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    _save_fig(save, "model_roc_curve")
    plt.show()


def plot_pr_curve(model, features_test, target_test, save=False):
    """
    Precision-Recall curve.
    """
    plt.figure(figsize=(7, 5))
    PrecisionRecallDisplay.from_estimator(model, features_test, target_test)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    _save_fig(save, "model_pr_curve")
    plt.show()


def plot_confusion_matrix(model, features_test, target_test, save=False):
    """
    Confusion matrix heatmap.
    """
    target_pred = model.predict(features_test)
    cm = confusion_matrix(target_test, target_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    _save_fig(save, "model_confusion_matrix")
    plt.show()


def plot_feature_importance(model, feature_names, save=False):
    """
    Feature importance for tree-based models.
    """
    importances = model.named_steps["model"].feature_importances_

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=importances,
        y=feature_names,
        orient="h"
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    _save_fig(save, "model_feature_importance")
    plt.show()