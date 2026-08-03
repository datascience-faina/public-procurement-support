# 7. Visualization
### Visualisierungen


"""
7_visualization.py
-----------------------------------------
Plot-Funktionen für Wettbewerb, Effizienz und Digitalisierung.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_competition(df):
    sns.histplot(df["NUMBER_OFFERS"], bins=40)
    plt.title("Verteilung der Anzahl der Angebote")
    plt.show()