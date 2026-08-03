# 3. Model training
### Training von ML‑Modellen


"""
3_model_training.py
-----------------------------------------
Training von ML-Modellen für Wettbewerb und Effizienz.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_competition_model(df: pd.DataFrame):
    """Modell zur Vorhersage der Wettbewerbsintensität."""
    features = ["LOTS_NUMBER", "TYPE_OF_CONTRACT", "VALUE_EURO"]
    df = df.dropna(subset=features + ["NUMBER_OFFERS"])

    X = df[features]
    y = df["NUMBER_OFFERS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    return model, X_test, y_test