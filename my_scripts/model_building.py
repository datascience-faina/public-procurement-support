# Model building & evaluation

"""
Training of ML-Models for Competition and Efficiency.
"""
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Categorical features encoding
# ---------------------------------------------------------

def fit_encoder(df):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    encoder.fit(df[categorical_cols])
    return encoder


def transform_with_encoder(df, encoder):

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # numeric
    df_num = df[numeric_cols]

    # categorical
    df_ohe = encoder.transform(df[categorical_cols])
    ohe_cols = encoder.get_feature_names_out(categorical_cols)

    df_cat = pd.DataFrame(df_ohe, columns=ohe_cols)

    # final
    df_encoded = pd.concat([df_num, df_cat], axis=1)
    return df_encoded


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model(model, features_train, features_test, target_train, target_test):
    """
    Train model inside a pipeline with StandardScaler.
    """

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(features_train, target_train)

    preds = pipe.predict(features_test)
    proba = pipe.predict_proba(features_test)[:, 1]

    print("\n=== Model:", model.__class__.__name__, "===")
    print("F1:", f1_score(target_test, preds))
    print("ROC-AUC:", roc_auc_score(target_test, proba))
    print(classification_report(target_test, preds))

    return pipe



# ---------------------------------------------------------
# Grid Search
# ---------------------------------------------------------

def tune_logreg(features_train, target_train):
    """
    Hyperparameter tuning for Logistic Regression.
    """

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid.fit(features_train, target_train)

    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)

    return grid.best_estimator_
