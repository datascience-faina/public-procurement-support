# Model training

"""
Training of ML-Models for Competition and Efficiency.
"""

import pandas as pd
import pickle
import json
import joblib

from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# -------------------------------------------------------------------
# 1. Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
DATA_PATH = BASE_DIR / "data" / "dataset_nlp.pkl"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "risk_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# -------------------------------------------------------------------
# Load final dataset (already after NLP)
# -------------------------------------------------------------------
with open(DATA_PATH, "rb") as f:
    df = pickle.load(f)

# -------------------------------------------------------------------
# Define target and features
# -------------------------------------------------------------------
# Adjust target column name to your actual one, e.g. "tender_failed"
TARGET_COL = "target_failed"  # <-- change to real target

y = df[TARGET_COL]

# Example: use all non-target columns as features
X = df.drop(columns=[TARGET_COL])

# If you want to restrict to 7 key parameters later for the simulator,
# you can still train on full feature set now and then analyse importance.

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# -------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------------
# Preprocessor (no NLP here, only encoding/scaling)
# -------------------------------------------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
clf.fit(X_train, y_train)

# -------------------------------------------------------------------
# 8. Evaluate
# -------------------------------------------------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "f1": float(f1_score(y_test, y_pred)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
}

print("Metrics:", metrics)

# -------------------------------------------------------------------
# Save artifacts
# -------------------------------------------------------------------
# Save full pipeline as model (preprocessor + model)
joblib.dump(clf, MODEL_PATH)

# If you want separate preprocessor for Streamlit:
joblib.dump(preprocessor, PREPROCESSOR_PATH)

# Save feature list (original columns)
feature_list = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "all_feature_cols": X.columns.tolist(),
    "target_col": TARGET_COL,
}
with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_list, f, indent=2)

# Save metrics
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
