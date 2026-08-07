# NLP processing

"""
NLP processing for TED CAN text fields.

Creates:
    - Combined text field (TITLE, CRIT_CRITERIA, CRIT_WEIGHTS)
    - TF-IDF representations
    - SVD (LSA) semantic components
    - NMF topic features
    - Text-based risk classifier (SVM)
    - Text risk scores and optional probabilities

Removes:
    - Raw text columns replaced by engineered NLP features
"""

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

stopWords = set(stopwords.words('english'))
punctuations = string.punctuation
nlp = spacy.load('en_core_web_sm')

# -----------------------------
# TEXT CLEANING
# -----------------------------

def text_preproceccing(df, cols):
   
    for c in cols:
        df[c] = (
            df[c]
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


# -----------------------------
# TF-IDF + SVD FEATURES
# -----------------------------
def build_tfidf_svd(texts, n_components=100):
    texts = pd.Series(texts).fillna("").astype(str)

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=5,
        max_features=20000,
        ngram_range=(1, 2),
    )
    X_tfidf = tfidf_vectorizer.fit_transform(texts)

    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd_model.fit_transform(X_tfidf)

    df_svd = pd.DataFrame(X_svd, columns=[f"NLP_SVD_{i}" for i in range(n_components)])

    return df_svd, X_tfidf, tfidf_vectorizer, svd_model


# -----------------------------
# TOPIC MODELLING (NMF)
# -----------------------------
def build_nmf_topics(tfidf_matrix, n_topics=15):
    nmf_model = NMF(n_components=n_topics, random_state=42)
    X_topics = nmf_model.fit_transform(tfidf_matrix)

    topic_cols = [f"NLP_TOPIC_{i}" for i in range(n_topics)]
    df_topics = pd.DataFrame(X_topics, columns=topic_cols)

    return df_topics, nmf_model


# -----------------------------
# TEXT RISK CLASSIFIER (SVM)
# -----------------------------
def build_text_risk_classifier(texts, labels):
    texts = pd.Series(texts).fillna("").astype(str)
    labels = pd.Series(labels).fillna("unknown").astype(str)

    tfidf = TfidfVectorizer(max_df=0.8, min_df=5, max_features=20000)
    X_tfidf = tfidf.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    svm = LinearSVC()
    svm.fit(X_tfidf, y)

    return svm, tfidf, le


def predict_text_risk(texts, svm, tfidf, label_encoder):
    X = tfidf.transform(texts)
    preds = svm.predict(X)
    return label_encoder.inverse_transform(preds)
