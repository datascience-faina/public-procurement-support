# 5. NLP processing
### NLP für Titel, Losbeschreibung, Kriterien


"""
5_nlp_processing.py
-----------------------------------------
NLP-Pipeline für TED-Textfelder.
"""

import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
import string
import spacy
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

stopWords = set(stopwords.words('english'))
punctuations = string.punctuation
nlp = spacy.load('en_core_web_sm')


# -----------------------------
# TEXT CLEANING
# -----------------------------

def text_cleaner(sentence):
    if pd.isna(sentence):
        return ""
    doc = nlp(sentence)
    lemma_token = [token.lemma_ for token in doc if token.pos_ != 'PRON']
    no_stopWords_lemma_token = [token.lower() for token in lemma_token if token not in stopWords]
    clean_doc = [token for token in no_stopWords_lemma_token if token not in punctuations]
    joined_clean_doc = " ".join(clean_doc)
    final_doc = re.sub('[\.\s]+', ' ', joined_clean_doc)
    return final_doc

def preprocess_text(df, text_cols):
    df["TEXT_ALL"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    df["TEXT_ALL"] = df["TEXT_ALL"].apply(text_cleaner)
    return df


# -----------------------------
# TF-IDF + SVD FEATURES
# -----------------------------
def build_tfidf_svd(texts, n_components=50, max_features=20000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )
    X_tfidf = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    svd_cols = [f"NLP_SVD_{i}" for i in range(n_components)]
    df_svd = pd.DataFrame(X_svd, columns=svd_cols)

    return df_svd, tfidf, svd


# -----------------------------
# TOPIC MODELLING (NMF)
# -----------------------------
def build_nmf_topics(texts, tfidf_vectorizer, n_topics=15):
    X_tfidf = tfidf_vectorizer.transform(texts)

    nmf = NMF(n_components=n_topics, random_state=42)
    X_topics = nmf.fit_transform(X_tfidf)

    topic_cols = [f"NLP_TOPIC_{i}" for i in range(n_topics)]
    df_topics = pd.DataFrame(X_topics, columns=topic_cols)

    return df_topics, nmf


# -----------------------------
# TEXT RISK CLASSIFIER (SVM)
# -----------------------------
def build_text_risk_classifier(texts, risk_labels):
    le = LabelEncoder()
    y = le.fit_transform(risk_labels)

    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )
    X = tfidf.fit_transform(texts)

    svm = LinearSVC()
    svm.fit(X, y)

    return svm, tfidf, le


def predict_text_risk(texts, svm, tfidf, label_encoder):
    X = tfidf.transform(texts)
    preds = svm.predict(X)
    return label_encoder.inverse_transform(preds)
