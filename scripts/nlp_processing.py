# 5. NLP processing
### NLP für Titel, Losbeschreibung, Kriterien


"""
5_nlp_processing.py
-----------------------------------------
NLP-Pipeline für TED-Textfelder.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(text_series):
    """TF-IDF-Vektorisierung."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    X = vectorizer.fit_transform(text_series.fillna(""))
    return X, vectorizer