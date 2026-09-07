"""Factify: an SVM-based fake-news text classifier."""

from pathlib import Path
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_DIR = Path(__file__).parent


def clean_text(text: str) -> str:
    """Normalize text while retaining word boundaries for TF-IDF."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Training the SVM model…")
def train_model() -> tuple[Pipeline, float]:
    fake = pd.read_csv(DATA_DIR / "Fake.csv.gz")
    real = pd.read_csv(DATA_DIR / "True.csv.gz")

    fake["label"] = 0
    real["label"] = 1
    data = pd.concat([fake, real], ignore_index=True)
    data = data.dropna(subset=["text"])

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42,
        stratify=data["label"],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                                  max_df=0.7, ngram_range=(1, 2))),
        ("svm", LinearSVC(class_weight="balanced", random_state=42)),
    ])
    model.fit(x_train, y_train)
    return model, model.score(x_test, y_test)


st.set_page_config(page_title="Factify", page_icon="📰")
st.title("📰 Factify")
st.caption("An SVM-powered fake-news detector")

if not (DATA_DIR / "Fake.csv.gz").exists() or not (DATA_DIR / "True.csv.gz").exists():
    st.error("Dataset files are missing. Add `Fake.csv.gz` and `True.csv.gz` beside app.py.")
    st.stop()

model, accuracy = train_model()
st.sidebar.metric("Held-out accuracy", f"{accuracy:.1%}")

article = st.text_area("Paste a news article or claim", height=220,
                       placeholder="Enter text to check…")
if st.button("Check news", type="primary"):
    if not article.strip():
        st.warning("Please paste some text first.")
    else:
        prediction = model.predict([article])[0]
        if prediction == 1:
            st.success("Likely real news")
        else:
            st.error("Likely fake news")
        st.caption("This is a statistical classification, not a fact-check. Verify important claims with reliable sources.")
