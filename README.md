# Factify — Fake News Detection with SVM

Factify is a small Streamlit app that classifies news text as likely real or fake. It uses a TF-IDF vectorizer and a **linear Support Vector Machine** (`sklearn.svm.LinearSVC`).

## Run locally

1. Put the compressed datasets `Fake.csv.gz` and `True.csv.gz` in this directory. Each must include a `text` column.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`

The model trains when the app starts, then is cached for later interactions. Its held-out accuracy is shown in the sidebar.

## Notes

The output is a model prediction, not evidence that a claim is true or false. Always fact-check consequential claims using trustworthy primary sources.
