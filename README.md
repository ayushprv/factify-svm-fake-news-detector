# Factify — Fake News Detection with SVM

Factify is a small Streamlit app that classifies news text as likely real or fake. It uses a TF-IDF vectorizer and a **linear Support Vector Machine** (`sklearn.svm.LinearSVC`).

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`

The included compressed datasets, `Fake.csv.gz` and `True.csv.gz`, each provide the `text` column used for training.

The model trains when the app starts, then is cached for later interactions. Its held-out accuracy is shown in the sidebar.

## Project Team

This project was prepared for academic exhibition by:

| Role | Name | Registration No. |
| --- | --- | --- |
| Team Leader | Omkar Kumar | 25BAI10893 |
| Team Member | Vaidant Udawat | 25BAI10266 |
| Team Member | Anirudh Arya | 25BAI11192 |
| Team Member | Samiksha Sinha | 25BAI10556 |
| Team Member | Nyasha Kumari | 25BAI10550 |
| Team Member | Ayush Yadav | 25BAI10946 |

## Notes

The output is a model prediction, not evidence that a claim is true or false. Always fact-check consequential claims using trustworthy primary sources.
