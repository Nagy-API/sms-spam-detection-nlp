"""Regenerate the original model artifacts as an explicit offline task."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from app.main import MODEL_PATH, PROJECT_ROOT, VECTORIZER_PATH, preprocess_text


def train_and_save_model():
    data_path = PROJECT_ROOT / "data" / "spam.csv"
    df = pd.read_csv(data_path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df.drop_duplicates(inplace=True)

    df["processed"] = df["text"].apply(preprocess_text)

    X_train_text, _, y_train, _ = train_test_split(
        df["processed"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()
