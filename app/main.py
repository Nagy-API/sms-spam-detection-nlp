import os
import re

import joblib
import nltk
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = "../data/spam.csv"
VECTORIZER_PATH = "../models/vectorizer.pkl"
MODEL_PATH = "../models/model.pkl"

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text.strip()


def _get_wordnet_pos(tag: str) -> str:
    if tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    tokens = nltk.word_tokenize(cleaned)
    tokens = [w for w in tokens if w not in _stop_words]
    tagged = nltk.pos_tag(tokens)
    tokens = [_lemmatizer.lemmatize(w, _get_wordnet_pos(tag)) for w, tag in tagged]
    return " ".join(tokens)


def train_and_save_model():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
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

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    return vectorizer, model


if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
else:
    vectorizer, model = train_and_save_model()


app = FastAPI(title="SMS Spam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    text: str


class Prediction(BaseModel):
    label: str
    spam_probability: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(message: Message):
    processed = preprocess_text(message.text)
    features = vectorizer.transform([processed])

    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    spam_probability = proba[list(model.classes_).index("spam")]

    return Prediction(label=label, spam_probability=round(float(spam_probability), 4))
