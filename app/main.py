import re
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

import joblib
import nltk
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
MAX_MESSAGE_CHARS = 5000

nltk.data.path[:] = [str(PROJECT_ROOT / "nltk_data")]
_lemmatizer = WordNetLemmatizer()


@cache
def _english_stopwords() -> set[str]:
    return set(stopwords.words("english"))


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
    stop_words = _english_stopwords()
    tokens = [w for w in tokens if w not in stop_words]
    tagged = nltk.pos_tag(tokens)
    tokens = [_lemmatizer.lemmatize(w, _get_wordnet_pos(tag)) for w, tag in tagged]
    return " ".join(tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vectorizer = joblib.load(VECTORIZER_PATH)
    app.state.model = joblib.load(MODEL_PATH)
    # Initialize NLTK's lazy readers once before serving concurrent requests.
    preprocess_text("message")
    yield


app = FastAPI(title="SMS Spam Detection API", lifespan=lifespan)


class Message(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Enter an SMS message.")
        return text


class Prediction(BaseModel):
    label: Literal["spam", "ham"]
    spam_probability: float


class InconclusivePrediction(BaseModel):
    label: Literal["inconclusive"]
    spam_probability: None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction | InconclusivePrediction)
def predict(message: Message, request: Request) -> Prediction | InconclusivePrediction:
    vectorizer = request.app.state.vectorizer
    model = request.app.state.model
    processed = preprocess_text(message.text)
    features = vectorizer.transform([processed])
    if features.nnz == 0:
        return InconclusivePrediction(label="inconclusive", spam_probability=None)

    label = str(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    spam_probability = proba[list(model.classes_).index("spam")]

    return Prediction(label=label, spam_probability=round(float(spam_probability), 4))


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
