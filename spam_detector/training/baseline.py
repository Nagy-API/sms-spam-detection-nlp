"""The original (v1) system, reproduced faithfully so it can be evaluated under the new protocol.

Recipe, unchanged from ``app/main.py`` / ``app/train.py`` at commit 5efa6a4:

1. remove ``http\\S+``; replace every non-ASCII-letter with a space; lowercase
2. NLTK ``word_tokenize``; drop NLTK English stop words
3. POS-tag with NLTK's perceptron tagger and lemmatize with WordNet
4. ``CountVectorizer(ngram_range=(1, 2))`` -> ``MultinomialNB()`` (alpha = 1)

Requires NLTK and its data (``python scripts/prepare_nltk.py``). It is used
only for offline comparison; the production API no longer depends on NLTK.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@cache
def _nltk():  # imported lazily so the rest of the training code works without NLTK
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    nltk.data.path[:] = [str(PROJECT_ROOT / "nltk_data")]
    return nltk, set(stopwords.words("english")), wordnet, WordNetLemmatizer()


@cache
def legacy_preprocess(text: str) -> str:
    nltk, stop_words, wordnet, lemmatizer = _nltk()
    cleaned = re.sub(r"[^a-zA-Z]", " ", re.sub(r"http\S+", "", text)).lower().strip()
    tokens = [w for w in nltk.word_tokenize(cleaned) if w not in stop_words]

    def pos(tag: str) -> str:
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    return " ".join(lemmatizer.lemmatize(w, pos(t)) for w, t in nltk.pos_tag(tokens))


@dataclass
class LegacyBaseline:
    vectorizer: CountVectorizer
    model: MultinomialNB

    @classmethod
    def fit(cls, texts: Sequence[str], labels: Sequence[int]) -> LegacyBaseline:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x = vectorizer.fit_transform([legacy_preprocess(t) for t in texts])
        return cls(vectorizer, MultinomialNB().fit(x, np.asarray(labels)))

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        """Log-odds log P(spam|x) - log P(ham|x)."""
        x = self.vectorizer.transform([legacy_preprocess(t) for t in texts])
        joint = self.model.predict_joint_log_proba(x)
        return joint[:, 1] - joint[:, 0]
