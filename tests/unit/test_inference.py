import math
from dataclasses import replace

import pytest

from spam_detector.artifact import AbstentionPolicy
from spam_detector.inference import SpamClassifier
from tests.conftest import build_toy_artifact


@pytest.fixture
def classifier() -> SpamClassifier:
    return SpamClassifier(build_toy_artifact())


def test_spam_and_ham(classifier: SpamClassifier) -> None:
    spam = classifier.predict("WIN a FREE prize, claim now")
    ham = classifier.predict("see you at lunch tomorrow")
    assert spam.label == "spam" and spam.spam_probability > 0.5
    assert ham.label == "ham" and ham.spam_probability < 0.5
    assert spam.reason is None and spam.recognized_features > 0


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("12345", "no_letters"),
        ("🙂 !!! ...", "no_letters"),
        ("مرحبا كيف حالك", "unsupported_script"),
        ("Привет как дела", "unsupported_script"),
        ("zzzzqqq", "no_recognized_features"),
    ],
)
def test_abstains_with_a_reason(classifier: SpamClassifier, text: str, reason: str) -> None:
    result = classifier.predict(text)
    assert (result.label, result.reason, result.spam_probability) == ("inconclusive", reason, None)


def test_full_width_latin_is_supported(classifier: SpamClassifier) -> None:
    assert classifier.predict("ＦＲＥＥ ＰＲＩＺＥ").label == "spam"


def test_mostly_english_with_some_other_script_is_still_predicted(classifier: SpamClassifier) -> None:
    assert classifier.predict("win a free prize now مرحبا").label == "spam"


def test_threshold_is_applied_to_calibrated_probability() -> None:
    text = "win lunch"
    base = SpamClassifier(build_toy_artifact(threshold=0.5)).predict(text)
    assert 0.5 < base.spam_probability < 0.99
    strict = SpamClassifier(build_toy_artifact(threshold=0.99)).predict(text)
    assert strict.spam_probability == base.spam_probability
    assert (base.label, strict.label) == ("spam", "ham")


def test_probability_is_platt_of_score(classifier: SpamClassifier) -> None:
    result = classifier.predict("win a free prize")
    a, b = classifier.artifact.calibration.a, classifier.artifact.calibration.b
    assert result.spam_probability == pytest.approx(1 / (1 + math.exp(-(a * result.score + b))))


def test_known_word_rule_only_applies_when_enabled() -> None:
    artifact = build_toy_artifact()
    lenient = SpamClassifier(replace(artifact, abstention=AbstentionPolicy(0.5, require_known_word=False)))
    strict = SpamClassifier(replace(artifact, abstention=AbstentionPolicy(0.5, require_known_word=True)))
    # The toy model has only a word block, so unknown words give an empty vector either way;
    # both must abstain, and a known word must be predicted by both.
    assert lenient.predict("zzzzqqq").label == strict.predict("zzzzqqq").label == "inconclusive"
    assert strict.predict("prize").label == lenient.predict("prize").label == "spam"


def test_explanation_sums_to_the_score(classifier: SpamClassifier) -> None:
    text = "win a free prize, claim now"
    contributions = classifier.explain(text, top_k=1000)
    total = classifier.artifact.intercept + sum(c.weight for c in contributions)
    assert total == pytest.approx(classifier.predict(text).score)
    assert all(c.feature.startswith("word:") for c in contributions)


def test_prediction_is_deterministic(classifier: SpamClassifier) -> None:
    assert classifier.predict("win a prize") == classifier.predict("win a prize")
