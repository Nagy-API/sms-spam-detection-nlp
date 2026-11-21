import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from spam_detector.artifact import LinearModelArtifact, PlattCalibration
from spam_detector.features import BlockConfig
from spam_detector.inference import SpamClassifier
from spam_detector.training.candidates import AnalysisCache, CandidateSpec, fit_platt, sigmoid
from spam_detector.training.metrics import (
    bayes_threshold,
    bootstrap_intervals,
    classification_metrics,
    expected_calibration_error,
)
from tests.conftest import HAM_EXAMPLES, SPAM_EXAMPLES

TEXTS = SPAM_EXAMPLES + HAM_EXAMPLES
LABELS = np.array([1] * len(SPAM_EXAMPLES) + [0] * len(HAM_EXAMPLES))


def test_classification_metrics_on_a_known_confusion_matrix() -> None:
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    predicted = [1, 1, 0, 1, 0, 0, 0, 0]
    result = classification_metrics(y, predicted, predicted, false_positive_cost=3, false_negative_cost=1)
    assert result["confusion_matrix"] == [[4, 1], [1, 2]]
    assert result["spam_precision"] == pytest.approx(2 / 3)
    assert result["spam_recall"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 5)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert result["expected_cost_per_message"] == pytest.approx((3 * 1 + 1 * 1) / 8)
    assert "brier" not in result


def test_calibration_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, perfect) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(4, 0.9)) == pytest.approx(0.4)
    result = classification_metrics(y, [0, 0, 1, 1], perfect, perfect)
    assert result["brier"] == pytest.approx(0.0)


def test_bayes_threshold() -> None:
    assert bayes_threshold(1, 1) == 0.5
    assert bayes_threshold(3, 1) == 0.75


def test_bootstrap_intervals_contain_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = y + rng.normal(0, 0.6, 300)
    predicted = (score > 0.5).astype(int)
    point = classification_metrics(y, predicted, score)
    intervals = bootstrap_intervals(y, predicted, score, resamples=300)
    for name, (low, high) in intervals.items():
        assert low <= point[name] <= high


def test_platt_fit_recovers_known_parameters() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 2, 20000)
    labels = rng.random(20000) < sigmoid(1.5 * scores - 0.5)
    a, b = fit_platt(scores, labels.astype(int))
    assert a == pytest.approx(1.5, abs=0.1) and b == pytest.approx(-0.5, abs=0.1)


@pytest.mark.parametrize(
    ("classifier", "params", "blocks", "reference"),
    [
        ("logistic_regression", {"C": 10.0}, [BlockConfig("word", (1, 2))], "decision_function"),
        ("linear_svm", {"C": 1.0}, [BlockConfig("word", (1, 2)), BlockConfig("char_wb", (2, 4))], "decision_function"),
        ("multinomial_nb", {"alpha": 0.5},
         [BlockConfig("word", (1, 2), weighting="count", sublinear_tf=False, l2_normalize=False)], "log_odds"),
    ],
)
def test_every_classifier_exports_to_an_equivalent_linear_artifact(classifier, params, blocks, reference) -> None:
    cache = AnalysisCache()
    spec = CandidateSpec("c", classifier, tuple(blocks), params)
    fitted = spec.fit(TEXTS, LABELS, cache)
    x = fitted.transform(TEXTS)

    factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=5000, random_state=0, **params),
        "linear_svm": lambda: LinearSVC(max_iter=20000, random_state=0, **params),
        "multinomial_nb": lambda: MultinomialNB(**params),
    }
    estimator = factories[classifier]().fit(x, LABELS)
    if reference == "log_odds":
        joint = estimator.predict_joint_log_proba(x)
        expected = joint[:, 1] - joint[:, 0]
    else:
        expected = estimator.decision_function(x)
    np.testing.assert_allclose(fitted.scores(TEXTS), expected, atol=1e-9)

    artifact = LinearModelArtifact(
        model_version="t", feature_space=fitted.space, coefficients=[float(c) for c in fitted.coefficients],
        intercept=fitted.intercept, calibration=PlattCalibration(), threshold=0.5,
    )
    runtime = SpamClassifier(artifact)
    exported = [runtime.score_vector(runtime.vectorize(t)) for t in TEXTS]
    np.testing.assert_allclose(exported, expected, atol=1e-9)


def test_vocabulary_comes_only_from_training_texts() -> None:
    cache = AnalysisCache()
    fitted = CandidateSpec("c", "logistic_regression", (BlockConfig("word", (1, 1)),), {}).fit(TEXTS, LABELS, cache)
    assert "unseenword" not in fitted.space.blocks[0].terms
    assert fitted.scores(["unseenword"])[0] == pytest.approx(fitted.intercept)


def test_fitting_is_reproducible() -> None:
    spec = CandidateSpec("c", "linear_svm", (BlockConfig("char_wb", (2, 4)),), {"C": 1.0})
    first = spec.fit(TEXTS, LABELS, AnalysisCache())
    second = spec.fit(TEXTS, LABELS, AnalysisCache())
    np.testing.assert_array_equal(first.coefficients, second.coefficients)


def test_unknown_classifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown classifier"):
        CandidateSpec("c", "random_forest", (BlockConfig("word", (1, 1)),)).fit(TEXTS, LABELS, AnalysisCache())
