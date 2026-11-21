"""Shared fixtures.

Most tests run without the (non-redistributable) dataset: they use a small
synthetic corpus or the committed release artifact. Tests that need
``data/spam.csv`` are marked ``dataset`` and skipped with an explicit reason
when it is absent (as in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spam_detector.artifact import (  # noqa: E402
    AbstentionPolicy,
    LinearModelArtifact,
    PlattCalibration,
    save_artifact,
)
from spam_detector.features import BlockConfig, FeatureSpace  # noqa: E402
from spam_detector.text import normalize  # noqa: E402

DATASET = PROJECT_ROOT / "data" / "spam.csv"
RELEASE_DIR = PROJECT_ROOT / "models" / "spam-v2"

SPAM_EXAMPLES = [
    "WINNER!! You have won a £900 prize. Call 09061701461 to claim now",
    "FREE entry to win a cash prize, txt WIN to 87121 now",
    "Urgent! Claim your free prize reward today, call 09050000301",
    "Congratulations you won free tickets, text CLAIM to 80608",
]
HAM_EXAMPLES = [
    "See you at lunch tomorrow",
    "Can you pick up milk on the way home",
    "I will call you when I get home tonight",
    "Are we still meeting for dinner later",
]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if DATASET.is_file():
        return
    skip = pytest.mark.skip(reason="data/spam.csv not present (not redistributable); see docs/DATASET.md")
    for item in items:
        if "dataset" in item.keywords:
            item.add_marker(skip)


def build_toy_artifact(version: str = "0.0.1-test", threshold: float = 0.5) -> LinearModelArtifact:
    """A tiny, hand-weighted model: 'win', 'prize', 'free', 'claim' push towards spam."""
    corpus = [normalize(t) for t in SPAM_EXAMPLES + HAM_EXAMPLES]
    space = FeatureSpace.from_configs([BlockConfig("word", (1, 1))])
    space.fit([space.analyze(t) for t in corpus])
    spam_words = {"win", "won", "prize", "free", "claim", "zzphone", "zzcode", "£", "!"}
    coefficients = [3.0 if term in spam_words else -0.5 for term in space.blocks[0].terms]
    return LinearModelArtifact(
        model_version=version,
        estimator="test:toy",
        feature_space=space,
        coefficients=coefficients,
        intercept=-1.0,
        calibration=PlattCalibration(1.0, 0.0),
        threshold=threshold,
        abstention=AbstentionPolicy(0.5),
    )


@pytest.fixture
def toy_artifact_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "toy-model"
    save_artifact(build_toy_artifact(), directory, {"intended_use": "tests"})
    return directory
