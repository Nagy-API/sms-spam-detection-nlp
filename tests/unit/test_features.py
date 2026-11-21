"""The pure-Python featurizer must match scikit-learn's reference implementation exactly."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from spam_detector.features import BlockConfig, FeatureSpace, char_wb_ngrams, word_ngrams, word_tokens
from spam_detector.text import normalize
from spam_detector.training.candidates import AnalysisCache, to_csr
from tests.conftest import HAM_EXAMPLES, SPAM_EXAMPLES

CORPUS = [normalize(t) for t in SPAM_EXAMPLES + HAM_EXAMPLES + ["a", "ok ok ok", "free free prize £ !"]]
UNSEEN = [normalize(t) for t in ["free prize now", "lunch at home", "zzz qqq", "win win win £"]]


def test_word_tokens_keep_currency_and_exclamation() -> None:
    assert word_tokens("win £ zznum !! now?") == ["win", "£", "zznum", "!", "!", "now", "?"]


def test_word_ngrams() -> None:
    assert word_ngrams(["a", "b", "c"], 1, 2) == ["a", "b", "c", "a b", "b c"]


def test_char_wb_short_word_counted_once() -> None:
    assert char_wb_ngrams("a", 2, 5) == [" a", "a ", " a "]


def _sklearn_reference(config: BlockConfig):
    common = {"ngram_range": config.ngram_range, "lowercase": False, "min_df": config.min_df}
    if config.analyzer == "word":
        common.update(tokenizer=word_tokens, token_pattern=None)
    else:
        common.update(analyzer="char_wb")
    if config.weighting == "count":
        return CountVectorizer(**common)
    return TfidfVectorizer(
        **common, sublinear_tf=config.sublinear_tf, norm="l2" if config.l2_normalize else None, smooth_idf=True
    )


@pytest.mark.parametrize(
    "config",
    [
        BlockConfig("word", (1, 2)),
        BlockConfig("word", (1, 1), sublinear_tf=False, l2_normalize=False),
        BlockConfig("char_wb", (2, 5), min_df=2),
        BlockConfig("char_wb", (3, 6)),
        BlockConfig("word", (1, 2), weighting="count", sublinear_tf=False, l2_normalize=False),
    ],
    ids=str,
)
def test_matches_scikit_learn_vectorizers(config: BlockConfig) -> None:
    space = FeatureSpace.from_configs([config])
    cache = AnalysisCache()
    space.fit([cache.analyze([config], t) for t in CORPUS])
    reference = _sklearn_reference(config).fit(CORPUS)

    assert space.blocks[0].terms == reference.get_feature_names_out().tolist()
    for texts in (CORPUS, UNSEEN):
        ours = to_csr(space, [cache.analyze([config], t) for t in texts]).toarray()
        theirs = reference.transform(texts).toarray()
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-12)


def test_blocks_are_concatenated_and_individually_normalized() -> None:
    space = FeatureSpace.from_configs([BlockConfig("word", (1, 1)), BlockConfig("char_wb", (2, 3))])
    space.fit([space.analyze(t) for t in CORPUS])
    assert space.offsets == [0, space.blocks[0].size]
    vector = space.transform(space.analyze(normalize("free prize")))
    split = space.offsets[1]
    for part in ({c: v for c, v in vector.items() if c < split}, {c: v for c, v in vector.items() if c >= split}):
        assert sum(v * v for v in part.values()) == pytest.approx(1.0)
    assert space.feature_name(0).startswith("word:")
    assert space.feature_name(split).startswith("char_wb:")


def test_out_of_vocabulary_text_gives_empty_vector() -> None:
    space = FeatureSpace.from_configs([BlockConfig("word", (1, 2))])
    space.fit([space.analyze(t) for t in CORPUS])
    assert space.transform(space.analyze("qwertyuiop asdfghjkl")) == {}


def test_vocabulary_is_independent_of_corpus_order() -> None:
    def fitted(corpus: list[str]) -> FeatureSpace:
        space = FeatureSpace.from_configs([BlockConfig("char_wb", (2, 4))])
        return space.fit([space.analyze(t) for t in corpus])

    first, second = fitted(CORPUS), fitted(list(reversed(CORPUS)))
    assert first.blocks[0].terms == second.blocks[0].terms
    assert first.blocks[0].idf == second.blocks[0].idf


def test_invalid_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        BlockConfig("word", (2, 1))
    with pytest.raises(ValueError):
        BlockConfig("word", (1, 1), min_df=0)
