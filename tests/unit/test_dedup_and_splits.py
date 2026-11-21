import numpy as np
import pytest

from spam_detector.data import Message
from spam_detector.training.dedup import (
    canonical_key,
    deduplicate_and_group,
    near_duplicate_groups,
    remove_exact_duplicates,
    template_key,
)
from spam_detector.training.splits import (
    assert_no_group_leakage,
    cross_validation_folds,
    split_development_test,
)


def m(i: int, label: str, text: str) -> Message:
    return Message(i, label, text)


def test_canonical_and_template_keys() -> None:
    assert canonical_key("  Hello\tWORLD ") == canonical_key("hello world")
    assert template_key("Call 0906 now! www.a.com") == template_key("call 0800 NOW www.b.com")


def test_exact_duplicates_and_label_conflicts() -> None:
    messages = [m(0, "ham", "Hi there"), m(1, "ham", "hi  THERE"), m(2, "spam", "Win"), m(3, "ham", "win")]
    kept, duplicates, conflicts = remove_exact_duplicates(messages)
    assert [x.row_id for x in kept] == [0]
    assert (duplicates, conflicts) == (1, 2)


def test_campaign_variants_are_grouped_and_unrelated_messages_are_not() -> None:
    messages = [
        m(0, "spam", "URGENT! You have won a £900 prize. Call 09061701461 to claim. Valid 12hrs"),
        m(1, "spam", "URGENT! You have won a £2000 prize. Call 09066364589 to claim. Valid 12hrs"),
        m(2, "spam", "Urgent!! you have WON a £800 prize, call 09050003091 to claim, valid 12 hrs"),
        m(3, "ham", "Are you coming to the party tonight?"),
        m(4, "ham", "I'll call you after the meeting"),
    ]
    groups, _, _ = near_duplicate_groups(messages, jaccard_threshold=0.5)
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[0], groups[3], groups[4]}) == 3


def test_grouping_is_deterministic() -> None:
    messages = [m(i, "ham", f"message number {i % 7} about topic {i % 3}") for i in range(40)]
    assert near_duplicate_groups(messages, 0.5) == near_duplicate_groups(messages, 0.5)


def _corpus(n_groups: int = 120) -> tuple[list[Message], list[int]]:
    rng = np.random.default_rng(0)
    messages, groups = [], []
    for g in range(n_groups):
        label = "spam" if g % 6 == 0 else "ham"
        for k in range(int(rng.integers(1, 4))):
            messages.append(m(len(messages), label, f"group {g} variant {k}"))
            groups.append(g)
    return messages, groups


def test_split_keeps_groups_together_preserves_ratio_and_is_reproducible() -> None:
    messages, groups = _corpus()
    split = split_development_test(messages, groups, test_fraction=0.2, seed=3)
    assert not set(split.development_groups) & set(split.test_groups)
    assert len(split.development) + len(split.test) == len(messages)
    overall = sum(x.is_spam for x in messages) / len(messages)
    assert abs(split.summary()["test"]["spam_rate"] - overall) < 0.08
    again = split_development_test(messages, groups, test_fraction=0.2, seed=3)
    assert split.fingerprint() == again.fingerprint()
    assert split.fingerprint() != split_development_test(messages, groups, 0.2, seed=4).fingerprint()


def test_cross_validation_folds_partition_without_group_leakage() -> None:
    messages, groups = _corpus()
    labels = [x.is_spam for x in messages]
    folds = list(cross_validation_folds(labels, groups, n_folds=5, repeats=2, seed=1))
    assert len(folds) == 10
    for repeat in (0, 1):
        validation = np.concatenate([v for r, _, _, v in folds if r == repeat])
        assert sorted(validation) == list(range(len(messages)))


def test_leakage_check_raises() -> None:
    with pytest.raises(AssertionError, match="near-duplicate groups"):
        assert_no_group_leakage([1, 2, 3], [3, 4])


def test_deduplicate_and_group_report() -> None:
    messages = [m(0, "spam", "Win cash now 123"), m(1, "spam", "WIN cash now 123"), m(2, "ham", "hello friend")]
    unique, groups, report = deduplicate_and_group(messages, 0.5)
    assert report.exact_duplicates_removed == 1
    assert report.output_messages == len(unique) == len(groups) == 2
    assert report.mixed_label_groups == 0
