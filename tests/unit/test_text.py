import pytest

from spam_detector.text import (
    CODE_TOKEN,
    EMAIL_TOKEN,
    NUM_TOKEN,
    PHONE_TOKEN,
    URL_TOKEN,
    letter_profile,
    normalize,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Call 09061701461 now", f"call {PHONE_TOKEN} now"),
        ("Txt WIN to 87121", f"txt win to {CODE_TOKEN}"),
        ("Win £900!", f"win £ {NUM_TOKEN} !"),
        ("Call 0800 123 4567", f"call {PHONE_TOKEN}"),
        ("visit www.example.com/win or http://x.co/a", f"visit {URL_TOKEN} or {URL_TOKEN}"),
        ("go to dbuk.net today", f"go to {URL_TOKEN} today"),
        ("mail win@prize4u.com", f"mail {EMAIL_TOKEN}"),
        # Dataset anonymization markers map to the placeholder of what they replaced.
        ("i am <#> mins late at <TIME>", f"i am {NUM_TOKEN} mins late at {NUM_TOKEN}"),
        ("see <URL> or <EMAIL>", f"see {URL_TOKEN} or {EMAIL_TOKEN}"),
    ],
)
def test_masks_volatile_values_but_keeps_their_kind(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_unicode_variants_produce_identical_text() -> None:
    # Curly vs straight apostrophe, full-width digits/letters (NFKC), case, and whitespace.
    assert normalize("Don’t  MISS\tthis") == normalize("don't miss this")
    assert normalize("ＷＩＮ １２３") == normalize("win 123")
    assert normalize("“quoted” – dash…") == '"quoted" - dash...'


def test_keeps_spam_signal_the_legacy_pipeline_discarded() -> None:
    normalized = normalize("FREE!!! £1000 cash")
    assert "!" in normalized and "£" in normalized


def test_is_idempotent_on_typical_messages() -> None:
    for text in ["WINNER!! Call 09061701461", "See you at 5pm :)", "Ok lar... Joking wif u oni..."]:
        once = normalize(text)
        assert normalize(once) == once


def test_empty_and_whitespace() -> None:
    assert normalize("") == ""
    assert normalize(" \t\n ") == ""


@pytest.mark.parametrize(
    ("text", "letters", "latin"),
    [("12345 !!!", 0, 0), ("🙂🙂", 0, 0), ("hello", 5, 5), ("Café", 4, 4), ("مرحبا", 5, 0), ("hi مرحبا", 7, 2)],
)
def test_letter_profile(text: str, letters: int, latin: int) -> None:
    assert letter_profile(text) == (letters, latin)
