"""Text normalization: the single authoritative preprocessing implementation.

Training, evaluation, and the production API all call :func:`normalize`. Nothing
else in the project is allowed to transform message text before featurization,
which is how train/serve skew is prevented by construction rather than by
convention. This module deliberately depends only on the Python standard library
so the API can import it without pulling in scikit-learn, NumPy, or NLTK.

Design choices (see ``docs/adr/0003-preprocessing.md`` for the full rationale):

* **Keep signal the legacy pipeline threw away.** The original preprocessing
  deleted every non-letter, so URLs, phone numbers, prices, ``£`` and ``!``
  never reached the model, even though they are among the strongest spam cues.
* **Mask volatile values, keep their shape.** A premium-rate number such as
  ``09061701461`` is specific to one campaign, but "this message contains a
  long phone number" generalizes to campaigns the model has never seen. Numbers,
  URLs and e-mail addresses are therefore replaced by placeholder words.
* **No stop-word removal, stemming, or lemmatization.** Short SMS messages have
  few tokens, and function words ("you", "to", "now") are discriminative in
  spam. Character n-grams already provide robustness to inflection and
  misspelling ("FREE", "fr33", "freee") without a heavyweight NLP dependency.
"""

from __future__ import annotations

import re
import unicodedata

#: Bump whenever the observable output of :func:`normalize` changes. Model
#: artifacts record the version they were trained with, and the loader refuses
#: to serve an artifact whose preprocessing version differs from this code.
PREPROCESSING_VERSION = "2.0"

# Placeholder words. They are prefixed with "zz" so they cannot collide with
# real English vocabulary, and contain only letters so every tokenizer keeps
# them intact. A unit test checks that none occurs in the training corpus.
URL_TOKEN = "zzurl"
EMAIL_TOKEN = "zzemail"
PHONE_TOKEN = "zzphone"  # seven or more digits: phone and premium-rate numbers
CODE_TOKEN = "zzcode"  # four to six digits: SMS short codes, PINs, years, prices
NUM_TOKEN = "zznum"  # one to three digits: quantities, times, ordinals
PLACEHOLDER_TOKENS = frozenset({URL_TOKEN, EMAIL_TOKEN, PHONE_TOKEN, CODE_TOKEN, NUM_TOKEN})

# Typographic punctuation is folded to ASCII so that a curly apostrophe typed on
# a phone and a straight one in the training data produce identical features.
_PUNCTUATION_FOLDING = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
    }
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    # Bare domains such as "dbuk.net" or "txt82228.co.uk/stop".
    r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|net|org|biz|info|co\.uk|org\.uk|uk|mobi|tv)\b(?:/\S*)?"
)
# Part of the corpus (the NUS SMS subset, legitimate messages only) was
# anonymized with markers such as "<#>", "<DECIMAL>", "<TIME>", "<URL>" and
# "<EMAIL>". Left alone, they would teach the model "contains <#> => ham", a
# shortcut that never occurs in real SMS. Each marker becomes the placeholder
# for the kind of value it replaced. (Runs after lowercasing.)
_REDACTED_NUMBER_RE = re.compile(r"<(?:#|decimal|time)>")
_REDACTED_URL_RE = re.compile(r"<url>")
_REDACTED_EMAIL_RE = re.compile(r"<email>")
# Digits separated by single spaces or hyphens ("0800 123 4567") form one number.
_NUMBER_RE = re.compile(r"\d+(?:[ -]\d+)*")
_WHITESPACE_RE = re.compile(r"\s+")


def _mask_number(match: re.Match[str]) -> str:
    digit_count = sum(character.isdigit() for character in match.group())
    if digit_count >= 7:
        token = PHONE_TOKEN
    elif digit_count >= 4:
        token = CODE_TOKEN
    else:
        token = NUM_TOKEN
    return f" {token} "


def normalize(text: str) -> str:
    """Return the canonical form of an SMS message used for featurization.

    The steps run in a fixed order because later patterns depend on earlier
    ones (for example, e-mail addresses must be masked before URLs, and both
    before numbers, so ``win@prize4u.com`` is not split into fragments).

    >>> normalize("WINNER!! Call 09061701461 to claim £900 at www.example.com")
    'winner!! call zzphone to claim £ zznum at zzurl'
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCTUATION_FOLDING)
    text = text.lower()
    text = _REDACTED_EMAIL_RE.sub(f" {EMAIL_TOKEN} ", text)
    text = _REDACTED_URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _EMAIL_RE.sub(f" {EMAIL_TOKEN} ", text)
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _REDACTED_NUMBER_RE.sub(f" {NUM_TOKEN} ", text)
    text = _NUMBER_RE.sub(_mask_number, text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def letter_profile(text: str) -> tuple[int, int]:
    """Count Unicode letters and how many of them belong to the Latin script.

    The model was trained exclusively on English SMS. The inference layer uses
    this profile to abstain on text that has no letters at all or is written
    mostly in another script, instead of returning a confident-looking guess
    based on a handful of shared punctuation n-grams.
    """
    letters = 0
    latin = 0
    for character in text:
        if character.isalpha():
            letters += 1
            if character.isascii() or unicodedata.name(character, "").startswith("LATIN"):
                latin += 1
    return letters, latin
