"""
TTS normalization helpers.
Focus on removing characters that some TTS engines speak literally (e.g., ":").
"""

from __future__ import annotations

import re


_TIME_AMPM_RE = re.compile(
    r"\b(\d{1,2})\s*[:.]\s*([0-5]\d)\s*([AaPp])\s*\.?\s*([Mm])\.?\b"
)
_TIME_AMPM_SPACE_RE = re.compile(
    r"\b(\d{1,2})\s+00\s*([AaPp])\s*\.?\s*([Mm])\.?\b"
)
_TIME_PLAIN_RE = re.compile(r"\b(\d{1,2})\s*[:.]\s*([0-5]\d)\b")
_PUNCT_NO_NUM_RE = re.compile(r"(?<!\d)[,;:](?!\d)")
_DOT_NO_NUM_RE = re.compile(r"(?<!\d)\.(?!\d)")
_BRACKETS_RE = re.compile(r"[()\[\]{}<>]")
_DASHES_RE = re.compile(r"[–—-]+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

# Read common Greek-call identifiers digit-by-digit for better clarity:
# order ids, postal codes, phones, and long digit runs.
_EL_ID_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"ταχυδρομ(?:ικ(?:ός|ο)?\s*κώδικ(?:ας|α)|ικός\s*κώδικας|ος\s*κώδικας)|"
    r"κώδικ(?:ας|α)|"
    r"zip(?:\s*code)?|postal(?:\s*code)?|"
    r"order(?:\s*number)?|"
    r"αριθμ(?:ός|ο)?(?:\s*παραγγελ(?:ίας|ιας))?|"
    r"παραγγελ(?:ία|ια)|"
    r"τηλέφων(?:ο|ου)|κινητ(?:ό|ο)|phone"
    r")\s*[:#]?\s*(\d{4,16})\b"
)
_EL_HASH_NUMBER_RE = re.compile(r"#\s*(\d{4,16})\b")
_EL_LONG_DIGITS_RE = re.compile(r"(?<![\d.,])(\d{5,16})(?![\d.,])")

_EL_DIGIT_WORDS = {
    "0": "μηδέν",
    "1": "ένα",
    "2": "δύο",
    "3": "τρία",
    "4": "τέσσερα",
    "5": "πέντε",
    "6": "έξι",
    "7": "επτά",
    "8": "οκτώ",
    "9": "εννέα",
}


def normalize_time_colons(text: str) -> str:
    """
    Replace time separators so TTS doesn't say "colon".

    Examples:
      9:00 AM -> 9 AM
      9:05 AM -> 9 05 AM
      9.05 AM -> 9 05 AM
      17:30 -> 17 30
      9 00 AM -> 9 AM
    """
    if not text:
        return text

    def _ampm_repl(match: re.Match) -> str:
        hour = match.group(1)
        minute = match.group(2)
        suffix = f"{match.group(3).upper()}{match.group(4).upper()}"
        if minute == "00":
            return f"{hour} {suffix}"
        return f"{hour} {minute} {suffix}"

    text = _TIME_AMPM_RE.sub(_ampm_repl, text)
    text = _TIME_AMPM_SPACE_RE.sub(
        lambda m: f"{m.group(1)} {m.group(2).upper()}{m.group(3).upper()}",
        text,
    )
    text = _TIME_PLAIN_RE.sub(r"\1 \2", text)
    return text


def normalize_punctuation_for_tts(text: str) -> str:
    """
    Remove punctuation that TTS may speak aloud (comma, colon, semicolon, full stop).
    Keeps decimal points between digits.
    """
    if not text:
        return text
    text = _PUNCT_NO_NUM_RE.sub(" ", text)
    text = _DOT_NO_NUM_RE.sub(" ", text)
    text = _BRACKETS_RE.sub(" ", text)
    text = _DASHES_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _digits_to_greek_words(raw_digits: str) -> str:
    return " ".join(_EL_DIGIT_WORDS.get(ch, ch) for ch in raw_digits)


def normalize_numeric_ids_for_tts(text: str, language: str | None = None) -> str:
    """
    Normalize numeric IDs for clearer TTS pronunciation.

    Greek behavior:
    - Reads ZIP/order/phone-like numbers digit-by-digit.
    - Reads standalone long digit runs (5-16) digit-by-digit.
    """
    if not text:
        return text
    if (language or "").lower() != "el":
        return text

    def _ctx_repl(match: re.Match) -> str:
        label = match.group(1)
        digits = match.group(2)
        return f"{label} {_digits_to_greek_words(digits)}"

    updated = _EL_ID_CONTEXT_RE.sub(_ctx_repl, text)
    updated = _EL_HASH_NUMBER_RE.sub(
        lambda m: f"αριθμός {_digits_to_greek_words(m.group(1))}",
        updated,
    )
    updated = _EL_LONG_DIGITS_RE.sub(
        lambda m: _digits_to_greek_words(m.group(1)),
        updated,
    )
    return updated
