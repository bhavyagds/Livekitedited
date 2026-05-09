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
_SSML_TAG_RE = re.compile(r"</?[^>]+>")
_MD_BULLET_RE = re.compile(r"^\s*(?:[-*]|\u2022)\s+", re.MULTILINE)
_MD_MARKER_RE = re.compile(r"[*_`~#]+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
_EN_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b",
    re.IGNORECASE
)

# Read common Greek-call identifiers for better clarity.
_EL_ID_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"ταχυδρομ(?:ικ(?:ός|ο)?\s*κώδικ(?:ας|α)|ικός\s*κώδικας|ος\s*κώδικας)|"
    r"κώδικ(?:ας|α)|"
    r"zip(?:\s*code)?|postal(?:\s*code)?|"
    r"order(?:\s*number)?|"
    r"αριθμ(?:ός|ο)?(?:\s*παραγγελ(?:ίας|ιας))?|"
    r"παραγγελ(?:ία|ια)|"
    r"τηλέφων(?:ο|ου)|κινητ(?:ό|ο)|phone|mobile"
    r")\s*[:#]?\s*(\d{4,16})\b"
)
_EL_HASH_NUMBER_RE = re.compile(r"#\s*(\d{4,16})\b")
_EL_LONG_DIGITS_RE = re.compile(r"(?<![\d.,])(\d{5,16})(?![\d.,])")

# Read common English-call identifiers digit by digit.
_EN_ID_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"order(?:\s*(?:number|id|no\.?))?|"
    r"phone(?:\s*number)?|mobile(?:\s*number)?|"
    r"zip(?:\s*code)?|postal(?:\s*code)?|"
    r"confirmation(?:\s*number)?|reference(?:\s*number)?|tracking(?:\s*number)?"
    r")\s*[:#]?\s*(\+?\d(?:[\s-]*\d){3,20})\b"
)
_EN_HASH_NUMBER_RE = re.compile(r"#\s*(\d{4,16})\b")

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

_EN_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
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
    # Handle time ranges like 10:00 - 12:00 -> 10 00 to 12 00
    text = re.sub(r"(\d{1,2}\s+\d{2})\s*[-–—]\s*(\d{1,2}\s+\d{2})", r"\1 to \2", text)
    return text


def normalize_punctuation_for_tts(text: str) -> str:
    """
    Remove punctuation that TTS may speak aloud (comma, colon, semicolon, full stop).
    Keeps decimal points between digits.
    """
    if not text:
        return text
    # Remove markup so engines do not read tags/markdown literally.
    text = _SSML_TAG_RE.sub(" ", text)
    text = _MD_BULLET_RE.sub(" ", text)
    text = _MD_MARKER_RE.sub(" ", text)
    # Keep commas and periods as they are essential for TTS prosody (pauses).
    # text = _PUNCT_NO_NUM_RE.sub("", text)
    # text = _DOT_NO_NUM_RE.sub("", text)
    text = _BRACKETS_RE.sub(" ", text)
    # Don't strip dashes if they look like "to" (already handled in time) or are between words
    # but for TTS, replacing most with space is safe.
    text = _DASHES_RE.sub(" ", text)
    text = _EMOJI_RE.sub("", text)
    # Strip any other non-essential symbols, but KEEP . , ! ? ; :
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _digits_to_greek_words(raw_digits: str) -> str:
    return " ".join(raw_digits)


def _digits_to_english_words(raw_digits: str) -> str:
    """Space out digits for TTS clarity without using words, keeping transcript numeric."""
    return " ".join(raw_digits)


def _english_spoken_id_from_raw(raw_value: str) -> str:
    value = (raw_value or "").strip()
    has_plus = value.startswith("+")
    digits = re.sub(r"\D", "", value)
    if not (4 <= len(digits) <= 16):
        return value
    spoken = _digits_to_english_words(digits)
    if has_plus:
        spoken = f"plus {spoken}"
    return spoken


_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")


def normalize_numeric_ids_for_tts(text: str, language: str | None = None) -> str:
    """
    Normalize numeric IDs for clearer TTS pronunciation.

    Greek behavior:
    - Reads ZIP/order/phone-like numbers digit-by-digit.
    - Reads standalone long digit runs (5-16) digit-by-digit.

    English behavior:
    - Reads order/phone/zip/reference-like IDs digit-by-digit.
    - Reads hash-prefixed IDs (#12345) digit-by-digit.
    - Converts numeric dates (DD/MM) to spoken month + ordinal day.
    """
    if not text:
        return text

    lang = (language or "").lower()

    if lang.startswith("en"):

        def _en_ctx_repl(match: re.Match) -> str:
            label = match.group(1)
            raw_value = match.group(2)
            spoken = _english_spoken_id_from_raw(raw_value)
            return f"{label} {spoken}"

        updated = _EN_ID_CONTEXT_RE.sub(_en_ctx_repl, text)
        updated = _EN_HASH_NUMBER_RE.sub(
            lambda m: f"number {_digits_to_english_words(m.group(1))}",
            updated,
        )

        def _date_repl(m: re.Match) -> str:
            month_name = m.group(1)
            day = m.group(2)
            # Simple ordinal logic: 1st, 2nd, 3rd, everything else th
            d_val = int(day)
            if d_val == 1 or d_val == 21 or d_val == 31: suffix = "st"
            elif d_val == 2 or d_val == 22: suffix = "nd"
            elif d_val == 3 or d_val == 23: suffix = "rd"
            else: suffix = "th"
            return f"{month_name} {day}{suffix}"

        updated = _EN_DATE_RE.sub(_date_repl, updated)

        def _num_date_repl(m: re.Match) -> str:
            day = int(m.group(1))
            month = int(m.group(2))
            month_names = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            if 1 <= month <= 12:
                m_name = month_names[month - 1]
                if day == 1 or day == 21 or day == 31: suffix = "st"
                elif day == 2 or day == 22: suffix = "nd"
                elif day == 3 or day == 23: suffix = "rd"
                else: suffix = "th"
                return f"{m_name} {day}{suffix}"
            return m.group(0)

        updated = _NUMERIC_DATE_RE.sub(_num_date_repl, updated)
        return updated

    if not lang.startswith("el"):
        return text

    def _to_greek_number_words(raw_digits: str) -> str:
        return " ".join(raw_digits)

    def _ctx_repl(match: re.Match) -> str:
        label = match.group(1)
        digits = match.group(2)
        return f"{label} {_to_greek_number_words(digits)}"

    updated = _EL_ID_CONTEXT_RE.sub(_ctx_repl, text)
    updated = _EL_HASH_NUMBER_RE.sub(
        lambda m: f"αριθμός {_to_greek_number_words(m.group(1))}",
        updated,
    )
    updated = _EL_LONG_DIGITS_RE.sub(
        lambda m: _to_greek_number_words(m.group(1)),
        updated,
    )
    return updated
