"""
Meallion Voice AI - Language Detection Helpers
Lightweight heuristics for Greek/English switching.
"""

import re
from typing import Optional

# Greek and Coptic + Greek Extended ranges.
_GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: Optional[str], default: str = "el") -> str:
    """
    Detect language from text using simple script heuristics.

    Args:
        text: Input text to analyze
        default: Fallback language code when detection is inconclusive

    Returns:
        "el" for Greek, "en" for English
    """
    if not text:
        return default

    greek_chars = _GREEK_RE.findall(text)
    latin_chars = _LATIN_RE.findall(text)
    greek_count = len(greek_chars)
    latin_count = len(latin_chars)

    # Strong Greek signal: keep/switch to Greek immediately.
    if greek_count > 0 and greek_count >= latin_count:
        return "el"

    # Switch to English only on a clear signal, not on short mixed tokens
    # (e.g., product names, IDs, "ok", "yes"), to avoid accidental flips.
    if latin_count >= 6 and greek_count == 0:
        return "en"

    if latin_count >= 8 and latin_count > (greek_count * 2):
        return "en"

    # If there is any Greek at all, bias to Greek for mixed utterances.
    if greek_count > 0:
        return "el"

    return default
