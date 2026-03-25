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

    if _GREEK_RE.search(text):
        return "el"

    if _LATIN_RE.search(text):
        return "en"

    return default
