"""
TTS normalization helpers.
Focus on removing characters that some TTS engines speak literally (e.g., ":").
"""

from __future__ import annotations

import re


_TIME_AMPM_RE = re.compile(r"\b(\d{1,2})\s*:\s*([0-5]\d)\s*([AaPp]\.?[Mm]\.?)\b")
_TIME_PLAIN_RE = re.compile(r"\b(\d{1,2})\s*:\s*([0-5]\d)\b")


def normalize_time_colons(text: str) -> str:
    """
    Replace time colons so TTS doesn't say "colon".

    Examples:
      9:00 AM -> 9 AM
      9:05 AM -> 9 05 AM
      17:30 -> 17 30
    """
    if not text:
        return text

    def _ampm_repl(match: re.Match) -> str:
        hour = match.group(1)
        minute = match.group(2)
        suffix = match.group(3)
        if minute == "00":
            return f"{hour} {suffix}"
        return f"{hour} {minute} {suffix}"

    text = _TIME_AMPM_RE.sub(_ampm_repl, text)
    text = _TIME_PLAIN_RE.sub(r"\1 \2", text)
    return text
