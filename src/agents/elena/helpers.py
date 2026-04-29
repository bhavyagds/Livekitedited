import re
import time
import json
import logging
from typing import Optional, List, Dict, Any
from src.agents.elena.context import _current_session
from src.agents.elena.logger import room_log
from src.agents.prompts import get_agent_language, get_agent_setting

logger = logging.getLogger(__name__)

def _as_bool(value: object, default: bool = False) -> bool:
    """Safely coerce string/number/bool values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_float(
    value: object,
    default: float,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Safely coerce values to float with optional bounds."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default

    if min_value is not None:
        result = max(min_value, result)
    if max_value is not None:
        result = min(max_value, result)
    return result


def _as_int(
    value: object,
    default: int,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Safely coerce values to int with optional bounds."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default

    if min_value is not None:
        result = max(min_value, result)
    if max_value is not None:
        result = min(max_value, result)
    return result


def _expected_order_digits() -> int:
    """Configured strict order-id length used across lookups/validation."""
    return _as_int(
        get_agent_setting("order_id_exact_digits", 5),
        5,
        min_value=4,
        max_value=8,
    )


_ORDER_WORD_TO_DIGIT: dict[str, str] = {
    # English
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    # Greek
    "\u03bc\u03b7\u03b4\u03ad\u03bd": "0",
    "\u03bc\u03b7\u03b4\u03b5\u03bd": "0",
    "\u03ad\u03bd\u03b1": "1",
    "\u03b5\u03bd\u03b1": "1",
    "\u03b4\u03cd\u03bf": "2",
    "\u03b4\u03c5\u03bf": "2",
    "\u03c4\u03c1\u03af\u03b1": "3",
    "\u03c4\u03c1\u03b9\u03b1": "3",
    "\u03c4\u03ad\u03c3\u03c3\u03b5\u03c1\u03b1": "4",
    "\u03c4\u03b5\u03c3\u03c3\u03b5\u03c1\u03b1": "4",
    "\u03c0\u03ad\u03bd\u03c4\u03b5": "5",
    "\u03c0\u03b5\u03bd\u03c4\u03b5": "5",
    "\u03ad\u03be\u03b9": "6",
    "\u03b5\u03be\u03b9": "6",
    "\u03b5\u03c0\u03c4\u03ac": "7",
    "\u03b5\u03c0\u03c4\u03b1": "7",
    "\u03b5\u03c6\u03c4\u03ac": "7",
    "\u03b5\u03c6\u03c4\u03b1": "7",
    "\u03bf\u03ba\u03c4\u03ce": "8",
    "\u03bf\u03ba\u03c4\u03c9": "8",
    "\u03b5\u03bd\u03bd\u03ad\u03b1": "9",
    "\u03b5\u03bd\u03bd\u03b5\u03b1": "9",
    # Common transliterations
    "ena": "1",
    "dyo": "2",
    "tria": "3",
    "tessera": "4",
    "pente": "5",
    "eksi": "6",
    "epta": "7",
    "okto": "8",
    "ennea": "9",
    # Common STT variants for Greek "εννιά"
    "\u03bd\u03b5\u03b1": "9",
    "\u03bd\u03b9\u03b1": "9",
    "\u03b5\u03bd\u03b9\u03b1": "9",
    "\u03b5\u03bd\u03b9\u03ac": "9",
}


def _normalize_digit_token(token: str) -> str:
    """Lowercase + strip accents so Greek spoken digits map reliably."""
    normalized = unicodedata.normalize("NFD", (token or "").strip().lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _extract_greek_mobile_from_digits(raw_digits: str) -> Optional[str]:
    """
    Extract best Greek mobile candidate (69XXXXXXXX) from a noisy digit stream.
    Handles extra prefixes/noise and picks the last valid 10-digit window.
    """
    digits = re.sub(r"\D", "", raw_digits or "")
    if not digits:
        return None

    candidates: list[str] = []

    def _collect(stream: str) -> None:
        if re.fullmatch(r"69\d{8}", stream):
            candidates.append(stream)
        for match in re.finditer(r"69\d{8}", stream):
            candidates.append(match.group(0))

    _collect(digits)

    if digits.startswith("0030"):
        _collect(digits[4:])
    elif digits.startswith("30") and len(digits) > 10:
        _collect(digits[2:])

    return candidates[-1] if candidates else None


def _digits_from_phrase(text: str) -> str:
    """Convert mixed spoken-number tokens into a compact digits-only string."""
    tokens = re.findall(r"[a-zA-Z\u0370-\u03FF0-9]+", (text or "").lower())
    digits: list[str] = []
    for token in tokens:
        normalized = _normalize_digit_token(token)
        if normalized in _ORDER_WORD_TO_DIGIT:
            digits.append(_ORDER_WORD_TO_DIGIT[normalized])
            continue
        if token.isdigit():
            digits.append(token)
            continue
        embedded_digits = re.sub(r"\D", "", token)
        if embedded_digits:
            digits.append(embedded_digits)
    return "".join(digits)


def _extract_digit_parts(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z\u0370-\u03FF0-9]+", (text or "").lower())
    parts: list[str] = []
    for token in tokens:
        normalized = _normalize_digit_token(token)
        if normalized in _ORDER_WORD_TO_DIGIT:
            parts.append(_ORDER_WORD_TO_DIGIT[normalized])
            continue
        if token.isdigit():
            parts.append(token)
            continue
        embedded_digits = re.sub(r"\D", "", token)
        if embedded_digits:
            parts.append(embedded_digits)
    return parts


def _normalize_order_id_strict(raw_text: str) -> Optional[str]:
    """Return strict order id candidate with exact configured length."""
    expected = _expected_order_digits()
    normalized = (raw_text or "").strip().lower()
    if not normalized:
        return None

    explicit_runs = re.findall(rf"\d{{{expected}}}", normalized)
    if explicit_runs:
        return explicit_runs[-1]

    parts = _extract_digit_parts(normalized)
    joined = "".join(parts)
    if len(joined) == expected:
        return joined
    return None


def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    """
    Normalize spoken phone text into a complete phone number for Shopify lookup.

    Rules:
    - Reject partial numbers.
    - Accept configured market pattern if provided.
    - By default, accept:
      - Greek mobile: 69XXXXXXXX
      - Greek international mobile: 3069XXXXXXXX or 003069XXXXXXXX
      - Generic full phone numbers: 10 to 15 digits
    """
    normalized = (raw_text or "").strip().lower()
    if not normalized:
        return None

    digits = _digits_from_phrase(normalized)
    compact = re.sub(r"\D", "", digits or "")
    if not compact:
        return None

    configured_regex = str(
        get_agent_setting(
            "phone_lookup_regex",
            r"^(?:69\d{8}|30\d{10}|0030\d{10}|\d{10,15})$",
        )
        or ""
    ).strip()
    if configured_regex:
        try:
            if re.fullmatch(configured_regex, compact):
                return compact
        except re.error:
            room_log("INVALID_PHONE_REGEX_SETTING", regex=configured_regex)

    greek_mobile = _extract_greek_mobile_from_digits(compact)
    if greek_mobile:
        return greek_mobile

    min_digits = _as_int(
        get_agent_setting("phone_lookup_min_digits", 10),
        10,
        min_value=7,
        max_value=15,
    )
    max_digits = _as_int(
        get_agent_setting("phone_lookup_max_digits", 15),
        15,
        min_value=min_digits,
        max_value=15,
    )
    if min_digits <= len(compact) <= max_digits:
        return compact
    return None


def _speak_digits(raw: str, language: str) -> str:
    """Convert digits into digit-by-digit spoken words for reliable confirmations."""
    digits = re.sub(r"\D", "", raw or "")
    if not digits:
        return ""

    if (language or "").lower() == "el":
        words = {
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
    else:
        words = {
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

    return " ".join(words[digit] for digit in digits)


def _require_setting(key: str, *, allow_empty: bool = False):
    """Fetch a required setting from DB. Raises if missing or empty."""
    value = get_agent_setting(key)
    if value is None:
        raise RuntimeError(f"Missing required setting: {key}")
    if isinstance(value, str) and not value.strip() and not allow_empty:
        raise RuntimeError(f"Missing required setting: {key}")
    return value


def _require_float_setting(
    key: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Fetch a required float setting from DB, with validation."""
    raw = _require_setting(key)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise RuntimeError(f"Invalid numeric setting: {key}")

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _require_bool_setting(key: str) -> bool:
    """Fetch a required boolean setting from DB, with coercion."""
    raw = _require_setting(key)
    return _as_bool(raw, default=False)


# =============================================================================
# CALL EVENT LOGGING
# =============================================================================
def _safe_slug(value: str) -> str:
    """Normalize strings for filenames."""
    if not value:
        return "unknown"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug or "unknown"


def _truncate(text: str, max_len: int = 500) -> str:
    """Keep log lines readable."""
    if text is None:
        return ""
    cleaned = str(text).replace("\r", "").replace("\n", "\\n")
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len] + "…"


def _strip_markup_for_output(text: str) -> str:
    """Strip SSML/markdown markers so logs/UI don't include literal markup."""
    if not text:
        return ""
    cleaned = re.sub(r"</?[^>]+>", " ", str(text))
    cleaned = re.sub(r"^\s*(?:[-*]|\u2022)\s+", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_`~#]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _strip_tts_style_leakage(text: str) -> str:
    """
    Remove style/prosody instruction leakage before sending text to TTS.
    Prevents speech like "high pitch", "medium volume", "style 0.7", etc.
    """
    if not text:
        return ""

    cleaned = str(text)
    # Remove common label-style fragments.
    cleaned = re.sub(
        r"(?i)\b(?:pitch|volume|rate|speed|tone|style|prosody|emotion|voice(?:\s*style)?)\s*[:=]\s*[a-z0-9_.-]+",
        " ",
        cleaned,
    )
    # Remove free-form sequences like "high pitch medium volume fast rate".
    cleaned = re.sub(
        r"(?i)\b(?:x-?low|low|medium|high|x-?high|soft|loud|x-?loud|slow|fast|x-?fast)\s+"
        r"(?:pitch|volume|rate|speed|tone|style|prosody)\b",
        " ",
        cleaned,
    )
    # Remove SSML prosody self references that sometimes leak as plain text.
    cleaned = re.sub(
        r'(?i)\bprosody\s+pitch\s+"?[a-z-]+"?\s+rate\s+"?[a-z-]+"?\s+volume\s+"?[a-z-]+"?\b',
        " ",
        cleaned,
    )
    # Remove repeated horizontal separators.
    cleaned = re.sub(r"\s*-{3,}\s*", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _normalize_intent_text(text: str) -> str:
    """Normalize text for robust intent checks."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _is_affirmative_utterance(text: str) -> bool:
    """Return True for short positive confirmations like 'yes'."""
    normalized = _normalize_intent_text(text)
    if not normalized:
        return False
    yes_tokens = {
        "yes", "yeah", "yep", "sure", "ok", "okay", "correct",
        "ναι", "ενταξει", "εντάξει", "σωστα", "σωστά",
    }
    return normalized in yes_tokens


def _is_negative_utterance(text: str) -> bool:
    """Return True for short negative confirmations like 'no'."""
    normalized = _normalize_intent_text(text)
    if not normalized:
        return False
    no_tokens = {"no", "nope", "nah", "not now", "οχι", "όχι", "οχι ευχαριστω", "όχι ευχαριστώ"}
    return normalized in no_tokens


def _is_issue_confirmation_utterance(text: str) -> bool:
    """Return True when user explicitly confirms the issue summary."""
    normalized = _normalize_intent_text(text)
    if not normalized:
        return False
    if _is_affirmative_utterance(normalized):
        return True
    confirmation_phrases = (
        "that is correct",
        "thats correct",
        "correct issue",
        "yes that is the issue",
        "this is the issue",
        "αυτό είναι το πρόβλημα",
        "αυτο ειναι το προβλημα",
        "σωστά",
        "σωστα",
    )
    return any(phrase in normalized for phrase in confirmation_phrases)


def _classify_lookup_result(result_text: str) -> str:
    """
    Classify lookup output into deterministic states.
    Returns one of: found, not_found, unknown.
    """
    normalized = _normalize_intent_text(result_text)
    if not normalized:
        return "unknown"

    not_found_markers = (
        "couldn t find order",
        "couldn't find order",
        "could not find order",
        "no order found",
        "no orders found",
        "order not found",
        "no matching order",
        "no matching orders",
        "0 orders found",
        "couldn t find any details",
        "could not find any details",
        "cannot find order",
        "couldn t find",
        "could not find",
        "doesn t look like a valid order number",
        "does not look like a valid order number",
        "double check the number",
        "no customer found",
        "no customers found",
        "no matching customer",
        "no matching customers",
        "no orders were found",
        "didn t find any orders",
        "did not find any orders",
        "δεν μπορώ να βρω την παραγγελία",
        "δεν μπορω να βρω την παραγγελια",
        "δεν βρήκα την παραγγελία",
        "δεν βρηκα την παραγγελια",
        "δεν βρέθηκε παραγγελία",
        "δεν βρεθηκε παραγγελια",
        "δεν βρέθηκαν παραγγελίες",
        "δεν βρεθηκαν παραγγελιες",
        "δεν φαίνεται έγκυρος αριθμός παραγγελίας",
        "δεν φαινεται εγκυρος αριθμος παραγγελιας",
        "δεν βρέθηκε καμία παραγγελία",
        "δεν βρεθηκε καμια παραγγελια",
        "δεν βρηκαμε παραγγελιες",
        "no orders found for this phone",
        "couldn't find any orders matching the phone",
        "couldn t find any orders matching the phone",
        "could not find any orders matching the phone",
        "couldn t find any orders matching this phone number",
        "could not find any orders matching this phone number",
        "couldn t find any orders for this phone",
        "could not find any orders for this phone",
        "couldn't find any orders for this phone",
        "σωστό αριθμό παραγγελίας",
        "σωστο αριθμο παραγγελιας",
    )
    if any(marker in normalized for marker in not_found_markers):
        return "not_found"

    if re.search(r"\border\s+\d{3,8}\s+(is|was)\s+\w+", normalized):
        return "found"

    if re.search(r"\border\s*#?\s*\d{3,8}\b", normalized):
        return "found"

    strong_found_markers = (
        "i found your order",
        "thanks for waiting i found your order",
        "order details for",
        "here are the details for order",
        "would you like more details about this order",
        "βρήκα την παραγγελία σας",
        "βρηκα την παραγγελια σας",
        "στοιχεία παραγγελίας",
        "λεπτομέρειες παραγγελίας",
        "θέλετε περισσότερες λεπτομέρειες",
    )
    if any(marker in normalized for marker in strong_found_markers):
        return "found"

    has_order_ref = bool(
        re.search(r"\border\s*#?\s*\d{3,8}\b", normalized)
        or re.search(r"\bπαραγγε\w*\s*#?\s*\d{3,8}\b", normalized)
    )
    has_status = bool(
        re.search(
            r"\b(status|is completed|was cancelled|completed|cancelled|fulfilled|unfulfilled|paid|"
            r"κατάσταση|ολοκληρώθηκε|ακυρώθηκε)\b",
            normalized,
        )
    )
    has_delivery_signal = bool(
        re.search(
            r"\b(delivery date|scheduled for delivery|delivery is scheduled|delivery|"
            r"παράδοση|προγραμματισμένη παράδοση)\b",
            normalized,
        )
    )
    has_total_signal = bool(
        re.search(r"\b(total|subtotal|σύνολο|συνολο)\b", normalized)
    )
    has_items_signal = bool(re.search(r"\b(line items|items \(|items:)\b", normalized))

    if has_order_ref and (has_status or has_delivery_signal or has_total_signal or has_items_signal):
        return "found"

    if (has_status and has_delivery_signal) or (has_status and has_total_signal):
        return "found"
    return "unknown"


def _extract_ticket_reference(text: str) -> Optional[str]:
    """Extract support ticket reference from tool output."""
    if not text:
        return None
    match = re.search(r"(?i)reference number is\s+([a-z0-9-]+)", text)
    if match:
        return match.group(1).strip()
    return None


def _create_room_logger(room_name: str, job_id: Optional[str]) -> tuple[logging.Logger, str]:
    """Create a per-room log file and logger."""
    log_dir = os.getenv("ROOM_LOG_DIR", "/app/data/room-logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_room = _safe_slug(room_name)
    safe_job = _safe_slug(job_id or "job")
    filename = f"room_{safe_room}_{safe_job}_{ts}.log"
    path = os.path.join(log_dir, filename)

    room_logger = logging.getLogger(f"room.{safe_room}.{safe_job}.{ts}")
    room_logger.setLevel(logging.INFO)
    room_logger.propagate = False
    if not room_logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        room_logger.addHandler(handler)

    return room_logger, path

def _should_block_silence_prompt(reason: str = "") -> bool:
    """
    Return True when silence prompts must be blocked during deterministic work.
    """
    now = time.time()
    block_reason: Optional[str] = None

    if bool(_current_session.get("lookup_pending")):
        block_reason = f"{reason}:lookup_pending"
    elif bool(_current_session.get("phone_lookup_inflight")):
        block_reason = f"{reason}:phone_lookup_inflight"
    elif bool(_current_session.get("details_lookup_inflight")):
        block_reason = f"{reason}:details_lookup_inflight"
    elif bool(_current_session.get("forced_response_manual_say_active")):
        block_reason = f"{reason}:forced_response_active"
    else:
        forced_suppress_until = float(_current_session.get("forced_response_suppress_llm_until") or 0.0)
        if forced_suppress_until and now <= forced_suppress_until:
            block_reason = f"{reason}:forced_response_suppress_window"
        elif _is_phone_confirmation_pending():
            block_reason = f"{reason}:phone_confirmation_pending"
        else:
            lookup_progress_until = float(_current_session.get("lookup_progress_prompt_until") or 0.0)
            if lookup_progress_until and now <= lookup_progress_until:
                block_reason = f"{reason}:lookup_progress_window"
            else:
                tracker = _current_session.get("silence_tracker")
                if isinstance(tracker, dict):
                    snooze_until = float(tracker.get("snooze_until") or 0.0)
                    if snooze_until and now <= snooze_until:
                        block_reason = f"{reason}:silence_snooze"
                if not block_reason:
                    pending_wait_phrase = str(_current_session.get("pending_lookup_wait_phrase") or "").strip()
                    pending_wait_set_at = float(_current_session.get("pending_lookup_wait_phrase_set_at") or 0.0)
                    lookup_wait_guard_s = _as_float(
                        get_agent_setting("lookup_wait_phrase_silence_guard_seconds", 30.0),
                        30.0,
                        min_value=10.0,
                        max_value=90.0,
                    )
                    if pending_wait_phrase and pending_wait_set_at and (now - pending_wait_set_at) <= lookup_wait_guard_s:
                        block_reason = f"{reason}:recent_wait_phrase"

    if block_reason:
        if _current_session.get("last_silence_block_reason") != block_reason:
            room_log("SILENCE_PROMPT_BLOCKED", reason=block_reason)
            _current_session["last_silence_block_reason"] = block_reason
        return True

    _current_session["last_silence_block_reason"] = None
    return False


def _build_order_voice_summary(result_text: str, language: str) -> str:
    """
    Convert raw lookup output into concise voice-safe summary.
    Keeps only status/date/total and formats order/date for speech.
    """
    text = _strip_markup_for_output(result_text or "")
    if not text:
        return ""

    lang = (language or "en").lower()
    lookup_state = _classify_lookup_result(text)

    if lookup_state == "not_found":
        if lang == "el":
            return (
                "Δεν μπορώ να βρω αυτή την παραγγελία. "
                "Μπορείτε να ελέγξετε ξανά τον αριθμό από την επιβεβαίωση παραγγελίας σας;"
            )
        return (
            "I couldn't find that order. "
            "Please double-check the order number from your confirmation."
        )
    if lookup_state == "unknown":
        if lang == "el":
            return (
                "Δεν μπόρεσα να επιβεβαιώσω τα στοιχεία αυτής της παραγγελίας. "
                "Μπορείτε να ελέγξετε τον αριθμό και να τον επαναλάβετε ψηφίο προς ψηφίο;"
            )
        return (
            "I couldn't verify this order from the details I received. "
            "Please check the order number and repeat it digit by digit."
        )

    def _digits_spaced(raw: str) -> str:
        digits = re.sub(r"\D", "", raw or "")
        if not digits:
            return ""
        if lang == "el":
            try:
                from src.utils.greek_numbers import number_to_greek
                return number_to_greek(int(digits))
            except Exception:
                return digits
        return digits

    def _month_name(month: int) -> str:
        if lang == "el":
            names = {
                1: "Ιανουαρίου", 2: "Φεβρουαρίου", 3: "Μαρτίου", 4: "Απριλίου",
                5: "Μαΐου", 6: "Ιουνίου", 7: "Ιουλίου", 8: "Αυγούστου",
                9: "Σεπτεμβρίου", 10: "Οκτωβρίου", 11: "Νοεμβρίου", 12: "Δεκεμβρίου",
            }
        else:
            names = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December",
            }
        return names.get(month, "")

    def _format_date(raw_date: str) -> str:
        m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", raw_date or "")
        if not m:
            return raw_date
        month = int(m.group(2))
        day = int(m.group(3))
        month_name = _month_name(month)
        if not month_name:
            return raw_date
        return f"{day} {month_name}" if lang == "el" else f"{month_name} {day}"

    order_match = re.search(r"(?i)\border\s*#?\s*(\d{3,8})\b", text)
    if not order_match:
        order_match = re.search(r"(?i)\bπαραγγε\w*\s*(\d{3,8})\b", text)
    order_number = order_match.group(1) if order_match else ""

    # Handle both "is completed" and "is currently completed" forms.
    status_match = re.search(r"(?i)\bis(?:\s+currently)?\s+([a-z_]+)\b", text)
    status = (status_match.group(1).lower() if status_match else "")
    if not status:
        if re.search(r"(?i)\bολοκληρ", text):
            status = "completed"
        elif re.search(r"(?i)\bακυρ", text):
            status = "cancelled"

    date_match = re.search(
        r"(?i)(?:delivery(?:\s+on)?|scheduled for delivery on|παράδοση(?:\s+στις)?)\s*[:\-]?\s*(\d{4}[/-]\d{2}[/-]\d{2})",
        text,
    )
    spoken_date = _format_date(date_match.group(1)) if date_match else ""

    total_match = re.search(r"(?i)\btotal\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not total_match:
        total_match = re.search(r"(?i)\bσύνολο\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)", text)
    amount = (total_match.group(1).replace(",", ".") if total_match else "")

    if lang == "el":
        intro = "Ευχαριστώ για την αναμονή. Βρήκα την παραγγελία σας."
        if status == "completed":
            status_phrase = "έχει ολοκληρωθεί"
        elif status == "cancelled":
            status_phrase = "έχει ακυρωθεί"
        elif status:
            status_phrase = f"είναι σε κατάσταση {status}"
        else:
            status_phrase = "βρέθηκε"
        parts = [intro]
        if order_number:
            parts.append(f"Ο αριθμός παραγγελίας {_digits_spaced(order_number)} {status_phrase}.")
        else:
            parts.append(f"Η παραγγελία σας {status_phrase}.")
        if spoken_date:
            parts.append(f"Η παράδοση είναι προγραμματισμένη για {spoken_date}.")
        if amount:
            whole, _, frac = amount.partition(".")
            if frac:
                parts.append(f"Το σύνολο είναι {int(whole)} ευρώ και {int(frac[:2]):02d} λεπτά.")
            else:
                parts.append(f"Το σύνολο είναι {int(whole)} ευρώ.")
        parts.append("Θέλετε περισσότερες λεπτομέρειες για αυτή την παραγγελία;")
        return re.sub(r"\s{2,}", " ", " ".join(parts)).strip()

    intro = "Thanks for waiting. I found your order."
    if status == "completed":
        status_phrase = "is completed"
    elif status == "cancelled":
        status_phrase = "was cancelled"
    elif status:
        status_phrase = f"is currently {status}"
    else:
        status_phrase = "was found"
    parts = [intro]
    if order_number:
        parts.append(f"Order number {_digits_spaced(order_number)} {status_phrase}.")
    else:
        parts.append(f"Your order {status_phrase}.")
    if spoken_date:
        parts.append(f"Delivery is scheduled for {spoken_date}.")
    if amount:
        whole, _, frac = amount.partition(".")
        if frac:
            parts.append(f"The total is {int(whole)} euros and {int(frac[:2]):02d} cents.")
        else:
            parts.append(f"The total is {int(whole)} euros.")
    parts.append("Would you like more details about this order?")
    return re.sub(r"\s{2,}", " ", " ".join(parts)).strip()


def _build_phone_lookup_voice_summary(result_text: str, language: str) -> str:
    """
    Build voice-safe summary for phone lookups.
    Preserve explicit phone not-found wording from the tool output.
    """
    text = _strip_markup_for_output(result_text or "")
    if not text:
        return ""

    lang = (language or "en").lower()
    lookup_state = _classify_lookup_result(text)
    normalized = _normalize_intent_text(text)

    if lookup_state == "not_found":
        if lang == "el":
            return (
                "Δεν μπόρεσα να βρω κάποια παραγγελία με αυτόν τον αριθμό τηλεφώνου. "
                "Μπορείτε να ελέγξετε τον αριθμό και να τον επαναλάβετε ψηφίο προς ψηφίο;"
            )
        return (
            "I couldn't find any order with this phone number. "
            "Please check the number and repeat it digit by digit."
        )

    if lookup_state == "unknown":
        if lang == "el":
            return (
                "Δεν μπόρεσα να επιβεβαιώσω κάποια παραγγελία με αυτόν τον αριθμό τηλεφώνου. "
                "Μπορείτε να ελέγξετε τον αριθμό και να τον επαναλάβετε ψηφίο προς ψηφίο;"
            )
        return (
            "I couldn't verify any order with this phone number. "
            "Please check the phone number and repeat it digit by digit."
        )

    if (
        "couldn t understand that phone number" in normalized
        or "could not understand that phone number" in normalized
        or "phone number must" in normalized
    ):
        return _repeat_number_prompt_for_mode("phone", lang)

    return _build_order_voice_summary(text, lang) or text


def _build_order_details_voice_summary(result_text: str, language: str) -> str:
    """
    Convert raw get_order_details output into a concise, voice-safe response.
    Never return raw multiline tool payload to avoid long or unstable speech.
    """
    raw = str(result_text or "").replace("\r", "")
    cleaned = _strip_markup_for_output(raw)
    if not cleaned:
        return ""

    lang = (language or "en").lower()
    lookup_state = _classify_lookup_result(cleaned)
    if lookup_state == "not_found":
        return _build_order_voice_summary(cleaned, lang)

    max_items = _as_int(
        get_agent_setting("order_details_voice_max_items", 5),
        5,
        min_value=1,
        max_value=8,
    )

    def _digits_spaced(raw_value: str) -> str:
        digits = re.sub(r"\D", "", raw_value or "")
        if not digits:
            return ""
        if lang == "el":
            try:
                from src.utils.greek_numbers import number_to_greek
                return number_to_greek(int(digits))
            except Exception:
                return digits
        return digits

    def _month_name(month: int) -> str:
        if lang == "el":
            names = {
                1: "Ιανουαρίου", 2: "Φεβρουαρίου", 3: "Μαρτίου", 4: "Απριλίου",
                5: "Μαΐου", 6: "Ιουνίου", 7: "Ιουλίου", 8: "Αυγούστου",
                9: "Σεπτεμβρίου", 10: "Οκτωβρίου", 11: "Νοεμβρίου", 12: "Δεκεμβρίου",
            }
        else:
            names = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December",
            }
        return names.get(month, "")

    def _format_date(raw_value: str) -> str:
        m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", raw_value or "")
        if not m:
            return ""
        month = int(m.group(2))
        day = int(m.group(3))
        month_name = _month_name(month)
        if not month_name:
            return ""
        return f"{day} {month_name}" if lang == "el" else f"{month_name} {day}"

    order_match = re.search(r"(?im)^ORDER DETAILS FOR\s*#?\s*(\d+)\s*:", raw)
    if not order_match:
        order_match = re.search(r"(?i)\border\s*number\s*(\d+)\b", raw)
    order_number = order_match.group(1) if order_match else ""

    status_match = re.search(r"(?im)^-\s*Status:\s*(.+)$", raw)
    status = status_match.group(1).strip().lower() if status_match else ""

    delivery_match = re.search(r"(?im)^-\s*Delivery Date:\s*(.+)$", raw)
    delivery_raw = delivery_match.group(1).strip() if delivery_match else ""
    delivery_spoken = _format_date(delivery_raw)

    total_match = re.search(r"(?im)^-\s*Total:\s*([0-9]+(?:[.,][0-9]+)?)\s*([A-Za-z€]{1,4})?", raw)
    amount = ""
    currency = "EUR"
    if total_match:
        amount = (total_match.group(1) or "").replace(",", ".")
        if total_match.group(2):
            currency = total_match.group(2).upper()

    item_lines: list[str] = []
    lines = raw.splitlines()
    items_start_idx = -1
    for idx, line in enumerate(lines):
        if re.search(r"(?i)^-\s*Items\s*\(\d+\)\s*:", line.strip()):
            items_start_idx = idx + 1
            break
    if items_start_idx != -1:
        for line in lines[items_start_idx:]:
            stripped = line.strip()
            if not stripped:
                if item_lines:
                    break
                continue
            if stripped.lower().startswith("use this information"):
                break
            if not stripped.startswith("-"):
                continue
            value = stripped[1:].strip()
            if not value:
                continue
            value = re.sub(r",\s*[0-9]+(?:[.,][0-9]+)?\s*[A-Za-z€]{1,4}(?:\s+each)?\s*$", "", value).strip()
            m_qty_en = re.match(r"^(\d+)\s+of\s+(.+)$", value, flags=re.IGNORECASE)
            m_qty_el = re.match(r"^(\d+)\s+τεμάχια\s+(.+)$", value, flags=re.IGNORECASE)
            if m_qty_en:
                qty = int(m_qty_en.group(1))
                name = m_qty_en.group(2).strip()
            elif m_qty_el:
                qty = int(m_qty_el.group(1))
                name = m_qty_el.group(2).strip()
            else:
                qty = 1
                name = value.strip()
            name = re.sub(r"\s{2,}", " ", name).strip(" .,;:-")
            if not name:
                continue
            if len(name) > 80:
                name = name[:80].rstrip() + "..."
            if lang == "el":
                item_lines.append(f"{qty} x {name}" if qty > 1 else name)
            else:
                item_lines.append(f"{qty} x {name}" if qty > 1 else name)
            if len(item_lines) >= max_items:
                break

    if lang == "el":
        parts = []
        if order_number:
            parts.append(f"Ορίστε οι λεπτομέρειες για την παραγγελία {_digits_spaced(order_number)}.")
        else:
            parts.append("Ορίστε οι λεπτομέρειες της παραγγελίας σας.")

        if status == "completed":
            parts.append("Η κατάσταση είναι ολοκληρωμένη.")
        elif status == "cancelled":
            parts.append("Η κατάσταση είναι ακυρωμένη.")
        elif status:
            parts.append(f"Η κατάσταση είναι {status}.")

        if delivery_spoken:
            parts.append(f"Η παράδοση είναι προγραμματισμένη για {delivery_spoken}.")

        if amount:
            whole, _, frac = amount.partition(".")
            if frac:
                parts.append(f"Το σύνολο είναι {int(whole)} ευρώ και {int(frac[:2]):02d} λεπτά.")
            else:
                parts.append(f"Το σύνολο είναι {int(whole)} ευρώ.")

        if item_lines:
            items_text = ", ".join(item_lines)
            parts.append(f"Τα βασικά προϊόντα είναι {items_text}.")

        parts.append("Θέλετε κάτι άλλο για αυτή την παραγγελία;")
        summary = " ".join(parts)
    else:
        parts = []
        if order_number:
            parts.append(f"Here are the details for order {_digits_spaced(order_number)}.")
        else:
            parts.append("Here are your order details.")

        if status == "completed":
            parts.append("The status is completed.")
        elif status == "cancelled":
            parts.append("The status is cancelled.")
        elif status:
            parts.append(f"The status is {status}.")

        delivery_total_parts: list[str] = []
        if delivery_spoken:
            delivery_total_parts.append(f"delivery is scheduled for {delivery_spoken}")

        if amount:
            whole, _, frac = amount.partition(".")
            if frac:
                delivery_total_parts.append(
                    f"the total is {int(whole)} euros and {int(frac[:2]):02d} cents"
                )
            else:
                delivery_total_parts.append(f"the total is {int(whole)} euros")

        if delivery_total_parts:
            parts.append("Also, " + ", and ".join(delivery_total_parts) + ".")

        if item_lines:
            items_text = ", ".join(item_lines)
            parts.append(f"The main items are {items_text}.")

        parts.append("Would you like help with anything else on this order?")
        summary = " ".join(parts)

    return re.sub(r"\s{2,}", " ", summary).strip()


