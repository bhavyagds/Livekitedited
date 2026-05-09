"""
Meallion Voice AI - Elena Voice Agent
Main voice agent implementation using LiveKit Agents SDK (2026 version).
"""


def _is_phone_number_prompt(text: str) -> bool:
    if not text: return False
    normalized = text.lower()
    return any(k in normalized for k in ["phone", "number", "τηλέφωνο", "αριθμό", "mobile", "κινητό"])

def _is_phone_number_collection_prompt(text: str) -> bool:
    return _is_phone_number_prompt(text)

import logging
import asyncio
import time
import os
import re
import json
import random
import threading
import unicodedata
from datetime import datetime
from typing import Annotated, Optional

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero, elevenlabs

# Try to use Deepgram for faster STT (optional)
try:
    from livekit.plugins import deepgram
    USE_DEEPGRAM = True
except ImportError:
    USE_DEEPGRAM = False

from src.config import settings
from src.agents.energy_vad import EnergyVAD
from src.agents.prompts import (
    get_system_prompt, get_system_prompt_async, get_greeting, get_closing, get_stt_language,
    get_agent_language, get_agent_setting, set_runtime_language
)
from src.agents.tools import order_lookup, support_ticket, knowledge_base
from src.utils import detect_language

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


def _order_digit_range() -> tuple[int, int]:
    """Return (min_digits, max_digits) for order ID validation. Defaults to 3-6."""
    min_d = _as_int(
        get_agent_setting("order_id_min_digits", 3),
        3,
        min_value=3,
        max_value=9,
    )
    max_d = _as_int(
        get_agent_setting("order_id_max_digits", 6),
        6,
        min_value=min_d,
        max_value=9,
    )
    return min_d, max_d


def _expected_order_digits() -> int:
    """Configured minimum order-id length — kept for backward compatibility."""
    min_d, _ = _order_digit_range()
    return min_d


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
}


def _normalize_digit_token(token: str) -> str:
    """Lowercase + strip accents so Greek spoken digits map reliably."""
    normalized = unicodedata.normalize("NFD", (token or "").strip().lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


# Greek mobile extraction removed in elena_en.py


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
    """Return strict order id candidate within the configured digit range (3-6 by default)."""
    min_len, max_len = _order_digit_range()
    normalized = (raw_text or "").strip().lower()
    if not normalized:
        return None

    # Convert spoken words to digits, then join
    parts = _extract_digit_parts(normalized)
    joined = "".join(parts)

    # Primary check: within configured range and not a phone number (10+ digits)
    if min_len <= len(joined) <= max_len:
        return joined

    # Fallback: find explicit digit sequences in the raw text within range
    matches = re.findall(rf"\d{{{min_len},{max_len}}}", normalized)
    if matches:
        return matches[-1]

    return None


def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    """
    Normalize spoken phone text into a complete phone number for Shopify lookup.
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
            r"^\d{10,15}$",
        )
        or ""
    ).strip()
    if configured_regex:
        try:
            if re.fullmatch(configured_regex, compact):
                return compact
        except re.error:
            room_log("INVALID_PHONE_REGEX_SETTING", regex=configured_regex)

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


def _speak_digits(raw: str) -> str:
    """Convert digits into digit-by-digit spoken words for reliable confirmations."""
    digits = re.sub(r"\D", "", raw or "")
    if not digits:
        return ""

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
async def log_call_event(
    event_type: str,
    room_name: str = None,
    call_type: str = "web",  # "web" or "sip"
    caller_number: str = None,
    caller_identity: str = None,
    trunk_id: str = None,
    trunk_name: str = None,
    call_id: str = None,
    duration_seconds: int = None,
    error_message: str = None,
    metadata: dict = None,
):
    """Log a call event to the database (works for both web and SIP calls)."""
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        
        # Add call type to metadata
        event_metadata = metadata or {}
        event_metadata["call_type"] = call_type
        if caller_identity:
            event_metadata["caller_identity"] = caller_identity
        
        await db.create_sip_event(
            event_type=event_type,
            room_name=room_name,
            caller_number=caller_number,
            trunk_id=trunk_id,
            trunk_name=trunk_name,
            call_id=call_id,
            duration_seconds=duration_seconds,
            error_message=error_message,
            metadata=event_metadata,
        )
        logger.debug(f"Logged call event: {event_type} ({call_type})")
    except Exception as e:
        logger.warning(f"Failed to log call event: {e}")


async def record_call_to_db(
    room_name: str,
    call_type: str = "web",
    caller_number: str = None,
    caller_identity: str = None,
) -> Optional[str]:
    """Record a new call in the database and return the call ID."""
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        
        call_id = await db.record_call_start(
            room_name=room_name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
        )
        return call_id
    except Exception as e:
        logger.warning(f"Failed to record call start: {e}")
        return None


async def end_call_in_db(
    call_id: str = None,
    room_name: str = None,
    status: str = "completed",
    duration_seconds: int = None,
    disconnect_reason: str = None,
    transcript: str = None,
) -> bool:
    """End a call in the database and update analytics."""
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        
        return await db.record_call_end(
            call_id=call_id,
            room_name=room_name,
            status=status,
            duration_seconds=duration_seconds,
            disconnect_reason=disconnect_reason,
            transcript=transcript,
        )
    except Exception as e:
        logger.warning(f"Failed to record call end: {e}")
        return False


async def save_transcript_to_db(
    call_id: str,
    text: str,
    speaker: str = "agent"
) -> bool:
    """Incrementally save a single transcript turn to the database."""
    if not call_id or not text:
        return False
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        
        entry = f"{speaker.capitalize()}: {text}"
        return await db.update_call_transcript(
            call_id=call_id,
            transcript=entry,
            append=True
        )
    except Exception as e:
        logger.debug(f"Failed to incrementally save transcript: {e}")
        return False


# =============================================================================
# LATENCY TRACKER - Measures timing for each service
# =============================================================================
class LatencyTracker:
    """Tracks latency for STT, LLM, TTS, and total response time."""
    
    def __init__(self):
        self.reset()
        self._turn_count = 0
    
    def reset(self):
        """Reset all timestamps for a new turn."""
        self._user_speech_start: Optional[float] = None
        self._user_speech_end: Optional[float] = None
        self._stt_complete: Optional[float] = None
        self._llm_start: Optional[float] = None
        self._llm_first_token: Optional[float] = None
        self._llm_complete: Optional[float] = None
        self._tts_start: Optional[float] = None
        self._tts_first_audio: Optional[float] = None
        self._agent_speaking_start: Optional[float] = None
        self._transcript: str = ""
    
    def user_started_speaking(self):
        """Called when VAD detects user started speaking."""
        self._user_speech_start = time.perf_counter()
        logger.info("⏱️ [TIMING] User started speaking")
        room_log("USER_SPEECH_START")
    
    def user_stopped_speaking(self):
        """Called when VAD detects user stopped speaking."""
        self._user_speech_end = time.perf_counter()
        if self._user_speech_start:
            duration = (self._user_speech_end - self._user_speech_start) * 1000
            logger.info(f"⏱️ [TIMING] User speech duration: {duration:.0f}ms")
            room_log("USER_SPEECH_END", duration_ms=round(duration))
    
    def stt_complete(self, transcript: str):
        """Called when STT returns the transcript."""
        self._stt_complete = time.perf_counter()
        self._transcript = transcript 
        print("", transcript)
        stt_time = None
        if self._user_speech_end:
            stt_time = (self._stt_complete - self._user_speech_end) * 1000
            logger.info(f"⏱️ [TIMING] STT processing: {stt_time:.0f}ms | Transcript: '{transcript[:50]}...'")
        room_log("STT_COMPLETE", transcript=_truncate(transcript), stt_ms=round(stt_time) if stt_time else None)
        self._llm_start = time.perf_counter()  # LLM starts right after STT
    
    def llm_first_token(self):
        """Called when LLM returns the first token (streaming)."""
        self._llm_first_token = time.perf_counter()
        if self._llm_start:
            ttft = (self._llm_first_token - self._llm_start) * 1000
            logger.info(f"⏱️ [TIMING] LLM time-to-first-token: {ttft:.0f}ms")
            room_log("LLM_TTFT", ms=round(ttft))
    
    def llm_complete(self, response: str):
        """Called when LLM completes its response."""
        self._llm_complete = time.perf_counter()
        if self._llm_start:
            llm_time = (self._llm_complete - self._llm_start) * 1000
            logger.info(f"⏱️ [TIMING] LLM total time: {llm_time:.0f}ms | Response: '{response[:50]}...'")
            room_log("LLM_COMPLETE", response=_truncate(response), ms=round(llm_time))
        self._tts_start = time.perf_counter()
    
    def tts_first_audio(self):
        """Called when TTS starts generating audio."""
        self._tts_first_audio = time.perf_counter()
        if self._tts_start:
            tts_time = (self._tts_first_audio - self._tts_start) * 1000
            logger.info(f"⏱️ [TIMING] TTS time-to-first-audio: {tts_time:.0f}ms")
            room_log("TTS_TTFB", ms=round(tts_time))
    
    def agent_started_speaking(self):
        """Called when agent actually starts speaking (audio plays)."""
        self._agent_speaking_start = time.perf_counter()
        self._turn_count += 1
        
        # Calculate total end-to-end latency
        if self._user_speech_end:
            total = (self._agent_speaking_start - self._user_speech_end) * 1000
            
            # Build breakdown
            breakdown = []
            if self._stt_complete and self._user_speech_end:
                stt = (self._stt_complete - self._user_speech_end) * 1000
                breakdown.append(f"STT:{stt:.0f}ms")
            if self._llm_complete and self._llm_start:
                llm = (self._llm_complete - self._llm_start) * 1000
                breakdown.append(f"LLM:{llm:.0f}ms")
            if self._tts_first_audio and self._tts_start:
                tts = (self._tts_first_audio - self._tts_start) * 1000
                breakdown.append(f"TTS:{tts:.0f}ms")
            
            breakdown_str = " | ".join(breakdown) if breakdown else "N/A"
            
            logger.info(
                f"🚀 [LATENCY] Turn #{self._turn_count} | "
                f"TOTAL: {total:.0f}ms | {breakdown_str}"
            )
            
            # Warn if latency is too high
            room_log("AGENT_SPEAKING_START", turn=self._turn_count, total_ms=round(total), breakdown=breakdown_str)
            if total > 3000:
                logger.warning(f"⚠️ High latency detected: {total:.0f}ms")
        
        # Reset for next turn
        self.reset()


# Global latency tracker
_latency_tracker = LatencyTracker()

# Deterministic support flow states (business logic, not LLM inference)
FLOW_IDLE = "idle"
FLOW_AWAITING_ORDER_NUMBER = "awaiting_order_number"
FLOW_CHECKING_ORDER_NUMBER = "checking_order_number"
FLOW_AWAITING_PHONE_NUMBER = "awaiting_phone_number"
FLOW_AWAITING_PHONE_CONFIRMATION = "awaiting_phone_confirmation"
FLOW_CHECKING_PHONE_NUMBER = "checking_phone_number"
FLOW_ORDER_FOUND = "order_found"
FLOW_ORDER_NOT_FOUND = "order_not_found"
PHONE_FLOW_STATES = {
    FLOW_AWAITING_PHONE_NUMBER,
    FLOW_AWAITING_PHONE_CONFIRMATION,
    FLOW_CHECKING_PHONE_NUMBER,
}

# Global reference to current session for termination/logging
_current_session: dict = {
    "agent": None,
    "room": None,
    "room_name": None,
    "job_id": None,
    "call_id": None,
    "tts_provider": "unknown",
    "room_logger": None,
    "should_end": False,
    "silence_tracker": None,
    "silence_pause_depth": 0,
    "last_silence_block_reason": None,
    "last_lookup_wait_phrase": None,
    "pending_lookup_wait_phrase": None,
    "pending_lookup_wait_phrase_set_at": 0.0,
    "last_agent_text": "",
    "last_forced_lookup_order": None,
    "last_forced_lookup_at": 0.0,
    "last_user_turn_id": 0,
    "last_lookup_tool_called_at": 0.0,
    "last_lookup_tool_order": None,
    "lookup_progress_prompt_until": 0.0,
    "number_mode_lock": None,
    "number_mode_turn_id": 0,
    "forced_response_manual_say_active": False,
    "forced_response_spoken_turn_id": 0,
    "forced_response_spoken_text": "",
    "forced_response_transcript_sent_at": 0.0,
    "forced_response_suppress_llm_until": 0.0,
    "last_manual_prompt_turn_id": 0,
    "last_manual_prompt_key": "",
    "lookup_pending": False,
    "lookup_pending_started_at": 0.0,
    "lookup_pending_order": None,
    "last_lookup_state": "unknown",
    "last_lookup_order": None,
    "pending_phone_candidate": None,
    "phone_digit_buffer": "",
    "phone_digit_buffer_updated_at": 0.0,
    "phone_lookup_inflight": False,
    "phone_lookup_active_attempt_id": 0,
    "phone_lookup_active_phone": None,
    "phone_lookup_active_source": None,
    "phone_forced_turn_id": 0,
    "phone_forced_pending_turn_id": 0,
    "details_lookup_inflight": False,
    "details_forced_turn_id": 0,
    "details_forced_pending_turn_id": 0,
    "details_last_spoken_order": None,
    "details_last_spoken_at": 0.0,
    "details_confirmation_pending": False,
    "details_confirmation_pending_until": 0.0,
    "full_order_details_allowed_until": 0.0,
    "full_details_unlocked_once": False,
    "last_more_details_prompt_at": 0.0,
    "ticket_confirmation_pending": False,
    "ticket_confirmation_pending_until": 0.0,
    "ticket_create_allowed_until": 0.0,
    "pending_ticket_payload": None,
    "ticket_created": False,
    "ticket_reference": None,
    "support_flow_state": FLOW_IDLE,
    "support_issue": None,
}


def _set_support_flow_state(new_state: str, reason: str = "") -> None:
    """Centralized flow-state transition with logging."""
    previous = str(_current_session.get("support_flow_state") or FLOW_IDLE)
    _current_session["support_flow_state"] = new_state
    room_log("SUPPORT_FLOW_STATE", previous=previous, current=new_state, reason=reason)


def _is_phone_flow_active() -> bool:
    """True when flow is currently collecting/checking phone-based lookup."""
    flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
    return flow_state in PHONE_FLOW_STATES


def _is_phone_confirmation_pending() -> bool:
    """Source-of-truth check for whether phone confirmation is currently required."""
    flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
    return flow_state == FLOW_AWAITING_PHONE_CONFIRMATION


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
    """Return True for positive confirmations. Handles full sentences not just single words."""
    normalized = _normalize_intent_text(text)
    if not normalized:
        return False
    yes_tokens = {
        "yes", "yeah", "yep", "sure", "ok", "okay", "correct",
        "right", "absolutely", "definitely", "of course", "exactly",
        "that s right", "that is right", "that s correct", "that is correct",
        "go ahead", "please go ahead", "yes please", "yes go ahead",
    }
    # Exact match first
    if normalized in yes_tokens:
        return True
    # Starts with any yes token (handles "yes that's my number", "ναι είναι σωστός" etc.)
    for token in yes_tokens:
        if normalized.startswith(token + " ") or normalized == token:
            return True
    return False


def _is_negative_utterance(text: str) -> bool:
    """Return True for short negative confirmations like 'no'."""
    normalized = _normalize_intent_text(text)
    if not normalized:
        return False
    no_tokens = {"no", "nope", "nah", "not now"}
    # Handle "No. Thanks."
    if "no thanks" in normalized or "no. thanks" in normalized:
        return True
    return normalized in no_tokens


def _is_closing_utterance(text: str) -> bool:
    """Check if the text indicates the caller/agent wants to end the interaction."""
    lowered = (text or "").lower().strip()
    # Remove common punctuation for robust matching
    lowered = re.sub(r'[.,!?]', '', lowered)
    
    closing_patterns = [
        r"have a (great|nice|wonderful|good) day",
        r"(good)?bye",
        r"that's (all|everything|it)",
        r"nothing else",
        r"that will be all",
        r"finished",
        r"done",
        r"ευχαριστώ.*αντίο",
        r"καλή σας μέρα",
        r"γεια σας",
        r"τίποτα άλλο",
        r"αυτά (είναι όλα|μόνο)",
    ]
    return any(re.search(pat, lowered) for pat in closing_patterns)


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
        "couldn t find that order",
        "could not find that order",
        "i couldn t find that order",
        "i could not find that order",
        "i couldn t find order",
        "i could not find order",
        "no order was found for this phone",
        "no order was found for this phone number",
        "couldn t find any order with this phone number",
        "could not find any order with this phone number",
        "didn t find any order for this phone",
        "did not find any order for this phone",
        "no orders found",
        "no order found",
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
        "no orders found for this phone",
        "couldn't find any orders matching the phone",
        "couldn t find any orders matching the phone",
        "could not find any orders matching the phone",
        "couldn t find any orders matching this phone number",
        "could not find any orders matching this phone number",
        "couldn t find any orders for this phone",
        "could not find any orders for this phone",
        "couldn't find any orders for this phone",
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
    )
    if any(marker in normalized for marker in strong_found_markers):
        return "found"

    has_order_ref = bool(
        re.search(r"\border\s*#?\s*\d{3,8}\b", normalized)
    )
    has_status = bool(
        re.search(
            r"\b(status|is completed|was cancelled|completed|cancelled|fulfilled|unfulfilled|paid)\b",
            normalized,
        )
    )
    has_delivery_signal = bool(
        re.search(
            r"\b(delivery date|scheduled for delivery|delivery is scheduled|delivery)\b",
            normalized,
        )
    )
    has_total_signal = bool(
        re.search(r"\b(total|subtotal)\b", normalized)
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
    match = re.search(r"(?i)reference number is\s+([#a-zA-Z0-9-]+)", text)
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


def room_log(event: str, **fields):
    """Write a structured per-room log entry if enabled."""
    room_logger = _current_session.get("room_logger")
    if not room_logger:
        return
    payload = {
        "event": event,
        "room": _current_session.get("room_name"),
        "job_id": _current_session.get("job_id"),
        "call_id": _current_session.get("call_id"),
    }
    payload.update(fields)
    room_logger.info(json.dumps(payload, ensure_ascii=False))


def _set_lookup_pending(order_number: Optional[str], reason: str) -> None:
    """Mark lookup flow as pending to suppress silence prompts until resolved/timeout."""
    now = time.time()
    _current_session["lookup_pending"] = True
    _current_session["lookup_pending_started_at"] = now
    _current_session["lookup_pending_order"] = str(order_number or "")
    progress_window_s = _as_float(
        get_agent_setting("lookup_progress_prompt_window_seconds", 30.0),
        30.0,
        min_value=5.0,
        max_value=120.0,
    )
    _current_session["lookup_progress_prompt_until"] = now + progress_window_s
    tracker = _current_session.get("silence_tracker")
    if isinstance(tracker, dict):
        tracker["last_lookup_progress_prompt_at"] = 0.0
        tracker["lookup_progress_prompt_count"] = 0
        # Immediate race guard: prevent stale silence prompts right after order capture.
        capture_snooze = _as_float(
            get_agent_setting("order_lookup_capture_snooze_seconds", 8.0),
            8.0,
            min_value=2.0,
            max_value=25.0,
        )
        tracker["snooze_until"] = max(float(tracker.get("snooze_until") or 0.0), now + capture_snooze)
        tracker["last_user_speech"] = now
        tracker["prompt_count"] = 0
        room_log("SILENCE_SNOOZED", reason=f"lookup_pending:{reason}", seconds=round(capture_snooze, 2))
    room_log("LOOKUP_PENDING_SET", order_number=order_number, reason=reason)


def _clear_lookup_pending(reason: str) -> None:
    """Clear pending lookup state."""
    if _current_session.get("lookup_pending"):
        room_log(
            "LOOKUP_PENDING_CLEARED",
            order_number=_current_session.get("lookup_pending_order"),
            reason=reason,
        )
    _current_session["lookup_pending"] = False
    _current_session["lookup_pending_started_at"] = 0.0
    _current_session["lookup_pending_order"] = None
    _current_session["lookup_progress_prompt_until"] = 0.0
    tracker = _current_session.get("silence_tracker")
    if isinstance(tracker, dict):
        tracker["last_lookup_progress_prompt_at"] = 0.0
        tracker["lookup_progress_prompt_count"] = 0


def _clear_pending_lookup_wait_phrase(reason: str) -> None:
    """Clear pending wait-phrase state to avoid stale prepends on unrelated turns."""
    if _current_session.get("pending_lookup_wait_phrase"):
        room_log("TOOL_WAIT_ACK_CLEARED", reason=reason)
    _current_session["pending_lookup_wait_phrase"] = None
    _current_session["pending_lookup_wait_phrase_set_at"] = 0.0


def _reset_phone_digit_buffer(reason: str = "") -> None:
    """Reset phone digit buffer used for chunked phone-number capture."""
    buffered = str(_current_session.get("phone_digit_buffer") or "")
    if buffered:
        room_log(
            "PHONE_DIGIT_BUFFER_RESET",
            reason=reason,
            buffer=buffered,
        )
    _current_session["phone_digit_buffer"] = ""
    _current_session["phone_digit_buffer_updated_at"] = 0.0


def _reset_phone_collection_state(reason: str = "") -> None:
    """Reset all phone collection state before entering/re-entering phone flow."""
    room_log(
        "PHONE_COLLECTION_RESET",
        reason=reason,
        pending_phone=_current_session.get("pending_phone_candidate"),
        buffer=_current_session.get("phone_digit_buffer"),
    )
    _current_session["pending_phone_candidate"] = None
    _current_session["phone_digit_buffer"] = ""
    _current_session["phone_digit_buffer_updated_at"] = 0.0
    _current_session["phone_lookup_inflight"] = False
    _current_session["phone_lookup_active_attempt_id"] = 0
    _current_session["phone_lookup_active_phone"] = None
    _current_session["phone_lookup_active_source"] = None
    _current_session["phone_forced_turn_id"] = 0
    _current_session["phone_forced_pending_turn_id"] = 0


def _start_phone_lookup_attempt(phone: str, source: str) -> int:
    """Start a single-flight phone lookup attempt and invalidate older attempts."""
    attempt_id = int(_current_session.get("phone_lookup_active_attempt_id") or 0) + 1
    _current_session["phone_lookup_active_attempt_id"] = attempt_id
    _current_session["phone_lookup_active_phone"] = str(phone or "")
    _current_session["phone_lookup_active_source"] = source
    room_log(
        "PHONE_LOOKUP_ATTEMPT_STARTED",
        attempt_id=attempt_id,
        phone=str(phone or ""),
        source=source,
    )
    return attempt_id


def _is_phone_lookup_attempt_active(attempt_id: int, phone: str = "") -> bool:
    """True only for the latest active attempt (optionally matching phone)."""
    active_id = int(_current_session.get("phone_lookup_active_attempt_id") or 0)
    if attempt_id != active_id:
        return False
    if phone:
        return str(_current_session.get("phone_lookup_active_phone") or "") == str(phone)
    return True


def _reset_support_session_state(reason: str = "") -> None:
    """Reset deterministic support/session state at call start."""
    room_log("SUPPORT_SESSION_RESET", reason=reason)
    _current_session["support_flow_state"] = FLOW_IDLE
    _current_session["support_issue"] = None
    _current_session["number_mode_lock"] = None
    _current_session["number_mode_turn_id"] = 0

    _reset_phone_collection_state(f"{reason}:phone")

    _current_session["lookup_pending"] = False
    _current_session["lookup_pending_started_at"] = 0.0
    _current_session["lookup_pending_order"] = None
    _current_session["lookup_progress_prompt_until"] = 0.0

    _current_session["details_lookup_inflight"] = False
    _current_session["details_confirmation_pending"] = False
    _current_session["details_confirmation_pending_until"] = 0.0
    _current_session["last_lookup_state"] = "unknown"
    _current_session["last_lookup_order"] = None

    _current_session["forced_response_manual_say_active"] = False
    _current_session["forced_response_spoken_turn_id"] = 0
    _current_session["forced_response_spoken_text"] = ""
    _current_session["forced_response_transcript_sent_at"] = 0.0
    _current_session["forced_response_suppress_llm_until"] = 0.0
    _current_session["last_manual_prompt_turn_id"] = 0
    _current_session["last_manual_prompt_key"] = ""

    _clear_pending_lookup_wait_phrase("support_session_reset")


def _append_phone_digits_from_turn(user_text: str) -> str:
    """Append digits from this turn into session buffer for chunked phone capture."""
    raw_digits = "".join(_extract_digit_parts(user_text or ""))
    if not raw_digits:
        return ""

    now = time.time()
    last_updated = float(_current_session.get("phone_digit_buffer_updated_at") or 0.0)
    buffer_timeout = _as_float(
        get_agent_setting("phone_digit_buffer_timeout_seconds", 20.0),
        20.0,
        min_value=5.0,
        max_value=60.0,
    )
    if last_updated and (now - last_updated) > buffer_timeout:
        _reset_phone_digit_buffer("timeout")

    current = str(_current_session.get("phone_digit_buffer") or "")
    max_digits = _as_int(
        get_agent_setting("phone_lookup_max_digits", 15),
        15,
        min_value=10,
        max_value=15,
    )
    min_digits = _as_int(
        get_agent_setting("phone_lookup_min_digits", 10),
        10,
        min_value=7,
        max_value=15,
    )

    # If the new chunk looks like the caller restarted from the beginning,
    # prefer the fresh chunk instead of concatenating stale digits.
    if current and raw_digits.startswith(current[: min(len(current), 4)]):
        combined = raw_digits
    else:
        combined = current + raw_digits

    if len(combined) > max_digits:
        # If the new turn itself is too long, treat it as malformed input rather than
        # silently reusing it as a fresh candidate.
        if len(raw_digits) > max_digits:
            combined = raw_digits
        else:
            combined = raw_digits

    # Guard against carrying obviously malformed oversize chunks into lookup logic.
    if len(combined) > max_digits:
        _current_session["phone_digit_buffer"] = combined
        _current_session["phone_digit_buffer_updated_at"] = now
        room_log("PHONE_DIGIT_BUFFER_UPDATED", raw_digits=raw_digits, buffer=combined, malformed=True)
        return combined

    # If we already have a partial prefix and the caller now gives another short partial
    # that still does not reach the minimum phone length, keep buffering.
    if current and len(combined) < min_digits:
        _current_session["phone_digit_buffer"] = combined
        _current_session["phone_digit_buffer_updated_at"] = now
        room_log("PHONE_DIGIT_BUFFER_UPDATED", raw_digits=raw_digits, buffer=combined)
        return combined

    _current_session["phone_digit_buffer"] = combined
    _current_session["phone_digit_buffer_updated_at"] = now
    room_log("PHONE_DIGIT_BUFFER_UPDATED", raw_digits=raw_digits, buffer=combined)
    return combined


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
            flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
            phone_buffer = str(_current_session.get("phone_digit_buffer") or "")
            phone_buffer_updated_at = float(_current_session.get("phone_digit_buffer_updated_at") or 0.0)
            phone_capture_grace = _as_float(
                get_agent_setting("phone_digit_capture_silence_block_seconds", 45.0),
                45.0,
                min_value=10.0,
                max_value=120.0,
            )
            if (
                flow_state == FLOW_AWAITING_PHONE_NUMBER
                and phone_buffer
                and phone_buffer_updated_at
                and (now - phone_buffer_updated_at) <= phone_capture_grace
            ):
                block_reason = f"{reason}:phone_digit_collection_active"

            if not block_reason:
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


def _build_order_voice_summary(result_text: str) -> str:
    """
    Convert raw lookup output into concise voice-safe summary.
    Keeps only status/date/total and formats order/date for speech.
    """
    text = _strip_markup_for_output(result_text or "")
    if not text:
        return ""

    lookup_state = _classify_lookup_result(text)

    if lookup_state == "not_found":
        return (
            "I'm sorry, but I couldn't find your order with the details provided. "
            "Please double-check the order number from the confirmation email you received."
        )
    if lookup_state == "unknown":
        return (
            "I couldn't verify this order from the details I received. "
            "Please check the order number and repeat it again."
        )

    def _digits_spaced(raw: str) -> str:
        digits = re.sub(r"\D", "", raw or "")
        if not digits:
            return ""
        
        # For long numbers (phone numbers, order IDs), always speak digit-by-digit
        if len(digits) > 4:
            return _speak_digits(digits)
            
        # For English, if not digit-by-digit, at least space them out for TTS
        return " ".join(digits)

    def _month_name(month: int) -> str:
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
        return f"{day:02d}/{month:02d}"

    order_match = re.search(r"(?i)\border\s*#?\s*(\d{3,8})\b", text)
    order_number = order_match.group(1) if order_match else ""

    # Handle both "is completed" and "is currently completed" forms.
    status_match = re.search(r"(?i)\bis(?:\s+currently)?\s+([a-z\s]{2,30}?)(?:\.|\s+delivery|\s+scheduled|$)", text)
    status = (status_match.group(1).strip().lower() if status_match else "")

    date_match = re.search(
        r"(?i)(?:delivery(?:\s+on)?|scheduled for delivery on)\s*[:\-]?\s*(\d{4}[/-]\d{2}[/-]\d{2})",
        text,
    )
    spoken_date = _format_date(date_match.group(1)) if date_match else ""

    total_match = re.search(r"(?i)\btotal\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", text)
    amount = (total_match.group(1).replace(",", ".") if total_match else "")

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
        parts.append(f"Order number {order_number} {status_phrase}.")
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




def _build_phone_lookup_voice_summary(result_text: str) -> str:
    """
    Build voice-safe summary for phone lookups.
    Preserve explicit phone not-found wording from the tool output.
    """
    text = _strip_markup_for_output(result_text or "")
    if not text:
        return ""

    lookup_state = _classify_lookup_result(text)
    normalized = _normalize_intent_text(text)

    if lookup_state == "not_found":
        return (
            "I couldn't find any order with this phone number. "
            "Please check the number and repeat it again."
        )

    if lookup_state == "unknown":
        return (
            "I couldn't verify any order with this phone number. "
            "Please check the phone number and repeat it again."
        )

    if (
        "couldn t understand that phone number" in normalized
        or "could not understand that phone number" in normalized
        or "phone number must" in normalized
    ):
        return _repeat_number_prompt_for_mode("phone")

    return _build_order_voice_summary(text) or text


def _build_order_details_voice_summary(result_text: str) -> str:
    """
    Convert raw get_order_details output into a concise, voice-safe response.
    Never return raw multiline tool payload to avoid long or unstable speech.
    """
    raw = str(result_text or "").replace("\r", "")
    cleaned = _strip_markup_for_output(raw)
    if not cleaned:
        return ""

    # Never let template placeholders leak into customer-facing speech.
    if re.search(r"\[[^\]]+\]", raw):
        return (
            "I can share the order details as soon as I verify them correctly. "
            "Would you like me to check that again?"
        )

    lookup_state = _classify_lookup_result(cleaned)
    if lookup_state == "not_found":
        return _build_order_voice_summary(cleaned)

    max_items = _as_int(
        get_agent_setting("order_details_voice_max_items", 5),
        5,
        min_value=1,
        max_value=8,
    )

    def _digits_spaced(raw_value: str) -> str:
        """Return the numeric string as-is. TTS normalization will handle the spacing."""
        return re.sub(r"\D", "", raw_value or "")

    def _month_name(month: int) -> str:
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
        return f"{day:02d}/{month:02d}"

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
            parts.append(f"Ορίστε οι λεπτομέρειες για την παραγγελία {order_number}.")
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
            parts.append(f"Here are the details for order {order_number}.")
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


def _pause_silence_for_tool(tool_name: str) -> None:
    """Temporarily pause silence prompts while a long-running tool executes."""
    tracker = _current_session.get("silence_tracker")
    if not isinstance(tracker, dict):
        return

    depth = int(_current_session.get("silence_pause_depth") or 0) + 1
    _current_session["silence_pause_depth"] = depth
    tracker["tool_pause_depth"] = depth
    tracker["paused_by_tool"] = True
    tracker["is_waiting_for_response"] = False
    if depth == 1:
        tracker["pause_started_at"] = time.time()
        room_log("SILENCE_PAUSED", reason=f"tool:{tool_name}", depth=depth)


def _resume_silence_for_tool(tool_name: str) -> None:
    """Resume silence prompts after tool execution completes."""
    tracker = _current_session.get("silence_tracker")
    if not isinstance(tracker, dict):
        return

    depth = max(0, int(_current_session.get("silence_pause_depth") or 0) - 1)
    _current_session["silence_pause_depth"] = depth
    tracker["tool_pause_depth"] = depth
    if depth > 0:
        return

    now = time.time()
    tracker["paused_by_tool"] = False
    tracker["last_user_speech"] = now
    tracker["last_agent_speech"] = now
    # Keep is_waiting_for_response=False — the LLM still needs to process the tool
    # result and start TTS (3-7s). mark_agent_speaking() will set it to True only
    # after the LLM response audio actually finishes playing.
    tracker["is_waiting_for_response"] = False
    started_at = float(tracker.get("pause_started_at") or now)
    room_log("SILENCE_RESUMED", reason=f"tool:{tool_name}", paused_s=round(now - started_at, 2))


def _snooze_silence_prompts(seconds: float, reason: str) -> None:
    """Temporarily prevent silence prompts to avoid overlap with pending tool replies."""
    tracker = _current_session.get("silence_tracker")
    if not isinstance(tracker, dict):
        return
    duration = max(0.0, float(seconds))
    now = time.time()
    until = now + duration
    tracker["snooze_until"] = max(float(tracker.get("snooze_until") or 0.0), until)
    room_log("SILENCE_SNOOZED", reason=reason, seconds=round(duration, 2))


def _is_silence_prompt_text(text: str) -> bool:
    """Return True when text is one of the stock silence prompts."""
    if not text:
        return False
    normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    silence_prompts = {
        "είστε εκεί",
        "με ακούτε",
        "φαίνεται ότι δεν είστε εκεί αντίο",
        "are you still there",
        "hello can you hear me",
        "it seems like you re not there goodbye",
    }
    return normalized in silence_prompts


def _is_lookup_wait_ack_only_text(text: str) -> bool:
    """Return True for short 'one moment while I check' style messages."""
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized or len(normalized) > 180:
        return False

    has_wait_ack = bool(
        re.search(
            r"(one moment|give me a moment|while i check|let me check|i ll check|thanks[, ]+got it|"
            r"μια στιγμή|περιμένετε|το ελέγχω|να το ελέγξω)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if not has_wait_ack:
        return False

    # Not an ack-only phrase if it already contains concrete lookup results.
    has_results = bool(
        re.search(
            r"(i found your order|order number\s*\d+|here are the details|delivery is scheduled|the total is|"
            r"would you like more details|status is)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    return not has_results


def _repeat_number_prompt_for_mode(mode: str) -> str:
    """Recovery prompt when number capture/validation is unclear."""
    if mode == "phone":
        return "Could you please repeat your mobile number?"
    return "Could you please repeat your order number?"


def _get_contextual_silence_prompt() -> str:
    """Return a silence prompt based on the current support flow state."""
    state = _current_session.get("support_flow_state")
    if state == FLOW_AWAITING_ORDER_NUMBER:
        return "Are you still there? Could you please provide your order number?"
    if state == FLOW_AWAITING_PHONE_NUMBER:
        return "Are you still there? Could you please provide your mobile number?"
    return "Are you still there?"


def _lookup_progress_prompt() -> str:
    """Progress prompt while deterministic lookup is still in progress."""
    return "I’m still checking your order details. One more moment, please."


def create_llm():
    """Create the LLM instance based on admin settings.
    
    Supports:
    - OpenAI: gpt-4o-mini (recommended), gpt-4o, gpt-3.5-turbo
    - Groq: llama-3.3-70b-versatile (fastest!), llama-3.1-8b-instant
    
    Provider and model are configured from admin panel.
    API keys come from environment variables.
    """
    import os
    
    # Read provider from database settings only (admin-controlled)
    provider = str(_require_setting("llm_provider")).strip().lower()
    
    if provider == "groq":
        # Groq is 10x faster than OpenAI - near instant responses
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not set, falling back to OpenAI")
        else:
            try:
                from livekit.plugins import openai as openai_plugin
                # Read model from database settings (admin-controlled)
                groq_model = str(_require_setting("groq_model")).strip()
                logger.info(f"⚡ Using Groq LLM: {groq_model} (ultra-fast)")
                room_log("LLM_PROVIDER", provider="groq", model=groq_model)
                return openai_plugin.LLM.with_groq(
                    model=groq_model,
                    temperature=0.3,  # Lower = faster, more focused responses
                )
            except Exception as e:
                logger.warning(f"Groq init failed, falling back to OpenAI: {e}")
    
    # Default: OpenAI
    # Read model from database settings (admin-controlled)
    openai_model = str(_require_setting("openai_model")).strip()
    logger.info(f"🤖 Using OpenAI LLM: {openai_model}")
    room_log("LLM_PROVIDER", provider="openai", model=openai_model)
    return openai.LLM(
        model=openai_model,
        temperature=0.3,  # Lower = faster, more deterministic
    )


def create_tts():
    """Create TTS with automatic fallback if ElevenLabs is unavailable."""
    import json
    import urllib.error
    import urllib.request

    class FailoverTTS:
        """Wrap a primary TTS and fall back to secondary on runtime failure."""

        def __init__(self, primary, fallback, *, primary_supports_ssml: bool = False):
            self._primary = primary
            self._fallback = fallback
            self._use_fallback = False
            self._locked_provider = None  # "primary" or "fallback"
            self._audio_emitted = False
            self._primary_supports_ssml = primary_supports_ssml
            self._lock_per_call = _as_bool(
                get_agent_setting("tts_failover_lock_per_call", True),
                default=True,
            )

        def _active(self):
            if self._locked_provider == "primary":
                return self._primary
            if self._locked_provider == "fallback":
                return self._fallback
            return self._fallback if self._use_fallback else self._primary

        def current_provider_name(self) -> str:
            return "elevenlabs" if self._active() is self._primary else "openai"

        def supports_ssml(self) -> bool:
            return self._active() is self._primary and self._primary_supports_ssml

        def _lock_to(self, provider: str):
            if not self._lock_per_call:
                return
            if self._locked_provider is None:
                self._locked_provider = provider
                _current_session["tts_provider"] = "elevenlabs" if provider == "primary" else "openai"
                logger.info("TTS provider locked to %s for this call", provider)
                room_log("TTS_LOCKED", provider=provider)

        def _switch_to_fallback(self, error: Exception):
            if self._use_fallback:
                return False
            if self._locked_provider == "primary" and self._lock_per_call:
                logger.warning(
                    "Primary TTS failed after lock; keeping provider to avoid tone change: %s",
                    error,
                )
                room_log("TTS_FAILOVER_BLOCKED", reason=str(error)[:200])
                return False
            logger.warning("Primary TTS failed, switching to fallback: %s", error)
            room_log("TTS_FAILOVER", reason=str(error)[:200])
            self._use_fallback = True
            self._lock_to("fallback")
            return True

        async def _stream_with_fallback(self, text, **kwargs):
            provider = self._active()
            try:
                async for seg in provider.stream(text, **kwargs):
                    if not self._audio_emitted:
                        self._audio_emitted = True
                        if provider is self._primary:
                            self._lock_to("primary")
                        else:
                            self._lock_to("fallback")
                    yield seg
                return
            except Exception as e:
                if provider is self._fallback:
                    raise
                if self._audio_emitted and self._lock_per_call:
                    logger.warning(
                        "Primary TTS failed mid-call; keeping provider locked to avoid tone change: %s",
                        e,
                    )
                    room_log("TTS_FAILOVER_BLOCKED", reason=str(e)[:200])
                    raise
                if not self._switch_to_fallback(e):
                    raise
                async for seg in self._fallback.stream(text, **kwargs):
                    if not self._audio_emitted:
                        self._audio_emitted = True
                        self._lock_to("fallback")
                    yield seg

        def stream(self, text=None, **kwargs):
            # LiveKit may call stream() with no args and push text later.
            if text is None:
                try:
                    return self._active().stream()
                except Exception as e:
                    if self._active() is self._fallback:
                        raise
                    self._switch_to_fallback(e)
                    return self._fallback.stream()
            return self._stream_with_fallback(text, **kwargs)

        async def synthesize(self, text, **kwargs):
            provider = self._active()
            try:
                result = await provider.synthesize(text, **kwargs)
                if not self._audio_emitted:
                    self._audio_emitted = True
                    if provider is self._primary:
                        self._lock_to("primary")
                    else:
                        self._lock_to("fallback")
                return result
            except Exception as e:
                if provider is self._fallback:
                    raise
                if self._audio_emitted and self._lock_per_call:
                    logger.warning(
                        "Primary TTS failed mid-call; keeping provider locked to avoid tone change: %s",
                        e,
                    )
                    room_log("TTS_FAILOVER_BLOCKED", reason=str(e)[:200])
                    raise
                if not self._switch_to_fallback(e):
                    raise
                result = await self._fallback.synthesize(text, **kwargs)
                if not self._audio_emitted:
                    self._audio_emitted = True
                    self._lock_to("fallback")
                return result

        def __getattr__(self, name):
            return getattr(self._active(), name)

    def create_openai_tts():
        """Fallback TTS provider using OpenAI audio API."""
        voice = str(get_agent_setting("openai_tts_voice", "alloy") or "alloy")
        model = str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1")
        speed = _as_float(
            get_agent_setting("openai_tts_speed", 1.0),
            1.0,
            min_value=0.25,
            max_value=4.0,
        )
        logger.warning(f"Falling back to OpenAI TTS: model={model}, voice={voice}, speed={speed}")
        room_log("TTS_PROVIDER", provider="openai", model=model, voice=voice, speed=speed)
        _current_session["tts_provider"] = "openai"
        tts = openai.TTS(model=model, voice=voice, speed=speed)
        setattr(tts, "_supports_ssml", False)
        setattr(tts, "_provider_name", "openai")
        return tts

    def elevenlabs_available() -> bool:
        """Check whether ElevenLabs key is valid for core voice endpoints."""
        if not settings.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY missing; using OpenAI TTS fallback")
            return False
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
            with urllib.request.urlopen(req, timeout=8):
                pass
            return True
        except urllib.error.HTTPError as e:
            logger.warning(f"ElevenLabs auth check HTTP {e.code}; using OpenAI TTS fallback")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs auth check failed: {e}; using OpenAI TTS fallback")
            return False

    def elevenlabs_voice_exists(voice_id: str) -> bool:
        """Validate the configured ElevenLabs voice id."""
        if not voice_id:
            return False
        try:
            req = urllib.request.Request(
                f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
            with urllib.request.urlopen(req, timeout=8):
                pass
            return True
        except urllib.error.HTTPError as e:
            logger.warning(f"ElevenLabs voice check failed for {voice_id}: HTTP {e.code}")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs voice check failed for {voice_id}: {e}")
            return False

    def elevenlabs_synthesis_available() -> bool:
        """Check if ElevenLabs account can still synthesize speech."""
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))

            character_count = payload.get("character_count")
            character_limit = payload.get("character_limit")
            if isinstance(character_count, int) and isinstance(character_limit, int) and character_limit > 0:
                if character_count >= character_limit:
                    logger.warning(
                        "ElevenLabs character quota exhausted (%s/%s). Using OpenAI TTS fallback.",
                        character_count,
                        character_limit,
                    )
                    return False
            return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 429):
                logger.warning(f"ElevenLabs synthesis unavailable (HTTP {e.code}). Using OpenAI TTS fallback.")
                return False
            logger.warning(f"ElevenLabs subscription check HTTP {e.code}; using OpenAI TTS fallback")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs subscription check failed: {e}; using OpenAI TTS fallback")
            return False

    tts_provider = str(get_agent_setting("tts_provider", "elevenlabs") or "elevenlabs").lower()
    if tts_provider == "openai":
        return create_openai_tts()

    enable_failover = _as_bool(get_agent_setting("tts_failover_enabled", True), default=True)

    if not elevenlabs_available():
        return create_openai_tts()
    if not elevenlabs_synthesis_available():
        return create_openai_tts()

    agent_lang = "en"
    
    # TTS model selection for English-only elena_en.py
    tts_model = settings.elevenlabs_model
    
    logger.info("TTS model selected: %s (language: %s)", tts_model, agent_lang)
    
    voice_id = str(get_agent_setting("agent_voice_id", settings.elevenlabs_voice_id) or settings.elevenlabs_voice_id)
    if not elevenlabs_voice_exists(voice_id):
        fallback_voice_id = settings.elevenlabs_voice_id
        if fallback_voice_id != voice_id and elevenlabs_voice_exists(fallback_voice_id):
            logger.warning("Configured voice_id '%s' invalid. Falling back to '%s'.", voice_id, fallback_voice_id)
            voice_id = fallback_voice_id
        else:
            logger.warning("No valid ElevenLabs voice_id available. Falling back to OpenAI TTS.")
            return create_openai_tts()

    voice_speed = _require_float_setting(
        "agent_voice_speed",
        min_value=0.5,
        max_value=1.2,
    )
    voice_stability = _as_float(
        get_agent_setting("agent_voice_stability", 0.75),
        0.75,
        min_value=0.0,
        max_value=1.0,
    )
    voice_similarity = _as_float(
        get_agent_setting("agent_voice_similarity", settings.elevenlabs_voice_similarity),
        settings.elevenlabs_voice_similarity,
        min_value=0.0,
        max_value=1.0,
    )

    logger.info(
        "TTS voice config: voice_id=%s speed=%.2f stability=%.2f similarity=%.2f",
        voice_id,
        voice_speed,
        voice_stability,
        voice_similarity,
    )
    room_log(
        "TTS_PROVIDER",
        provider="elevenlabs",
        model=tts_model,
        voice_id=voice_id,
        speed=voice_speed,
        stability=voice_stability,
        similarity=voice_similarity,
    )
    _current_session["tts_provider"] = "elevenlabs"

    allow_advanced = _as_bool(
        get_agent_setting("elevenlabs_allow_advanced_settings", False),
        default=False,
    )

    voice_settings = elevenlabs.VoiceSettings(
        stability=voice_stability,
        similarity_boost=voice_similarity,
        # These advanced knobs can cause ElevenLabs 400/500 on some plans/voices.
        style=0.0 if allow_advanced else None,
        speed=voice_speed if allow_advanced else None,
        use_speaker_boost=True if allow_advanced else False,
    )

    if not allow_advanced:
        logger.info("ElevenLabs advanced voice settings disabled for compatibility")

    voice = elevenlabs.Voice(
        id=voice_id,
        name="Eleni",
        category="premade",
        settings=voice_settings,
    )
    tts_use_ssml = _as_bool(get_agent_setting("tts_use_ssml", False), default=False)
    primary_tts = elevenlabs.TTS(
        voice=voice,
        model=tts_model,
        enable_ssml_parsing=tts_use_ssml,
    )
    setattr(primary_tts, "_supports_ssml", tts_use_ssml)
    setattr(primary_tts, "_provider_name", "elevenlabs")
    if enable_failover:
        return FailoverTTS(
            primary_tts,
            create_openai_tts(),
            primary_supports_ssml=tts_use_ssml,
        )
    return primary_tts


def create_stt(*, is_sip_call: bool = False):
    """Create the Speech-to-Text instance optimized for speed."""
    agent_lang = "en"
    stt_lang = "en-US"
    effective_stt_auto_detect = False
    openai_stt_model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1").strip()
    deepgram_stt_model = str(get_agent_setting("deepgram_stt_model", "nova-3") or "nova-3").strip()
    # Bias OpenAI STT to preserve source language (Greek/English) instead of translating.
    openai_stt_prompt = str(
        get_agent_setting(
            "openai_stt_prompt",
            (
                "Transcribe exactly what is spoken. "
                "Do not translate. "
                "Keep the original spoken language. "
                "Likely languages are Greek and English."
            ),
        )
        or ""
    ).strip()

    def _create_openai_stt(*, language: Optional[str]) -> object:
        """
        Create OpenAI STT with best-effort compatibility across plugin versions.
        Tries prompt + language hints first, then gracefully falls back.
        """
        attempts: list[dict] = []
        base = {"model": openai_stt_model}

        if language:
            attempts.append({**base, "language": language, "prompt": openai_stt_prompt})
            attempts.append({**base, "language": language})
        else:
            attempts.append({**base, "prompt": openai_stt_prompt})
            attempts.append(base.copy())

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                logger.info(
                    "Creating OpenAI STT: model=%s language=%s prompt=%s",
                    kwargs.get("model"),
                    kwargs.get("language", "auto"),
                    bool(kwargs.get("prompt")),
                )
                return openai.STT(**kwargs)
            except TypeError as e:
                last_error = e
                logger.warning("OpenAI STT args not supported, retrying with fallback args: %s", e)
                continue

        if last_error:
            raise last_error
        return openai.STT(model=openai_stt_model)

    def _create_deepgram_stt(*, language: Optional[str], auto_detect: bool) -> object:
        """
        Create Deepgram STT with best-effort compatibility across plugin versions.
        For auto language switching, prefer Deepgram auto language detection.
        """
        if not USE_DEEPGRAM:
            raise RuntimeError("Deepgram plugin is not available")

        base = {
            "model": deepgram_stt_model,
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }
        attempts: list[dict] = []

        if auto_detect:
            # Deepgram streaming mode in this SDK does not support detect_language=True.
            # "language=multi" is the best-effort auto-language option for streaming.
            attempts.append({**base, "language": "multi"})
            attempts.append(base.copy())
        elif language:
            attempts.append({**base, "language": language})
            attempts.append(base.copy())
        else:
            attempts.append(base.copy())

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                logger.info(
                    "Creating Deepgram STT: model=%s language=%s detect_language=%s",
                    kwargs.get("model"),
                    kwargs.get("language", "auto"),
                    kwargs.get("detect_language", False),
                )
                return deepgram.STT(**kwargs)
            except TypeError as e:
                last_error = e
                logger.warning("Deepgram STT args not supported, retrying with fallback args: %s", e)
                continue

        if last_error:
            raise last_error
        return deepgram.STT(model=deepgram_stt_model)

    class FailoverSTT:
        """Wrap a primary STT and fall back to secondary on runtime failure."""

        class _StreamWrapper:
            def __init__(self, parent, stream):
                self._parent = parent
                self._stream = stream

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return await self._stream.__anext__()
                except Exception as e:
                    if self._parent._active() is self._parent._fallback:
                        raise
                    self._parent._switch_to_fallback(e)
                    raise

            def __getattr__(self, name):
                return getattr(self._stream, name)

        def __init__(self, primary, fallback):
            self._primary = primary
            self._fallback = fallback
            self._use_fallback = False

        def _active(self):
            return self._fallback if self._use_fallback else self._primary

        def _switch_to_fallback(self, error: Exception):
            if self._use_fallback:
                return
            logger.warning("Primary STT failed, switching to fallback: %s", error)
            room_log("STT_FAILOVER", reason=str(error)[:200])
            self._use_fallback = True

        def stream(self, *args, **kwargs):
            provider = self._active()
            try:
                stream = provider.stream(*args, **kwargs)
                return self._StreamWrapper(self, stream)
            except Exception as e:
                if provider is self._fallback:
                    raise
                self._switch_to_fallback(e)
                stream = self._fallback.stream(*args, **kwargs)
                return self._StreamWrapper(self, stream)

        async def transcribe(self, *args, **kwargs):
            provider = self._active()
            try:
                return await provider.transcribe(*args, **kwargs)
            except Exception as e:
                if provider is self._fallback:
                    raise
                self._switch_to_fallback(e)
                return await self._fallback.transcribe(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._active(), name)
    
    provider = str(get_agent_setting("stt_provider", "") or "").strip().lower()
    if not provider:
        provider = "deepgram" if USE_DEEPGRAM else "openai"

    # Use Deepgram as primary when selected; OpenAI remains the fallback.
    if provider == "deepgram" and USE_DEEPGRAM:
        use_auto_detect = False  # English-only: no auto language detection
        fallback_language = None if use_auto_detect else stt_lang
        fallback = _create_openai_stt(language=fallback_language)
        try:
            primary = _create_deepgram_stt(language=stt_lang, auto_detect=use_auto_detect)
        except Exception as e:
            logger.warning("Deepgram STT init failed, falling back to OpenAI STT: %s", e)
            room_log("STT_FAILOVER", reason=f"deepgram_init_failed:{str(e)[:180]}")
            if use_auto_detect:
                logger.info("Using OpenAI STT - model: %s - language: auto", openai_stt_model)
                room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language="auto")
                return _create_openai_stt(language=None)
            logger.info("Using OpenAI STT - model: %s - language: %s", openai_stt_model, stt_lang)
            room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language=stt_lang)
            return _create_openai_stt(language=stt_lang)

        stt_language_for_log = "auto" if use_auto_detect else stt_lang
        logger.info("Using Deepgram STT (priority) - model: %s - language: %s", deepgram_stt_model, stt_language_for_log)
        room_log(
            "STT_PROVIDER",
            provider="deepgram",
            model=deepgram_stt_model,
            language=stt_language_for_log,
            auto_detect=use_auto_detect,
        )
        return FailoverSTT(primary, fallback)

    # Fallback to OpenAI Whisper (English only)
    if provider == "deepgram" and not USE_DEEPGRAM:
        logger.warning("Deepgram requested but not available; falling back to OpenAI Whisper")
    logger.info("Using OpenAI STT - model: %s - language: %s", openai_stt_model, stt_lang)
    room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language=stt_lang)
    return _create_openai_stt(language=stt_lang)


def create_vad(*, is_sip_call: bool = False):
    """Create Voice Activity Detection tuned for better transcript completeness."""
    min_speech_duration = _as_float(
        get_agent_setting("vad_min_speech_duration", 0.15),
        0.15,
        min_value=0.1,
        max_value=0.8,
    )
    # Language-aware silence delay: Greek speakers tend to pause more between digits.
    initial_lang = 'en'
    if initial_lang == "el":
        default_silence = 2.0 if is_sip_call else 1.8
    else:
        default_silence = 1.4 if is_sip_call else 1.2
    
    min_silence_duration = _as_float(
        get_agent_setting("vad_min_silence_duration", default_silence),
        default_silence,
        min_value=0.2,
        max_value=3.0,
    )

    vad_backend = str(get_agent_setting("vad_backend", "silero") or "").strip().lower()
    if vad_backend in {"energy", "rms", "simple"}:
        energy_threshold = _as_float(
            get_agent_setting("energy_vad_threshold", 0.02),
            0.012,
            min_value=0.001,
            max_value=0.2,
        )
        prefix_padding = _as_float(
            get_agent_setting("energy_vad_prefix_padding", 0.15),
            0.15,
            min_value=0.0,
            max_value=0.8,
        )
        logger.info(
            "VAD backend: energy threshold=%.4f min_speech_duration=%.2fs min_silence_duration=%.2fs prefix_padding=%.2fs",
            energy_threshold,
            min_speech_duration,
            min_silence_duration,
            prefix_padding,
        )
        room_log(
            "VAD_CONFIG",
            backend="energy",
            threshold=energy_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding=prefix_padding,
        )
        return EnergyVAD(
            threshold=energy_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding,
        )

    vad_sample_rate = _as_int(
        get_agent_setting("vad_sample_rate", 8000),
        8000,
        min_value=8000,
        max_value=16000,
    )
    vad_activation_threshold = _as_float(
        get_agent_setting(
            "vad_activation_threshold",
            0.58 if (initial_lang == "el" and is_sip_call) else 0.64 if initial_lang == "el" else 0.72,
        ),
        0.58 if initial_lang == "el" else 0.6,
        min_value=0.1,
        max_value=0.9,
    )
    vad_force_cpu = _as_bool(
        get_agent_setting("vad_force_cpu", True),
        default=True,
    )
    logger.info(
        "VAD config: sample_rate=%sHz activation_threshold=%.2f min_speech_duration=%.2fs min_silence_duration=%.2fs",
        vad_sample_rate,
        vad_activation_threshold,
        min_speech_duration,
        min_silence_duration,
    )
    room_log(
        "VAD_CONFIG",
        backend="silero",
        sample_rate=vad_sample_rate,
        activation_threshold=vad_activation_threshold,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
    )
    return silero.VAD.load(
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        activation_threshold=vad_activation_threshold,
        sample_rate=vad_sample_rate,
        force_cpu=vad_force_cpu,
    )


class ElenaFunctionContext(llm.FunctionContext):
    """Function context with all Elena's tools as methods."""

    def _pick_lookup_wait_phrase(self) -> str:
        """Pick a natural, non-repeating wait phrase based on current language."""
        silence_grace_s = _as_float(
            get_agent_setting("order_lookup_silence_grace_seconds", 30.0),
            30.0,
            min_value=20.0,
            max_value=60.0,
        )
        phrases = (
            "Got it. Please give me a moment to check the details for you.",
            "Okay, I have it. One moment while I pull up the details.",
            "Thanks, let me check that for you right away.",
            "Perfect, I’ll quickly look this up for you now.",
        )

        last_phrase = _current_session.get("last_lookup_wait_phrase")
        options = [p for p in phrases if p != last_phrase] or list(phrases)
        phrase = random.choice(options)
        _current_session["last_lookup_wait_phrase"] = phrase
        _current_session["pending_lookup_wait_phrase"] = phrase
        _current_session["pending_lookup_wait_phrase_set_at"] = time.time()
        _snooze_silence_prompts(silence_grace_s, reason="lookup_wait_ack")
        room_log("TOOL_WAIT_ACK_SELECTED", phrase=_truncate(phrase))
        return phrase

    async def _run_tool_with_silence_pause(self, name: str, coro):
        """Pause silence prompts during tool I/O and resume afterward."""
        _pause_silence_for_tool(name)
        try:
            return await coro
        finally:
            _resume_silence_for_tool(name)

    @llm.ai_callable()
    async def lookup_order(
        self,
        order_number: Annotated[str, llm.TypeInfo(description="The order number (min 4 digits)")],
    ) -> str:
        """Look up an order. Returns brief status first. Use get_order_details for more info."""
        lang = 'en'

        # 0. Empty/Non-numeric Guard: If user says they don't have it, don't try to validate
        if not any(char.isdigit() for char in order_number or ""):
            room_log("TOOL_EMPTY_INPUT", name="lookup_order", input=order_number)
            return "No digits were provided. If the customer does not have their order number, please ask for their phone number instead."

        # 1. Redirection Guard: If the input looks like a phone number, redirect to lookup_order_by_phone
        clean_input = re.sub(r"\D", "", order_number or "")
        if len(clean_input) >= 10:
            room_log("TOOL_REDIRECT", name="lookup_order", target="lookup_order_by_phone", input=order_number)
            logger.info(f"🔄 Redirecting lookup_order to lookup_order_by_phone for {order_number}")
            return await self.lookup_order_by_phone(order_number)

        # 2. Validation Guard: Ensure it's not a mismatched mode if we're locked
        lock_mode = str(_current_session.get("number_mode_lock") or "")
        lock_turn = int(_current_session.get("number_mode_turn_id") or 0)
        latest_turn = int(_current_session.get("last_user_turn_id") or 0)
        if lock_mode == "phone" and lock_turn == latest_turn:
            if not _normalize_order_id_strict(order_number):
                room_log("TOOL_RESULT_BLOCKED", name="lookup_order", reason="number_mode_mismatch")
                expected = _expected_order_digits()
                return f"That doesn't look like an order number. Please share your order number (at least {expected} digits)."

        strict_order = _normalize_order_id_strict(order_number)
        if not strict_order:
            min_d, max_d = _order_digit_range()
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order", reason="invalid_order_id_format", input=order_number)
            _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason="invalid_order_format")
            return f"The order number must be between {min_d} and {max_d} digits. Could you say it again?"

        # We now have a valid order id, so move flow authority back to order lookup.
        _set_support_flow_state(FLOW_CHECKING_ORDER_NUMBER, reason="lookup_order_called")
        _set_lookup_pending(strict_order, reason="lookup_order_called")
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = strict_order
        _current_session["last_lookup_progress_at"] = time.time()
        room_log("TOOL_CALL", name="lookup_order", order_number=strict_order)
        logger.info(f"🔎 Starting order lookup for #{strict_order}")
        # No agent.say() here — the LLM's pre-tool message handles the wait UX
        try:
            result = await self._run_tool_with_silence_pause(
                "lookup_order",
                order_lookup.lookup_order(strict_order),
            )
            room_log("TOOL_RESULT", name="lookup_order", result=_truncate(result))
            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            _current_session["last_lookup_order"] = strict_order if lookup_state == "found" else ""
            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason="lookup_order_found")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason=f"lookup_order_{lookup_state}")
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0
                _current_session["number_mode_lock"] = "order"
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("back_to_order_flow")
                tracker = _current_session.get("silence_tracker")
                silence_timeout = 12.0
                if isinstance(tracker, dict):
                    silence_timeout = float(tracker.get("silence_timeout") or 12.0)
                _snooze_silence_prompts(
                    silence_timeout + 5.0,
                    reason="order_lookup_not_found_grace",
                )
                # Suppress LLM from adding its own extra paraphrase on top of the tool result
                _current_session["forced_response_suppress_llm_until"] = time.time() + 15.0
                room_log("LLM_SUPPRESSED_AFTER_NOT_FOUND", order_number=strict_order)

            summary = _build_order_voice_summary(result) or result
            return summary
        finally:
            _clear_lookup_pending("lookup_order_finished")
            _clear_pending_lookup_wait_phrase("lookup_order_finished")
            _current_session["lookup_progress_prompt_until"] = 0.0
            _snooze_silence_prompts(15.0, reason="lookup_order_finished")

    @llm.ai_callable()
    async def get_order_details(
        self,
        order_number: Annotated[str, llm.TypeInfo(description="Order number or 'last' for most recent")] = "last",
    ) -> str:
        """Get FULL order details (items, prices, address). Use after lookup_order when customer wants more info."""
        lang = 'en'
        current_turn = int(_current_session.get("last_user_turn_id") or 0)
        forced_turn = int(_current_session.get("details_forced_turn_id") or 0)
        forced_pending_turn = int(_current_session.get("details_forced_pending_turn_id") or 0)
        if current_turn and (forced_turn == current_turn or forced_pending_turn == current_turn):
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="forced_turn_in_progress")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            return "I am already fetching those details and will share them in a moment."
        if bool(_current_session.get("details_lookup_inflight")):
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="forced_lookup_inflight")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            return "I am already fetching those details and will share them in a moment."
        now = time.time()
        allowed_until = float(_current_session.get("full_order_details_allowed_until") or 0.0)
        if now > allowed_until:
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="explicit_details_required")
            return "I can share full order details as soon as you say yes. Would you like the complete order details now?"

        last_state = str(_current_session.get("last_lookup_state") or "unknown")
        anchor_order = re.sub(r"\D", "", str(_current_session.get("last_lookup_order") or ""))
        expected = _expected_order_digits()
        if last_state != "found" or not re.fullmatch(rf"\d{{{expected}}}", anchor_order):
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="missing_found_lookup_anchor")
            return "I first need a valid found order before I can share full details."

        requested_order = anchor_order if str(order_number or "").lower() == "last" else (
            _normalize_order_id_strict(order_number or "") or ""
        )
        if requested_order != anchor_order:
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="order_anchor_mismatch")
            return "I can only share details for the last found order. Please provide that same order number again."

        _current_session["full_order_details_allowed_until"] = 0.0
        _current_session["details_confirmation_pending"] = False
        _current_session["details_confirmation_pending_until"] = 0.0
        _set_lookup_pending(requested_order, reason="get_order_details_called")
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = requested_order
        _current_session["last_lookup_progress_at"] = time.time()
        _current_session["details_lookup_inflight"] = True
        room_log("TOOL_CALL", name="get_order_details", order_number=requested_order)
        try:
            result = await self._run_tool_with_silence_pause(
                "get_order_details",
                order_lookup.get_order_details(requested_order),
            )
            room_log("TOOL_RESULT", name="get_order_details", result=_truncate(result))
            spoken_summary = _build_order_details_voice_summary(result)
            if not spoken_summary:
                spoken_summary = _build_order_voice_summary(result) or (
                    "I could not find details for this order."
                )
            room_log("ORDER_DETAILS_FORMATTED", order_number=order_number, result=_truncate(spoken_summary))
            if _as_bool(get_agent_setting("order_lookup_wait_phrase_enabled", True), default=True):
                self._pick_lookup_wait_phrase()
            return spoken_summary
        finally:
            _current_session["details_lookup_inflight"] = False
            _clear_lookup_pending("get_order_details_finished")
            _clear_pending_lookup_wait_phrase("get_order_details_finished")
            _current_session["lookup_progress_prompt_until"] = 0.0
            _snooze_silence_prompts(5.0, reason="get_order_details_finished")

    @llm.ai_callable()
    async def lookup_order_by_phone(
        self,
        phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
    ) -> str:
        """Look up orders by customer phone number. Use when order number is unknown."""
        lang = 'en'

        # 0. Empty/Non-numeric Guard: If user says they don't have it, don't try to validate
        if not any(char.isdigit() for char in phone or ""):
            room_log("TOOL_EMPTY_INPUT", name="lookup_order_by_phone", input=phone)
            return "No digits were provided. If the customer does not have their phone number, please ask them to check their confirmation email or provide an order ID."

        # 1. Redirection Guard: If input looks like an order ID (min 4 digits, but not a phone)
        clean_input = re.sub(r"\D", "", phone or "")
        min_order_len = _expected_order_digits()
        # Guard: Greek mobile numbers start with 69. Never redirect those to order lookup if short.
        is_greek_mobile_prefix = clean_input.startswith("69") or clean_input.startswith("3069")
        
        if len(clean_input) >= min_order_len and len(clean_input) < 10 and not is_greek_mobile_prefix:
            room_log("TOOL_REDIRECT", name="lookup_order_by_phone", target="lookup_order", input=phone)
            logger.info(f"🔄 Redirecting lookup_order_by_phone to lookup_order for {phone}")
            return await self.lookup_order(phone)

        current_turn = int(_current_session.get("last_user_turn_id") or 0)
        forced_turn = int(_current_session.get("phone_forced_turn_id") or 0)
        forced_pending_turn = int(_current_session.get("phone_forced_pending_turn_id") or 0)
        if current_turn and (forced_turn == current_turn or forced_pending_turn == current_turn):
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="forced_turn_in_progress")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            return "I am already checking that phone number and will respond in a moment."
        if bool(_current_session.get("phone_lookup_inflight")):
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="forced_lookup_inflight")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            return "I am already checking that phone number and will respond in a moment."

        # 2. Validation Guard: Ensure it's not a mismatched mode if we're locked
        lock_mode = str(_current_session.get("number_mode_lock") or "")
        lock_turn = int(_current_session.get("number_mode_turn_id") or 0)
        latest_turn = int(_current_session.get("last_user_turn_id") or 0)
        if lock_mode == "order" and lock_turn == latest_turn:
            # Only block if it doesn't look like a valid phone number
            if not _normalize_phone_for_lookup(phone):
                room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="number_mode_mismatch")
                return "That doesn't look like a phone number. Please share your phone number."

        normalized_phone = _normalize_phone_for_lookup(phone)
        if not normalized_phone:
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="invalid_phone_pattern")
            _clear_pending_lookup_wait_phrase("invalid_phone_pattern")
            _clear_lookup_pending("invalid_phone_pattern")
            _current_session["phone_lookup_inflight"] = False
            _current_session["pending_phone_candidate"] = None
            _reset_phone_digit_buffer("invalid_phone_pattern")
            _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="invalid_phone_pattern")
            invalid_recovery_grace = _as_float(
                get_agent_setting("invalid_number_recovery_silence_grace_seconds", 12.0),
                12.0,
                min_value=4.0,
                max_value=40.0,
            )
            _snooze_silence_prompts(invalid_recovery_grace, reason="invalid_phone_recovery")
            return _repeat_number_prompt_for_mode("phone")

        # removed hard gate here to allow parity with order number lookup
        attempt_id = _start_phone_lookup_attempt(normalized_phone, "tool:lookup_order_by_phone")
        _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="lookup_order_by_phone_called")
        _current_session["phone_lookup_inflight"] = True
        _set_lookup_pending(normalized_phone, reason="phone_lookup_started")
        _snooze_silence_prompts(60.0, reason="phone_lookup_started")
        _current_session["lookup_progress_prompt_until"] = time.time() + 60.0
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = normalized_phone
        _current_session["last_lookup_progress_at"] = time.time()
        room_log("TOOL_CALL", name="lookup_order_by_phone", phone=normalized_phone)
        logger.info(f"🔍 Starting phone lookup for {normalized_phone}")
        # No agent.say() here — the LLM's pre-tool message handles the wait UX
        try:
            result = await self._run_tool_with_silence_pause(
                "lookup_order_by_phone",
                order_lookup.lookup_order_by_phone(normalized_phone),
            )

            if not _is_phone_lookup_attempt_active(attempt_id, normalized_phone):
                room_log(
                    "PHONE_LOOKUP_RESULT_DISCARDED",
                    attempt_id=attempt_id,
                    phone=normalized_phone,
                    reason="stale_tool_attempt",
                )
                return "I am already checking with the latest phone number and will respond in a moment."

            room_log("TOOL_RESULT", name="lookup_order_by_phone", result=_truncate(result))
            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            snapshot = order_lookup.get_last_order_snapshot() or {}
            snapshot_order = str(snapshot.get("order_number") or "")
            strict_snapshot_order = _normalize_order_id_strict(snapshot_order) if snapshot_order else None
            _current_session["last_lookup_order"] = strict_snapshot_order if (lookup_state == "found" and strict_snapshot_order) else ""

            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason="lookup_order_by_phone_found")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason=f"lookup_order_by_phone_{lookup_state}")
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0
                # Suppress LLM from adding its own paraphrase on top of the not-found tool result
                _current_session["forced_response_suppress_llm_until"] = time.time() + 15.0
                room_log("LLM_SUPPRESSED_AFTER_PHONE_NOT_FOUND", phone=normalized_phone)

            summary = _build_phone_lookup_voice_summary(result) or result
            return summary
        finally:
            if _is_phone_lookup_attempt_active(attempt_id, normalized_phone):
                _current_session["phone_lookup_inflight"] = False
                _clear_lookup_pending("phone_lookup_finished")
                _clear_pending_lookup_wait_phrase("phone_lookup_finished")
                _reset_phone_digit_buffer("phone_lookup_finished")
                _snooze_silence_prompts(15.0, reason="phone_lookup_finished")

    @llm.ai_callable()
    async def create_support_ticket(
        self,
        customer_name: Annotated[str, llm.TypeInfo(description="Customer's full name")],
        customer_phone: Annotated[str, llm.TypeInfo(description="Customer's phone number")],
        customer_email: Annotated[str, llm.TypeInfo(description="Customer's email address")],
        issue_description: Annotated[str, llm.TypeInfo(description="Description of the issue")],
    ) -> str:
        """Create a support ticket. Collect ALL 4 fields one by one before calling this."""
        lang = 'en'
        existing_ref = str(_current_session.get("ticket_reference") or "").strip()
        if _current_session.get("ticket_created"):
            room_log("TOOL_RESULT_BLOCKED", name="create_support_ticket", reason="already_created")
            if existing_ref:
                return (
                    f"I already created your support request with reference number {existing_ref}. "
                    "One of our colleagues will contact you soon."
                )
            return "I already created your support request. One of our colleagues will contact you soon."

        now = time.time()
        allowed_until = float(_current_session.get("ticket_create_allowed_until") or 0.0)
        pending_payload = {
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "customer_email": customer_email,
            "issue_description": issue_description,
        }
        if now > allowed_until:
            _current_session["pending_ticket_payload"] = pending_payload
            _current_session["ticket_confirmation_pending"] = True
            _current_session["ticket_confirmation_pending_until"] = now + 180.0
            room_log("TOOL_RESULT_BLOCKED", name="create_support_ticket", reason="issue_confirmation_required")
            return (
                "Before I create the support request, I need one final confirmation of the issue you described. "
                "If that's correct, say yes and I will submit it right away."
            )

        _current_session["ticket_create_allowed_until"] = 0.0
        _current_session["ticket_confirmation_pending"] = False
        _current_session["ticket_confirmation_pending_until"] = 0.0
        payload = _current_session.get("pending_ticket_payload") or pending_payload
        _current_session["pending_ticket_payload"] = None
        room_log(
            "TOOL_CALL",
            name="create_support_ticket",
            customer_name=payload.get("customer_name"),
            customer_phone=payload.get("customer_phone"),
            customer_email=payload.get("customer_email"),
        )
        result = await self._run_tool_with_silence_pause(
            "create_support_ticket",
            support_ticket.create_support_ticket(
                payload.get("customer_name"),
                payload.get("customer_phone"),
                payload.get("customer_email"),
                payload.get("issue_description"),
            ),
        )
        room_log("TOOL_RESULT", name="create_support_ticket", result=_truncate(result))
        normalized_result = _normalize_intent_text(result)
        if "cannot create ticket" in normalized_result or "couldn t create the support ticket" in normalized_result:
            return result
        reference = _extract_ticket_reference(result)
        _current_session["ticket_created"] = True
        _current_session["ticket_reference"] = reference
        if lang == "el":
            if reference:
                return (
                    f"Το αίτημα υποστήριξης δημιουργήθηκε με επιτυχία. Ο αριθμός αναφοράς σας είναι {reference}. "
                    "Ένας συνάδελφός μας θα επικοινωνήσει μαζί σας σύντομα."
                )
            return "Το αίτημα υποστήριξης δημιουργήθηκε με επιτυχία. Ένας συνάδελφός μας θα επικοινωνήσει μαζί σας σύντομα."
        if reference:
            return (
                f"Your support request has been created successfully. Your reference number is {reference}. "
                "One of our colleagues will contact you soon."
            )
        return "Your support request has been created successfully. One of our colleagues will contact you soon."

    @llm.ai_callable()
    async def validate_ticket_field(
        self,
        field_name: Annotated[str, llm.TypeInfo(description="Field name: name, phone, email, or issue")],
        value: Annotated[str, llm.TypeInfo(description="Value to validate")],
    ) -> str:
        """Validate a support ticket field value."""
        room_log("TOOL_CALL", name="validate_ticket_field", field=field_name, value=value)
        result = await self._run_tool_with_silence_pause(
            "validate_ticket_field",
            support_ticket.validate_ticket_field(field_name, value),
        )
        room_log("TOOL_RESULT", name="validate_ticket_field", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def log_customer_query(
        self,
        customer_question: Annotated[str, llm.TypeInfo(description="The question or issue you cannot answer")],
        customer_name: Annotated[Optional[str], llm.TypeInfo(description="Customer name if known")] = None,
        customer_phone: Annotated[Optional[str], llm.TypeInfo(description="Customer phone if known")] = None,
    ) -> str:
        """
        Log a customer query you cannot answer for team follow-up.
        Use when:
        - You don't know the answer
        - The question requires human expertise
        - The issue is too complex to resolve
        """
        room_log(
            "TOOL_CALL",
            name="log_customer_query",
            customer_question=customer_question,
            customer_name=customer_name,
            customer_phone=customer_phone,
        )
        result = await self._run_tool_with_silence_pause(
            "log_customer_query",
            support_ticket.log_customer_query(
                customer_question, customer_name, customer_phone
            ),
        )
        room_log("TOOL_RESULT", name="log_customer_query", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def search_knowledge_base(
        self,
        query: Annotated[str, llm.TypeInfo(description="The question to search for")],
    ) -> str:
        """Search the knowledge base for answers to common questions."""
        language = "en"
        room_log("TOOL_CALL", name="search_knowledge_base", query=query, language=language)
        result = await self._run_tool_with_silence_pause(
            "search_knowledge_base",
            knowledge_base.search_knowledge_base(query, language=language),
        )
        room_log("TOOL_RESULT", name="search_knowledge_base", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def get_brand_info(self) -> str:
        """Get information about the Meallion brand."""
        room_log("TOOL_CALL", name="get_brand_info")
        result = await self._run_tool_with_silence_pause(
            "get_brand_info",
            knowledge_base.get_brand_info(),
        )
        room_log("TOOL_RESULT", name="get_brand_info", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def save_agent_memory(
        self,
        question: Annotated[str, llm.TypeInfo(description="The question asked by the user, or topic")],
        answer: Annotated[str, llm.TypeInfo(description="The answer given or the key information")],
        comments: Annotated[str, llm.TypeInfo(description="Any extra comments or feedback provided by user")] = None,
    ) -> str:
        """Save a long-term memory about the user's question, answer, and comments for future training. Use this when you get feedback or an interesting Q/A."""
        lang = 'en'
        try:
            from src.services.database import get_database_service
            db = get_database_service()
            success = await db.add_agent_memory(question, answer, comments, lang)
            if success:
                room_log("TOOL_CALL", name="save_agent_memory", question=question, answer=answer)
                return "Successfully saved to memory."
            return "Failed to save to memory."
        except Exception as e:
            logger.error(f"Error saving memory in tool: {e}")
            return "Failed to save memory due to error."

    @llm.ai_callable()
    async def end_session(self) -> str:
        """
        End the voice call session gracefully.
        Use this when the customer says goodbye, thanks, or indicates they're done.
        
        Examples of when to use:
        - "Goodbye", "Bye", "Thanks, bye"
        - "That's all I needed", "I'm done"
        - "Have a nice day", "Thank you, that's it"
        
        Returns:
            Goodbye message - you MUST speak this message, the call will end after
        """
        global _current_session
        logger.info("Session end requested - scheduling disconnect after goodbye")
        room_log("SESSION_END_REQUESTED")
        
        # Schedule the disconnect with a delay to allow goodbye to be spoken
        async def delayed_end():
            # Wait for LLM to process response + TTS to generate + speak
            # This needs to be long enough for the full goodbye to be heard
            await asyncio.sleep(6.0)  # 6 seconds should be plenty
            _current_session["should_end"] = True
            logger.info("Delayed session end triggered")
        
        asyncio.create_task(delayed_end())
        
        # Return closing message based on language
        goodbye = get_closing("en")
        room_log("SESSION_END_MESSAGE", text=_truncate(goodbye))
        return goodbye


async def create_initial_context(cache_task: asyncio.Task = None) -> llm.ChatContext:
    """Create the initial chat context with system prompt based on configured language.
    
    Args:
        cache_task: Optional task from _fetch_from_db() to wait for before getting prompts.
                   This ensures the cache is ready without re-fetching.
    """
    # Wait for cache to be ready if a task was provided
    if cache_task is not None:
        try:
            await cache_task
        except Exception as e:
            logger.warning(f"Cache task exception (will retry): {e}")
    
    ctx = llm.ChatContext()
    agent_lang = 'en'
    
    # Use async version to ensure KB and prompts are loaded from DB
    system_prompt = await get_system_prompt_async(agent_lang)

    has_memory_block = "LONG-TERM MEMORY" in system_prompt
    logger.info(
        "Using %s system prompt (from database), length=%s chars, memory_block=%s",
        agent_lang,
        len(system_prompt),
        has_memory_block,
    )
    ctx.append(role="system", text=system_prompt)
    return ctx


async def entrypoint(ctx: JobContext):
    """LiveKit Agent entrypoint. Called when a new participant joins the room."""
    global _current_session
    
    startup_time = time.time()
    logger.info(f"Elena EN agent starting for room: {ctx.room.name} (job: {ctx.job.id})")
    logger.warning("VOICE_AGENT_VERSION: phone-confirmation-hard-gate-2026-04-28-v4")
    
    # Track call timing for metrics
    call_start_time = time.time()
    caller_number = None
    caller_identity = None
    db_call_id = None
    
    # Determine call type (SIP or Web)
    is_sip_call = ctx.room.name.startswith("sip-") or "sip" in ctx.room.name.lower()
    call_type = "sip" if is_sip_call else "web"
    
    # Extract caller info from room name if available (e.g., sip-call-+30211234567)
    if is_sip_call:
        parts = ctx.room.name.split("-")
        if len(parts) >= 3:
            caller_number = parts[-1] if parts[-1].startswith("+") else None

    # Create per-room log file (full lifecycle)
    job_obj = getattr(ctx, "job", None)
    job_id = getattr(job_obj, "id", None) or getattr(job_obj, "job_id", None)
    room_logger, room_log_path = _create_room_logger(ctx.room.name, job_id)
    _current_session["room_logger"] = room_logger
    _current_session["room_name"] = ctx.room.name
    _current_session["job_id"] = job_id
    room_log("ROOM_START", call_type=call_type, caller_number=caller_number)
    logger.info(f"Per-room log file: {room_log_path}")
    _reset_support_session_state("new_call_started")
    
    # Reset session state
    _current_session["should_end"] = False
    _current_session["silence_tracker"] = None
    _current_session["silence_pause_depth"] = 0
    _current_session["last_silence_block_reason"] = None
    _current_session["last_lookup_wait_phrase"] = None
    _current_session["pending_lookup_wait_phrase"] = None
    _current_session["pending_lookup_wait_phrase_set_at"] = 0.0
    _current_session["last_agent_text"] = ""
    _current_session["last_forced_lookup_order"] = None
    _current_session["last_forced_lookup_at"] = 0.0
    _current_session["last_user_turn_id"] = 0
    _current_session["last_lookup_tool_called_at"] = 0.0
    _current_session["last_lookup_tool_order"] = None
    _current_session["lookup_progress_prompt_until"] = 0.0
    _current_session["number_mode_lock"] = None
    _current_session["number_mode_turn_id"] = 0
    _current_session["forced_response_manual_say_active"] = False
    _current_session["forced_response_spoken_turn_id"] = 0
    _current_session["forced_response_spoken_text"] = ""
    _current_session["forced_response_suppress_llm_until"] = 0.0
    _current_session["lookup_pending"] = False
    _current_session["lookup_pending_started_at"] = 0.0
    _current_session["lookup_pending_order"] = None
    _current_session["last_lookup_state"] = "unknown"
    _current_session["last_lookup_order"] = None
    _current_session["pending_phone_candidate"] = None
    _current_session["phone_digit_buffer"] = ""
    _current_session["phone_digit_buffer_updated_at"] = 0.0
    _current_session["phone_lookup_inflight"] = False
    _current_session["phone_forced_turn_id"] = 0
    _current_session["phone_forced_pending_turn_id"] = 0
    _current_session["details_lookup_inflight"] = False
    _current_session["details_forced_turn_id"] = 0
    _current_session["details_forced_pending_turn_id"] = 0
    _current_session["details_last_spoken_order"] = None
    _current_session["details_last_spoken_at"] = 0.0
    _current_session["details_confirmation_pending"] = False
    _current_session["details_confirmation_pending_until"] = 0.0
    _current_session["full_order_details_allowed_until"] = 0.0
    _current_session["full_details_unlocked_once"] = False
    _current_session["last_more_details_prompt_at"] = 0.0
    _current_session["ticket_confirmation_pending"] = False
    _current_session["ticket_confirmation_pending_until"] = 0.0
    _current_session["ticket_create_allowed_until"] = 0.0
    _current_session["pending_ticket_payload"] = None
    _current_session["ticket_created"] = False
    _current_session["ticket_reference"] = None
    _current_session["support_flow_state"] = FLOW_IDLE
    _current_session["support_issue"] = None
    
    # =========================================================================
    # PARALLEL STARTUP - Run ALL independent operations concurrently for speed
    # Key insight: DB fetch is slowest (~10s), so start it first and run
    # everything else in parallel
    # =========================================================================
    
    # 1. Start cache refresh FIRST (this is the slowest operation - ~10s)
    from src.agents.prompts import _fetch_from_db
    cache_task = asyncio.create_task(_fetch_from_db())
    
    # 2. Start context creation immediately - it will wait for cache internally
    # This runs in parallel with room connection
    context_task = asyncio.create_task(create_initial_context(cache_task))
    
    # 3. Start other background tasks (fire and forget - don't wait for these)
    asyncio.create_task(log_call_event(
        event_type="call_incoming",
        room_name=ctx.room.name,
        call_type=call_type,
        caller_number=caller_number,
        metadata={"source": "livekit_agent"},
    ))
    
    # 4. Connect to room (runs in parallel with cache fetch + context creation)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"⏱️ Connected to room ({time.time() - startup_time:.1f}s)")
    room_log("ROOM_CONNECTED", elapsed_s=round(time.time() - startup_time, 1))
    
    # 5. Wait for participant (user connects) - runs in parallel with cache fetch
    participant = await ctx.wait_for_participant()
    logger.info(f"⏱️ Participant connected: {participant.identity} ({time.time() - startup_time:.1f}s)")
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity, elapsed_s=round(time.time() - startup_time, 1))
    caller_identity = participant.identity
    
    # Try to get caller number from participant identity for SIP calls
    if is_sip_call and not caller_number:
        caller_number = participant.identity if participant.identity.startswith("+") else None
    
    # Record call start in background (don't block greeting)
    async def record_call_async():
        nonlocal db_call_id
        db_call_id = await record_call_to_db(
            room_name=ctx.room.name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
        )
        await log_call_event(
            event_type="call_connected",
            room_name=ctx.room.name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
            call_id=db_call_id,
            metadata={"source": "livekit_agent"},
        )
        logger.info(f"Call recorded in DB with ID: {db_call_id}")
        _current_session["call_id"] = db_call_id
        room_log("CALL_RECORDED", db_call_id=db_call_id)
    
    record_task = asyncio.create_task(record_call_async())
    
    # Initialize transcript collection
    conversation_transcript = []
    # Track if we already handled call end to avoid duplicate processing and late async sends.
    call_ended = {"value": False}
    
    # Context creation was started earlier (line ~520) in parallel with room connection
    logger.info(f"⏱️ Context creation in progress ({time.time() - startup_time:.1f}s)")
    
    # Track sent payloads to avoid duplicate UI updates.
    _sent_agent_transcripts = set()
    _sent_agent_info_payloads = set()
    _last_user_interim = ""
    _last_user_interim_sent_at = 0.0
    _last_user_final = ""
    
    # Ensure settings cache is ready before requiring DB-backed settings.
    if cache_task is not None:
        try:
            await cache_task
        except Exception as e:
            logger.warning(f"Cache task exception (settings): {e}")

    # Initialize runtime language (per-call) from DB defaults.
    set_runtime_language(None)
    base_language = 'en'
    set_runtime_language(base_language)

    def _normalize_switch_text(text: str) -> str:
        """Normalize text for robust language-switch intent detection."""
        lowered = (text or "").strip().lower()
        if not lowered:
            return ""
        # Keep letters/numbers/spaces only to make phrase matching resilient.
        lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _explicit_more_order_details_request(text: str) -> bool:
        """Detect when caller explicitly asks for full order details."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False

        en_phrases = (
            "more details",
            "full details",
            "complete details",
            "show details",
            "order details",
            "item details",
            "what are the items",
            "what did i order",
            "details please",
        )
        el_phrases = (
            "περισσότερες λεπτομέρειες",
            "όλες τις λεπτομέρειες",
            "αναλυτικές λεπτομέρειες",
            "λεπτομέρειες παραγγελίας",
            "τι περιέχει η παραγγελία",
            "ποια είδη",
        )
        if any(p in lowered for p in en_phrases):
            return True
        if any(p in lowered for p in el_phrases):
            return True

        yes_tokens = {"yes", "yeah", "yep", "sure", "ok", "okay", "ναι", "εντάξει"}
        if lowered in yes_tokens:
            prompted_at = float(_current_session.get("last_more_details_prompt_at") or 0.0)
            if prompted_at and (time.time() - prompted_at) <= 25.0:
                return True
        return False

    # Reuse module-level numeric parsers to avoid divergence between scopes.
    _digits_from_phrase = globals()["_digits_from_phrase"]
    _extract_digit_parts = globals()["_extract_digit_parts"]
    _normalize_order_id_strict = globals()["_normalize_order_id_strict"]
    _normalize_phone_for_lookup = globals()["_normalize_phone_for_lookup"]

    def _format_user_text_for_transcript(raw_text: str) -> str:
        """
        Normalize number-heavy utterances to raw digits for saved transcripts.
        Avoid "[digits: ...]" annotations and keep transcript clean.
        """
        text = (raw_text or "").strip()
        if not text:
            return text

        phone_candidate = _normalize_phone_for_lookup(text)
        if phone_candidate:
            return phone_candidate

        order_candidate = _normalize_order_id_strict(text)
        if order_candidate:
            return order_candidate

        parts = _extract_digit_parts(text)
        joined = "".join(parts)
        if len(joined) >= 4 and _is_digit_collection_utterance(text):
            return joined

        return text

    def _format_agent_text_for_transcript(raw_text: str) -> str:
        """
        Normalize agent phone-confirmation lines to show numeric digits in transcripts.
        This affects transcript/log text only, not tool inputs.
        """
        text = (raw_text or "").strip()
        if not text:
            return text
        if True:
            return text

        if not _is_phone_confirmation_prompt(text):
            return text

        pending_phone = str(_current_session.get("pending_phone_candidate") or "").strip()
        if pending_phone and re.fullmatch(r"\d{7,15}", pending_phone):
            return f"Phone confirmation: {pending_phone}. Is that correct?"

        digits = "".join(_extract_digit_parts(text))
        if len(digits) < 4:
            digits = ""

        if digits:
            return f"Phone confirmation: {digits}. Is that correct?"

        return text

    def _mentions_no_order_number(text: str) -> bool:
        """Detect caller intent that they do not have an order number."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
            
        # Stiffen detection: Check for clear negative markers
        has_negative = bool(re.search(r"(no|don t|do not|doesn t|didn t|δεν|όχι|οχι)", lowered))
        has_order_keyword = bool(re.search(r"(order|number|παραγγελ|αριθμ)", lowered))
        has_have_keyword = bool(re.search(r"(have|έχω|εχω|βρήκα|βρηκα)", lowered))
        
        # Guard: If they say "I have it" (positive), definitely not a "no"
        if has_have_keyword and not has_negative:
            return False
            
        # Must have a negative combined with either "order" or "have"
        return (has_negative and (has_order_keyword or has_have_keyword)) or (lowered in {"no", "όχι", "οχι"})

    def _mentions_missing_confirmation_email(text: str) -> bool:
        """
        Detect when caller indicates they cannot access confirmation email,
        which usually means they cannot provide an order number.
        """
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        return bool(
            re.search(
                r"(didn t get|did not get|don t have|do not have|can t find|cannot find|didn t receive|did not receive).*(confirmation|email|mail)",
                lowered,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"(confirmation|email|mail).*(didn t get|did not get|don t have|do not have|can t find|cannot find|didn t receive|did not receive)",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    def _infer_number_mode(user_text: str, last_agent_text: str) -> Optional[str]:
        """
        Infer whether current numeric capture should be treated as order id or phone.
        Returns 'order', 'phone', or None.
        """
        lowered = _normalize_switch_text(user_text)
        last_lowered = _normalize_switch_text(last_agent_text)
        if not lowered:
            return None

        force_phone_context = _is_phone_flow_active()
        if _mentions_no_order_number(lowered):
            return "phone"

        order_hint = bool(re.search(r"(order|παραγγελ|αριθμ.*παραγγελ)", lowered))
        phone_hint = bool(re.search(r"(phone|mobile|τηλέφων|τηλεφων|κινητ)", lowered))

        asked_for_order = bool(re.search(r"(order number|παραγγελ|αριθμ.*παραγγελ)", last_lowered))
        asked_for_phone = bool(re.search(r"(phone|mobile|τηλέφων|τηλεφων|κινητ)", last_lowered))

        if force_phone_context and not order_hint:
            if phone_hint or asked_for_phone or bool(_extract_digit_parts(lowered)):
                return "phone"

        if phone_hint and not order_hint:
            return "phone"
        if order_hint and not phone_hint:
            return "order"
        if asked_for_phone and asked_for_order and bool(_extract_digit_parts(lowered)):
            return "phone"
        if asked_for_phone and not asked_for_order:
            return "phone"
        if asked_for_order and not asked_for_phone:
            return "order"
        return None

    def _is_digit_collection_utterance(text: str) -> bool:
        lowered = (text or "").lower().strip()
        if not lowered:
            return False
        parts = _extract_digit_parts(lowered)
        return len(parts) >= 3

    def _is_short_utterance(text: str) -> bool:
        lowered = _normalize_switch_text(text)
        if not lowered:
            return True
        words = [w for w in lowered.split() if w]
        return len(words) <= 2

    def _is_phone_confirmation_prompt(text: str) -> bool:
        """Detect prompts that ask the caller to confirm a captured phone number."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        strict_patterns = (
            r"just to confirm,?\s*(?:your )?(?:phone|mobile)(?: number)?\s+is",
            r"is this (?:your )?(?:phone|mobile)(?: number)? correct",
            r"did i get your (?:phone|mobile)(?: number)? right",
            r"can you confirm (?:this|your) (?:phone|mobile)(?: number)?",
            r"(?:επιβεβαι|επιβεβαιώ).*(?:κινητ|τηλέφων)",
            r"(?:τηλέφων|κινητ).*(?:σωστό|σωστα)",
            r"ξανα πειτε το κινητο",
            r"ξανά πείτε το κινητό",
        )
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in strict_patterns)

    def _is_phone_number_collection_prompt(text: str) -> bool:
        """Detect prompts asking user to provide phone number (not confirmation)."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        patterns = (
            r"(provide|give|share|say|repeat).*(phone|mobile)(?: number)?",
            r"(phone|mobile)(?: number)?.*(provide|give|share|say|repeat)",
            r"used for the order",
        )
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

    def _can_unlock_full_details(user_text: str) -> tuple[bool, str]:
        """Allow full details only when we have a found lookup for the same order."""
        if bool(_current_session.get("full_details_unlocked_once")):
            return False, "already_unlocked_once"
        state = str(_current_session.get("last_lookup_state") or "unknown")
        if state != "found":
            return False, "lookup_not_found"
        anchor = re.sub(r"\D", "", str(_current_session.get("last_lookup_order") or ""))
        expected = _expected_order_digits()
        if not re.fullmatch(rf"\d{{{expected}}}", anchor):
            return False, "missing_anchor_order"
        mentioned = _normalize_order_id_strict(user_text)
        if mentioned and mentioned != anchor:
            return False, "order_mismatch"
        return True, "ok"

    def _is_clarification_prompt_text(text: str) -> bool:
        """Detect clarification/repair prompts so silence grace can be extended."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        return bool(
            re.search(
                r"(repeat|say it again|digit by digit|not clear|couldn t hear|didn t hear|did not hear|could not hear|δεν .*άκουσα|δεν .*κατάλαβ|επαναλάβ|ξανά|ψηφίο προς ψηφίο)",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    def _extract_order_number_candidate(text: str) -> Optional[str]:
        """
        Extract likely order number within the configured digit range (e.g. 3-6 digits).
        Prioritizes chunks near order-related keywords to avoid false captures.
        Uses word boundaries to avoid capturing substrings of longer numbers (like phones).
        """
        normalized = (text or "").lower().strip()
        if not normalized:
            return None
        min_d, max_d = _order_digit_range()

        # 1. Try to find explicit digit sequences within range first (using boundaries)
        explicit_runs = re.findall(rf"\b\d{{{min_d},{max_d}}}\b", normalized)
        if explicit_runs:
            # Pick the longest match if multiple exist, or the last one if same length
            best = sorted(explicit_runs, key=len, reverse=True)[0]
            return best

        # 2. Look for digits near keywords
        windows = []
        for match in re.finditer(r"(order(?:\s+number)?|παραγγε\w*|αριθμ\w*)", normalized, flags=re.IGNORECASE):
            windows.append(normalized[match.start(): match.start() + 96])

        if windows:
            for segment in windows:
                candidate = _normalize_order_id_strict(segment)
                if candidate and min_d <= len(candidate) <= max_d:
                    return candidate

        # 3. Last resort: normalize full text
        fallback = _normalize_order_id_strict(normalized)
        if fallback and min_d <= len(fallback) <= max_d:
            return fallback
        return None


    def _should_force_order_lookup(user_text: str, order_number: Optional[str]) -> bool:
        """Return True when we should force immediate lookup_order for this turn."""
        if not order_number:
            return False

        normalized_user = _normalize_switch_text(user_text)
        if re.search(r"(order|παραγγελ|αριθμ)", normalized_user, flags=re.IGNORECASE):
            return True

        last_agent = _normalize_switch_text(str(_current_session.get("last_agent_text") or ""))
        return bool(
            re.search(
                r"(order number|number from your confirmation|παραγγελ|αριθμ)",
                last_agent,
                flags=re.IGNORECASE,
            )
        )


    # Initialize abuse tracker for this session
    from src.utils.abuse_handler import AbuseTracker, check_and_respond_to_abuse
    _abuse_tracker = AbuseTracker()
    abuse_detection_enabled = _require_bool_setting("abuse_detection_enabled")
    
    # =========================================================================
    # SILENCE DETECTION - Prompt user if no response (2 prompts then disconnect)
    # =========================================================================
    silence_timeout_raw = get_agent_setting("silence_timeout_seconds", 12.0)
    max_prompts_raw = get_agent_setting("silence_max_prompts", 3)
    try:
        silence_timeout = float(silence_timeout_raw)
    except (TypeError, ValueError):
        silence_timeout = 12.0
    try:
        max_prompts = int(max_prompts_raw)
    except (TypeError, ValueError):
        max_prompts = 3
    silence_timeout = max(5.0, min(60.0, silence_timeout))
    max_prompts = max(1, min(10, max_prompts))

    silence_tracker = {
        "last_user_speech": time.time(),
        "last_agent_speech": time.time(),
        "prompt_count": 0,
        "max_prompts": max_prompts,
        "silence_timeout": silence_timeout,
        "is_waiting_for_response": False,
        "agent_is_speaking": False,
        "enabled": True,
        "paused_by_tool": False,
        "tool_pause_depth": 0,
        "snooze_until": 0.0,
        "last_lookup_progress_prompt_at": 0.0,
        "lookup_progress_prompt_count": 0,
        "prompt_in_progress": False,  # Prevent concurrent silence prompts
    }
    _current_session["silence_tracker"] = silence_tracker
    
    def reset_silence_timer():
        """Reset the silence timer when user speaks."""
        silence_tracker["last_user_speech"] = time.time()
        silence_tracker["prompt_count"] = 0
        silence_tracker["is_waiting_for_response"] = False
    
    def mark_agent_speaking():
        """Mark that agent just spoke - start silence countdown from now.
        
        We reset BOTH timestamps so the countdown starts fresh after the agent
        finishes speaking. This fixes the bug where the silence prompt would
        trigger immediately after the greeting because last_user_speech was
        set during initialization (before greeting).
        """
        now = time.time()
        silence_tracker["last_agent_speech"] = now
        silence_tracker["last_user_speech"] = now  # Reset user timer too - countdown starts fresh
        silence_tracker["is_waiting_for_response"] = True
    
    import re
    import json as json_module

    async def extract_key_info_ai(text: str) -> list:
        """Keep extraction lightweight for realtime UX."""
        return extract_key_info_fallback(text)

    def extract_key_info_fallback(text: str) -> list:
        """Fallback regex extraction if AI fails."""
        info_items = []
        seen_values = set()
        
        def add_item(item_type, icon, label, value):
            if value and value not in seen_values:
                seen_values.add(value)
                info_items.append({"type": item_type, "icon": icon, "label": label, "value": value})
        
        # Order numbers
        order_matches = re.findall(r'#(\d{4,6})|order\s*(?:number\s*)?#?(\d{4,6})', text, re.I)
        for match in order_matches:
            order_num = match[0] or match[1]
            if order_num:
                add_item("order", "📦", "Order", f"#{order_num}")
        
        # Status
        status_match = re.search(r'\b(delivered|pending|processing|being prepared|shipped|confirmed|cancelled)\b', text, re.I)
        if status_match:
            add_item("status", "✓", "Status", status_match.group(1).title())
        
        # Dates
        date_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b', text, re.I)
        if date_match:
            date_str = f"{date_match.group(1)} {date_match.group(2)}"
            if date_match.group(3):
                date_str += f", {date_match.group(3)}"
            add_item("date", "📅", "Date", date_str)
        
        # Prices
        price_match = re.search(r'(\d+(?:[.,]\d{2})?)\s?(?:EUR|€|euros?)', text, re.I)
        if price_match:
            add_item("price", "💰", "Total", f"€{price_match.group(1)}")
        
        # Customer name
        name_match = re.search(r'customer\s+name[^:]*?is\s+([^\.,]+)', text, re.I)
        if name_match:
            add_item("customer", "👤", "Customer", name_match.group(1).strip())
        
        # Ticket
        ticket_match = re.search(r'(TICKET|TKT|TASK)-([A-Z0-9]+)', text, re.I)
        if ticket_match:
            add_item("ticket", "🎫", "Ticket", f"{ticket_match.group(1).upper()}-{ticket_match.group(2)}")
        
        return info_items
    
    async def send_agent_transcript(text: str, *, source: str = "manual"):
        """Send spoken agent text to frontend chat in realtime.

        To prevent duplicate or unsaid transcript lines, we only publish
        entries that come from committed speech callbacks.
        """
        nonlocal conversation_transcript
        try:
            # Don't drop transcripts if the call is ending; we still want to log them.
            # Only return if we've already handled the final DB recording.
            if call_ended["value"]:
                return
            if source != "commit":
                return
            cleaned = _strip_markup_for_output(text)
            if not cleaned:
                return
            # Short-term de-duplication to avoid double transcripts from manual + automatic paths
            # while still allowing repetitions later in the call.
            last_sent_text = _current_session.get("last_agent_transcript_text")
            last_sent_at = float(_current_session.get("last_agent_transcript_at") or 0.0)
            if cleaned == last_sent_text and (time.time() - last_sent_at) < 3.0:
                return
            _current_session["last_agent_transcript_text"] = cleaned
            _current_session["last_agent_transcript_at"] = time.time()

            import json
            transcript_data = json.dumps({
                "type": "transcript",
                "speaker": "agent",
                "text": cleaned,
            }, ensure_ascii=False)
            await ctx.room.local_participant.publish_data(
                transcript_data.encode('utf-8'),
                reliable=True,
            )
            # Centralized recording for DB persistence
            conversation_transcript.append(f"Agent: {cleaned}")
            
            # Incremental DB save
            call_id = _current_session.get("call_id")
            if call_id:
                asyncio.create_task(save_transcript_to_db(call_id, cleaned, speaker="agent"))
        except Exception as e:
            logger.error(f"Failed to send agent transcript: {e}")

    async def send_agent_info(text: str):
        """Extract and send important information cards without blocking audio."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                return
            cleaned = (text or "").strip()
            if not cleaned:
                return

            key_info = await extract_key_info_ai(cleaned)
            if not key_info:
                return

            payload_key = json_module.dumps(key_info, sort_keys=True, ensure_ascii=False)
            if payload_key in _sent_agent_info_payloads:
                return
            _sent_agent_info_payloads.add(payload_key)
            if len(_sent_agent_info_payloads) > 100:
                _sent_agent_info_payloads.clear()
                _sent_agent_info_payloads.add(payload_key)

            import json
            info_data = json.dumps({
                "type": "info",
                "items": key_info,
            })
            await ctx.room.local_participant.publish_data(
                info_data.encode('utf-8'),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send agent info: {e}")

    last_lookup_status = {
        "status": None,
        "order_number": None,
        "language": None,
        "updated_at": 0.0,
    }

    def _status_sentence(status: str, language: str) -> str:
        sentences = {
            "en": {
                "processing": "Your order is being prepared.",
                "in_transit": "Your order is on the way.",
                "delivered": "Your order has been delivered.",
                "completed": "Your order is completed.",
                "cancelled": "Your order was cancelled.",
            },
            "el": {
                "processing": "Η παραγγελία σας ετοιμάζεται.",
                "in_transit": "Η παραγγελία σας είναι καθ οδόν.",
                "delivered": "Η παραγγελία σας παραδόθηκε.",
                "completed": "Η παραγγελία σας ολοκληρώθηκε.",
                "cancelled": "Η παραγγελία σας ακυρώθηκε.",
            },
        }
        return sentences.get(language, sentences["en"]).get(status, "")

    def _status_keywords(language: str) -> dict:
        if language == "el":
            return {
                "processing": ["ετοιμάζεται"],
                "in_transit": ["καθ οδόν", "σε μεταφορά", "στο δρόμο"],
                "delivered": ["παραδόθηκε"],
                "completed": ["ολοκληρώθηκε"],
                "cancelled": ["ακυρώθηκε", "ακυρωθηκε"],
            }
        return {
            "processing": ["processing", "being processed", "being prepared", "preparing"],
            "in_transit": ["on the way", "in transit", "out for delivery"],
            "delivered": ["delivered"],
            "completed": ["completed"],
            "cancelled": ["cancelled", "canceled"],
        }

    def _sentence_mentions_status(sentence: str, language: str) -> bool:
        lower = sentence.lower()
        for keywords in _status_keywords(language).values():
            for keyword in keywords:
                if keyword in lower:
                    return True
        return False

    def _enforce_order_status(text: str) -> str:
        if not text:
            return text
        if not last_lookup_status["status"]:
            return text
        # Only enforce shortly after lookup to avoid stale overrides
        if time.time() - last_lookup_status["updated_at"] > 180:
            return text

        language = last_lookup_status["language"] or 'en'
        expected_status = last_lookup_status["status"]
        keywords = _status_keywords(language)
        lower = text.lower()

        found_status = None
        for status, words in keywords.items():
            if any(word in lower for word in words):
                found_status = status
                break

        if not found_status:
            return text
        if found_status == expected_status:
            return text

        status_sentence = _status_sentence(expected_status, language)
        if not status_sentence:
            return text

        # Replace any sentence that mentions a status with the correct one
        parts = re.split(r'(?<=[.!?])\s+', text)
        if len(parts) <= 1:
            return status_sentence

        new_parts = []
        replaced = False
        for sentence in parts:
            if _sentence_mentions_status(sentence, language):
                if not replaced:
                    new_parts.append(status_sentence)
                    replaced = True
                continue
            new_parts.append(sentence)

        if not replaced:
            return status_sentence
        return " ".join(p for p in new_parts if p).strip()

    def _is_details_retry_fallback_text(text: str) -> bool:
        """Detect generic retry/failure filler that appears after details were already spoken."""
        normalized = _normalize_switch_text(text)
        if not normalized:
            return False
        return bool(
            re.search(
                r"(issue retrieving the details|trouble getting the detailed information|let me try again|let me try that again|πρόβλημα .*λεπτομέρει|δυσκολ.*λεπτομέρει|ας το ξαναδοκιμάσω)",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    def _should_suppress_tts_text(candidate_text: str) -> tuple[bool, str]:
        """Shared suppression guard for both string and streaming TTS paths."""
        text_value = (candidate_text or "").strip()
        if not text_value:
            return False, ""

        # Never suppress stock silence prompts with this guard.
        if _is_silence_prompt_text(text_value):
            return False, ""

        if not _current_session.get("forced_response_manual_say_active"):
            suppress_until = float(_current_session.get("forced_response_suppress_llm_until") or 0.0)
            suppress_turn = int(_current_session.get("forced_response_spoken_turn_id") or 0)
            latest_turn = int(_current_session.get("last_user_turn_id") or 0)
            if suppress_until and time.time() <= suppress_until and suppress_turn == latest_turn:
                return True, "forced_response_mutual_exclusion"

        # Prevent repeated "please provide phone number" variants in the same user turn
        # after a deterministic/manual phone prompt has already been spoken.
        latest_turn = int(_current_session.get("last_user_turn_id") or 0)
        last_manual_turn = int(_current_session.get("last_manual_prompt_turn_id") or 0)
        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        if (
            flow_state in {FLOW_AWAITING_PHONE_NUMBER, FLOW_CHECKING_PHONE_NUMBER}
            and latest_turn
            and last_manual_turn == latest_turn
            and _is_phone_number_collection_prompt(text_value)
        ):
            return True, "duplicate_phone_collection_prompt_same_turn"

        retry_guard_s = _as_float(
            get_agent_setting("forced_details_retry_guard_seconds", 120.0),
            120.0,
            min_value=20.0,
            max_value=300.0,
        )
        last_details_spoken_at = float(_current_session.get("details_last_spoken_at") or 0.0)
        if (
            last_details_spoken_at
            and (time.time() - last_details_spoken_at) <= retry_guard_s
            and _is_details_retry_fallback_text(text_value)
        ):
            return True, "forced_details_retry_guard"

        return False, ""

    def before_tts_callback(agent_instance, text: str | llm.LLMStream):
        """Callback that fires BEFORE text is sent to TTS - apply prosody."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                room_log("LATE_EVENT_DROPPED", source="before_tts_callback")
                return ""
            # text can be a string or LLMStream - handle both
            if isinstance(text, str):
                logger.info(f"before_tts_cb processing: {text[:50]}...")
                suppress, suppress_reason = _should_suppress_tts_text(text)
                if suppress:
                    room_log(
                        "TTS_SUPPRESSED",
                        reason=suppress_reason,
                        turn_id=int(_current_session.get("last_user_turn_id") or 0),
                        text=_truncate(text),
                    )
                    return ""
                if _is_lookup_wait_ack_only_text(text):
                    lookup_recent_s = _as_float(
                        get_agent_setting("lookup_wait_ack_require_tool_window_seconds", 4.0),
                        4.0,
                        min_value=1.0,
                        max_value=20.0,
                    )
                    last_lookup_called_at = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                    lookup_started = bool(
                        _current_session.get("lookup_pending")
                        or _current_session.get("phone_lookup_inflight")
                        or _current_session.get("details_lookup_inflight")
                    ) or (
                        last_lookup_called_at and (time.time() - last_lookup_called_at) <= lookup_recent_s
                    )
                    if not lookup_started:
                        mode = str(_current_session.get("number_mode_lock") or "phone")
                        replacement = _repeat_number_prompt_for_mode(mode)
                        _current_session["pending_lookup_wait_phrase"] = None
                        _current_session["pending_lookup_wait_phrase_set_at"] = 0.0
                        invalid_recovery_grace = _as_float(
                            get_agent_setting("invalid_number_recovery_silence_grace_seconds", 12.0),
                            12.0,
                            min_value=4.0,
                            max_value=40.0,
                        )
                        _snooze_silence_prompts(invalid_recovery_grace, reason="invalid_number_recovery")
                        room_log(
                            "LOOKUP_WAIT_ACK_REPLACED",
                            reason="lookup_not_started",
                            original=_truncate(text),
                            replacement=_truncate(replacement),
                        )
                        text = replacement
                strict_wait_phrase = _as_bool(
                    get_agent_setting("order_lookup_wait_phrase_strict", True),
                    default=True,
                )
                if strict_wait_phrase:
                    pending_phrase = _current_session.get("pending_lookup_wait_phrase")
                    pending_set_at = float(_current_session.get("pending_lookup_wait_phrase_set_at") or 0.0)
                    wait_phrase_window_s = _as_float(
                        get_agent_setting("lookup_wait_ack_require_tool_window_seconds", 4.0),
                        4.0,
                        min_value=1.0,
                        max_value=20.0,
                    )
                    last_lookup_called_at = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                    lookup_active = bool(
                        _current_session.get("lookup_pending")
                        or _current_session.get("phone_lookup_inflight")
                        or _current_session.get("details_lookup_inflight")
                    ) or (
                        last_lookup_called_at and (time.time() - last_lookup_called_at) <= wait_phrase_window_s
                    )
                    if pending_phrase and (time.time() - pending_set_at) <= 20.0:
                        if _is_silence_prompt_text(text):
                            room_log("TOOL_WAIT_ACK_SKIPPED", reason="silence_prompt")
                        elif not lookup_active:
                            _clear_pending_lookup_wait_phrase("lookup_not_active")
                        else:
                            if pending_phrase.lower() not in text.lower():
                                text = f"{pending_phrase} {text}".strip()
                                room_log("TOOL_WAIT_ACK_ENFORCED", phrase=_truncate(pending_phrase))
                            _clear_pending_lookup_wait_phrase("consumed")
                    elif pending_phrase:
                        # Expire stale pending phrase to avoid polluting unrelated turns.
                        _clear_pending_lookup_wait_phrase("expired")
                # Language lock removed: English-only module

                from src.utils import (
                    apply_prosody,
                    normalize_time_colons,
                    normalize_numeric_ids_for_tts,
                    normalize_punctuation_for_tts,
                )
                agent_lang = 'en'
                use_ssml = _as_bool(get_agent_setting("tts_use_ssml", False), default=False)
                tts_engine = getattr(agent_instance, "_tts", None) or getattr(agent_instance, "tts", None)
                tts_provider = "unknown"
                supports_ssml = False
                if tts_engine is not None:
                    if hasattr(tts_engine, "current_provider_name"):
                        try:
                            tts_provider = tts_engine.current_provider_name()
                        except Exception:
                            tts_provider = str(getattr(tts_engine, "_provider_name", "unknown"))
                    else:
                        tts_provider = str(getattr(tts_engine, "_provider_name", "unknown"))

                    if hasattr(tts_engine, "supports_ssml"):
                        try:
                            supports_ssml = bool(tts_engine.supports_ssml())
                        except Exception:
                            supports_ssml = bool(getattr(tts_engine, "_supports_ssml", False))
                    else:
                        supports_ssml = bool(getattr(tts_engine, "_supports_ssml", False))
                if tts_provider == "unknown":
                    tts_provider = str(_current_session.get("tts_provider") or "unknown")

                if use_ssml and not supports_ssml:
                    room_log(
                        "SSML_DISABLED_FOR_PROVIDER",
                        requested=True,
                        provider=tts_provider,
                    )
                    use_ssml = False

                tts_text = _enforce_order_status(text)
                sanitized_tts_text = _strip_tts_style_leakage(tts_text)
                if sanitized_tts_text != tts_text:
                    room_log(
                        "TTS_STYLE_TEXT_STRIPPED",
                        before=_truncate(tts_text, max_len=200),
                        after=_truncate(sanitized_tts_text, max_len=200),
                    )
                tts_text = sanitized_tts_text
                tts_text = normalize_time_colons(tts_text)
                tts_text = normalize_numeric_ids_for_tts(tts_text, language=agent_lang)
                tts_text = normalize_punctuation_for_tts(tts_text)
                cleaned_ui_text = _strip_markup_for_output(tts_text)
                if cleaned_ui_text:
                    room_log(
                        "TTS_TEXT_FINAL",
                        provider=tts_provider,
                        ssml=use_ssml,
                        text=_truncate(cleaned_ui_text),
                    )
                processed_text = apply_prosody(tts_text, language=agent_lang, use_ssml=use_ssml)
                return processed_text
            else:
                logger.debug("before_tts_cb got LLMStream (applying streaming normalization)")
                from src.utils import (
                    normalize_time_colons,
                    normalize_numeric_ids_for_tts,
                    normalize_punctuation_for_tts,
                )
                agent_lang = 'en'

                def _extract_chunk_text(chunk) -> str:
                    if isinstance(chunk, str):
                        return chunk
                    for attr in ("text", "delta", "content"):
                        if hasattr(chunk, attr):
                            value = getattr(chunk, attr)
                            if isinstance(value, str):
                                return value
                    return str(chunk)

                async def _normalized_stream(stream):
                    raw_buffer = ""
                    normalized_buffer = ""
                    async for chunk in stream:
                        if _current_session.get("forced_response_manual_say_active"):
                            room_log("LLM_STREAM_CANCELLED", reason="manual_response_active", turn_id=int(_current_session.get("last_user_turn_id") or 0))
                            return
                        chunk_text = _extract_chunk_text(chunk)
                        if not chunk_text:
                            continue
                        raw_buffer += chunk_text
                        updated = _enforce_order_status(raw_buffer)
                        updated = _strip_tts_style_leakage(updated)
                        # Language lock removed: English-only module
                        updated = normalize_time_colons(updated)
                        updated = normalize_numeric_ids_for_tts(updated, language=agent_lang)
                        updated = normalize_punctuation_for_tts(updated)
                        suppress, suppress_reason = _should_suppress_tts_text(updated)
                        if suppress:
                            room_log(
                                "TTS_SUPPRESSED",
                                reason=suppress_reason,
                                turn_id=int(_current_session.get("last_user_turn_id") or 0),
                                text=_truncate(updated),
                            )
                            return
                        if updated.startswith(normalized_buffer):
                            delta = updated[len(normalized_buffer):]
                        else:
                            delta = updated
                        normalized_buffer = updated
                        if delta:
                            yield delta

                return _normalized_stream(text)
        except Exception as e:
            logger.error(f"Error in before_tts_callback: {e}")
        return text  # Return text unchanged on error
    # Wait for context that was started earlier (should be ready by now)
    logger.info("⏱️ Awaiting context_task...")
    initial_ctx = await context_task
    logger.info(f"⏱️ Context ready ({time.time() - startup_time:.1f}s)")
    
    # Language-aware endpointing delay: Greek requires more patience for complete transcripts.
    initial_lang = 'en'
    if initial_lang == "el":
        default_endpointing = 2.8 if is_sip_call else 2.5
    else:
        default_endpointing = 2.5 if is_sip_call else 2.2
    
    # Create the voice pipeline agent - tuned to avoid clipping user speech.
    min_endpointing_delay = _as_float(
        get_agent_setting("min_endpointing_delay", default_endpointing),
        default_endpointing,
        min_value=0.2,
        max_value=3.0,
    )
    interrupt_min_words = _as_int(
        get_agent_setting("interrupt_min_words", 3),
        3,
        min_value=1,
        max_value=10,
    )
    logger.info(
        "Turn config: min_endpointing_delay=%.2fs interrupt_min_words=%s",
        min_endpointing_delay,
        interrupt_min_words,
    )
    logger.info("Language mode: English-only (no auto switch), call_type=%s", call_type)
    preemptive_synthesis = _as_bool(
        get_agent_setting("preemptive_synthesis", True),
        default=True,
    )
    logger.info("TTS preemptive_synthesis=%s", preemptive_synthesis)
    logger.info(f"⏱️ Creating agent ({time.time() - startup_time:.1f}s)")
    agent = VoicePipelineAgent(
        vad=create_vad(is_sip_call=is_sip_call),
        stt=create_stt(is_sip_call=is_sip_call),
        llm=create_llm(),
        tts=create_tts(),
        chat_ctx=initial_ctx,
        fnc_ctx=ElenaFunctionContext(),
        max_nested_fnc_calls=3,
        # Turn segmentation settings.
        min_endpointing_delay=min_endpointing_delay,
        preemptive_synthesis=preemptive_synthesis,  # Allow toggling to avoid punctuation leaks
        allow_interruptions=True,       # Allow user to interrupt
        interrupt_min_words=interrupt_min_words,
        before_tts_cb=before_tts_callback,  # Capture text when possible
    )
    logger.info(f"⏱️ Agent created ({time.time() - startup_time:.1f}s)")
    
    # ==========================================================================
    # LATENCY TRACKING - Register event handlers
    # ==========================================================================
    async def send_state_update(state: str):
        """Helper to send agent state updates to frontend."""
        try:
            import json
            state_data = json.dumps({
                "type": "state",
                "state": state
            })
            await ctx.room.local_participant.publish_data(
                state_data.encode('utf-8'),
                reliable=True
            )
            logger.debug(f"🎯 Agent state sent: {state}")
        except Exception as e:
            logger.error(f"Failed to send state update: {e}")
    
    thinking_task: Optional[asyncio.Task] = None

    def cancel_thinking_task():
        nonlocal thinking_task
        if thinking_task and not thinking_task.done():
            thinking_task.cancel()
        thinking_task = None

    def schedule_thinking_state(delay_s: float = 0.35):
        nonlocal thinking_task
        cancel_thinking_task()

        async def _set_thinking():
            try:
                await asyncio.sleep(delay_s)
                await send_state_update("thinking")
            except asyncio.CancelledError:
                return

        thinking_task = asyncio.create_task(_set_thinking())

    @agent.on("user_started_speaking")
    def on_user_started_speaking():
        _latency_tracker.user_started_speaking()
        cancel_thinking_task()
        asyncio.create_task(send_state_update("listening"))
        # Immediately stop silence detection when user starts talking
        # This prevents "are you there?" from interrupting the user mid-speech
        silence_tracker["is_waiting_for_response"] = False
        silence_tracker["user_is_speaking"] = True
        silence_tracker["last_user_speech"] = time.time()
    
    @agent.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        _latency_tracker.user_stopped_speaking()
        silence_tracker["user_is_speaking"] = False
        schedule_thinking_state()

    async def send_user_transcript(text: str, *, interim: bool = False):
        """Helper to send user transcript to frontend."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                return
            cleaned = (text or "").strip()
            if not cleaned:
                return

            nonlocal _last_user_interim, _last_user_interim_sent_at, _last_user_final
            now = time.monotonic()

            if interim:
                # Throttle interim updates to avoid flooding the UI.
                if cleaned == _last_user_interim and (now - _last_user_interim_sent_at) < 0.35:
                    return
                _last_user_interim = cleaned
                _last_user_interim_sent_at = now
            else:
                # Avoid duplicate finals, but always override interim if present.
                if cleaned == _last_user_final and cleaned != _last_user_interim:
                    return
                _last_user_final = cleaned
                _last_user_interim = ""

            import json
            transcript_data = json.dumps({
                "type": "transcript",
                "speaker": "user",
                "text": cleaned,
                "interim": interim,
            }, ensure_ascii=False)
            await ctx.room.local_participant.publish_data(
                transcript_data.encode('utf-8'),
                reliable=True
            )
            logger.debug(f"?? User transcript sent: {cleaned[:50]}...")
            if not interim:
                room_log("USER_TEXT", text=_truncate(cleaned))
        except Exception as e:
            logger.error(f"Failed to send user transcript: {e}")

    async def _force_get_order_details(turn_id: int, trigger_reason: str) -> None:
        """Deterministically fetch and speak order details after explicit unlock."""
        if call_ended["value"] or _current_session.get("should_end"):
            room_log("ORDER_DETAILS_FORCED_SKIP", reason="call_ended", turn_id=turn_id)
            return

        if bool(_current_session.get("details_lookup_inflight")):
            room_log("ORDER_DETAILS_FORCED_SKIP", reason="already_inflight", turn_id=turn_id)
            return

        last_forced_turn = int(_current_session.get("details_forced_turn_id") or 0)
        if turn_id <= last_forced_turn:
            room_log(
                "ORDER_DETAILS_FORCED_SKIP",
                reason="duplicate_turn",
                turn_id=turn_id,
                last_forced_turn=last_forced_turn,
            )
            return

        expected = _expected_order_digits()
        order_number = re.sub(r"\D", "", str(_current_session.get("last_lookup_order") or ""))
        if not re.fullmatch(rf"\d{{{expected}}}", order_number):
            room_log(
                "ORDER_DETAILS_FORCED_SKIP",
                reason="missing_anchor_order",
                order_number=order_number,
                turn_id=turn_id,
            )
            return

        last_spoken_order = re.sub(r"\D", "", str(_current_session.get("details_last_spoken_order") or ""))
        last_spoken_at = float(_current_session.get("details_last_spoken_at") or 0.0)
        if (
            order_number
            and last_spoken_order == order_number
            and (time.time() - last_spoken_at) <= 20.0
        ):
            room_log(
                "ORDER_DETAILS_FORCED_SKIP",
                reason="recently_spoken",
                order_number=order_number,
                turn_id=turn_id,
            )
            return

        forced_suppress_s = _as_float(
            get_agent_setting("forced_details_llm_suppress_seconds", 120.0),
            120.0,
            min_value=20.0,
            max_value=300.0,
        )
        _current_session["details_forced_pending_turn_id"] = turn_id
        _current_session["details_lookup_inflight"] = True
        _current_session["details_forced_turn_id"] = turn_id
        _current_session["forced_response_spoken_turn_id"] = turn_id
        _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
        _set_lookup_pending(order_number, reason="forced_get_order_details")
        _pause_silence_for_tool("forced_get_order_details")

        try:
            room_log(
                "TOOL_CALL",
                name="get_order_details",
                order_number=order_number,
                forced=True,
                trigger=trigger_reason,
            )
            result = await order_lookup.get_order_details(order_number)
            room_log(
                "TOOL_RESULT",
                name="get_order_details",
                result=_truncate(result),
                forced=True,
                trigger=trigger_reason,
            )

            spoken_summary = _build_order_details_voice_summary(result)
            if not spoken_summary:
                spoken_summary = _build_order_voice_summary(result) or (
                    "I could not find details for this order."
                    
                )

            prefix = (
                "Thanks for waiting."
                
            )
            final_text = f"{prefix} {spoken_summary}".strip()
            room_log(
                "ORDER_DETAILS_FORCED_FORMATTED",
                order_number=order_number,
                result=_truncate(final_text),
            )
            room_log("ORDER_DETAILS_FORCED_SPEAKING", order_number=order_number, turn_id=turn_id)
            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_text"] = final_text
            await agent.say(final_text, allow_interruptions=True)
            _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
            mark_agent_speaking()
            _snooze_silence_prompts(10.0, reason="post_forced_order_details_spoken")
            _clear_lookup_pending(reason="forced_order_details_spoken")
            _current_session["details_last_spoken_order"] = order_number
            _current_session["details_last_spoken_at"] = time.time()
            room_log("ORDER_DETAILS_FORCED_SPOKEN", order_number=order_number, turn_id=turn_id)
        except Exception as e:
            room_log(
                "ORDER_DETAILS_FORCED_ERROR",
                order_number=order_number,
                trigger=trigger_reason,
                error=_truncate(str(e), max_len=200),
            )
            _clear_lookup_pending(reason="forced_order_details_error")
        finally:
            _current_session["forced_response_manual_say_active"] = False
            _current_session["details_lookup_inflight"] = False
            if int(_current_session.get("details_forced_pending_turn_id") or 0) == turn_id:
                _current_session["details_forced_pending_turn_id"] = 0
            _resume_silence_for_tool("forced_get_order_details")

    async def _force_lookup_order_by_number(turn_id: int, order_number: str, scheduled_at: float) -> None:
        """
        Watchdog: wait a few seconds for the LLM to call lookup_order itself.
        If the LLM doesn't start the tool within the grace window, call it directly.
        """
        # Ensure silence monitor doesn't fire while we wait for the LLM
        _snooze_silence_prompts(30.0, reason="order_watchdog_grace")
        _pause_silence_for_tool("order_watchdog_grace")

        try:
            # Grace window — give the LLM time to initiate the tool call itself.
            watchdog_grace_s = 4.0
            await asyncio.sleep(watchdog_grace_s)

            if call_ended["value"] or _current_session.get("should_end"):
                return

            # Check if the LLM already started the lookup via the tool
            last_tool_call = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
            if last_tool_call >= scheduled_at:
                room_log("ORDER_WATCHDOG_SKIPPED", reason="llm_started_tool", turn_id=turn_id)
                # Skip without clearing lookup_pending (LLM is currently using it)
                _resume_silence_for_tool("order_watchdog_grace")
                _current_session["forced_response_manual_say_active"] = False
                return

            # If the flow already advanced past checking (e.g. order found), do nothing.
            current_flow = str(_current_session.get("support_flow_state") or FLOW_IDLE)
            if current_flow in {FLOW_ORDER_FOUND}:
                room_log("ORDER_WATCHDOG_SKIPPED", reason="order_already_found", turn_id=turn_id)
                _resume_silence_for_tool("order_watchdog_grace")
                _current_session["forced_response_manual_say_active"] = False
                return

            # Deduplication: skip if a newer turn has already started.
            latest_turn = int(_current_session.get("last_user_turn_id") or 0)
            if latest_turn > turn_id:
                room_log("ORDER_WATCHDOG_SKIPPED", reason="newer_turn", turn_id=turn_id, latest=latest_turn)
                return

            # Check if order_number is still valid.
            strict_order = _normalize_order_id_strict(order_number)
            if not strict_order:
                room_log("ORDER_WATCHDOG_SKIPPED", reason="invalid_order_number", order=order_number)
                return

            room_log("ORDER_WATCHDOG_FIRED", turn_id=turn_id, order_number=strict_order)
            logger.info(f"🚨 Order watchdog fired: LLM didn't call lookup_order for #{strict_order}, running directly")

            forced_suppress_s = _as_float(
                get_agent_setting("forced_phone_llm_suppress_seconds", 90.0),
                90.0,
                min_value=15.0,
                max_value=300.0,
            )
            _set_support_flow_state(FLOW_CHECKING_ORDER_NUMBER, reason="order_watchdog_fired")
            _set_lookup_pending(strict_order, reason="order_watchdog_started")
            _current_session["forced_response_spoken_turn_id"] = turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
            _current_session["forced_response_manual_say_active"] = True
            _snooze_silence_prompts(45.0, reason="order_watchdog_started")
            _pause_silence_for_tool("order_watchdog")

            result = await order_lookup.lookup_order(strict_order)
            room_log("ORDER_WATCHDOG_RESULT", result=_truncate(result), order_number=strict_order)

            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            _current_session["last_lookup_order"] = strict_order if lookup_state == "found" else ""

            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason="order_watchdog_found")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason=f"order_watchdog_{lookup_state}")
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0

            spoken_summary = _build_order_voice_summary(result) or result
            await send_agent_transcript(spoken_summary)
            agent.chat_ctx.append(role="assistant", text=spoken_summary)
            live_agent = _current_session.get("agent")
            if live_agent:
                await live_agent.say(spoken_summary, allow_interruptions=True)
                mark_agent_speaking()
                _snooze_silence_prompts(10.0, reason="order_watchdog_spoken")
        except Exception as e:
            room_log("ORDER_WATCHDOG_ERROR", error=_truncate(str(e), max_len=200), order_number=str(order_number))
            logger.error(f"Order watchdog error for #{order_number}: {e}")
        finally:
            _resume_silence_for_tool("order_watchdog_grace")
            _resume_silence_for_tool("order_watchdog")
            _current_session["forced_response_manual_say_active"] = False
            _clear_lookup_pending(reason="order_watchdog_finished")

    async def _force_lookup_by_phone(turn_id: int, phone: str, trigger_reason: str, scheduled_at: float = 0.0) -> None:
        """
        Watchdog/Force: run lookup_order_by_phone and speak the result.
        If scheduled_at > 0, it acts as a watchdog (waiting for LLM first).
        """
        # Ensure silence monitor doesn't fire while we wait/work
        _snooze_silence_prompts(30.0, reason="phone_watchdog_grace")
        _pause_silence_for_tool("phone_watchdog_grace")

        try:
            if scheduled_at > 0.0:
                # Watchdog mode: wait for LLM to act
                watchdog_grace_s = 4.0
                await asyncio.sleep(watchdog_grace_s)

                if call_ended["value"] or _current_session.get("should_end"):
                    return

                # Check if the LLM already started a lookup via a tool
                last_tool_call = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                if last_tool_call >= scheduled_at:
                    room_log("PHONE_WATCHDOG_SKIPPED", reason="llm_started_tool", turn_id=turn_id)
                    # Important: skip WITHOUT clearing lookup_pending.
                    _resume_silence_for_tool("phone_watchdog_grace")
                    _current_session["phone_lookup_inflight"] = False
                    _current_session["forced_response_manual_say_active"] = False
                    return

                # If flow already advanced (e.g. order found), skip watchdog
                current_flow = str(_current_session.get("support_flow_state") or FLOW_IDLE)
                if current_flow in {FLOW_ORDER_FOUND}:
                    room_log("PHONE_WATCHDOG_SKIPPED", reason="order_already_found", turn_id=turn_id)
                    _resume_silence_for_tool("phone_watchdog_grace")
                    _current_session["phone_lookup_inflight"] = False
                    _current_session["forced_response_manual_say_active"] = False
                    return

            # Proceed with forced lookup
            if call_ended["value"] or _current_session.get("should_end"):
                return

            normalized_phone = _normalize_phone_for_lookup(phone or "")
            if not normalized_phone:
                room_log("PHONE_LOOKUP_FORCED_SKIP", reason="invalid_phone", phone=phone)
                return

            if bool(_current_session.get("phone_lookup_inflight")):
                room_log("PHONE_LOOKUP_FORCED_SKIP", reason="already_inflight", turn_id=turn_id)
                return

            last_forced_turn = int(_current_session.get("phone_forced_turn_id") or 0)
            if turn_id < last_forced_turn:
                return

            room_log("PHONE_WATCHDOG_FIRED", turn_id=turn_id, phone=normalized_phone, trigger=trigger_reason)
            logger.info(f"🚨 Phone watchdog fired for {normalized_phone} (reason: {trigger_reason})")

            forced_suppress_s = _as_float(
                get_agent_setting("forced_phone_llm_suppress_seconds", 90.0),
                90.0,
                min_value=15.0,
                max_value=300.0,
            )
            _current_session["phone_forced_pending_turn_id"] = turn_id
            _current_session["phone_lookup_inflight"] = True
            attempt_id = _start_phone_lookup_attempt(normalized_phone, f"forced:{trigger_reason}")
            _current_session["phone_forced_turn_id"] = turn_id
            _current_session["forced_response_spoken_turn_id"] = turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
            _current_session["forced_response_manual_say_active"] = True
            _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason=f"forced_phone_lookup:{trigger_reason}")
            _set_lookup_pending(normalized_phone, reason="phone_watchdog_started")
            _snooze_silence_prompts(45.0, reason="phone_watchdog_started")
            _pause_silence_for_tool("phone_watchdog_work")

            wait_msg = "Just a moment, I am searching for your order."
            live_agent = _current_session.get("agent")
            if live_agent and scheduled_at > 0.0: # Only say wait msg if we are taking over from LLM
                 await live_agent.say(wait_msg, allow_interruptions=False)
                 await send_agent_transcript(wait_msg)
                 agent.chat_ctx.append(role="assistant", text=wait_msg)

            result = await order_lookup.lookup_order_by_phone(normalized_phone)
            if not _is_phone_lookup_attempt_active(attempt_id, normalized_phone):
                room_log(
                    "PHONE_WATCHDOG_RESULT_DISCARDED",
                    attempt_id=attempt_id,
                    phone=normalized_phone,
                    reason="stale_forced_attempt",
                )
                return
            room_log("PHONE_WATCHDOG_RESULT", result=_truncate(result), phone=normalized_phone)

            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            
            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason="phone_watchdog_found")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason=f"phone_watchdog_{lookup_state}")
                _current_session["details_confirmation_pending"] = False

            spoken_summary = _build_phone_lookup_voice_summary(result) or result
            await send_agent_transcript(spoken_summary)
            agent.chat_ctx.append(role="assistant", text=spoken_summary)
            if live_agent:
                await live_agent.say(spoken_summary, allow_interruptions=True)
                mark_agent_speaking()
                _snooze_silence_prompts(10.0, reason="phone_watchdog_spoken")

        except Exception as e:
            room_log("PHONE_WATCHDOG_ERROR", error=str(e), phone=phone)
            logger.error(f"Phone watchdog error: {e}")
        finally:
            _resume_silence_for_tool("phone_watchdog_grace")
            _resume_silence_for_tool("phone_watchdog_work")
            if _is_phone_lookup_attempt_active(locals().get("attempt_id", 0), normalized_phone if 'normalized_phone' in locals() else ""):
                _current_session["phone_lookup_inflight"] = False
                _current_session["forced_response_manual_say_active"] = False
                _clear_lookup_pending(reason="phone_watchdog_finished")

    async def _speak_phone_confirmation_prompt(
        turn_id: int,
        phone: str,
        trigger_reason: str,
        *,
        reprompt: bool = False,
    ) -> None:
        """Speak phone confirmation directly from code, independent of LLM instruction following."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                room_log("PHONE_CONFIRMATION_SKIP", reason="call_ended", turn_id=turn_id)
                return

            normalized_phone = _normalize_phone_for_lookup(phone or "")
            if not normalized_phone:
                room_log(
                    "PHONE_CONFIRMATION_SKIP",
                    reason="invalid_phone_pattern",
                    turn_id=turn_id,
                    phone=_truncate(phone, max_len=64),
                )
                return

            latest_turn = int(_current_session.get("last_user_turn_id") or 0)
            if latest_turn > turn_id:
                room_log(
                    "PHONE_CONFIRMATION_SKIP",
                    reason="newer_user_turn",
                    turn_id=turn_id,
                    latest_turn=latest_turn,
                )
                return

            pending_phone = str(_current_session.get("pending_phone_candidate") or "")
            if pending_phone and pending_phone != normalized_phone:
                room_log(
                    "PHONE_CONFIRMATION_SKIP",
                    reason="pending_phone_mismatch",
                    turn_id=turn_id,
                    pending_phone=pending_phone,
                    phone=normalized_phone,
                )
                return

            if not _is_phone_confirmation_pending():
                room_log("PHONE_CONFIRMATION_SKIP", reason="not_awaiting_confirmation", turn_id=turn_id)
                return

            suppress_s = _as_float(
                get_agent_setting("phone_confirmation_llm_suppress_seconds", 18.0),
                18.0,
                min_value=8.0,
                max_value=60.0,
            )
            confirmation_snooze_s = _as_float(
                get_agent_setting("phone_confirmation_silence_snooze_seconds", 10.0),
                10.0,
                min_value=4.0,
                max_value=30.0,
            )

            _current_session["forced_response_spoken_turn_id"] = turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
            spoken_phone = _speak_digits(normalized_phone)

            if False:
                if reprompt:
                    confirmation_text = (
                        f"Για να συνεχίσουμε, απαντήστε μόνο ναι ή όχι. Ο αριθμός είναι {spoken_phone}. Είναι σωστός;"
                    )
                else:
                    confirmation_text = f"Για επιβεβαίωση, ο αριθμός τηλεφώνου σας είναι {spoken_phone}. Είναι σωστός;"
            else:
                if reprompt:
                    confirmation_text = (
                        f"To continue, please answer only yes or no. The number is {spoken_phone}. Is that correct?"
                    )
                else:
                    confirmation_text = f"Just to confirm, your phone number is {spoken_phone}. Is that correct?"

            room_log(
                "PHONE_CONFIRMATION_PROMPT",
                turn_id=turn_id,
                phone=normalized_phone,
                reason=trigger_reason,
                reprompt=reprompt,
            )

            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_text"] = confirmation_text
            _current_session["forced_response_transcript_sent_at"] = time.time()
            await send_agent_transcript(confirmation_text)
            agent.chat_ctx.append(role="assistant", text=confirmation_text)
            await agent.say(confirmation_text, allow_interruptions=True)
            _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
            mark_agent_speaking()
            _snooze_silence_prompts(confirmation_snooze_s, reason="phone_confirmation_prompt")
            room_log("PHONE_CONFIRMATION_SPOKEN", turn_id=turn_id, reason=trigger_reason)
        finally:
            _current_session["forced_response_manual_say_active"] = False

    @agent.on("user_speech_committed")
    def on_user_speech_committed(message):
        """Send user transcript to frontend and check for abuse."""
        silence_tracker["llm_is_generating"] = True
        # Extract text first so we can log it even if the call is ending
        user_text = message.content
        user_text_for_transcript = _format_user_text_for_transcript(user_text)

        # Always add to internal transcript list and DB
        conversation_transcript.append(f"User: {user_text_for_transcript}")
        
        call_id = _current_session.get("call_id")
        if call_id:
            asyncio.create_task(save_transcript_to_db(call_id, user_text_for_transcript, speaker="user"))

        # Now check if we should skip the rest of the processing (LLM/Tools)
        if call_ended["value"] or _current_session.get("should_end"):
            room_log("LATE_USER_UTTERANCE_LOGGED_BUT_SKIPPED", source="user_speech_committed")
            return
        current_turn_id = int(_current_session.get("last_user_turn_id") or 0) + 1
        _current_session["last_user_turn_id"] = current_turn_id
        
        # Determine initial flow state
        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        is_affirmative = _is_affirmative_utterance(user_text)
        is_negative = _is_negative_utterance(user_text)

        # --- Early Intent/Mode Inference ---
        # We infer the number mode early to handle explicit requests to switch between 
        # Order ID and Phone Number lookups (e.g., "please search with order ID instead").
        inferred_mode = _infer_number_mode(user_text, str(_current_session.get("last_agent_text") or ""))
        
        if inferred_mode == "order" and flow_state in PHONE_FLOW_STATES:
            _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason="explicit_mode_switch_to_order")
            flow_state = FLOW_AWAITING_ORDER_NUMBER
            _reset_phone_digit_buffer("switched_to_order_mode")
            _current_session["pending_phone_candidate"] = None
            # Inject a system hint to guide the LLM back to order-id collection
            agent.chat_ctx.append(
                role="system",
                text="The user wants to search by Order ID instead. Please ask for their order number now."
            )
        elif inferred_mode == "phone" and flow_state in {FLOW_IDLE, FLOW_AWAITING_ORDER_NUMBER, FLOW_CHECKING_ORDER_NUMBER}:
            _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="explicit_mode_switch_to_phone")
            flow_state = FLOW_AWAITING_PHONE_NUMBER
            # Inject a system hint to guide the LLM to phone-number collection
            agent.chat_ctx.append(
                role="system",
                text="The user wants to search by Phone Number instead. Please ask for the mobile number used for the order."
            )

        phone_flow_active = flow_state in PHONE_FLOW_STATES
        normalized_order_candidate = (
            _normalize_order_id_strict(user_text)
            if flow_state in {FLOW_IDLE, FLOW_AWAITING_ORDER_NUMBER, FLOW_CHECKING_ORDER_NUMBER}
            else None
        )

        phone_lookup_schedule_snooze = _as_float(
            get_agent_setting("phone_lookup_schedule_snooze_seconds", 20.0),
            20.0,
            min_value=5.0,
            max_value=90.0,
        )

        # Clear stale pending forced-turn markers from previous turns.
        if (
            int(_current_session.get("phone_forced_pending_turn_id") or 0) < current_turn_id
            and not bool(_current_session.get("phone_lookup_inflight"))
        ):
            _current_session["phone_forced_pending_turn_id"] = 0
        if (
            int(_current_session.get("details_forced_pending_turn_id") or 0) < current_turn_id
            and not bool(_current_session.get("details_lookup_inflight"))
        ):
            _current_session["details_forced_pending_turn_id"] = 0
        # New user turn cancels stale forced-response suppression window.
        last_forced_response_turn = int(_current_session.get("forced_response_spoken_turn_id") or 0)
        if current_turn_id > last_forced_response_turn:
            _current_session["forced_response_suppress_llm_until"] = 0.0
            _current_session["forced_response_spoken_text"] = ""

        def _handle_phone_flow_turn(
            *,
            user_text: str,
            current_turn_id: int,
            is_affirmative: bool,
            is_negative: bool,
        ) -> bool:
            """Deterministic phone-flow state machine. Returns True when turn is fully handled."""
            local_flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
            if local_flow_state not in PHONE_FLOW_STATES:
                return False

            raw_digits = "".join(_extract_digit_parts(user_text or ""))
            pending_phone = str(_current_session.get("pending_phone_candidate") or "")

            def _schedule_manual_prompt(
                message_text: str,
                *,
                reason: str,
                suppress_s: float = 8.0,
                snooze_s: float = 8.0,
                delay_s: float = 0.0,
            ) -> None:
                prompt_key = f"{current_turn_id}:{reason}:{message_text}"
                if (
                    int(_current_session.get("last_manual_prompt_turn_id") or 0) == current_turn_id
                    and str(_current_session.get("last_manual_prompt_key") or "") == prompt_key
                ):
                    room_log("MANUAL_PROMPT_SKIPPED", reason="duplicate_same_turn", trigger=reason)
                    return

                # Robustness: Never double-respond if a manual response is already active 
                # or if a tool was called in this exact turn.
                if _current_session.get("forced_response_manual_say_active"):
                    room_log("MANUAL_PROMPT_SKIPPED", reason="already_active", trigger=reason)
                    return

                last_tool_call = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                # If a tool was called within the last 500ms, assume the LLM/Watchdog is handling it.
                if (time.time() - last_tool_call) < 0.5:
                    room_log("MANUAL_PROMPT_SKIPPED", reason="recent_tool_call", trigger=reason)
                    return

                _current_session["last_manual_prompt_turn_id"] = current_turn_id
                _current_session["last_manual_prompt_key"] = prompt_key
                _current_session["forced_response_manual_say_active"] = True
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
                _current_session["forced_response_spoken_text"] = message_text
                _current_session["forced_response_transcript_sent_at"] = time.time()
                _clear_pending_lookup_wait_phrase(reason)
                # DO NOT clear_lookup_pending here! It suppresses silence prompts 
                # while we are preparing/speaking the manual prompt.
                _current_session["phone_lookup_inflight"] = False
                _snooze_silence_prompts(snooze_s, reason=reason)

                async def _say_prompt() -> None:
                    try:
                        if delay_s > 0:
                            await asyncio.sleep(delay_s)
                            if int(_current_session.get("last_user_turn_id") or 0) != current_turn_id:
                                room_log(
                                    "MANUAL_PROMPT_SKIPPED",
                                    reason="superseded_by_newer_user_turn",
                                    trigger=reason,
                                )
                                return
                        # Final check before speaking to avoid race conditions
                        if _current_session.get("should_end"):
                             return
                        await send_agent_transcript(message_text)
                        agent.chat_ctx.append(role="assistant", text=message_text)
                        await agent.say(message_text, allow_interruptions=True)
                        _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
                        mark_agent_speaking()
                    finally:
                        _current_session["forced_response_manual_say_active"] = False

                asyncio.create_task(_say_prompt())

            def _schedule_phone_confirmation_from_turn(phone_candidate: str, trigger_reason: str) -> bool:
                """Only confirm phone when current turn actually contains digits."""
                if not raw_digits:
                    room_log(
                        "PHONE_CONFIRMATION_BLOCKED",
                        reason="no_digits_in_current_user_turn",
                        user_text=_truncate(user_text),
                        stale_pending_phone=_current_session.get("pending_phone_candidate"),
                        stale_buffer=_current_session.get("phone_digit_buffer"),
                    )
                    _reset_phone_collection_state("blocked_stale_phone_confirmation")
                    _set_support_flow_state(
                        FLOW_AWAITING_PHONE_NUMBER,
                        reason="blocked_stale_phone_confirmation",
                    )
                    return False

                _reset_phone_digit_buffer("phone_candidate_captured")
                _current_session["pending_phone_candidate"] = phone_candidate
                _set_support_flow_state(FLOW_AWAITING_PHONE_CONFIRMATION, reason="phone_candidate_captured")
                room_log("PHONE_CANDIDATE_CAPTURED", phone=phone_candidate, turn_id=current_turn_id)
                _current_session["forced_response_manual_say_active"] = True
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 30.0
                _clear_pending_lookup_wait_phrase("phone_confirmation_prompt")
                # Removed _clear_lookup_pending here to avoid dead state while confirmation speaks
                _current_session["phone_lookup_inflight"] = False
                _snooze_silence_prompts(30.0, reason="phone_confirmation_prompt")
                asyncio.create_task(
                    _speak_phone_confirmation_prompt(
                        current_turn_id,
                        phone_candidate,
                        trigger_reason,
                    )
                )
                return True

            if local_flow_state == FLOW_CHECKING_PHONE_NUMBER:
                # Keep order-flow parity: only ignore while lookup is truly in-flight.
                # If checking state is stale, recover and trigger lookup immediately.
                if bool(_current_session.get("phone_lookup_inflight")):
                    room_log("PHONE_FLOW_TURN_IGNORED", reason="lookup_already_running", turn_id=current_turn_id)
                    return True

                pending_lookup_phone = str(_current_session.get("lookup_pending_order") or "").strip()
                recovered_phone = (
                    _normalize_phone_for_lookup(pending_phone)
                    or _normalize_phone_for_lookup(pending_lookup_phone)
                    or _normalize_phone_for_lookup(raw_digits)
                )
                if recovered_phone:
                    _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="recover_stale_checking_phone")
                    _set_lookup_pending(recovered_phone, reason="recover_stale_checking_phone")
                    _current_session["phone_lookup_inflight"] = True
                    room_log(
                        "PHONE_FLOW_RECOVERY_TRIGGERED",
                        turn_id=current_turn_id,
                        phone=recovered_phone,
                        source="stale_checking_state",
                    )
                    asyncio.create_task(
                        _force_lookup_by_phone(current_turn_id, recovered_phone, "recover_stale_checking_phone")
                    )
                    return True

                room_log("PHONE_FLOW_RECOVERY_RESET", reason="stale_checking_without_phone", turn_id=current_turn_id)
                _current_session["phone_lookup_inflight"] = False
                _clear_lookup_pending("stale_checking_without_phone")
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="stale_checking_without_phone")
                local_flow_state = FLOW_AWAITING_PHONE_NUMBER

            if local_flow_state == FLOW_AWAITING_PHONE_NUMBER:
                # 1. Check for exit or negative intent before forcing digits.
                if is_negative or _is_closing_utterance(user_text):
                    room_log("PHONE_FLOW_EXIT", reason="negative_or_closing", turn_id=current_turn_id)
                    _reset_phone_collection_state("user_exit")
                    _set_support_flow_state(FLOW_IDLE, reason="user_exit")
                    return False

                if not raw_digits:
                    if _is_short_utterance(user_text) or not (user_text or "").strip():
                        if False:
                            msg = "Παρακαλώ πείτε τον αριθμό τηλεφώνου που χρησιμοποιήσατε για την παραγγελία."
                        else:
                            msg = "Please provide the phone number used for the order."
                        _schedule_manual_prompt(msg, reason="awaiting_phone_digits")
                        return True
                    return False

                combined_digits = _append_phone_digits_from_turn(user_text)
                phone_candidate = _normalize_phone_for_lookup(combined_digits)
                if phone_candidate:
                    # Parity: Trigger lookup directly instead of confirmation
                    _current_session["pending_phone_candidate"] = None
                    _reset_phone_digit_buffer("phone_candidate_captured")
                    _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="phone_candidate_captured")
                    asyncio.create_task(
                        _force_lookup_by_phone(current_turn_id, phone_candidate, "phone_candidate_captured")
                    )
                    return True

                min_digits = _as_int(
                    get_agent_setting("phone_lookup_min_digits", 10),
                    10,
                    min_value=7,
                    max_value=15,
                )
                if 0 < len(combined_digits) < min_digits:
                    # Guard: If the user is trying to exit or say no, don't force partial digit prompt.
                    if is_negative or _is_closing_utterance(user_text):
                        room_log("PHONE_PARTIAL_PROMPT_SKIPPED", reason="negative_or_closing")
                        return False
                    
                    # Ignore single-digit turns that look like noise (e.g. noise transcribed as "six")
                    if len(raw_digits) < 2 and len(combined_digits) < 2 and not _is_digit_collection_utterance(user_text):
                        room_log("PHONE_PARTIAL_PROMPT_SKIPPED", reason="potential_noise", turn_digits=raw_digits)
                        return False

                    if False:
                        msg = "Συνεχίστε με τα υπόλοιπα ψηφία του αριθμού, παρακαλώ."
                    else:
                        msg = "I’m listening. Please continue with the remaining digits of the phone number."
                    _schedule_manual_prompt(
                        msg,
                        reason="phone_digits_partial",
                        suppress_s=6.0,
                        delay_s=1.2,
                    )
                    return True

                _reset_phone_digit_buffer("invalid_complete_phone")
                _current_session["pending_phone_candidate"] = None
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="invalid_complete_phone")
                if False:
                    msg = (
                        "Αυτό δεν φαίνεται να είναι πλήρης αριθμός τηλεφώνου. "
                        f"Παρακαλώ πείτε ολόκληρο τον αριθμό ξανά, με τουλάχιστον {min_digits} ψηφία."
                    )
                else:
                    msg = (
                        "That does not look like a complete phone number. "
                        f"Please repeat the full number, at least {min_digits} digits."
                    )
                _schedule_manual_prompt(msg, reason="invalid_complete_phone")
                room_log("INVALID_OR_PARTIAL_PHONE_REJECTED", digits=combined_digits, turn_id=current_turn_id)
                return True

            # FLOW_AWAITING_PHONE_CONFIRMATION
            phone_candidate = _normalize_phone_for_lookup(user_text or "")
            if phone_candidate:
                # Parity: Trigger lookup directly instead of re-confirming
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("phone_candidate_captured_override")
                _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="phone_candidate_captured_override")
                asyncio.create_task(
                    _force_lookup_by_phone(current_turn_id, phone_candidate, "phone_candidate_captured_override")
                )
                return True

            if pending_phone and is_affirmative:
                _current_session["pending_phone_candidate"] = None
                _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="phone_confirmed")
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 30.0
                _current_session["phone_forced_pending_turn_id"] = current_turn_id
                _set_lookup_pending(pending_phone, reason="forced_phone_lookup_scheduled")
                _snooze_silence_prompts(phone_lookup_schedule_snooze, reason="phone_lookup_scheduled")
                asyncio.create_task(
                    _force_lookup_by_phone(current_turn_id, pending_phone, "phone_confirmed")
                )
                room_log("PHONE_LOOKUP_FORCED_TRIGGERED", phone=pending_phone, turn_id=current_turn_id)
                return True

            if is_negative:
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("phone_confirmation_rejected")
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="phone_confirmation_rejected")
                if False:
                    msg = "Εντάξει. Πείτε ξανά τον αριθμό τηλεφώνου σας."
                else:
                    msg = "Okay. Please repeat your phone number again."
                _schedule_manual_prompt(msg, reason="phone_confirmation_rejected")
                return True

            if raw_digits:
                combined_digits = _append_phone_digits_from_turn(user_text)
                min_digits = _as_int(
                    get_agent_setting("phone_lookup_min_digits", 10),
                    10,
                    min_value=7,
                    max_value=15,
                )
                if len(combined_digits) < min_digits:
                    if False:
                        msg = "Συνεχίστε με τα υπόλοιπα ψηφία του αριθμού, παρακαλώ."
                    else:
                        msg = "I’m listening. Please continue with the remaining digits of the phone number."
                    _schedule_manual_prompt(
                        msg,
                        reason="phone_digits_partial",
                        suppress_s=6.0,
                        delay_s=1.2,
                    )
                else:
                    _reset_phone_digit_buffer("invalid_complete_phone")
                    if False:
                        msg = (
                            "Αυτό δεν φαίνεται να είναι πλήρης αριθμός τηλεφώνου. "
                            f"Παρακαλώ πείτε ολόκληρο τον αριθμό ξανά, με τουλάχιστον {min_digits} ψηφία."
                        )
                    else:
                        msg = (
                            "That does not look like a complete phone number. "
                            f"Please repeat the full number, at least {min_digits} digits."
                        )
                    _schedule_manual_prompt(msg, reason="invalid_complete_phone")
                return True

            if pending_phone:
                if not raw_digits:
                    room_log(
                        "PHONE_CONFIRMATION_BLOCKED",
                        reason="no_digits_in_current_user_turn",
                        user_text=_truncate(user_text),
                        stale_pending_phone=_current_session.get("pending_phone_candidate"),
                        stale_buffer=_current_session.get("phone_digit_buffer"),
                    )
                    _reset_phone_collection_state("blocked_stale_phone_confirmation")
                    _set_support_flow_state(
                        FLOW_AWAITING_PHONE_NUMBER,
                        reason="blocked_stale_phone_confirmation",
                    )
                    return True
                
                # Parity: Trigger lookup directly
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("non_confirmation_reply_direct")
                _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="non_confirmation_reply_direct")
                asyncio.create_task(
                    _force_lookup_by_phone(current_turn_id, pending_phone, "non_confirmation_reply_direct")
                )
                return True
            if False:
                msg = "Παρακαλώ πείτε ξανά τον αριθμό τηλεφώνου σας."
            else:
                msg = "Please say your phone number again."
            _schedule_manual_prompt(msg, reason="awaiting_phone_recovery")
            return True

        # --- DETERMINISTIC WATCHDOGS (TOP PRIORITY) ---
        # We detect numbers and start watchdogs immediately to block silence prompts
        # before any flow handlers return early.
        detected_order_number = _extract_order_number_candidate(user_text)
        detected_phone = _normalize_phone_for_lookup(user_text)
        number_mode = str(_current_session.get("number_mode_lock") or "")
        
        order_lookup_blocked_by_flow = (
            (
                flow_state == FLOW_AWAITING_PHONE_CONFIRMATION
                or (
                    flow_state == FLOW_CHECKING_PHONE_NUMBER
                    and bool(_current_session.get("phone_lookup_inflight"))
                )
            )
            and not detected_order_number
        )

        # Hard fallback: if mode is already phone and we detect a valid phone number,
        # start phone lookup immediately even if flow is still on order states.
        if (
            number_mode == "phone"
            and detected_phone
            and flow_state in {FLOW_AWAITING_ORDER_NUMBER, FLOW_CHECKING_ORDER_NUMBER}
            and not bool(_current_session.get("phone_lookup_inflight"))
        ):
            room_log(
                "PHONE_MODE_FORCE_TRIGGERED",
                turn_id=current_turn_id,
                flow_state=flow_state,
                phone=detected_phone,
            )
            _set_support_flow_state(
                FLOW_CHECKING_PHONE_NUMBER,
                reason="phone_mode_force_triggered",
            )
            _set_lookup_pending(detected_phone, reason="phone_mode_force_triggered")
            asyncio.create_task(
                _force_lookup_by_phone(
                    current_turn_id,
                    detected_phone,
                    "phone_mode_force_triggered",
                )
            )
            return
        
        # 1. Order Number Watchdog
        if (
            _should_force_order_lookup(user_text, detected_order_number)
            and number_mode != "phone"
            and not order_lookup_blocked_by_flow
            and not phone_flow_active
        ):
            now = time.time()
            last_forced_order = str(_current_session.get("last_forced_lookup_order") or "")
            last_forced_at = float(_current_session.get("last_forced_lookup_at") or 0.0)
            if (
                detected_order_number != last_forced_order
                or (now - last_forced_at) > 15.0
            ):
                _current_session["last_forced_lookup_order"] = detected_order_number
                _current_session["last_forced_lookup_at"] = now
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "ORDER LOOKUP PRIORITY:\n"
                        f"- The caller already provided order number {detected_order_number}.\n"
                        "- In your next response, call lookup_order with this exact number immediately.\n"
                        "- Do not ask for the order number again unless lookup_order says invalid or not found.\n"
                        "- After the tool returns, provide the status without extra delay."
                    ),
                )
                room_log("ORDER_LOOKUP_HINT_INJECTED", order_number=detected_order_number)
                _set_lookup_pending(detected_order_number, reason="order_detected_awaiting_tool")
                start_time = time.time()
                asyncio.create_task(
                    _force_lookup_order_by_number(current_turn_id, detected_order_number, start_time)
                )

        # 2. Phone Number Watchdog
        elif (
            detected_phone
            and not order_lookup_blocked_by_flow
            and (flow_state == FLOW_AWAITING_PHONE_NUMBER or _should_force_order_lookup(user_text, None))
            and not phone_flow_active
        ):
            now = time.time()
            last_forced_phone = str(_current_session.get("last_forced_lookup_phone") or "")
            last_forced_phone_at = float(_current_session.get("last_forced_lookup_phone_at") or 0.0)
            
            if (
                detected_phone != last_forced_phone
                or (now - last_forced_phone_at) > 15.0
            ):
                _current_session["last_forced_lookup_phone"] = detected_phone
                _current_session["last_forced_lookup_phone_at"] = now
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "PHONE LOOKUP PRIORITY:\n"
                        f"- The caller provided phone number {detected_phone}.\n"
                        "- In your next response, call lookup_order_by_phone with this phone number immediately.\n"
                        "- Do not ask for confirmation or repeat the number unless the tool fails.\n"
                        "- After the tool returns, provide the status without extra delay."
                    ),
                )
                room_log("PHONE_LOOKUP_HINT_INJECTED", phone=detected_phone)
                _set_lookup_pending(detected_phone, reason="phone_detected_awaiting_tool")
                start_time = time.time()
                asyncio.create_task(
                    _force_lookup_by_phone(current_turn_id, detected_phone, "detected_in_speech", start_time)
                )

        if phone_flow_active:
            handled = _handle_phone_flow_turn(
                user_text=user_text,
                current_turn_id=current_turn_id,
                is_affirmative=is_affirmative,
                is_negative=is_negative,
            )
            if handled:
                return

        # Deterministic state bootstrap: treat the first open-ended complaint as issue context,
        # then ask for order number unless caller explicitly says they don't have it.
        if flow_state == FLOW_IDLE and user_text and not _is_short_utterance(user_text):
            if not _mentions_no_order_number(user_text) and not normalized_order_candidate:
                _current_session["support_issue"] = user_text
                _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason="issue_collected")
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT FLOW:\n"
                        "- The issue has been collected.\n"
                        "- Ask the caller for their order number next.\n"
                        "- If they don't have an order number, ask for the phone number used in the order.\n"
                        "- IMPORTANT: When reporting delivery dates, always use numeric format (e.g., 14/03) instead of words.\n"
                        "- For Order IDs and Phone numbers, use numeric digits (e.g., 12752) in your text responses."
                    ),
                )

        has_digits = bool(_extract_digit_parts(user_text))
        raw_digits = "".join(_extract_digit_parts(user_text or ""))
        expected_digits = _expected_order_digits()
        if flow_state == FLOW_AWAITING_ORDER_NUMBER and raw_digits:
            normalized_phone_candidate = _normalize_phone_for_lookup(user_text or "")
            last_agent_text = str(_current_session.get("last_agent_text") or "")
            if normalized_phone_candidate and (
                _is_phone_number_prompt(last_agent_text)
                or str(_current_session.get("number_mode_lock") or "") == "phone"
            ):
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("phone_from_order_flow_fast_path")
                _set_support_flow_state(
                    FLOW_CHECKING_PHONE_NUMBER,
                    reason="phone_provided_while_awaiting_order_number",
                )
                room_log(
                    "PHONE_FAST_PATH_TRIGGERED",
                    turn_id=current_turn_id,
                    phone=normalized_phone_candidate,
                    last_agent_text=_truncate(last_agent_text, max_len=120),
                )
                asyncio.create_task(
                    _force_lookup_by_phone(
                        current_turn_id,
                        normalized_phone_candidate,
                        "phone_provided_while_awaiting_order_number",
                    )
                )
                return

            normalized_order = _normalize_order_id_strict(user_text or "")
            if not normalized_order:
                min_d, max_d = _order_digit_range()
                _current_session["number_mode_lock"] = "order"
                _set_support_flow_state(
                    FLOW_AWAITING_ORDER_NUMBER,
                    reason="invalid_order_digits",
                )
                _current_session["forced_response_manual_say_active"] = True
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 8.0
                _clear_lookup_pending("invalid_order_digits")
                _clear_pending_lookup_wait_phrase("invalid_order_digits")
                _snooze_silence_prompts(8.0, reason="invalid_order_digits")

                async def _say_invalid_order_digits() -> None:
                    live_agent = _current_session.get("agent")
                    if not live_agent:
                        _current_session["forced_response_manual_say_active"] = False
                        return
                    try:
                        if False:
                            msg = (
                                f"Ο αριθμός παραγγελίας πρέπει να έχει από {min_d} έως {max_d} ψηφία. "
                                f"Μπορείτε να τον πείτε ξανά;"
                            )
                        else:
                            msg = "I didn't quite catch the order number. Could you please repeat it?"
                        await send_agent_transcript(msg)
                        agent.chat_ctx.append(role="assistant", text=msg)
                        await live_agent.say(msg, allow_interruptions=True)
                    finally:
                        _current_session["forced_response_manual_say_active"] = False

                asyncio.create_task(_say_invalid_order_digits())
                return

        if (
            flow_state == FLOW_AWAITING_ORDER_NUMBER
            and not _mentions_no_order_number(user_text)
            and not normalized_order_candidate
        ):
            min_d, max_d = _order_digit_range()
            if has_digits:
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT FLOW - ORDER NUMBER STILL MISSING:\n"
                        f"- The caller has not provided a valid {min_d}-{max_d} digit order number yet.\n"
                        "- Ask them to repeat the order number.\n"
                        "- Do not switch to phone lookup unless they explicitly say they don't have an order number."
                    ),
                )
            else:
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT FLOW - REQUEST ORDER NUMBER:\n"
                        "- Ask for the order number now.\n"
                        "- If caller says they don't have it, then request the phone number used in the order."
                    ),
                )

        no_order_recovery = _mentions_no_order_number(user_text) or (
            flow_state == FLOW_AWAITING_ORDER_NUMBER
            and _mentions_missing_confirmation_email(user_text)
        )
        if no_order_recovery:
            _reset_phone_collection_state("user_has_no_order_number")
            _clear_lookup_pending("user_has_no_order_number")
            _clear_pending_lookup_wait_phrase("user_has_no_order_number")
            _current_session["number_mode_lock"] = "phone"
            _set_support_flow_state(
                FLOW_AWAITING_PHONE_NUMBER,
                reason="user_has_no_order_number",
            )
            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_turn_id"] = current_turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + 8.0
            _snooze_silence_prompts(8.0, reason="ask_phone_after_no_order_number")

            async def _ask_phone_number() -> None:
                live_agent = _current_session.get("agent")
                if not live_agent:
                    _current_session["forced_response_manual_say_active"] = False
                    return
                try:
                    if False:
                        msg = (
                            "Εντάξει. Μπορείτε να μου δώσετε τον αριθμό τηλεφώνου "
                            "που χρησιμοποιήσατε για την παραγγελία;"
                        )
                    else:
                        msg = (
                            "No problem. Please give me the phone number used for the order, "
                            "one more time?"
                        )
                    prompt_key = f"{current_turn_id}:ask_phone_after_no_order_number:{msg}"
                    if (
                        int(_current_session.get("last_manual_prompt_turn_id") or 0) == current_turn_id
                        and str(_current_session.get("last_manual_prompt_key") or "") == prompt_key
                    ):
                        return
                    _current_session["last_manual_prompt_turn_id"] = current_turn_id
                    _current_session["last_manual_prompt_key"] = prompt_key
                    await send_agent_transcript(msg)
                    agent.chat_ctx.append(role="assistant", text=msg)
                    await live_agent.say(msg, allow_interruptions=True)
                finally:
                    _current_session["forced_response_manual_say_active"] = False

            asyncio.create_task(_ask_phone_number())
            return
        elif (
            flow_state == FLOW_AWAITING_ORDER_NUMBER
            and not normalized_order_candidate
            and not has_digits
        ):
            # Deterministic safety net:
            # If caller stays in support flow but provides contextual text (e.g. "I didn't get the email"),
            # do not wait for LLM/tool timing. Respond immediately with the next valid options.
            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_turn_id"] = current_turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + 8.0
            _snooze_silence_prompts(8.0, reason="awaiting_order_number_clarify_next_step")

            async def _clarify_order_or_phone_next_step() -> None:
                live_agent = _current_session.get("agent")
                if not live_agent:
                    _current_session["forced_response_manual_say_active"] = False
                    return
                try:
                    msg = (
                        "I understand. If you have the order number, please share it. "
                        "If you don't have it, I can check by phone number instead."
                    )
                    prompt_key = f"{current_turn_id}:awaiting_order_number_clarify:{msg}"
                    if (
                        int(_current_session.get("last_manual_prompt_turn_id") or 0) == current_turn_id
                        and str(_current_session.get("last_manual_prompt_key") or "") == prompt_key
                    ):
                        return
                    _current_session["last_manual_prompt_turn_id"] = current_turn_id
                    _current_session["last_manual_prompt_key"] = prompt_key
                    await send_agent_transcript(msg)
                    agent.chat_ctx.append(role="assistant", text=msg)
                    await live_agent.say(msg, allow_interruptions=True)
                finally:
                    _current_session["forced_response_manual_say_active"] = False

            asyncio.create_task(_clarify_order_or_phone_next_step())
            return
        elif normalized_order_candidate and flow_state == FLOW_AWAITING_ORDER_NUMBER:
            # Caller provided a valid order id, so clear phone-capture context.
            _current_session["pending_phone_candidate"] = None
            _reset_phone_digit_buffer("back_to_order_flow")
            _set_support_flow_state(FLOW_CHECKING_ORDER_NUMBER, reason="order_number_provided")
        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        active_number_collection_states = {
            FLOW_AWAITING_ORDER_NUMBER,
            FLOW_CHECKING_ORDER_NUMBER,
            FLOW_AWAITING_PHONE_NUMBER,
            FLOW_AWAITING_PHONE_CONFIRMATION,
            FLOW_CHECKING_PHONE_NUMBER,
        }
        if flow_state not in active_number_collection_states:
            _current_session["number_mode_lock"] = None
            _current_session["number_mode_turn_id"] = 0
        # inferred_mode already calculated at turn start
        phone_flow_states = PHONE_FLOW_STATES
        if (
            flow_state == FLOW_AWAITING_ORDER_NUMBER
            and inferred_mode == "phone"
            and not _mentions_no_order_number(user_text)
        ):
            inferred_mode = None
            room_log("NUMBER_MODE_SUPPRESSED", flow_state=flow_state, candidate="phone")
        if (
            inferred_mode is None
            and _is_phone_flow_active()
            and bool(_extract_digit_parts(user_text))
        ):
            inferred_mode = "phone"
        # Removed strict mode forcing block that prevented switching from phone back to order flow.
        if inferred_mode in {"order", "phone"}:
            _current_session["number_mode_lock"] = inferred_mode
            _current_session["number_mode_turn_id"] = current_turn_id
            room_log("NUMBER_MODE_LOCKED", mode=inferred_mode, turn_id=current_turn_id)

        now = time.time()
        explicit_details_request = _explicit_more_order_details_request(user_text)

        # Drop stale phone candidate if this turn is clearly back to order-id flow.
        if inferred_mode == "order":
            _current_session["pending_phone_candidate"] = None

        details_pending = bool(_current_session.get("details_confirmation_pending"))

        details_pending_until = float(_current_session.get("details_confirmation_pending_until") or 0.0)
        if details_pending and now <= details_pending_until:
            if explicit_details_request or is_affirmative:
                can_unlock, unlock_reason = _can_unlock_full_details(user_text)
                if can_unlock:
                    _current_session["full_order_details_allowed_until"] = now + 120.0
                    _current_session["full_details_unlocked_once"] = True
                    _current_session["details_confirmation_pending"] = False
                    _current_session["details_confirmation_pending_until"] = 0.0
                    room_log("FULL_DETAILS_ALLOWED", ttl_s=120, reason="single_yes_unlock")
                    # Deterministic path: suppress generic LLM chatter and force details tool fetch.
                    _current_session["forced_response_spoken_turn_id"] = current_turn_id
                    _current_session["forced_response_suppress_llm_until"] = time.time() + 20.0
                    _current_session["details_forced_pending_turn_id"] = current_turn_id
                    asyncio.create_task(
                        _force_get_order_details(current_turn_id, "single_yes_unlock")
                    )
                    return
                else:
                    room_log("FULL_DETAILS_BLOCKED", reason=unlock_reason)
                    _current_session["full_order_details_allowed_until"] = 0.0
                    _current_session["details_confirmation_pending"] = False
                    _current_session["details_confirmation_pending_until"] = 0.0
            elif is_negative:
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0
        elif explicit_details_request:
            can_unlock, unlock_reason = _can_unlock_full_details(user_text)
            if can_unlock:
                _current_session["full_order_details_allowed_until"] = now + 120.0
                _current_session["full_details_unlocked_once"] = True
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                room_log("FULL_DETAILS_ALLOWED", ttl_s=120, reason="explicit_request")
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 20.0
                _current_session["details_forced_pending_turn_id"] = current_turn_id
                asyncio.create_task(
                    _force_get_order_details(current_turn_id, "explicit_request")
                )
                return
            else:
                room_log("FULL_DETAILS_BLOCKED", reason=unlock_reason)

        ticket_pending = bool(_current_session.get("ticket_confirmation_pending"))
        ticket_pending_until = float(_current_session.get("ticket_confirmation_pending_until") or 0.0)
        if ticket_pending and now <= ticket_pending_until:
            if _is_issue_confirmation_utterance(user_text):
                _current_session["ticket_create_allowed_until"] = now + 120.0
                _current_session["ticket_confirmation_pending"] = False
                _current_session["ticket_confirmation_pending_until"] = 0.0
                room_log("TICKET_CREATE_ALLOWED", ttl_s=120, reason="issue_confirmed")
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT TICKET CONFIRMED:\n"
                        "- The user gave final confirmation for the issue.\n"
                        "- Call create_support_ticket now with the collected customer details.\n"
                        "- Create the ticket only once."
                    ),
                )
            elif is_negative:
                _current_session["ticket_confirmation_pending"] = False
                _current_session["ticket_confirmation_pending_until"] = 0.0
                _current_session["ticket_create_allowed_until"] = 0.0
                _current_session["pending_ticket_payload"] = None

        # Snooze silence monitor on closing intent (e.g., "No, that's all").
        if is_negative and not details_pending and not ticket_pending:
            _snooze_silence_prompts(120.0, reason="user_negative_closing")

        set_runtime_language("en")

        try:
            detected_order_number = _extract_order_number_candidate(user_text)
            detected_phone = _normalize_phone_for_lookup(user_text)
            number_mode = str(_current_session.get("number_mode_lock") or "")
            flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
            
            order_lookup_blocked_by_flow = (
                (
                    flow_state == FLOW_AWAITING_PHONE_CONFIRMATION
                    or (
                        flow_state == FLOW_CHECKING_PHONE_NUMBER
                        and bool(_current_session.get("phone_lookup_inflight"))
                    )
                )
                and not detected_order_number
            )

            # Hard fallback duplicate in second watchdog pass:
            if (
                number_mode == "phone"
                and detected_phone
                and flow_state in {FLOW_AWAITING_ORDER_NUMBER, FLOW_CHECKING_ORDER_NUMBER}
                and not bool(_current_session.get("phone_lookup_inflight"))
            ):
                room_log(
                    "PHONE_MODE_FORCE_TRIGGERED",
                    turn_id=int(_current_session.get("last_user_turn_id") or 0),
                    flow_state=flow_state,
                    phone=detected_phone,
                    source="secondary_watchdog_pass",
                )
                _set_support_flow_state(
                    FLOW_CHECKING_PHONE_NUMBER,
                    reason="phone_mode_force_triggered_secondary",
                )
                _set_lookup_pending(detected_phone, reason="phone_mode_force_triggered_secondary")
                current_turn = int(_current_session.get("last_user_turn_id") or 0)
                asyncio.create_task(
                    _force_lookup_by_phone(
                        current_turn,
                        detected_phone,
                        "phone_mode_force_triggered_secondary",
                    )
                )
                return
            
            # --- 1. Order Number Watchdog ---
            if (
                _should_force_order_lookup(user_text, detected_order_number)
                and number_mode != "phone"
                and not order_lookup_blocked_by_flow
                and flow_state not in PHONE_FLOW_STATES
            ):
                now = time.time()
                last_forced_order = str(_current_session.get("last_forced_lookup_order") or "")
                last_forced_at = float(_current_session.get("last_forced_lookup_at") or 0.0)
                if (
                    detected_order_number != last_forced_order
                    or (now - last_forced_at) > 15.0
                ):
                    _current_session["last_forced_lookup_order"] = detected_order_number
                    _current_session["last_forced_lookup_at"] = now
                    agent.chat_ctx.append(
                        role="system",
                        text=(
                            "ORDER LOOKUP PRIORITY:\n"
                            f"- The caller already provided order number {detected_order_number}.\n"
                            "- In your next response, call lookup_order with this exact number immediately.\n"
                            "- Do not ask for the order number again unless lookup_order says invalid or not found.\n"
                            "- After the tool returns, provide the status without extra delay."
                        ),
                    )
                    room_log("ORDER_LOOKUP_HINT_INJECTED", order_number=detected_order_number)

                    # Mark as pending immediately to block silence prompts during the grace window
                    _set_lookup_pending(detected_order_number, reason="order_detected_awaiting_tool")
                    start_time = time.time()
                    current_turn = int(_current_session.get("last_user_turn_id") or 0)
                    asyncio.create_task(
                        _force_lookup_order_by_number(current_turn, detected_order_number, start_time)
                    )
                    room_log("ORDER_WATCHDOG_SCHEDULED", turn_id=current_turn, order_number=detected_order_number)

            # --- 2. Phone Number Watchdog ---
            elif (
                detected_phone
                and not order_lookup_blocked_by_flow
                and (flow_state == FLOW_AWAITING_PHONE_NUMBER or _should_force_order_lookup(user_text, None))
                and flow_state not in PHONE_FLOW_STATES
            ):
                now = time.time()
                last_forced_phone = str(_current_session.get("last_forced_lookup_phone") or "")
                last_forced_phone_at = float(_current_session.get("last_forced_lookup_phone_at") or 0.0)
                
                if (
                    detected_phone != last_forced_phone
                    or (now - last_forced_phone_at) > 15.0
                ):
                    _current_session["last_forced_lookup_phone"] = detected_phone
                    _current_session["last_forced_lookup_phone_at"] = now
                    agent.chat_ctx.append(
                        role="system",
                        text=(
                            "PHONE LOOKUP PRIORITY:\n"
                            f"- The caller provided phone number {detected_phone}.\n"
                            "- In your next response, call lookup_order_by_phone with this phone number immediately.\n"
                            "- Do not ask for confirmation or repeat the number unless the tool fails.\n"
                            "- After the tool returns, provide the status without extra delay."
                        ),
                    )
                    room_log("PHONE_LOOKUP_HINT_INJECTED", phone=detected_phone)

                    # Mark as pending to block silence prompts
                    _set_lookup_pending(detected_phone, reason="phone_detected_awaiting_tool")
                    start_time = time.time()
                    current_turn = int(_current_session.get("last_user_turn_id") or 0)
                    asyncio.create_task(
                        _force_lookup_by_phone(current_turn, detected_phone, "detected_in_speech", start_time)
                    )
                    room_log("PHONE_WATCHDOG_SCHEDULED", turn_id=current_turn, phone=detected_phone)


            elif order_lookup_blocked_by_flow and _should_force_order_lookup(user_text, detected_order_number):
                room_log(
                    "ORDER_LOOKUP_HINT_SKIPPED",
                    reason="phone_flow_in_progress",
                    flow_state=flow_state,
                )
        except Exception as e:
            logger.debug(f"Order lookup forcing skipped: {e}")

        user_text_for_transcript = _format_user_text_for_transcript(user_text)
        asyncio.create_task(send_user_transcript(user_text_for_transcript))

        # Reset silence timer - user is responding
        reset_silence_timer()
        if _is_digit_collection_utterance(user_text):
            digit_grace = _as_float(
                get_agent_setting("digit_collection_silence_grace_seconds", 30.0),
                30.0,
                min_value=15.0,
                max_value=60.0,
            )
            _snooze_silence_prompts(digit_grace, reason="digit_collection")
        elif _is_short_utterance(user_text):
            short_grace = _as_float(
                get_agent_setting("short_utterance_silence_grace_seconds", 8.0),
                8.0,
                min_value=2.0,
                max_value=20.0,
            )
            _snooze_silence_prompts(short_grace, reason="short_utterance")

        # Digit collection silence management
        if abuse_detection_enabled:
            # Check for abusive language
            abuse_detected, abuse_response = check_and_respond_to_abuse(
                user_text,
                language='en',
                tracker=_abuse_tracker,
                use_ssml=True
            )

            if abuse_detected:
                logger.warning("Abuse detected in: %s...", user_text[:50])
                # The agent will continue normally, but we log the incident
                # The abuse response will be handled by the LLM with special instructions
                # For now, we just track it for escalation purposes

    @agent.on("agent_started_speaking")
    def on_agent_started_speaking():
        _latency_tracker.agent_started_speaking()
        silence_tracker["llm_is_generating"] = False
        cancel_thinking_task()
        logger.info("audio_publish_start: agent_started_speaking")
        silence_tracker["agent_is_speaking"] = True
        # Never fire silence prompts while the agent is actively speaking.
        silence_tracker["is_waiting_for_response"] = False
        asyncio.create_task(send_state_update("speaking"))
    
    @agent.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        silence_tracker["agent_is_speaking"] = False
        asyncio.create_task(send_state_update("idle"))
        # Mark that agent finished speaking - now waiting for user response
        mark_agent_speaking()
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(message):
        """Send committed agent speech to UI (transcript + info cards)."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                room_log("LATE_EVENT_DROPPED", source="agent_speech_committed")
                return
            text = message.content if hasattr(message, 'content') else None
            if text:
                normalized_text = text  # English-only: no language enforcement needed
                display_text = _strip_markup_for_output(normalized_text)
                suppress_commit, suppress_reason = _should_suppress_tts_text(display_text or text)
                if suppress_commit:
                    # Keep the deterministic forced-response summary in transcript even inside suppression windows.
                    expected_forced_response_text = str(_current_session.get("forced_response_spoken_text") or "")
                    expected_norm = _normalize_switch_text(expected_forced_response_text)
                    candidate_norm = _normalize_switch_text(display_text or text)
                    same_as_forced_response = bool(
                        expected_norm
                        and candidate_norm
                        and (
                            candidate_norm == expected_norm
                            or candidate_norm in expected_norm
                            or expected_norm in candidate_norm
                        )
                    )
                    # Also allow core lookup outcomes and number-collection prompts through transcript,
                    # even during forced-response windows, to avoid "spoken but missing in transcript".
                    lookup_state_in_text = _classify_lookup_result(display_text or text)
                    keep_core_message = (
                        lookup_state_in_text in {"found", "not_found"}
                        or _is_phone_number_collection_prompt(display_text or text)
                        or bool(
                            re.search(
                                r"(provide|repeat|share).*(order number)",
                                _normalize_switch_text(display_text or text),
                                flags=re.IGNORECASE,
                            )
                        ) or bool(
                            re.search(
                                r"(checking|searching|looking up|one moment|moment please|wait a second|μια στιγμή|περιμένετε|το ελέγχω)",
                                _normalize_switch_text(display_text or text),
                                flags=re.IGNORECASE,
                            )
                        )
                    )
                    if not same_as_forced_response and not keep_core_message:
                        room_log(
                            "AGENT_TEXT_SUPPRESSED",
                            reason=suppress_reason,
                            turn_id=int(_current_session.get("last_user_turn_id") or 0),
                            text=_truncate(display_text or text),
                        )
                        return
                transcript_text = _format_agent_text_for_transcript(display_text or text)
                asyncio.create_task(send_agent_transcript(transcript_text, source="commit"))
                asyncio.create_task(send_agent_info(transcript_text))
                _current_session["last_agent_text"] = transcript_text
                
                # Snooze silence monitor if agent is saying goodbye or closing the call.
                if _is_closing_utterance(transcript_text):
                    _snooze_silence_prompts(180.0, reason="agent_said_goodbye")
                
                normalized_display = _normalize_switch_text(transcript_text)
                details_prompted = bool(
                    re.search(
                        r"(would you like .*details.*order|complete order details|more details about this order)",
                        normalized_display,
                    )
                )
                if details_prompted:
                    _current_session["last_more_details_prompt_at"] = time.time()
                    if not bool(_current_session.get("full_details_unlocked_once")):
                        _current_session["details_confirmation_pending"] = True
                        _current_session["details_confirmation_pending_until"] = time.time() + 120.0
                    else:
                        _current_session["details_confirmation_pending"] = False
                        _current_session["details_confirmation_pending_until"] = 0.0
                if _is_clarification_prompt_text(transcript_text):
                    clarification_grace = _as_float(
                        get_agent_setting("clarification_silence_grace_seconds", 12.0),
                        12.0,
                        min_value=4.0,
                        max_value=40.0,
                    )
                    _snooze_silence_prompts(clarification_grace, reason="clarification_prompt")
                if _current_session.get("lookup_pending"):
                    lookup_state_in_text = _classify_lookup_result(transcript_text)
                    if details_prompted or lookup_state_in_text in {"found", "not_found"}:
                        _clear_lookup_pending(reason="agent_committed_lookup_result")
                logger.info(f"agent_speech_committed: {transcript_text[:50]}...")
                room_log("AGENT_TEXT", text=_truncate(transcript_text))
        except Exception as e:
            logger.error(f"Error in agent_speech_committed: {e}")
    # Track detailed metrics from pipeline
    @agent.on("metrics_collected")
    def on_metrics_collected(metrics):
        """Collect detailed metrics from the pipeline - logs individual service timings."""
        try:
            parts = []
            
            # Try different attribute names (SDK versions vary)
            # STT metrics
            stt_duration = getattr(metrics, 'stt_duration', None) or getattr(metrics, 'transcription_delay', None)
            if stt_duration:
                parts.append(f"STT:{stt_duration*1000:.0f}ms")
            
            # LLM metrics
            llm_ttft = getattr(metrics, 'llm_ttft', None) or getattr(metrics, 'llm_first_token_delay', None)
            if llm_ttft:
                parts.append(f"LLM-TTFT:{llm_ttft*1000:.0f}ms")
            
            llm_total = getattr(metrics, 'llm_duration', None) or getattr(metrics, 'llm_total_duration', None)
            if llm_total:
                parts.append(f"LLM-Total:{llm_total*1000:.0f}ms")
            
            # TTS metrics
            tts_ttfb = getattr(metrics, 'tts_ttfb', None) or getattr(metrics, 'speech_start_delay', None)
            if tts_ttfb:
                parts.append(f"TTS:{tts_ttfb*1000:.0f}ms")
            
            # End-to-end delay
            eou = getattr(metrics, 'eou_delay', None) or getattr(metrics, 'end_of_utterance_delay', None)
            if eou:
                parts.append(f"EoU:{eou*1000:.0f}ms")
            
            if parts:
                logger.info(f"📊 [METRICS] {' | '.join(parts)}")
                room_log("METRICS", details=" | ".join(parts))
            else:
                # Log all available metrics for debugging
                attrs = [a for a in dir(metrics) if not a.startswith('_')]
                logger.debug(f"Available metrics: {attrs}")
                
        except Exception as e:
            logger.debug(f"Metrics collection error: {e}")
    
    # Additional event for function calls timing
    @agent.on("function_calls_finished")
    def on_function_calls_finished(called_functions):
        """Track tool call execution time."""
        for fn in called_functions:
            # Try different attribute names for the function name
            name = (
                getattr(fn, 'function_name', None) or 
                getattr(fn, 'name', None) or 
                getattr(fn, 'tool_name', None) or
                getattr(fn, '__name__', None) or
                str(type(fn).__name__)
            )
            result = getattr(fn, 'result', None)
            logger.info(f"🔧 [TOOL] {name} executed" + (f" - result: {str(result)[:100]}" if result else ""))
            room_log("TOOL_EXECUTED", name=name, result=_truncate(str(result)) if result else None)
            if name == "lookup_order":
                snapshot = order_lookup.get_last_order_snapshot()
                if snapshot:
                    last_lookup_status["status"] = snapshot.get("status")
                    last_lookup_status["order_number"] = snapshot.get("order_number")
                    last_lookup_status["language"] = "en"
                    last_lookup_status["updated_at"] = time.time()
    
    # Store references for session management
    _current_session["agent"] = agent
    _current_session["room"] = ctx.room
    
    async def handle_call_end(reason: str = "normal"):
        """Handle call ending - save transcript and clean up."""
        if call_ended["value"]:
            logger.debug("Call end already handled, skipping")
            return
        call_ended["value"] = True
        _current_session["should_end"] = True
        
        try:
            # Calculate call duration
            call_duration = int(time.time() - call_start_time)
            
            # Build transcript string
            full_transcript = "\n".join(conversation_transcript)
            
            logger.info(f"Handling call end: {reason}, duration={call_duration}s, transcript_lines={len(conversation_transcript)}")
            room_log("CALL_END", reason=reason, duration_s=call_duration, transcript_lines=len(conversation_transcript))
            room_log("FULL_TRANSCRIPT", transcript=full_transcript)
            
            # Log call completed event (for all calls)
            await log_call_event(
                event_type="call_completed",
                room_name=ctx.room.name,
                call_type=call_type,
                caller_number=caller_number,
                caller_identity=caller_identity,
                call_id=db_call_id,
                duration_seconds=call_duration,
                metadata={
                    "source": "livekit_agent",
                    "disconnect_reason": reason,
                    "transcript_lines": len(conversation_transcript),
                },
            )
            
            # Record call end in database (updates analytics) with transcript
            await end_call_in_db(
                call_id=db_call_id,
                room_name=ctx.room.name,
                status="completed",
                duration_seconds=call_duration,
                disconnect_reason=reason,
                transcript=full_transcript if full_transcript else None,
            )
            
            logger.info(f"Call recorded: {db_call_id}, transcript saved ({len(full_transcript)} chars)")
            
        except Exception as e:
            logger.error(f"Error handling call end: {e}")
        finally:
            _current_session["silence_tracker"] = None
            _current_session["silence_pause_depth"] = 0
            _current_session["last_silence_block_reason"] = None
            _current_session["last_lookup_wait_phrase"] = None
            _current_session["pending_lookup_wait_phrase"] = None
            _current_session["pending_lookup_wait_phrase_set_at"] = 0.0
            _current_session["last_agent_text"] = ""
            _current_session["last_forced_lookup_order"] = None
            _current_session["last_forced_lookup_at"] = 0.0
            _current_session["last_user_turn_id"] = 0
            _current_session["last_lookup_tool_called_at"] = 0.0
            _current_session["last_lookup_tool_order"] = None
            _current_session["lookup_progress_prompt_until"] = 0.0
            _current_session["number_mode_lock"] = None
            _current_session["number_mode_turn_id"] = 0
            _current_session["forced_response_manual_say_active"] = False
            _current_session["forced_response_spoken_turn_id"] = 0
            _current_session["forced_response_spoken_text"] = ""
            _current_session["forced_response_transcript_sent_at"] = 0.0
            _current_session["forced_response_suppress_llm_until"] = 0.0
            _current_session["last_manual_prompt_turn_id"] = 0
            _current_session["last_manual_prompt_key"] = ""
            _current_session["lookup_pending"] = False
            _current_session["lookup_pending_started_at"] = 0.0
            _current_session["lookup_pending_order"] = None
            _current_session["last_lookup_state"] = "unknown"
            _current_session["last_lookup_order"] = None
            _current_session["pending_phone_candidate"] = None
            _current_session["phone_lookup_inflight"] = False
            _current_session["phone_forced_turn_id"] = 0
            _current_session["phone_forced_pending_turn_id"] = 0
            _current_session["details_lookup_inflight"] = False
            _current_session["details_forced_turn_id"] = 0
            _current_session["details_forced_pending_turn_id"] = 0
            _current_session["details_last_spoken_order"] = None
            _current_session["details_last_spoken_at"] = 0.0
            _current_session["details_confirmation_pending"] = False
            _current_session["details_confirmation_pending_until"] = 0.0
            _current_session["full_order_details_allowed_until"] = 0.0
            _current_session["full_details_unlocked_once"] = False
            _current_session["last_more_details_prompt_at"] = 0.0
            _current_session["ticket_confirmation_pending"] = False
            _current_session["ticket_confirmation_pending_until"] = 0.0
            _current_session["ticket_create_allowed_until"] = 0.0
            _current_session["pending_ticket_payload"] = None
            _current_session["ticket_created"] = False
            _current_session["ticket_reference"] = None
            _current_session["support_flow_state"] = FLOW_IDLE
            _current_session["support_issue"] = None
            set_runtime_language("en")
    # Handle participant disconnection
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant_info):
        """Handle when a participant (user) disconnects."""
        logger.info(f"Participant disconnected: {participant_info.identity}")
        
        # Check if this is the main participant (not the agent)
        if participant_info.identity != "agent" and participant_info.identity == caller_identity:
            logger.info("User disconnected - ending call and closing room")
            asyncio.create_task(handle_call_end("user_disconnected"))
            # Disconnect room after a short delay to allow cleanup
            async def delayed_disconnect():
                await asyncio.sleep(0.5)
                try:
                    if ctx.room and ctx.room.isconnected():
                        await ctx.room.disconnect()
                except Exception as e:
                    logger.debug(f"Room disconnect: {e}")
            asyncio.create_task(delayed_disconnect())
    
    # Handle room disconnection
    @ctx.room.on("disconnected")
    def on_room_disconnected():
        """Handle when the room is disconnected."""
        logger.info("Room disconnected")
        asyncio.create_task(handle_call_end("room_disconnected"))
    # Ensure call is recorded before starting to have db_call_id
    if record_task:
        await record_task
        
    agent.start(ctx.room, participant)
    logger.info(f"⏱️ Agent started ({time.time() - startup_time:.1f}s)")

    # Stream interim user transcripts to the UI for realtime feel.
    human_input = getattr(agent, "_human_input", None)
    if human_input:
        @human_input.on("interim_transcript")
        def on_interim_transcript(ev):
            try:
                text = ev.alternatives[0].text
            except Exception:
                text = None
            if text:
                cancel_thinking_task()
                asyncio.create_task(send_user_transcript(text, interim=True))
    
    # Start background audio if enabled (runs completely independently)
    bg_audio_player = None
    async def start_background_audio():
        nonlocal bg_audio_player
        try:
            from src.services.background_audio import create_background_audio_player
            bg_audio_player = await create_background_audio_player()
            if bg_audio_player:
                success = await bg_audio_player.start(ctx.room)
                if success:
                    logger.info("🎵 Background audio playing continuously")
                else:
                    logger.warning("🎵 Background audio failed to start")
        except Exception as e:
            logger.debug(f"Background audio not available: {e}")
    
    # Get greeting based on configured language
    agent_lang = "en"
    greeting_enabled = _require_bool_setting("agent_greeting_enabled")
    if greeting_enabled:
        greeting = get_greeting("en")
        logger.info(f"⏱️ Saying greeting ({time.time() - startup_time:.1f}s): {greeting[:50]}...")
        # No manual send_agent_transcript here; on_agent_speech_committed will handle it automatically.
        agent.chat_ctx.append(role="assistant", text=greeting)
        await agent.say(greeting, allow_interruptions=True)
    else:
        logger.info("Greeting disabled by settings")
    
    total_startup = time.time() - startup_time
    logger.info(f"✅ Elena ready! Total startup: {total_startup:.1f}s, language: en")
    
    # Reset silence timer after greeting (or initial ready state if greeting disabled)
    mark_agent_speaking()

    # Start background audio after the greeting to avoid delaying first response.
    bg_audio_task = asyncio.create_task(start_background_audio())
    bg_audio_task.set_name("bg_audio_init")

    # Defer order prefetch until after greeting to reduce initial latency.
    asyncio.create_task(order_lookup.prefetch_orders())
    
    # =========================================================================
    # SILENCE MONITORING - Prompt user if no response
    # =========================================================================
    async def monitor_silence():
        """Monitor for user silence and prompt them."""
        # Contextual prompts for elena_en.py
        def _get_contextual_silence_prompt(count):
            state = _current_session.get("support_flow_state", FLOW_IDLE)
            
            # Goodbye prompt (last one)
            if count >= silence_tracker["max_prompts"] - 1:
                return "I'll end the call for now. You can call us again anytime."

            # State-specific prompts
            if state == FLOW_AWAITING_ORDER_NUMBER:
                if count == 0:
                    return "Whenever you are ready, please share your order number."
                return "I'm still here. Please provide your order number so I can check the details."
            
            if state in PHONE_FLOW_STATES:
                if count == 0:
                    return "Please share the phone number used for the order when you are ready."
                return "I'm ready whenever you are. Please provide the full phone number."

            # Default / Idle contextual nudge
            # Try to nudge based on the last thing the assistant said
            last_msg = ""
            for msg in reversed(agent.chat_ctx.messages):
                if msg.role == "assistant" and msg.text:
                    last_msg = msg.text
                    break
            
            if last_msg:
                # Simple rephrasing or reminder
                if count == 0:
                    return "I'm still here to help. Would you like to continue?"
                return "Take your time. I'm still here if you need help with anything else."
            
            return "I'm here when you're ready."

        try:
            while not _current_session["should_end"] and silence_tracker["enabled"]:
                await asyncio.sleep(1.0)  # Check every second
                
                # Only check silence if we're waiting for a response
                if not silence_tracker["is_waiting_for_response"]:
                    continue

                now = time.time()

                lookup_active = (
                    bool(_current_session.get("lookup_pending"))
                    or bool(_current_session.get("phone_lookup_inflight"))
                    or bool(_current_session.get("details_lookup_inflight"))
                )
                # Extra guard: after any lookup tool call, suppress idle prompts briefly
                # even if state flags clear early due timing races.
                last_lookup_called_at = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                lookup_recent_window = _as_float(
                    get_agent_setting("lookup_silence_recent_window_seconds", 20.0),
                    20.0,
                    min_value=5.0,
                    max_value=120.0,
                )
                if last_lookup_called_at and (now - last_lookup_called_at) < lookup_recent_window:
                    continue
                if lookup_active:
                    pending_started = float(_current_session.get("lookup_pending_started_at") or 0.0)
                    max_lookup_block_s = _as_float(
                        get_agent_setting("lookup_silence_block_max_seconds", 90.0),
                        90.0,
                        min_value=15.0,
                        max_value=300.0,
                    )

                    # Periodic progress updates while searching
                    last_progress = float(_current_session.get("last_lookup_progress_at") or 0.0)
                    progress_interval = _as_float(get_agent_setting("lookup_progress_interval_seconds", 15.0), 15.0)
                    
                    if last_progress and (now - last_progress) >= progress_interval:
                        # Guard: don't interrupt if user is speaking or a manual prompt is active
                        if not silence_tracker.get("user_is_speaking") and not _current_session.get("forced_response_manual_say_active"):
                            _current_session["last_lookup_progress_at"] = now
                            progress_msg = "I'm still searching for your order, thank you for your patience."
                            logger.info(f"⏳ Sending periodic search update: {progress_msg}")
                            await send_agent_transcript(progress_msg)
                            agent.chat_ctx.append(role="assistant", text=progress_msg)
                            await agent.say(progress_msg, allow_interruptions=False)

                    # If lookup state is stale, clear it silently.
                    if pending_started and (now - pending_started) > max_lookup_block_s:
                        room_log(
                            "LOOKUP_SILENCE_BLOCK_STALE_CLEARED",
                            age_s=round(now - pending_started, 2),
                        )
                        _clear_lookup_pending("lookup_silence_block_stale")
                        _current_session["phone_lookup_inflight"] = False
                        _current_session["details_lookup_inflight"] = False
                        tracker = _current_session.get("silence_tracker")
                        if isinstance(tracker, dict):
                            tracker["last_user_speech"] = now
                            tracker["last_agent_speech"] = now
                            tracker["prompt_count"] = 0

                    # Important:
                    # While lookup is active, silence monitor must not speak "are you there".
                    continue

                # Pause silence prompts while tool calls are executing.
                if silence_tracker.get("paused_by_tool"):
                    continue

                # Skip checks while agent is speaking or LLM is thinking.
                if silence_tracker.get("agent_is_speaking") or \
                   silence_tracker.get("llm_is_generating") or \
                   silence_tracker.get("prompt_in_progress"):
                    continue

                # Strong global guard: never say silence prompts during deterministic work.
                if _should_block_silence_prompt("monitor_loop"):
                    continue
                
                # Calculate time since last activity
                time_since_user = now - silence_tracker["last_user_speech"]
                time_since_agent = now - silence_tracker["last_agent_speech"]
                
                # Only trigger if:
                # 1. User hasn't spoken for silence_timeout seconds
                # 2. Agent finished speaking at least silence_timeout seconds ago
                # 3. Extra grace period of 2s since last agent activity to avoid overlapping prompts
                if time_since_user >= silence_tracker["silence_timeout"] and \
                   time_since_agent >= max(silence_tracker["silence_timeout"], 2.0):
                    
                    prompt_count = silence_tracker["prompt_count"]
                    
                    if prompt_count < silence_tracker["max_prompts"]:
                        # Prompt the user
                        prompt_text = _get_contextual_silence_prompt(prompt_count)
                        # Race-condition guard right before speaking.
                        if _should_block_silence_prompt("before_silence_prompt_say"):
                            continue
                        logger.info(f"🔇 Silence detected ({time_since_user:.1f}s), prompting user: {prompt_text}")
                        
                        silence_tracker["prompt_count"] += 1
                        silence_tracker["is_waiting_for_response"] = False  # Will be set again after agent speaks
                        
                        # Say the prompt
                        try:
                            silence_tracker["prompt_in_progress"] = True
                            await send_agent_transcript(prompt_text)
                            agent.chat_ctx.append(role="assistant", text=prompt_text)
                            await agent.say(prompt_text, allow_interruptions=True)
                        finally:
                            silence_tracker["prompt_in_progress"] = False
                        
                    else:
                        # Max prompts reached - disconnect
                        goodbye_text = _get_contextual_silence_prompt(prompt_count)
                        logger.info(f"🔇 Max silence prompts reached, disconnecting: {goodbye_text}")
                        
                        silence_tracker["enabled"] = False
                        await agent.say(goodbye_text, allow_interruptions=False)
                        await send_agent_transcript(goodbye_text)
                        agent.chat_ctx.append(role="assistant", text=goodbye_text)
                        
                        # Wait for goodbye to finish, then disconnect
                        await asyncio.sleep(3.0)
                        _current_session["should_end"] = True
                        break
                        
        except asyncio.CancelledError:
            logger.debug("Silence monitor cancelled")
        except Exception as e:
            logger.error(f"Silence monitor error: {e}")
    
    # Monitor for session end request (agent-initiated end)
    async def monitor_session_end():
        try:
            while not _current_session["should_end"]:
                await asyncio.sleep(0.5)
            
            logger.info("Session end flag set - disconnecting now")
            await asyncio.sleep(0.5)
            
            # Stop background audio
            if bg_audio_player:
                await bg_audio_player.stop()
            
            await handle_call_end("agent_ended")
            
            # Clean disconnect
            if ctx.room and ctx.room.isconnected():
                await ctx.room.disconnect()
            
            logger.info("Session ended by agent successfully")
            
        except Exception as e:
            logger.debug(f"Session end cleanup: {e} (expected during disconnect)")
    
    # Start monitoring tasks in background
    asyncio.create_task(monitor_silence())
    asyncio.create_task(monitor_session_end())


def prewarm(proc: JobProcess):
    """
    Prewarm function - Keep it lightweight to avoid issues.
    
    NOTE: We intentionally do NOT pre-fetch database here because:
    1. It creates connections in a separate event loop that can't be reused
    2. It exhausts the database connection pool
    3. The connections get "attached to a different loop" errors
    
    The database fetch happens quickly during call startup with parallel queries.
    """
    logger.info("Prewarm: ready (lightweight)")

    def _warm_tts():
        # Best-effort TTS warm-up to reduce first greeting latency.
        try:
            import urllib.request
            import urllib.error

            # Warm ElevenLabs if configured.
            if settings.elevenlabs_api_key:
                voice_id = settings.elevenlabs_voice_id
                model_id = settings.elevenlabs_model or "eleven_multilingual_v2"
                payload = {
                    "text": "Hi",
                    "model_id": model_id,
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    data=data,
                    headers={
                        "xi-api-key": settings.elevenlabs_api_key,
                        "Content-Type": "application/json",
                        "accept": "audio/mpeg",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        resp.read(1)
                    logger.info("TTS prewarm: ElevenLabs warmed")
                except Exception as e:
                    logger.debug(f"TTS prewarm: ElevenLabs skipped ({e})")

            # Warm OpenAI TTS if configured.
            if settings.openai_api_key:
                payload = {
                    "model": "tts-1",
                    "voice": "alloy",
                    "input": "Hi",
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/audio/speech",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        resp.read(1)
                    logger.info("TTS prewarm: OpenAI warmed")
                except Exception as e:
                    logger.debug(f"TTS prewarm: OpenAI skipped ({e})")
        except Exception as e:
            logger.debug(f"TTS prewarm error: {e}")

    threading.Thread(target=_warm_tts, daemon=True).start()


def run_agent():
    """
    Run the Elena voice agent as a LiveKit worker.
    """
    import os
    
    # Configure logging
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Ensure our services log at least at INFO level
    logging.getLogger("src.services.shopify").setLevel(logging.INFO)
    logging.getLogger("src.agents.elena").setLevel(logging.INFO)
    logger.info(f"Elena Worker starting (log_level={settings.log_level})")

    def _get_float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _get_int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    initialize_timeout = _get_float_env("LIVEKIT_AGENTS_INITIALIZE_TIMEOUT", 60.0)
    shutdown_timeout = _get_float_env("LIVEKIT_AGENTS_SHUTDOWN_TIMEOUT", 60.0)
    num_idle_processes = _get_int_env("LIVEKIT_AGENTS_NUM_IDLE_PROCESSES", 1)
    load_threshold = _get_float_env("LIVEKIT_AGENTS_LOAD_THRESHOLD", 0.90)

    initialize_timeout = max(5.0, min(300.0, initialize_timeout))
    shutdown_timeout = max(10.0, min(300.0, shutdown_timeout))
    num_idle_processes = max(0, min(8, num_idle_processes))
    load_threshold = max(0.1, min(1.0, load_threshold))

    logger.info(
        "Worker options: initialize_timeout=%ss, shutdown_timeout=%ss, num_idle_processes=%s, load_threshold=%s",
        initialize_timeout,
        shutdown_timeout,
        num_idle_processes,
        load_threshold,
    )

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
            initialize_process_timeout=initialize_timeout,
            shutdown_process_timeout=shutdown_timeout,
            num_idle_processes=num_idle_processes,
            load_threshold=load_threshold,
        ),
    )


# For running directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_agent()
