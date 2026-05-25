"""
Meallion Voice AI - Elena English Agent (Patch 9 - Ticket Flow Fix)
- Removed phone from ticket flow (name → email → issue only)
- Added create_ticket_without_phone
- Guarded phone lookup against ticket states
- Early LLM suppression for ticket flow
- Fixed interrupted speech race condition
"""

AGENT_BUILD = "patch9d-suppress-only-20260525"

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Optional

from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm, JobRequest
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import elevenlabs, openai, silero

try:
    from livekit.plugins import deepgram
    USE_DEEPGRAM = True
except ImportError:
    USE_DEEPGRAM = False

from src.config import settings
from src.agents.en.prompts import (
    _fetch_from_db,
    get_agent_setting,
    get_closing,
    get_greeting,
    get_system_prompt_async,
)
from src.agents.en import tools as order_lookup
from src.agents.en import tools as knowledge_base
from src.agents.en import tools as support_ticket

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PATCH 8: 5-minute automatic cache expiry
# -----------------------------------------------------------------------------

_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes

_agent_cache: dict = {
    "memory_items":  None,   # list | None
    "last_refresh":  0.0,    # epoch float
}
_agent_cache_lock = threading.Lock()


def _agent_cache_is_valid() -> bool:
    """Return True if the cache was refreshed within the last 5 minutes."""
    return (time.time() - _agent_cache["last_refresh"]) < _CACHE_TTL_SECONDS


def flush_agent_cache() -> None:
    """Manually flush all agent cache entries (thread-safe). Forces next call to re-fetch from DB."""
    with _agent_cache_lock:
        _agent_cache["memory_items"] = None
        _agent_cache["last_refresh"] = 0.0
    logger.info("PATCH 8: Agent cache flushed")


async def _refresh_agent_cache(force: bool = False) -> None:
    """Populate _agent_cache from DB if stale or if force=True."""
    if not force and _agent_cache_is_valid():
        return

    logger.info("PATCH 8: Refreshing agent cache (TTL=5min, force=%s)...", force)
    # Force prompts.py to re-fetch KB/settings/memory from DB
    try:
        from src.agents.en.prompts import _fetch_from_db as _prompts_fetch
        await _prompts_fetch(force=True)
    except Exception as e:
        logger.warning("PATCH 8: prompts re-fetch error: %s", e)

    # Re-load memory items
    memory_items: list = []
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        memory_items = await db.get_memory_items(active_only=True)
    except Exception as e:
        logger.warning("PATCH 8: memory_items refresh error: %s", e)
        # Keep previous value if available
        memory_items = _agent_cache.get("memory_items") or []

    with _agent_cache_lock:
        _agent_cache["memory_items"] = memory_items
        _agent_cache["last_refresh"] = time.time()

    logger.info(
        "PATCH 8: Agent cache refreshed at %s UTC | memory_items=%d | next refresh in 5 min",
        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        len(memory_items),
    )


async def _cache_auto_refresh_loop() -> None:
    """Background task: wake every 60 s, flush and refresh cache when 5-min TTL has expired."""
    while True:
        await asyncio.sleep(60.0)   # check interval: every 1 minute
        if not _agent_cache_is_valid():
            logger.info("PATCH 8: Cache TTL expired — auto-flushing and refreshing")
            flush_agent_cache()
            try:
                await _refresh_agent_cache(force=True)
            except Exception as e:
                logger.warning("PATCH 8: Auto-refresh error: %s", e)


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------

def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_float(value: object, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _as_int(value: object, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _truncate(text: str, max_len: int = 180) -> str:
    s = str(text or "")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


_ORDER_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_MEMORY_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "did", "for", "and", "or", "to",
    "of", "in", "on", "with", "any", "have", "has", "can", "i", "you", "we", "our",
    "your", "my", "it", "this", "that", "what", "how",
}


def _extract_digit_parts(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    parts: list[str] = []
    for token in tokens:
        if token in _ORDER_WORDS:
            parts.append(_ORDER_WORDS[token])
            continue
        if token.isdigit():
            parts.append(token)
            continue
        embedded = re.sub(r"\D", "", token)
        if embedded:
            parts.append(embedded)
    return parts


def _order_digit_range() -> tuple[int, int]:
    min_d = _as_int(get_agent_setting("order_id_min_digits", 3), 3, min_value=3, max_value=9)
    max_d = _as_int(get_agent_setting("order_id_max_digits", 6), 6, min_value=min_d, max_value=9)
    return min_d, max_d


def _normalize_order_id_strict(raw_text: str) -> Optional[str]:
    min_d, max_d = _order_digit_range()
    joined = "".join(_extract_digit_parts(raw_text or ""))
    
    # PATCH 3: If the total number of digits is too high, it's likely a phone number.
    # Do not extract a partial order ID from it.
    if len(joined) > max_d:
        return None
        
    if min_d <= len(joined) <= max_d:
        return joined
        
    # Fallback only if the string is very short or doesn't look like a phone number.
    normalized = (raw_text or "").strip().lower()
    matches = re.findall(rf"\d{{{min_d},{max_d}}}", normalized)
    return matches[-1] if matches else None


def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    digits = "".join(_extract_digit_parts(raw_text or ""))
    if not digits:
        return None
    min_digits = _as_int(get_agent_setting("phone_lookup_min_digits", 10), 10, min_value=7, max_value=15)
    max_digits = _as_int(get_agent_setting("phone_lookup_max_digits", 15), 15, min_value=min_digits, max_value=15)
    rx = str(get_agent_setting("phone_lookup_regex", r"^\d{10,15}$") or "").strip()
    if rx:
        try:
            if re.fullmatch(rx, digits):
                return digits
        except re.error:
            pass
    if min_digits <= len(digits) <= max_digits:
        return digits
    return None


def _clean_transcript_phone(text: str) -> str:
    """PATCH 4: Clean transcript text by replacing formatted phone numbers with pure digits."""
    cleaned = text
    # Matches patterns like (694) 263-3977, 694-263-3977, or (694) 263 3977
    matches = re.finditer(r"\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}", text)
    for m in matches:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        cleaned = cleaned.replace(raw, digits)
    return cleaned


def snooze_silence(seconds: float):
    """PATCH 4: Globally accessible snooze function."""
    state = _current.get("state")
    if state:
        now_ts = time.time()
        state.silence_snooze_until = max(state.silence_snooze_until, now_ts + max(0.0, seconds))
        room_log("SILENCE_SNOOZE", until=state.silence_snooze_until, seconds=seconds)


def _mentions_no_order_number(text: str) -> bool:
    t = (text or "").lower()
    return bool(
        re.search(r"(i do not have|i don't have|no order number|dont have order|don't have order)", t)
        or re.search(r"(didn't get|did not get|didn't receive|did not receive).*(email|confirmation)", t)
    )


def _mentions_order_lookup_intent(text: str) -> bool:
    """Check if the user wants to switch back to looking up by order number."""
    t = (text or "").lower()
    return bool(re.search(r"(order number|order id|order #|check order|search order|another order)", t))


def _mentions_phone_lookup_intent(text: str) -> bool:
    t = (text or "").lower()
    # PATCH 2/3: narrowed intent matching
    has_phone_keyword = bool(re.search(r"\b(phone|mobile|cell|contact)\b", t))
    has_explicit_request = bool(re.search(r"(check by phone|use phone|with phone|phone number)", t))
    return has_phone_keyword or has_explicit_request


def _is_order_relevant(text: str) -> bool:
    """Check if the text is likely an attempt to provide order info or ask about it."""
    t = (text or "").lower()
    # If it has any digits, it's likely an attempt at a number
    if re.search(r"\d", t):
        return True
    # Keywords that suggest they are still in the flow
    keywords = {
        "order", "number", "phone", "check", "find", "look", "track", "where",
        "problem", "issue", "id", "help", "yes", "ok", "sure", "here", "ready"
    }
    tokens = set(re.findall(r"\w+", t))
    return bool(tokens & keywords)


# -----------------------------------------------------------------------------
# Room logs and state
# -----------------------------------------------------------------------------


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value or "")


def _create_room_logger(room_name: str, job_id: Optional[str]) -> tuple[logging.Logger, str]:
    log_dir = os.getenv("ROOM_LOG_DIR", "./data/room-logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"room_{_safe_slug(room_name)}_{_safe_slug(job_id or 'job')}_{ts}.log"
    path = os.path.join(log_dir, filename)

    room_logger = logging.getLogger(f"room.{_safe_slug(room_name)}.{_safe_slug(job_id or 'job')}.{ts}")
    room_logger.setLevel(logging.INFO)
    room_logger.propagate = False
    if not room_logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        room_logger.addHandler(handler)
    return room_logger, path


@dataclass
class SessionState:
    support_state: str = "idle"  # idle|awaiting_order|checking_order|awaiting_phone|checking_phone|ticket_name|ticket_email|ticket_issue|ticket_confirm|creating_ticket
    ui_state: str = "idle"  # idle|listening|thinking|speaking
    last_issue: str = ""
    last_order_number: str = ""
    last_phone_number: str = ""
    lookup_inflight: bool = False
    ticket_inflight: bool = False
    ticket_name: str = ""
    ticket_email: str = ""
    ticket_issue: str = ""
    ticket_id: str = ""
    should_end: bool = False
    disconnect_reason: str = "session_end"
    silence_enabled: bool = True
    silence_timeout_s: float = 12.0
    silence_max_prompts: int = 2
    silence_prompt_count: int = 0
    silence_snooze_until: float = 0.0
    waiting_for_user: bool = False
    last_user_activity: float = 0.0
    last_agent_activity: float = 0.0
    last_user_transcript_text: str = ""
    last_user_transcript_at: float = 0.0
    last_agent_transcript_text: str = ""
    last_agent_transcript_at: float = 0.0
    last_clarification_prompt_text: str = ""
    last_clarification_prompt_at: float = 0.0
    suppress_llm_until: float = 0.0


_current = {
    "room_logger": None,
    "room_name": None,
    "job_id": None,
    "call_id": None,
    "state": SessionState(),
}


def room_log(event: str, **fields):
    rl = _current.get("room_logger")
    if not rl:
        return
    payload = {
        "event": event,
        "room": _current.get("room_name"),
        "job_id": _current.get("job_id"),
        "call_id": _current.get("call_id"),
    }
    payload.update(fields)
    rl.info(json.dumps(payload, ensure_ascii=False))


# -----------------------------------------------------------------------------
# DB call lifecycle helpers
# -----------------------------------------------------------------------------


async def record_call_to_db(room_name: str, call_type: str = "web", caller_number: str = None, caller_identity: str = None) -> Optional[str]:
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        return await db.record_call_start(
            room_name=room_name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
        )
    except Exception as e:
        logger.warning("Failed to record call start: %s", e)
        return None


async def end_call_in_db(call_id: str = None, room_name: str = None, status: str = "completed", duration_seconds: int = None, disconnect_reason: str = None, transcript: str = None) -> bool:
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
        logger.warning("Failed to record call end: %s", e)
        return False


async def save_transcript_to_db(call_id: str, text: str, speaker: str = "agent", append: bool = True) -> bool:
    if not call_id or not text:
        return False
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        transcript = f"{speaker.capitalize()}: {text}" if append else text
        return await db.update_call_transcript(call_id=call_id, transcript=transcript, append=append)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class ElenaFunctionContext(llm.FunctionContext):
    @llm.ai_callable()
    async def lookup_order(self, order_number: Annotated[str, llm.TypeInfo(description="Order number")]) -> str:
        """Look up an order by order number and return a customer-facing summary."""
        # Strict Guard: If it looks like a phone number (7+ digits), reject it.
        # This prevents the LLM from talking over the phone lookup handler.
        clean_num = re.sub(r"\D", "", str(order_number))
        if len(clean_num) >= 7:
            return "ERROR: This tool is only for 3-6 digit order numbers. For phone numbers, please wait for the automated lookup."

        room_log("TOOL_CALL", name="lookup_order", order_number=order_number)
        result = await order_lookup.lookup_order(order_number)
        room_log("TOOL_RESULT", name="lookup_order", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def get_order_details(self, order_number: Annotated[str, llm.TypeInfo(description="Order number or last")] = "last") -> str:
        """Fetch detailed information for a specific order or the last looked-up order."""
        room_log("TOOL_CALL", name="get_order_details", order_number=order_number)
        result = await order_lookup.get_order_details(order_number)
        room_log("TOOL_RESULT", name="get_order_details", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def lookup_order_by_phone(self, phone: Annotated[str, llm.TypeInfo(description="10-digit phone number")]) -> str:
        """Look up an order by phone number and return a customer-facing summary."""
        # Strict Guard: Only process 10 digits.
        clean_phone = re.sub(r"\D", "", str(phone))
        if len(clean_phone) < 10:
            return "ERROR: This tool requires a full 10-digit phone number."

        room_log("TOOL_CALL", name="lookup_order_by_phone", phone=phone)
        result = await order_lookup.lookup_order_by_phone(phone)
        room_log("TOOL_RESULT", name="lookup_order_by_phone", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def search_knowledge_base(self, query: Annotated[str, llm.TypeInfo(description="Question to search")]) -> str:
        """Search the knowledge base for policy, menu, and general support answers."""
        room_log("TOOL_CALL", name="search_knowledge_base", query=query)
        result = await knowledge_base.search_knowledge_base(query, language="en")
        room_log("TOOL_RESULT", name="search_knowledge_base", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def create_support_ticket(
        self,
        customer_name: Annotated[str, llm.TypeInfo(description="Customer full name")],
        customer_email: Annotated[str, llm.TypeInfo(description="Customer email")],
        issue_description: Annotated[str, llm.TypeInfo(description="Issue")],
    ) -> str:
        """Create a support ticket when an issue cannot be resolved during the call. Only collect name, email, and issue. Do NOT ask for phone number."""
        room_log("TOOL_CALL", name="create_support_ticket")
        result = await support_ticket.create_ticket_without_phone(
            customer_name=customer_name,
            customer_email=customer_email,
            issue_description=issue_description,
        )
        msg = result.get("message", "Sorry, I couldn't create the support ticket.")
        room_log("TOOL_RESULT", name="create_support_ticket", result=_truncate(msg))
        return msg

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
        logger.info("Session end requested - scheduling disconnect after goodbye")
        room_log("SESSION_END_REQUESTED")

        # Disable silence monitor immediately!
        _current["state"].silence_enabled = False

        # Schedule the disconnect with a delay to allow goodbye to be spoken
        async def delayed_end():
            # Wait for LLM to process response + TTS to generate + speak
            # This needs to be long enough for the full goodbye to be heard
            await asyncio.sleep(6.0)  # 6 seconds should be plenty
            _current["state"].should_end = True
            logger.info("Delayed session end triggered")

        asyncio.create_task(delayed_end())

        # Return closing message
        goodbye = get_closing("en")
        room_log("SESSION_END_MESSAGE", text=_truncate(goodbye))
        return goodbye


# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------


def create_llm():
    model = str(get_agent_setting("llm_model", "gpt-4o-mini") or "gpt-4o-mini")
    return openai.LLM(model=model, api_key=settings.openai_api_key)


def create_stt(is_sip_call: bool = False):
    provider = str(get_agent_setting("stt_provider", "deepgram") or "deepgram").lower()
    deepgram_api_key = getattr(settings, "deepgram_api_key", None)
    if provider == "deepgram" and USE_DEEPGRAM and deepgram_api_key:
        # Use the model configured in the DB, defaulting to nova-3 (same as original).
        model = str(get_agent_setting("deepgram_stt_model", "nova-3") or "nova-3").strip()

        # Base config matching elena_original.py — smart_format=True is critical
        # for reliable digit transcription (e.g. "seven seven three" → "773").
        base = {
            "model": model,
            "language": "en-US",
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }

        # Try with explicit api_key first; fall back without it (some SDK versions
        # reject api_key as an unknown kwarg — the original does NOT pass it).
        for kwargs in [
            {**base, "api_key": deepgram_api_key},
            base,
        ]:
            try:
                logger.info(
                    "Creating Deepgram STT: model=%s language=en-US smart_format=True api_key_passed=%s",
                    model,
                    "api_key" in kwargs,
                )
                return deepgram.STT(**kwargs)
            except TypeError as e:
                logger.warning("Deepgram STT args not supported, retrying with fallback args: %s", e)
                continue

        # Hard fallback: minimal args only (mirrors original's final safety net)
        return deepgram.STT(model=model)

    openai_model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1")
    return openai.STT(model=openai_model, api_key=settings.openai_api_key, language="en")


def create_tts():
    provider = str(get_agent_setting("tts_provider", "elevenlabs") or "elevenlabs").lower()
    if provider == "openai":
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )

    # ElevenLabs default (SDK-compatible signature), with safe fallback.
    eleven_api_key = getattr(settings, "elevenlabs_api_key", None)
    if not eleven_api_key:
        logger.warning("ELEVENLABS_API_KEY missing, falling back to OpenAI TTS")
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )

    voice_id = str(
        get_agent_setting("agent_voice_id", getattr(settings, "elevenlabs_voice_id", "")) or getattr(settings, "elevenlabs_voice_id", "")
    )
    similarity = _as_float(
        get_agent_setting("agent_voice_similarity", getattr(settings, "elevenlabs_voice_similarity", 0.9)),
        0.9,
        min_value=0.0,
        max_value=1.0,
    )
    stability = _as_float(
        get_agent_setting("agent_voice_stability", getattr(settings, "elevenlabs_voice_stability", 0.65)),
        0.65,
        min_value=0.0,
        max_value=1.0,
    )
    model = str(
        get_agent_setting("elevenlabs_model", getattr(settings, "elevenlabs_model", "eleven_turbo_v2_5") or "eleven_turbo_v2_5")
        or "eleven_turbo_v2_5"
    )

    try:
        voice = elevenlabs.Voice(
            id=voice_id,
            name="Elena",
            category="premade",
            settings=elevenlabs.VoiceSettings(stability=stability, similarity_boost=similarity),
        )
        return elevenlabs.TTS(
            voice=voice,
            model=model,
        )
    except TypeError as e:
        logger.warning("ElevenLabs TTS init failed (%s), falling back to OpenAI TTS", e)
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )


def _looks_like_email(text: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (text or "").strip()))


def _is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"yes", "y", "confirm", "confirmed", "correct", "go ahead", "please do"}


def _is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"no", "n", "cancel", "stop", "not now"}


def _normalize_intent_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _intent_tokens(text: str) -> set[str]:
    toks = _normalize_intent_text(text).split()
    normalized: set[str] = set()
    for tok in toks:
        if len(tok) < 3 or tok in _MEMORY_STOPWORDS:
            continue
        t = tok
        # Generic light stemming (language-agnostic enough for EN, no domain hardcoding).
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("es") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 4:
            t = t[:-1]
        elif t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 4:
            t = t[:-2]
        normalized.add(t)
    return normalized


def _find_memory_match(user_text: str, memory_items: list[dict]) -> Optional[str]:
    user_norm = _normalize_intent_text(user_text)
    user_tokens = _intent_tokens(user_text)
    if not user_norm or not user_tokens:
        return None

    best_answer: Optional[str] = None
    best_score = 0.0

    for item in memory_items:
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        if not q or not a:
            continue
        q_norm = _normalize_intent_text(q)
        q_tokens = _intent_tokens(q)
        if not q_tokens:
            continue

        overlap = len(user_tokens & q_tokens)
        coverage = overlap / max(1, len(q_tokens))
        recall = overlap / max(1, len(user_tokens))
        phrase_hit = 1.0 if (q_norm in user_norm or user_norm in q_norm) else 0.0
        score = (coverage * 0.55) + (recall * 0.35) + (phrase_hit * 0.25)

        if score > best_score and (overlap >= 2 or phrase_hit > 0):
            best_score = score
            best_answer = a

    return best_answer if best_score >= 0.40 else None


def _build_memory_prompt_block(memory_items: list[dict]) -> str:
    lines: list[str] = []
    for item in memory_items:
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        c = str(item.get("comment") or item.get("comments") or "").strip()
        if not q or not a:
            continue
        lines.append(f'SCENARIO (match by intent, not exact words): "{q}"')
        lines.append(f'EXPECTED RESPONSE: "{a}"')
        if c:
            lines.append(f"GUIDELINE: {c}")
        lines.append("-" * 20)
    if not lines:
        return ""
    return (
        "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
        "When user intent matches any memory scenario, respond using that memory response first.\n"
        "Treat scenario matching as semantic/intention-based (not exact wording).\n\n"
        "### ABSOLUTE VERBAL GUARDRAILS (INTERNAL ONLY)\n"
        "- NEVER speak section headers (e.g., '## Closing', '### CORE BEHAVIOR', 'GUIDELINE').\n"
        "- NEVER speak behavioral instructions or meta-rules (e.g., phrases starting with 'Never say...', 'Always say...', 'Avoid...', 'Close naturally...').\n"
        "- Bullet points in the 'CORE BEHAVIOR' or 'Closing' sections are for your logic reasoning ONLY.\n"
        "- NEVER speak the internal tags: 'SCENARIO', 'EXPECTED RESPONSE', or 'GUIDELINE'.\n"
        "- ONLY speak the actual conversational text intended for the customer.\n\n"
        + "\n".join(lines)
    )


def create_vad(is_sip_call: bool = False):
    min_speech = _as_float(get_agent_setting("vad_min_speech_duration", 0.15), 0.15, min_value=0.05, max_value=1.0)
    min_silence = _as_float(get_agent_setting("vad_min_silence_duration", 1.2 if is_sip_call else 1.0), 1.0, min_value=0.1, max_value=3.0)
    # threshold=0.6: more aggressive to filter out background noise hallucinations
    return silero.VAD.load(min_speech_duration=min_speech, min_silence_duration=min_silence, activation_threshold=0.6)


# -----------------------------------------------------------------------------
# Room logs and state
# -----------------------------------------------------------------------------


def _is_order_not_found_text(text: str) -> bool:
    t = (text or "").lower()
    return "could not find" in t or "no order" in t


async def _publish_transcript(ctx: JobContext, speaker: str, text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return
    payload = json.dumps({"type": "transcript", "speaker": speaker, "text": cleaned}, ensure_ascii=False)
    try:
        await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
    except Exception:
        pass


async def _publish_state(ctx: JobContext, state_name: str):
    payload = json.dumps({"type": "state", "state": state_name}, ensure_ascii=False)
    try:
        await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
    except Exception:
        pass


async def _run_order_lookup(agent: VoicePipelineAgent, order_number: str):
    state: SessionState = _current["state"]
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_order"
    room_log("ORDER_LOOKUP_STARTED", order_number=order_number)
    try:
        # PATCH 4: Add 10-second timeout to prevent dead states on slow APIs
        result = await asyncio.wait_for(order_lookup.lookup_order(order_number), timeout=10.0)
        room_log("ORDER_LOOKUP_RESULT", result=_truncate(result))
        
        from src.utils.voice_formatting import clean_text_for_tts
        cleaned_result = clean_text_for_tts(result, lang="en")
        await agent.say(cleaned_result, allow_interruptions=True)
        
        state.last_order_number = order_number
        if _is_order_not_found_text(result):
            state.support_state = "awaiting_order"
            snooze_silence(20.0)
        else:
            state.support_state = "idle"
    except asyncio.TimeoutError:
        room_log("ORDER_LOOKUP_TIMEOUT")
        await agent.say("I'm sorry, it's taking me a bit longer than usual to find your order. Could you please repeat the order number for me?", allow_interruptions=True)
        state.support_state = "awaiting_order"
    except Exception as e:
        logger.error("Order lookup error: %s", e)
        await agent.say("I'm sorry, I'm having trouble checking that order number right now. Please try again in a moment.", allow_interruptions=True)
        state.support_state = "awaiting_order"
    finally:
        state.lookup_inflight = False


async def _run_phone_lookup(agent: VoicePipelineAgent, phone_number: str):
    state: SessionState = _current["state"]
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_phone"
    room_log("PHONE_LOOKUP_STARTED", phone=phone_number)
    try:
        # PATCH 4: Add 10-second timeout to prevent dead states on slow APIs
        result = await asyncio.wait_for(order_lookup.lookup_order_by_phone(phone_number), timeout=10.0)
        room_log("PHONE_LOOKUP_RESULT", result=_truncate(result))
        
        from src.utils.voice_formatting import clean_text_for_tts
        cleaned_result = clean_text_for_tts(result, lang="en")
        await agent.say(cleaned_result, allow_interruptions=True)
        
        state.last_phone_number = phone_number
        if "no order" in (result or "").lower() or "could not" in (result or "").lower():
            state.support_state = "awaiting_phone"
            snooze_silence(20.0)
            # PATCH 5: Clear suppression so agent can respond to next user turn
            state.suppress_llm_until = 0.0
        else:
            state.support_state = "idle"
    except asyncio.TimeoutError:
        room_log("PHONE_LOOKUP_TIMEOUT")
        await agent.say("I'm sorry, I'm having a little trouble looking that up. Could you please share the phone number one more time?", allow_interruptions=True)
        state.support_state = "awaiting_phone"
    except Exception as e:
        logger.error("Phone lookup error: %s", e)
        await agent.say("I'm sorry, I'm having trouble checking that phone number right now. Please try again in a moment.", allow_interruptions=True)
        state.support_state = "awaiting_phone"
    finally:
        state.lookup_inflight = False


async def _run_create_ticket(agent: VoicePipelineAgent):
    state: SessionState = _current["state"]
    if state.ticket_inflight:
        return
    state.ticket_inflight = True
    state.support_state = "creating_ticket"
    room_log("TICKET_CREATE_STARTED")
    try:
        result = await support_ticket.create_ticket_without_phone(
            customer_name=state.ticket_name or "Customer",
            customer_email=state.ticket_email,
            issue_description=state.ticket_issue,
        )
        msg = result.get("message", "Sorry, I couldn't create the support ticket.")
        room_log("TICKET_CREATE_RESULT", result=_truncate(msg))
        await agent.say(msg, allow_interruptions=True)
        state.support_state = "idle"
        state.ticket_name = ""
        state.ticket_email = ""
        state.ticket_issue = ""
        state.ticket_id = ""
    finally:
        state.ticket_inflight = False


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    # PATCH 8: Warm the 5-min agent cache at startup (includes prompts + memory_items).
    cache_task = asyncio.create_task(_refresh_agent_cache(force=True))

    state = SessionState(
        silence_timeout_s=_as_float(get_agent_setting("silence_timeout_seconds", 12.0), 12.0, min_value=6.0, max_value=60.0),
        silence_max_prompts=_as_int(get_agent_setting("silence_max_prompts", 2), 2, min_value=1, max_value=5),
        last_user_activity=time.time(),
        last_agent_activity=time.time(),
    )
    _current["state"] = state
    memory_items: list[dict] = []

    async def set_ui_state(new_state: str):
        if state.ui_state == new_state:
            return
        state.ui_state = new_state
        room_log("UI_STATE", state=new_state)
        await _publish_state(ctx, new_state)

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
                await set_ui_state("thinking")
            except asyncio.CancelledError:
                return

        thinking_task = asyncio.create_task(_set_thinking())

    def _should_suppress_clarification(text: str, min_gap_s: float = 6.0) -> bool:
        now_ts = time.time()
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return True
        if (
            normalized == state.last_clarification_prompt_text
            and (now_ts - state.last_clarification_prompt_at) < max(0.0, min_gap_s)
        ):
            room_log("CLARIFICATION_SUPPRESSED", text=normalized, min_gap_s=min_gap_s)
            return True
        state.last_clarification_prompt_text = normalized
        state.last_clarification_prompt_at = now_ts
        return False

    job = getattr(ctx, "job", None)
    job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
    room_logger, room_log_path = _create_room_logger(ctx.room.name, job_id)
    _current["room_logger"] = room_logger
    _current["room_name"] = ctx.room.name
    _current["job_id"] = job_id
    room_log("ROOM_START", call_type="web", build=AGENT_BUILD)
    logger.info("Per-room log: %s | BUILD: %s", room_log_path, AGENT_BUILD)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity)

    call_id = await record_call_to_db(ctx.room.name, call_type="web", caller_identity=participant.identity)
    _current["call_id"] = call_id

    try:
        await asyncio.wait_for(cache_task, timeout=8.0)
    except Exception as e:
        logger.warning("Initial cache warmup did not complete in time: %s", e)

    # PATCH 8: Pull memory_items from the shared 5-min cache (populated by _refresh_agent_cache).
    try:
        cached_mi = _agent_cache.get("memory_items")
        if cached_mi is not None:
            memory_items = cached_mi
            room_log("MEMORY_ITEMS_FROM_CACHE", count=len(memory_items))
        else:
            from src.services.database import get_database_service
            db = get_database_service()
            memory_items = await db.get_memory_items(active_only=True)
            with _agent_cache_lock:
                _agent_cache["memory_items"] = memory_items
            room_log("MEMORY_ITEMS_LOADED", count=len(memory_items))
    except Exception as e:
        logger.warning("Failed loading memory items for direct matcher: %s", e)
        room_log("MEMORY_ITEMS_LOAD_FAILED", error=str(e))

    try:
        system_prompt = await get_system_prompt_async("en")
    except Exception as e:
        logger.warning("System prompt load failed, using fallback prompt: %s", e)
        system_prompt = (
            "You are Elena from Meallion. Reply in concise, friendly English. "
            "For order issues, ask for order number first, then phone number if needed."
        )
    # Ensure memory guidance is always present even if prompts cache misses memory context.
    memory_block = _build_memory_prompt_block(memory_items)
    if memory_block and "CRITICAL: LONG-TERM MEMORY" not in (system_prompt or ""):
        system_prompt = f"{memory_block}\n\n{system_prompt}"
    has_memory_block = "LONG-TERM MEMORY" in (system_prompt or "")
    room_log("SYSTEM_PROMPT_READY", length=len(system_prompt or ""), has_memory_block=has_memory_block)
    chat_ctx = llm.ChatContext()
    chat_ctx.append(role="system", text=system_prompt)

    configured_endpointing_delay = _as_float(
        get_agent_setting("min_endpointing_delay", 1.2),
        1.2,
        min_value=0.2,
        max_value=3.0,
    )
    # Patience: wait at least ~5s after user stops speaking before replying.
    # This gives the deterministic handler time to fire and suppress the LLM.
    effective_endpointing_delay = max(5.0, configured_endpointing_delay)

    def _before_llm_cb(agent_instance, chat_ctx):
        """Gate the LLM when the deterministic handler has already replied via agent.say()."""
        # Block LLM when suppress timer is active
        if time.time() < state.suppress_llm_until:
            room_log("LLM_SUPPRESSED", until=state.suppress_llm_until)
            return False
        from livekit.agents.pipeline.pipeline_agent import _default_before_llm_cb
        return _default_before_llm_cb(agent_instance, chat_ctx)

    agent = VoicePipelineAgent(
        vad=create_vad(),
        stt=create_stt(),
        llm=create_llm(),
        tts=create_tts(),
        chat_ctx=chat_ctx,
        fnc_ctx=ElenaFunctionContext(),
        allow_interruptions=True,
        interrupt_min_words=_as_int(get_agent_setting("interrupt_min_words", 2), 2, min_value=1, max_value=10),
        min_endpointing_delay=effective_endpointing_delay,
        # Keep disabled here to avoid race where LLM starts replying before deterministic
        # memory/order flow handlers finish, which can produce double answers.
        preemptive_synthesis=_as_bool(get_agent_setting("preemptive_synthesis", False), default=False),
        before_llm_cb=_before_llm_cb,
    )

    room_log(
        "TURN_CONFIG",
        configured_endpointing_delay=configured_endpointing_delay,
        effective_endpointing_delay=effective_endpointing_delay,
    )


    conversation_transcript: list[str] = []

    def suppress_llm(seconds: float = 10.0):
        """Suppress LLM synthesis for the next N seconds (used when handler replies deterministically)."""
        state.suppress_llm_until = time.time() + seconds
        room_log("LLM_SUPPRESS_SET", seconds=seconds)

    async def send_agent_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        now_ts = time.time()
        # Dedup window of 2.5s (agent.say is direct).
        if cleaned == state.last_agent_transcript_text and (now_ts - state.last_agent_transcript_at) < 2.5:
            room_log("AGENT_TEXT_DEDUPED", text=cleaned)
            return
        state.last_agent_transcript_text = cleaned
        state.last_agent_transcript_at = now_ts
        conversation_transcript.append(f"Agent: {cleaned}")
        await _publish_transcript(ctx, "agent", cleaned)
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="agent")
            transcript_text = "\n".join(conversation_transcript)
            await save_transcript_to_db(call_id, transcript_text, speaker="full", append=False)
        room_log("AGENT_TEXT", text=cleaned)

    _last_user_interim = ""
    _last_user_interim_sent_at = 0.0

    async def send_user_transcript(text: str, *, interim: bool = False):
        nonlocal _last_user_interim, _last_user_interim_sent_at
        # PATCH 4: Clean transcript text by replacing formatted phone numbers with pure digits.
        cleaned = _clean_transcript_phone((text or "").strip())
        if not cleaned:
            return
        now_ts = time.time()
        if interim:
            # Throttle interim updates to keep UI smooth and avoid flooding.
            if cleaned == _last_user_interim and (now_ts - _last_user_interim_sent_at) < 0.35:
                return
            _last_user_interim = cleaned
            _last_user_interim_sent_at = now_ts
            payload = json.dumps(
                {"type": "transcript", "speaker": "user", "text": cleaned, "interim": True},
                ensure_ascii=False,
            )
            try:
                # Use reliable=True to match the original — unreliable drops packets,
                # causing the user's text to not appear until the agent starts speaking.
                await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
            except Exception:
                pass
            return

        if cleaned == state.last_user_transcript_text and (now_ts - state.last_user_transcript_at) < 5.0:
            room_log("USER_TEXT_DEDUPED", text=cleaned)
            return

        state.last_user_transcript_text = cleaned
        state.last_user_transcript_at = now_ts
        _last_user_interim = ""
        conversation_transcript.append(f"User: {cleaned}")
        payload = json.dumps(
            {"type": "transcript", "speaker": "user", "text": cleaned, "interim": False},
            ensure_ascii=False,
        )
        try:
            await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
        except Exception:
            pass
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="user")
            transcript_text = "\n".join(conversation_transcript)
            await save_transcript_to_db(call_id, transcript_text, speaker="full", append=False)
        room_log("USER_TEXT", text=cleaned)

    @agent.on("agent_started_speaking")
    def _on_agent_started_speaking():
        cancel_thinking_task()
        state.waiting_for_user = False
        asyncio.create_task(set_ui_state("speaking"))

    @agent.on("agent_stopped_speaking")
    def _on_agent_stopped_speaking():
        state.last_agent_activity = time.time()
        state.waiting_for_user = True
        asyncio.create_task(set_ui_state("idle"))


    @agent.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        text = msg.content if hasattr(msg, "content") else None
        if text:
            asyncio.create_task(send_agent_transcript(text))

    @agent.on("agent_speech_interrupted")
    def _on_agent_speech_interrupted(msg):
        # Fallback: when user barges in before commit, capture whatever text exists.
        text = msg.content if hasattr(msg, "content") else None
        if text:
            room_log("AGENT_TEXT_INTERRUPTED_CAPTURE", text=_truncate(text))
            asyncio.create_task(send_agent_transcript(text))

    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        cancel_thinking_task()
        state.waiting_for_user = False
        # PATCH 2/3: Immediately snooze silence monitor when user starts speaking
        snooze_silence(5.0)
        asyncio.create_task(set_ui_state("listening"))

    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking():
        schedule_thinking_state()

    @agent.on("user_speech_committed")
    def _on_user_speech_committed(msg):
        user_text = str(getattr(msg, "content", "") or "").strip()
        if not user_text:
            return

        state.last_user_activity = time.time()
        state.silence_prompt_count = 0
        asyncio.create_task(send_user_transcript(user_text))

        # PATCH 5: Early check for digits to suppress LLM immediately.
        # This prevents the LLM from starting "Thanks, got it..." fillers
        # only to be cut off by the deterministic handler.
        all_digits = "".join(_extract_digit_parts(user_text))
        if len(all_digits) >= 3:
            suppress_llm(5.0)
            # agent.interrupt() # Removed in PATCH 5 to avoid interfering with pipeline state in phone flow
            room_log("EARLY_DIGIT_SUPPRESSION", digits=len(all_digits))

        # Early suppression for ticket flow states — prevent LLM from responding
        # while the deterministic state machine handles the turn
        if state.support_state in {"ticket_name", "ticket_email", "ticket_issue", "ticket_confirm"}:
            suppress_llm(10.0)
            agent.interrupt()

        # Early suppression for ticket intent detection — prevent LLM from starting
        # a response that will be interrupted by the ticket escape handler
        if state.support_state not in {"ticket_name", "ticket_email", "ticket_issue", "ticket_confirm", "creating_ticket"}:
            if re.search(r"\b(human|representative|call me|callback|support ticket|open\s*(a\s*)?ticket|create\s*(a\s*)?ticket|raise\s*(a\s*)?ticket|make\s*(a\s*)?ticket|want\s*(a\s*)?ticket|need\s*(a\s*)?ticket|complaint)\b", user_text.lower()):
                suppress_llm(15.0)
                agent.interrupt()

        # PATCH 4: Diagnostic logging
        all_digits = "".join(_extract_digit_parts(user_text))
        room_log("USER_TURN_DEBUG", state=state.support_state, text=user_text, extracted_digits=all_digits)

        # PATCH 4: Detect incomplete numbers (7-9 digits)
        if 7 <= len(all_digits) <= 9 and state.support_state == "awaiting_phone":
            room_log("INCOMPLETE_PHONE_DETECTED", digits=len(all_digits))
            agent.interrupt() # PATCH 4: Kill pending LLM
            suppress_llm(10.0)
            snooze_silence(20.0)
            asyncio.create_task(agent.say(
                f"I heard a {len(all_digits)}-digit number. Could you please provide the full 10-digit phone number?",
                allow_interruptions=True
            ))
            return

        # 0) Farewell detection
        _t = user_text.lower().strip()
        _farewell_intent = bool(re.search(
            r"\b(bye|by\b|goodbye|good bye|good night|see you|take care|"
            r"thanks? bye|that.?s all|no thank|nothing else|"
            r"i.?m done|all good|that will be all|have a good|have a great|"
            r"no more|no further|no other)\b",
            _t
        ))
        if not _farewell_intent:
            _has_thanks = bool(re.search(r"\bthank", _t))
            _has_close = bool(re.search(r"\b(no|okay|ok|alright|all right|done|that.?s it|enough)\b", _t))
            _is_short = len(_t.split()) <= 10
            if _has_thanks and _has_close and _is_short:
                _farewell_intent = True
        if _farewell_intent and len(re.findall(r"\d", user_text)) >= 3:
            _farewell_intent = False
            room_log("FAREWELL_SHIELDED", text=user_text)

        if _farewell_intent:
            room_log("FAREWELL_DETECTED", text=user_text)
            state.silence_enabled = False
            suppress_llm(15.0)
            goodbye_msg = get_closing("en")
            asyncio.create_task(agent.say(goodbye_msg, allow_interruptions=True))
            async def _delayed_end_farewell():
                await asyncio.sleep(2.0)
                state.should_end = True
                state.disconnect_reason = "farewell"
            asyncio.create_task(_delayed_end_farewell())
            return

        # 1) If lookup is in progress, keep caller informed and do not branch.
        if state.lookup_inflight:
            asyncio.create_task(set_ui_state("thinking"))
            snooze_silence(8.0)
            asyncio.create_task(agent.say("I am still checking that now. One moment please.", allow_interruptions=True))
            return

        if state.ticket_inflight:
            asyncio.create_task(set_ui_state("thinking"))
            snooze_silence(8.0)
            asyncio.create_task(agent.say("I am creating your support ticket now. One moment please.", allow_interruptions=True))
            return

        # 1.5) Ticket-creation escape
        _ticket_escape = bool(re.search(
            r"\b(human|representative|call me|callback|support ticket|open\s*(a\s*)?ticket|create\s*(a\s*)?ticket|raise\s*(a\s*)?ticket|make\s*(a\s*)?ticket|want\s*(a\s*)?ticket|need\s*(a\s*)?ticket|complaint)\b",
            user_text.lower()
        ))
        _in_ticket_flow = state.support_state in {
            "ticket_name", "ticket_email",
            "ticket_issue", "ticket_confirm", "creating_ticket"
        }
        if _ticket_escape and not _in_ticket_flow:
            room_log("FLOW_TRANSITION", from_state=state.support_state, to_state="ticket_name", reason="ticket_escape")
            state.support_state = "ticket_name"
            suppress_llm(15.0)
            agent.interrupt()

            async def _say_ticket_greeting():
                await asyncio.sleep(0.3)
                await agent.say(
                    "I can help you with that. First, could you please tell me your full name?",
                    allow_interruptions=True
                )

            asyncio.create_task(_say_ticket_greeting())
            return

        if state.support_state in {"awaiting_order", "checking_order"}:
            # PATCH 3: Check for PHONE number first, as it's more specific (10+ digits).
            # This prevents phone numbers from being misidentified as order IDs.
            phone_candidate = _normalize_phone_for_lookup(user_text)
            if phone_candidate:
                agent.interrupt() # PATCH 4: Kill pending LLM
                state.support_state = "awaiting_phone"
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(20.0) # Longer snooze for lookups
                asyncio.create_task(_run_phone_lookup(agent, phone_candidate))
                return

            # Then check for Order ID.
            order_id = _normalize_order_id_strict(user_text)
            if order_id:
                agent.interrupt() # PATCH 4: Kill pending LLM
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(20.0)
                asyncio.create_task(_run_order_lookup(agent, order_id))
                return

            # If user indicates phone lookup path, move flow to phone collection.
            if _mentions_no_order_number(user_text) or _mentions_phone_lookup_intent(user_text):
                state.support_state = "awaiting_phone"
                room_log("FLOW_TRANSITION", from_state="awaiting_order", to_state="awaiting_phone", reason="no_order_or_phone_intent")
                return

            if _is_order_relevant(user_text):
                prompt = "Whenever you are ready, please share your order number. If you do not have it, say that and I will check by phone number."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return
            return

        # 3) Active phone-support flow
        # GUARD: Skip phone lookup entirely when user is in the ticket creation flow
        _in_ticket_creation_flow = state.support_state in {
            "ticket_name", "ticket_email", "ticket_issue", "ticket_confirm", "creating_ticket"
        }
        if not _in_ticket_creation_flow and (
            state.support_state in {"awaiting_phone", "checking_phone"} or len(all_digits) >= 10
        ):
            # Check for Order ID first as an escape path, even in phone flow.
            order_id_escape = _normalize_order_id_strict(user_text)
            if order_id_escape:
                room_log("ORDER_ESCAPE_MATCH", order_id=order_id_escape)
                agent.interrupt()
                state.support_state = "awaiting_order"
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(20.0)
                asyncio.create_task(_run_order_lookup(agent, order_id_escape))
                return

            # Then check for phone number digits.
            phone = _normalize_phone_for_lookup(user_text)
            if phone:
                room_log("PHONE_MATCH_FOUND", phone=phone, state=state.support_state)
                agent.interrupt() # PATCH 4: Kill pending LLM
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(20.0)
                asyncio.create_task(_run_phone_lookup(agent, phone))
                return

            # PATCH 4: Detect intent to switch back to order number lookup
            if _mentions_order_lookup_intent(user_text):
                state.support_state = "awaiting_order"
                room_log("FLOW_TRANSITION", from_state="awaiting_phone", to_state="awaiting_order", reason="order_id_intent_given")
                # Do NOT suppress LLM; let it provide the natural transition message.
                return

            if _mentions_phone_lookup_intent(user_text):
                prompt = "Sure. Please provide the full phone number used for the order."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return

            if _is_order_relevant(user_text):
                prompt = "I need the full phone number to check the order. Please share it once."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return
            return

        # 3b) Support ticket flow
        if state.support_state == "ticket_name":
            state.ticket_name = user_text
            state.support_state = "ticket_email"
            suppress_llm()
            agent.interrupt()
            asyncio.create_task(agent.say("Thanks. Now please share your email address.", allow_interruptions=True))
            return

        if state.support_state == "ticket_email":
            # Try to extract email from surrounding text (e.g. "My email is bhavya@gmail.com")
            email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", user_text)
            if not email_match:
                suppress_llm()
                agent.interrupt()
                asyncio.create_task(agent.say("Please share a valid email address.", allow_interruptions=True))
                return
            state.ticket_email = email_match.group(0).lower().strip()
            state.support_state = "ticket_issue"
            suppress_llm()
            agent.interrupt()
            asyncio.create_task(agent.say("Please describe the issue in one or two sentences.", allow_interruptions=True))
            return

        if state.support_state == "ticket_issue":
            state.ticket_issue = user_text
            state.support_state = "ticket_confirm"
            confirm_text = (
                f"I have your details as name {state.ticket_name} and email {state.ticket_email}. "
                "Should I create the support ticket now?"
            )
            suppress_llm()
            agent.interrupt()
            asyncio.create_task(agent.say(confirm_text, allow_interruptions=True))
            return

        if state.support_state == "ticket_confirm":
            if _is_yes(user_text):
                suppress_llm()
                agent.interrupt()
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(10.0)
                asyncio.create_task(agent.say("Thanks. Creating your support ticket now.", allow_interruptions=True))
                asyncio.create_task(_run_create_ticket(agent))
                return
            if _is_no(user_text):
                state.support_state = "idle"
                state.ticket_name = ""
                state.ticket_email = ""
                state.ticket_issue = ""
                suppress_llm()
                agent.interrupt()
                asyncio.create_task(agent.say("No problem. I have cancelled the ticket request.", allow_interruptions=True))
                return
            suppress_llm()
            agent.interrupt()
            asyncio.create_task(agent.say("Please say yes to create the ticket, or no to cancel.", allow_interruptions=True))
            return

        # 4) Detect support intent from any general turn.
        support_intent = bool(re.search(r"(problem|issue|complaint|order problem|wrong order|late order|my order)", user_text.lower()))
        if support_intent:
            state.support_state = "awaiting_order"
            room_log("FLOW_TRANSITION", from_state="idle", to_state="awaiting_order", reason="support_intent")
            
            # PATCH 3: Check for PHONE first here too.
            phone = _normalize_phone_for_lookup(user_text)
            if phone:
                agent.interrupt() # PATCH 4: Kill pending LLM
                state.support_state = "awaiting_phone"
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(20.0)
                asyncio.create_task(_run_phone_lookup(agent, phone))
                return

            order_id = _normalize_order_id_strict(user_text)
            if order_id:
                agent.interrupt() # PATCH 4: Kill pending LLM
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(20.0)
                asyncio.create_task(_run_order_lookup(agent, order_id))
                return

        ticket_intent = bool(re.search(r"(human|representative|call me|callback|support ticket|open\s*(a\s*)?ticket|create\s*(a\s*)?ticket|raise\s*(a\s*)?ticket|make\s*(a\s*)?ticket|want\s*(a\s*)?ticket|need\s*(a\s*)?ticket|complaint)", user_text.lower()))
        if ticket_intent:
            state.support_state = "ticket_name"
            suppress_llm(15.0)
            agent.interrupt()

            async def _say_ticket_greeting2():
                await asyncio.sleep(0.3)
                await agent.say("I can help you with that. First, could you please tell me your full name?", allow_interruptions=True)

            asyncio.create_task(_say_ticket_greeting2())
            return

    # Participant disconnect
    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(participant_info):
        if participant_info.identity == participant.identity:
            state.should_end = True
            state.disconnect_reason = "participant_disconnected"

    # Start agent
    agent.start(ctx.room, participant)
    # PATCH 8: Start the background cache auto-refresh loop (fires every 60s, refreshes on 5-min TTL expiry)
    cache_refresh_task = asyncio.create_task(_cache_auto_refresh_loop())

    human_input = getattr(agent, "human_input", None) or getattr(agent, "_human_input", None)
    if human_input:
        @human_input.on("interim_transcript")
        def _on_interim_transcript(ev):
            try:
                text = ev.alternatives[0].text if ev.alternatives else None
            except Exception:
                text = None
            if text:
                cancel_thinking_task()
                asyncio.create_task(send_user_transcript(text, interim=True))

    # Greet
    greeting_enabled = _as_bool(get_agent_setting("agent_greeting_enabled", True), default=True)
    if greeting_enabled:
        greeting = get_greeting("en")
        chat_ctx.append(role="assistant", text=greeting)
        await agent.say(greeting, allow_interruptions=True)
    await set_ui_state("idle")

    bg_audio_player = None
    async def _start_background_audio():
        nonlocal bg_audio_player
        try:
            from src.services.background_audio import create_background_audio_player
            bg_audio_player = await create_background_audio_player()
            if bg_audio_player:
                await bg_audio_player.start(ctx.room)
        except Exception:
            pass
    asyncio.create_task(_start_background_audio())

    now = time.time()
    state.last_user_activity = now
    state.last_agent_activity = now
    asyncio.create_task(order_lookup.prefetch_orders())

    def _contextual_silence_prompt() -> str:
        phase = state.silence_prompt_count
        support_state = state.support_state
        if support_state in {"awaiting_order", "checking_order"}:
            if phase == 0:
                return "Whenever you are ready, please share your order number. If you do not have it, say that and I will check by phone number."
            return "I am still here. Please share your order number, or say you do not have it so I can check by phone."
        if support_state in {"awaiting_phone", "checking_phone"}:
            if phase == 0:
                return "Please share the phone number used for the order when you are ready."
            return "I am ready whenever you are. Please repeat the full phone number."
        if support_state == "ticket_name":
            return "Whenever you are ready, please tell me your full name so I can create the support ticket."
        if support_state == "ticket_email":
            return "Please share your email address when you are ready."
        if support_state == "ticket_issue":
            return "Please describe the issue in one or two sentences when you are ready."
        if support_state == "ticket_confirm":
            return "Please say yes to create the ticket, or no to cancel."
        if phase == 0:
            return "I am here whenever you are ready."
        return "I can continue whenever you are ready."

    async def _silence_monitor():
        room_log("SILENCE_MONITOR_START")
        while not state.should_end:
            await asyncio.sleep(1.0)
            if not state.silence_enabled or not state.waiting_for_user:
                continue
            
            now = time.time()
            # PATCH 4: Detailed debug logging for silence monitor (throttled)
            if int(now) % 15 == 0:
                room_log("SILENCE_DEBUG", 
                         now=now, 
                         user_idle=now - state.last_user_activity, 
                         agent_idle=now - state.last_agent_activity,
                         snooze_until=state.silence_snooze_until,
                         lookup_inflight=state.lookup_inflight,
                         ui_state=state.ui_state)

            if state.lookup_inflight:
                continue
            if now < state.silence_snooze_until:
                continue
            if (now - state.last_user_activity) < state.silence_timeout_s:
                continue
            if (now - state.last_agent_activity) < state.silence_timeout_s:
                continue
            
            # PATCH 4: Allow silence prompt even if 'thinking' to prevent dead states
            if state.ui_state in {"speaking", "listening"}:
                continue
                
            if state.silence_prompt_count >= state.silence_max_prompts:
                room_log("SILENCE_MAX_REACHED", count=state.silence_prompt_count)
                state.should_end = True
                break
                
            text = "I am still here. Please share your order number or phone number."
            if state.support_state == "awaiting_phone":
                text = "I'm still ready to help. Please share the phone number for the order whenever you can."
            
            room_log("SILENCE_PROMPT_TRIGGERED", text=text, count=state.silence_prompt_count)
            state.silence_prompt_count += 1
            state.silence_snooze_until = time.time() + 15.0
            suppress_llm(15.0)
            await agent.say(text, allow_interruptions=True)

    silence_task = asyncio.create_task(_silence_monitor())

    while not state.should_end:
        await asyncio.sleep(0.5)
    await asyncio.sleep(3.0)
    silence_task.cancel()
    cache_refresh_task.cancel()  # PATCH 8: Stop the background cache refresh loop
    transcript_text = "\n".join(conversation_transcript)
    await end_call_in_db(
        call_id=call_id,
        room_name=ctx.room.name,
        status="completed",
        disconnect_reason=state.disconnect_reason,
        transcript=transcript_text or None,
    )
    try:
        if bg_audio_player:
            await bg_audio_player.stop()
        if ctx.room and ctx.room.isconnected():
            await ctx.room.disconnect()
    except Exception:
        pass


async def request_fnc(req: JobRequest) -> None:
    """Determine if the English agent should accept this job request based on DB language setting."""
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        settings_dict = await db.get_all_settings()
        lang = settings_dict.get("agent_language", "en")
        logger.info("English agent: request received. Active DB language: %s", lang)
        if lang == "en":
            logger.info("English agent: accepting job request")
            await req.accept()
        else:
            logger.info("English agent: rejecting job request because active language is Greek (%s)", lang)
            await req.reject()
    except Exception as e:
        logger.warning("English agent: failed to check language in request_fnc, accepting anyway: %s", e)
        try:
            await req.accept()
        except Exception:
            pass


def prewarm(proc: JobProcess):
    """Prewarm the English agent process (lightweight to prevent connection pool exhaustion)."""
    logger.info("Prewarm: English Elena ready (lightweight)")


def run_agent():
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            request_fnc=request_fnc,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )

if __name__ == "__main__":
    run_agent();
