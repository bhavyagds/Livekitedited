"""
Meallion Voice AI - Elena English Agent (clean rewrite)
English-only voice agent with deterministic order/phone support flow.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Optional

from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import elevenlabs, openai, silero

try:
    from livekit.plugins import deepgram
    USE_DEEPGRAM = True
except ImportError:
    USE_DEEPGRAM = False

from src.config import settings
from src.agents.prompts import (
    _fetch_from_db,
    get_agent_setting,
    get_closing,
    get_greeting,
    get_system_prompt_async,
    set_runtime_language,
)
from src.agents.tools import knowledge_base, order_lookup, support_ticket

logger = logging.getLogger(__name__)


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
    if min_d <= len(joined) <= max_d:
        return joined
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


def _mentions_no_order_number(text: str) -> bool:
    t = (text or "").lower()
    return bool(
        re.search(r"(i do not have|i don't have|no order number|dont have order|don't have order)", t)
        or re.search(r"(didn't get|did not get|didn't receive|did not receive).*(email|confirmation)", t)
    )


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
    support_state: str = "idle"  # idle|awaiting_order|checking_order|awaiting_phone|checking_phone|ticket_name|ticket_phone|ticket_email|ticket_issue|ticket_confirm|creating_ticket
    ui_state: str = "idle"  # idle|listening|thinking|speaking
    last_issue: str = ""
    last_order_number: str = ""
    last_phone_number: str = ""
    lookup_inflight: bool = False
    ticket_inflight: bool = False
    ticket_name: str = ""
    ticket_phone: str = ""
    ticket_email: str = ""
    ticket_issue: str = ""
    should_end: bool = False
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


async def save_transcript_to_db(call_id: str, text: str, speaker: str = "agent") -> bool:
    if not call_id or not text:
        return False
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        return await db.update_call_transcript(call_id=call_id, transcript=f"{speaker.capitalize()}: {text}", append=True)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class ElenaFunctionContext(llm.FunctionContext):
    @llm.ai_callable()
    async def lookup_order(self, order_number: Annotated[str, llm.TypeInfo(description="Order number")]) -> str:
        """Look up an order by order number and return a customer-facing summary."""
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
    async def lookup_order_by_phone(self, phone: Annotated[str, llm.TypeInfo(description="Phone number")]) -> str:
        """Look up the most relevant order using the customer's phone number."""
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
        customer_phone: Annotated[str, llm.TypeInfo(description="Customer phone")],
        customer_email: Annotated[str, llm.TypeInfo(description="Customer email")],
        issue_description: Annotated[str, llm.TypeInfo(description="Issue")],
    ) -> str:
        """Create a support ticket when an issue cannot be resolved during the call."""
        room_log("TOOL_CALL", name="create_support_ticket")
        result = await support_ticket.create_support_ticket(
            customer_name,
            customer_phone,
            customer_email,
            issue_description,
        )
        room_log("TOOL_RESULT", name="create_support_ticket", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def end_session(self) -> str:
        """End the current voice session gracefully with a closing message."""
        _current["state"].should_end = True
        msg = get_closing("en")
        room_log("SESSION_END_REQUESTED", message=msg)
        return msg


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
        model = str(get_agent_setting("deepgram_stt_model", "nova-2") or "nova-2")
        return deepgram.STT(model=model, language="en-US", api_key=deepgram_api_key)
    model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1")
    return openai.STT(model=model, api_key=settings.openai_api_key, language="en")


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
        + "\n".join(lines)
    )


def create_vad(is_sip_call: bool = False):
    min_speech = _as_float(get_agent_setting("vad_min_speech_duration", 0.15), 0.15, min_value=0.05, max_value=1.0)
    min_silence = _as_float(get_agent_setting("vad_min_silence_duration", 1.2 if is_sip_call else 1.0), 1.0, min_value=0.1, max_value=3.0)
    return silero.VAD.load(min_speech_duration=min_speech, min_silence_duration=min_silence)


# -----------------------------------------------------------------------------
# Core flow helpers
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
        result = await order_lookup.lookup_order(order_number)
        room_log("ORDER_LOOKUP_RESULT", result=_truncate(result))
        await agent.say(result, allow_interruptions=True)
        state.last_order_number = order_number
        if _is_order_not_found_text(result):
            state.support_state = "awaiting_order"
        else:
            state.support_state = "idle"
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
        result = await order_lookup.lookup_order_by_phone(phone_number)
        room_log("PHONE_LOOKUP_RESULT", result=_truncate(result))
        await agent.say(result, allow_interruptions=True)
        state.last_phone_number = phone_number
        if "no order" in (result or "").lower() or "could not" in (result or "").lower():
            state.support_state = "awaiting_phone"
        else:
            state.support_state = "idle"
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
        result = await support_ticket.create_support_ticket(
            state.ticket_name or "Customer",
            state.ticket_phone,
            state.ticket_email,
            state.ticket_issue,
        )
        room_log("TICKET_CREATE_RESULT", result=_truncate(result))
        await agent.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.ticket_name = ""
        state.ticket_phone = ""
        state.ticket_email = ""
        state.ticket_issue = ""
    finally:
        state.ticket_inflight = False


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    set_runtime_language("en")
    # Warm cache in background; do not block first response path on DB latency.
    cache_task = asyncio.create_task(_fetch_from_db())

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

    def snooze_silence(seconds: float):
        now_ts = time.time()
        state.silence_snooze_until = max(state.silence_snooze_until, now_ts + max(0.0, seconds))
        room_log("SILENCE_SNOOZE", until=state.silence_snooze_until, seconds=seconds)

    job = getattr(ctx, "job", None)
    job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
    room_logger, room_log_path = _create_room_logger(ctx.room.name, job_id)
    _current["room_logger"] = room_logger
    _current["room_name"] = ctx.room.name
    _current["job_id"] = job_id
    room_log("ROOM_START", call_type="web")
    logger.info("Per-room log: %s", room_log_path)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity)

    call_id = await record_call_to_db(ctx.room.name, call_type="web", caller_identity=participant.identity)
    _current["call_id"] = call_id

    try:
        await asyncio.wait_for(cache_task, timeout=8.0)
    except Exception as e:
        logger.warning("Initial cache warmup did not complete in time: %s", e)

    try:
        from src.services.database import get_database_service
        db = get_database_service()
        memory_items = await db.get_memory_items(active_only=True)
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

    agent = VoicePipelineAgent(
        vad=create_vad(),
        stt=create_stt(),
        llm=create_llm(),
        tts=create_tts(),
        chat_ctx=chat_ctx,
        fnc_ctx=ElenaFunctionContext(),
        allow_interruptions=True,
        interrupt_min_words=_as_int(get_agent_setting("interrupt_min_words", 2), 2, min_value=1, max_value=10),
        min_endpointing_delay=_as_float(get_agent_setting("min_endpointing_delay", 1.2), 1.2, min_value=0.2, max_value=3.0),
        # Keep disabled here to avoid race where LLM starts replying before deterministic
        # memory/order flow handlers finish, which can produce double answers.
        preemptive_synthesis=_as_bool(get_agent_setting("preemptive_synthesis", False), default=False),
    )

    conversation_transcript: list[str] = []

    async def send_agent_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        now_ts = time.time()
        if cleaned == state.last_agent_transcript_text and (now_ts - state.last_agent_transcript_at) < 2.5:
            room_log("AGENT_TEXT_DEDUPED", text=cleaned)
            return
        state.last_agent_transcript_text = cleaned
        state.last_agent_transcript_at = now_ts
        conversation_transcript.append(f"Agent: {cleaned}")
        await _publish_transcript(ctx, "agent", cleaned)
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="agent")
        room_log("AGENT_TEXT", text=cleaned)

    async def send_user_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        now_ts = time.time()
        if cleaned == state.last_user_transcript_text and (now_ts - state.last_user_transcript_at) < 1.2:
            room_log("USER_TEXT_DEDUPED", text=cleaned)
            return
        state.last_user_transcript_text = cleaned
        state.last_user_transcript_at = now_ts
        conversation_transcript.append(f"User: {cleaned}")
        await _publish_transcript(ctx, "user", cleaned)
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="user")
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

        # 2) Active order-support flow
        if state.support_state in {"awaiting_order", "checking_order"}:
            order_id = _normalize_order_id_strict(user_text)
            if order_id:
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(10.0)
                asyncio.create_task(_run_order_lookup(agent, order_id))
                return

            if _mentions_no_order_number(user_text):
                state.support_state = "awaiting_phone"
                return
            return

        # 3) Active phone-support flow
        if state.support_state in {"awaiting_phone", "checking_phone"}:
            phone = _normalize_phone_for_lookup(user_text)
            if phone:
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(10.0)
                asyncio.create_task(_run_phone_lookup(agent, phone))
                return
            return

        # 3b) Support ticket flow
        if state.support_state == "ticket_name":
            state.ticket_name = user_text
            state.support_state = "ticket_phone"
            asyncio.create_task(agent.say("Thanks. Please share your phone number digit by digit.", allow_interruptions=True))
            return

        if state.support_state == "ticket_phone":
            ticket_phone = _normalize_phone_for_lookup(user_text)
            if not ticket_phone:
                asyncio.create_task(agent.say("Please repeat a valid phone number, digit by digit.", allow_interruptions=True))
                return
            state.ticket_phone = ticket_phone
            state.support_state = "ticket_email"
            asyncio.create_task(agent.say("Got it. Now please share your email address.", allow_interruptions=True))
            return

        if state.support_state == "ticket_email":
            if not _looks_like_email(user_text):
                asyncio.create_task(agent.say("Please share a valid email address.", allow_interruptions=True))
                return
            state.ticket_email = user_text.strip()
            state.support_state = "ticket_issue"
            asyncio.create_task(agent.say("Please describe the issue in one or two sentences.", allow_interruptions=True))
            return

        if state.support_state == "ticket_issue":
            state.ticket_issue = user_text
            state.support_state = "ticket_confirm"
            confirm_text = (
                f"I have your details as name {state.ticket_name}, phone {state.ticket_phone}, and email {state.ticket_email}. "
                "Should I create the support ticket now?"
            )
            asyncio.create_task(agent.say(confirm_text, allow_interruptions=True))
            return

        if state.support_state == "ticket_confirm":
            if _is_yes(user_text):
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(10.0)
                asyncio.create_task(agent.say("Thanks. Creating your support ticket now.", allow_interruptions=True))
                asyncio.create_task(_run_create_ticket(agent))
                return
            if _is_no(user_text):
                state.support_state = "idle"
                state.ticket_name = ""
                state.ticket_phone = ""
                state.ticket_email = ""
                state.ticket_issue = ""
                asyncio.create_task(agent.say("No problem. I have cancelled the ticket request.", allow_interruptions=True))
                return
            asyncio.create_task(agent.say("Please say yes to create the ticket, or no to cancel.", allow_interruptions=True))
            return

        # 3c) Memory is handled through system prompt context (single-response path).

        # 4) Detect support intent from any general turn.
        support_intent = bool(re.search(r"(problem|issue|complaint|order problem|wrong order|late order|my order)", user_text.lower()))
        if support_intent:
            state.support_state = "awaiting_order"
            state.last_issue = user_text
            asyncio.create_task(agent.say("I am sorry to hear that. Please provide your order number.", allow_interruptions=True))
            return

        ticket_intent = bool(re.search(r"(human|representative|call me|callback|support ticket|open ticket|create ticket)", user_text.lower()))
        if ticket_intent:
            state.support_state = "ticket_name"
            asyncio.create_task(agent.say("Sure, I can create a support ticket. Please tell me your full name.", allow_interruptions=True))
            return

        # 5) Otherwise let LLM handle general query naturally.

    # Participant disconnect
    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(participant_info):
        if participant_info.identity == participant.identity:
            state.should_end = True

    # Start agent
    agent.start(ctx.room, participant)

    # Greet
    greeting_enabled = _as_bool(get_agent_setting("agent_greeting_enabled", True), default=True)
    if greeting_enabled:
        greeting = get_greeting("en")
        chat_ctx.append(role="assistant", text=greeting)
        await agent.say(greeting, allow_interruptions=True)
    await set_ui_state("idle")

    # Prevent immediate silence prompt right after startup/greeting delays.
    now = time.time()
    state.last_user_activity = now
    state.last_agent_activity = now

    # Prefetch orders in background
    asyncio.create_task(order_lookup.prefetch_orders())

    # Simple silence monitor
    async def _silence_monitor():
        prompts = [
            "I am here when you are ready.",
            "Take your time. I am still here.",
            "I can continue whenever you are ready.",
        ]
        while not state.should_end:
            await asyncio.sleep(1.0)
            if not state.silence_enabled or not state.waiting_for_user or state.lookup_inflight:
                continue
            now = time.time()
            if now < state.silence_snooze_until:
                continue
            if (now - state.last_user_activity) < state.silence_timeout_s:
                continue
            if (now - state.last_agent_activity) < state.silence_timeout_s:
                continue

            if state.silence_prompt_count >= state.silence_max_prompts:
                # Hard stop to prevent repeated prompt loops.
                continue

            idx = min(state.silence_prompt_count, len(prompts) - 1)
            text = prompts[idx]
            state.silence_prompt_count += 1
            room_log("SILENCE_PROMPT", count=state.silence_prompt_count, text=text)
            await agent.say(text, allow_interruptions=True)

    silence_task = asyncio.create_task(_silence_monitor())

    # Wait until session ends
    while not state.should_end:
        await asyncio.sleep(0.5)

    # Cleanup and call end
    silence_task.cancel()
    transcript_text = "\n".join(conversation_transcript)
    await end_call_in_db(
        call_id=call_id,
        room_name=ctx.room.name,
        status="completed",
        duration_seconds=None,
        disconnect_reason="session_end",
        transcript=transcript_text or None,
    )
    room_log("CALL_END", transcript_lines=len(conversation_transcript))

    try:
        if ctx.room and ctx.room.isconnected():
            await ctx.room.disconnect()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Worker boot
# -----------------------------------------------------------------------------


def prewarm(proc: JobProcess):
    logger.info("Elena EN prewarm complete")


def run_agent():
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    init_timeout = _as_float(os.getenv("LIVEKIT_AGENTS_INITIALIZE_TIMEOUT", "60"), 60.0, min_value=5.0, max_value=300.0)
    shutdown_timeout = _as_float(os.getenv("LIVEKIT_AGENTS_SHUTDOWN_TIMEOUT", "60"), 60.0, min_value=10.0, max_value=300.0)
    idle_procs = _as_int(os.getenv("LIVEKIT_AGENTS_NUM_IDLE_PROCESSES", "1"), 1, min_value=0, max_value=8)
    load_threshold = _as_float(os.getenv("LIVEKIT_AGENTS_LOAD_THRESHOLD", "0.9"), 0.9, min_value=0.1, max_value=1.0)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
            initialize_process_timeout=init_timeout,
            shutdown_process_timeout=shutdown_timeout,
            num_idle_processes=idle_procs,
            load_threshold=load_threshold,
        )
    )


if __name__ == "__main__":
    run_agent()
