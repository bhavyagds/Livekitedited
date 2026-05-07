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
    support_state: str = "idle"  # idle|awaiting_order|checking_order|awaiting_phone|checking_phone
    last_issue: str = ""
    last_order_number: str = ""
    last_phone_number: str = ""
    lookup_inflight: bool = False
    should_end: bool = False
    silence_enabled: bool = True
    silence_timeout_s: float = 12.0
    silence_max_prompts: int = 2
    silence_prompt_count: int = 0
    waiting_for_user: bool = False
    last_user_activity: float = 0.0
    last_agent_activity: float = 0.0


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
        room_log("TOOL_CALL", name="lookup_order", order_number=order_number)
        result = await order_lookup.lookup_order(order_number)
        room_log("TOOL_RESULT", name="lookup_order", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def get_order_details(self, order_number: Annotated[str, llm.TypeInfo(description="Order number or last")] = "last") -> str:
        room_log("TOOL_CALL", name="get_order_details", order_number=order_number)
        result = await order_lookup.get_order_details(order_number)
        room_log("TOOL_RESULT", name="get_order_details", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def lookup_order_by_phone(self, phone: Annotated[str, llm.TypeInfo(description="Phone number")]) -> str:
        room_log("TOOL_CALL", name="lookup_order_by_phone", phone=phone)
        result = await order_lookup.lookup_order_by_phone(phone)
        room_log("TOOL_RESULT", name="lookup_order_by_phone", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def search_knowledge_base(self, query: Annotated[str, llm.TypeInfo(description="Question to search")]) -> str:
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
    if provider == "deepgram" and USE_DEEPGRAM and settings.deepgram_api_key:
        model = str(get_agent_setting("deepgram_stt_model", "nova-2") or "nova-2")
        return deepgram.STT(model=model, language="en-US", api_key=settings.deepgram_api_key)
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

    # ElevenLabs default
    voice_id = str(get_agent_setting("agent_voice_id", settings.elevenlabs_voice_id) or settings.elevenlabs_voice_id)
    similarity = _as_float(get_agent_setting("agent_voice_similarity", settings.elevenlabs_voice_similarity), 0.9, min_value=0.0, max_value=1.0)
    stability = _as_float(get_agent_setting("agent_voice_stability", settings.elevenlabs_voice_stability), 0.65, min_value=0.0, max_value=1.0)

    return elevenlabs.TTS(
        api_key=settings.elevenlabs_api_key,
        voice_id=voice_id,
        model=str(get_agent_setting("elevenlabs_model", settings.elevenlabs_model or "eleven_turbo_v2_5") or "eleven_turbo_v2_5"),
        voice_settings=elevenlabs.VoiceSettings(stability=stability, similarity_boost=similarity),
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
        state.last_agent_activity = time.time()
        state.waiting_for_user = True


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
        state.last_agent_activity = time.time()
        state.waiting_for_user = True


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    set_runtime_language("en")
    await _fetch_from_db()

    state = SessionState(
        silence_timeout_s=_as_float(get_agent_setting("silence_timeout_seconds", 12.0), 12.0, min_value=6.0, max_value=60.0),
        silence_max_prompts=_as_int(get_agent_setting("silence_max_prompts", 2), 2, min_value=1, max_value=5),
        last_user_activity=time.time(),
        last_agent_activity=time.time(),
    )
    _current["state"] = state

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

    system_prompt = await get_system_prompt_async("en")
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
        preemptive_synthesis=_as_bool(get_agent_setting("preemptive_synthesis", True), default=True),
    )

    conversation_transcript: list[str] = []

    async def send_agent_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        conversation_transcript.append(f"Agent: {cleaned}")
        await _publish_transcript(ctx, "agent", cleaned)
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="agent")
        room_log("AGENT_TEXT", text=cleaned)

    async def send_user_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        conversation_transcript.append(f"User: {cleaned}")
        await _publish_transcript(ctx, "user", cleaned)
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="user")
        room_log("USER_TEXT", text=cleaned)

    @agent.on("agent_started_speaking")
    def _on_agent_started_speaking():
        state.waiting_for_user = False

    @agent.on("agent_stopped_speaking")
    def _on_agent_stopped_speaking():
        state.last_agent_activity = time.time()
        state.waiting_for_user = True

    @agent.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        text = msg.content if hasattr(msg, "content") else None
        if text:
            asyncio.create_task(send_agent_transcript(text))

    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        state.waiting_for_user = False

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
            asyncio.create_task(agent.say("I am still checking that now. One moment please.", allow_interruptions=True))
            return

        # 2) Active order-support flow
        if state.support_state in {"awaiting_order", "checking_order"}:
            order_id = _normalize_order_id_strict(user_text)
            if order_id:
                asyncio.create_task(agent.say("Thanks, got it. Give me a moment while I check the details.", allow_interruptions=True))
                asyncio.create_task(_run_order_lookup(agent, order_id))
                return

            if _mentions_no_order_number(user_text):
                state.support_state = "awaiting_phone"
                asyncio.create_task(agent.say("No problem. Please give me the phone number used for the order, digit by digit.", allow_interruptions=True))
                return

            asyncio.create_task(
                agent.say(
                    "I understand. If you have the order number, please share it digit by digit. "
                    "If you do not have it, say that and I will check by phone number.",
                    allow_interruptions=True,
                )
            )
            return

        # 3) Active phone-support flow
        if state.support_state in {"awaiting_phone", "checking_phone"}:
            phone = _normalize_phone_for_lookup(user_text)
            if phone:
                asyncio.create_task(agent.say("Thanks. Let me check that phone number now.", allow_interruptions=True))
                asyncio.create_task(_run_phone_lookup(agent, phone))
                return

            asyncio.create_task(agent.say("Please repeat the full phone number, digit by digit.", allow_interruptions=True))
            return

        # 4) Detect support intent from any general turn.
        support_intent = bool(re.search(r"(problem|issue|complaint|order problem|wrong order|late order|my order)", user_text.lower()))
        if support_intent:
            state.support_state = "awaiting_order"
            state.last_issue = user_text
            asyncio.create_task(agent.say("I am sorry to hear that. Please provide your order number.", allow_interruptions=True))
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

    # Prefetch orders in background
    asyncio.create_task(order_lookup.prefetch_orders())

    # Simple silence monitor
    async def _silence_monitor():
        prompts = [
            "I am here when you are ready.",
            "Take your time. I am still here.",
            "I will end the call for now. You can call us again anytime.",
        ]
        while not state.should_end:
            await asyncio.sleep(1.0)
            if not state.silence_enabled or not state.waiting_for_user or state.lookup_inflight:
                continue
            now = time.time()
            if (now - state.last_user_activity) < state.silence_timeout_s:
                continue
            if (now - state.last_agent_activity) < state.silence_timeout_s:
                continue

            idx = min(state.silence_prompt_count, len(prompts) - 1)
            text = prompts[idx]
            state.silence_prompt_count += 1
            await agent.say(text, allow_interruptions=True)

            if state.silence_prompt_count > state.silence_max_prompts:
                state.should_end = True
                break

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
