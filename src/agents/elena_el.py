"""
Meallion Voice AI - Elena Greek Agent (clean rewrite)
Greek-only voice agent with deterministic order/phone support flow.
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
from src.agents.flows import base, termination, farewell, ticket_flow, order_flow, phone_flow, greeting_flow, silence_flow, memory_flow
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
    last_user_text_norm: str = ""
    user_repetition_count: int = 0
    deterministic_replied: bool = False


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


async def _publish_state(ctx: JobContext, state_name: str):
    try:
        payload = json.dumps({"type": "state", "state": state_name})
        await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
    except Exception:
        pass


async def _publish_transcript(ctx: JobContext, speaker: str, text: str):
    try:
        payload = json.dumps({"type": "transcript", "speaker": speaker, "text": text, "interim": False}, ensure_ascii=False)
        await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
    except Exception:
        pass


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
        result = await knowledge_base.search_knowledge_base(query, language="el")
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

        # Sync deterministic state
        state = _current["state"]
        state.support_state = "idle"
        state.ticket_name = ""
        state.ticket_phone = ""
        state.ticket_email = ""
        state.ticket_issue = ""

        return result


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
        model = str(get_agent_setting("deepgram_stt_model", "nova-3") or "nova-3").strip()
        base_cfg = {
            "model": model,
            "language": "el",
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }
        try:
            return deepgram.STT(api_key=deepgram_api_key, **base_cfg)
        except Exception:
            return deepgram.STT(**base_cfg)

    openai_model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1")
    return openai.STT(model=openai_model, api_key=settings.openai_api_key, language="el")


def create_tts():
    provider = str(get_agent_setting("tts_provider", "elevenlabs") or "elevenlabs").lower()
    if provider == "openai":
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            api_key=settings.openai_api_key,
        )

    eleven_api_key = getattr(settings, "elevenlabs_api_key", None)
    voice_id = str(get_agent_setting("agent_voice_id", getattr(settings, "elevenlabs_voice_id", "")) or "")
    model = str(get_agent_setting("elevenlabs_model", "eleven_turbo_v2_5") or "eleven_turbo_v2_5")

    try:
        return elevenlabs.TTS(
            voice=elevenlabs.Voice(id=voice_id, settings=elevenlabs.VoiceSettings(stability=0.65, similarity_boost=0.9)),
            model=model,
        )
    except Exception:
        return openai.TTS(api_key=settings.openai_api_key)


def create_vad(is_sip_call: bool = False):
    min_speech = _as_float(get_agent_setting("vad_min_speech_duration", 0.15), 0.15)
    min_silence = _as_float(get_agent_setting("vad_min_silence_duration", 1.0), 1.0)
    return silero.VAD.load(min_speech_duration=min_speech, min_silence_duration=min_silence, activation_threshold=0.6)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    set_runtime_language("el")
    cache_task = asyncio.create_task(_fetch_from_db())

    state = SessionState(
        silence_timeout_s=_as_float(get_agent_setting("silence_timeout_seconds", 5.0), 5.0),
        silence_max_prompts=_as_int(get_agent_setting("silence_max_prompts", 0), 0),
        last_user_activity=time.time(),
        last_agent_activity=time.time(),
    )
    _current["state"] = state

    async def set_ui_state(new_state: str):
        if state.ui_state == new_state: return
        state.ui_state = new_state
        room_log("UI_STATE", state=new_state)
        await _publish_state(ctx, new_state)

    thinking_task: Optional[asyncio.Task] = None

    def cancel_thinking_task():
        nonlocal thinking_task
        if thinking_task:
            thinking_task.cancel()
            thinking_task = None

    async def _thinking_logic():
        await asyncio.sleep(1.0)
        await set_ui_state("thinking")

    def schedule_thinking_state():
        nonlocal thinking_task
        cancel_thinking_task()
        thinking_task = asyncio.create_task(_thinking_logic())

    def snooze_silence(seconds: float = 15.0):
        state.silence_snooze_until = time.time() + seconds
        room_log("SILENCE_SNOOZE", seconds=seconds)

    def _should_suppress_clarification(prompt: str, min_interval: float = 5.0) -> bool:
        now = time.time()
        if prompt == state.last_clarification_prompt_text and (now - state.last_clarification_prompt_at) < min_interval:
            return True
        state.last_clarification_prompt_text = prompt
        state.last_clarification_prompt_at = now
        return False

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    _current["room_name"] = ctx.room.name
    _current["job_id"] = ctx.job.id
    _current["room_logger"] = logger

    room_log("ROOM_START", room=ctx.room.name, job_id=ctx.job.id, call_type="web")
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity)

    call_id = await record_call_to_db(ctx.room.name, caller_identity=participant.identity)
    _current["call_id"] = call_id

    # Memory and Prompt
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        memory_items = await db.get_memory_items(active_only=True)
    except Exception:
        memory_items = []
    
    system_prompt = await get_system_prompt_async("el")
    memory_block = memory_flow.build_memory_prompt_block(memory_items)
    if memory_block: system_prompt = f"{memory_block}\n\n{system_prompt}"
    
    chat_ctx = llm.ChatContext()
    chat_ctx.append(role="system", text=system_prompt)

    configured_endpointing_delay = _as_float(get_agent_setting("min_endpointing_delay", 1.2), 1.2)
    effective_endpointing_delay = max(4.0, configured_endpointing_delay)

    def _before_llm_cb(agent_instance, chat_ctx):
        if state.deterministic_replied or time.time() < state.suppress_llm_until:
            room_log("LLM_SUPPRESSED", deterministic=state.deterministic_replied, until=state.suppress_llm_until)
            return False
        from livekit.agents.pipeline.pipeline_agent import _default_before_llm_cb
        return _default_before_llm_cb(agent_instance, chat_ctx)

    agent = VoicePipelineAgent(
        vad=create_vad(), stt=create_stt(), llm=create_llm(), tts=create_tts(),
        chat_ctx=chat_ctx, fnc_ctx=ElenaFunctionContext(), allow_interruptions=True,
        interrupt_min_words=_as_int(get_agent_setting("interrupt_min_words", 2), 2),
        min_endpointing_delay=effective_endpointing_delay,
        preemptive_synthesis=False,
        before_llm_cb=_before_llm_cb,
    )

    conversation_transcript: list[str] = []

    def suppress_llm(seconds: float = 10.0):
        state.suppress_llm_until = time.time() + seconds
        room_log("LLM_SUPPRESS_SET", seconds=seconds)

    async def send_agent_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned: return
        now_ts = time.time()
        if cleaned == state.last_agent_transcript_text and (now_ts - state.last_agent_transcript_at) < 2.5:
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

    async def send_user_transcript(text: str, *, interim: bool = False):
        cleaned = (text or "").strip()
        if not cleaned: return
        now_ts = time.time()
        if interim:
            payload = json.dumps({"type": "transcript", "speaker": "user", "text": cleaned, "interim": True}, ensure_ascii=False)
            try: await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
            except Exception: pass
            return
        if cleaned == state.last_user_transcript_text and (now_ts - state.last_user_transcript_at) < 5.0: return
        state.last_user_transcript_text = cleaned
        state.last_user_transcript_at = now_ts
        conversation_transcript.append(f"User: {cleaned}")
        payload = json.dumps({"type": "transcript", "speaker": "user", "text": cleaned, "interim": False}, ensure_ascii=False)
        try: await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
        except Exception: pass
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
        if text: asyncio.create_task(send_agent_transcript(text))

    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        cancel_thinking_task()
        state.waiting_for_user = False
        asyncio.create_task(set_ui_state("listening"))

    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking(): schedule_thinking_state()

    @agent.on("user_speech_committed")
    def _on_user_speech_committed(msg):
        user_text = str(getattr(msg, "content", "") or "").strip()
        if not user_text: return
        state.last_user_activity = time.time()
        state.silence_prompt_count = 0
        state.deterministic_replied = False
        asyncio.create_task(send_user_transcript(user_text))
        
        async def _route_flows():
            flow_ctx = base.FlowContext(
                state=state, agent=agent, suppress_llm=suppress_llm, snooze_silence=snooze_silence,
                set_ui_state=set_ui_state, room_log=room_log, should_suppress_clarification=_should_suppress_clarification,
                cancel_thinking_task=cancel_thinking_task, send_transcript=send_agent_transcript, lang="el"
            )
            if await termination.handle(flow_ctx, user_text): return
            if await farewell.handle(flow_ctx, user_text): return
            if await ticket_flow.handle(flow_ctx, user_text): return
            if await order_flow.handle(flow_ctx, user_text): return
            if await phone_flow.handle(flow_ctx, user_text): return
            room_log("FLOW_FALLTHROUGH", text=user_text)

        asyncio.create_task(_route_flows())

    agent.start(ctx.room, participant)
    
    greeting_ctx = base.FlowContext(
        state=state, agent=agent, suppress_llm=suppress_llm, snooze_silence=snooze_silence,
        set_ui_state=set_ui_state, room_log=room_log, should_suppress_clarification=_should_suppress_clarification,
        cancel_thinking_task=cancel_thinking_task, send_transcript=send_agent_transcript, lang="el"
    )
    asyncio.create_task(greeting_flow.handle(greeting_ctx))

    async def _silence_monitor():
        silence_ctx = base.FlowContext(
            state=state, agent=agent, suppress_llm=suppress_llm, snooze_silence=snooze_silence,
            set_ui_state=set_ui_state, room_log=room_log, should_suppress_clarification=_should_suppress_clarification,
            cancel_thinking_task=cancel_thinking_task, send_transcript=send_agent_transcript, lang="el"
        )
        while not state.should_end:
            await asyncio.sleep(1.0)
            if await silence_flow.monitor_iteration(silence_ctx): break

    asyncio.create_task(_silence_monitor())

    human_input = getattr(agent, "human_input", None) or getattr(agent, "_human_input", None)
    if human_input:
        @human_input.on("interim_transcript")
        def _on_interim_transcript(ev):
            text = ev.alternatives[0].text if ev.alternatives else None
            if text:
                cancel_thinking_task()
                asyncio.create_task(send_user_transcript(text, interim=True))

    bg_audio_player = None
    async def _start_background_audio():
        nonlocal bg_audio_player
        try:
            from src.services.background_audio import create_background_audio_player
            bg_audio_player = await create_background_audio_player()
            if bg_audio_player: await bg_audio_player.start(ctx.room)
        except Exception: pass

    asyncio.create_task(_start_background_audio())

    while not state.should_end:
        if (time.time() - state.last_user_activity) > 300: break
        await asyncio.sleep(1.0)

    room_log("CALL_END", transcript_lines=len(conversation_transcript))

    try:
        if bg_audio_player: await bg_audio_player.stop()
    except Exception: pass
    try:
        if ctx.room and ctx.room.isconnected(): await ctx.room.disconnect()
    except Exception: pass


# -----------------------------------------------------------------------------
# Worker boot
# -----------------------------------------------------------------------------


def prewarm(proc: JobProcess):
    logger.info("Elena EL prewarm complete")


def run_agent():
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )


if __name__ == "__main__":
    run_agent()
