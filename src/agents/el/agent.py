"""
Meallion Voice AI - Elena (Local Standalone Greek Agent)
New-SDK style (Agent / AgentSession / AgentServer) with production features:
  - Long-term memory from DB (5-min cache)
  - Dynamic system prompt from DB (KB + prompts + memory)
  - Background audio from src.services.background_audio
  - Frontend data channel (transcript + state via publish_data)
  - DB call lifecycle (record / end / transcript save)
  - Room-level per-call log files
  - Interim transcript streaming
  - conversation_transcript list + DB save
  - Provider config: ElevenLabs TTS, Deepgram STT, GPT-4o-mini, Silero VAD
Run: python agent_elena.py dev
"""

AGENT_BUILD = "elena-v1-local-20260530-mcp-fix-greek"

import asyncio
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    JobRequest,
    TurnHandlingOptions,
    cli,
    mcp,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import elevenlabs, openai, silero

try:
    from livekit.plugins import deepgram
    USE_DEEPGRAM = True
except ImportError:
    USE_DEEPGRAM = False

try:
    from livekit.plugins import ai_coustics
    HAS_AI_COUSTICS = True
except ImportError:
    HAS_AI_COUSTICS = False


# ── Local src imports ─────────────────────────────────────────────────────────
load_dotenv(".env")  # Load .env from project root

from src.config import settings
from src.agents.el.prompts import (
    _fetch_from_db,
    get_agent_setting,
    get_closing,
    get_greeting,
    get_system_prompt_async,
)

logger = logging.getLogger("agent-elena-el")


# =============================================================================
# LONG-TERM MEMORY (Memory items are fetched fresh from the database per session)
# =============================================================================



# =============================================================================
# MEMORY MATCHING UTILS (from agent.py)
# =============================================================================

_MEMORY_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "did", "for", "and", "or", "to",
    "of", "in", "on", "with", "any", "have", "has", "can", "i", "you", "we", "our",
    "your", "my", "it", "this", "that", "what", "how",
}


def _normalize_intent_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _intent_tokens(text: str) -> set[str]:
    toks = _normalize_intent_text(text).split()
    normalized: set[str] = set()
    for tok in toks:
        if len(tok) < 3 or tok in _MEMORY_STOPWORDS:
            continue
        t = tok
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
        "- NEVER speak behavioral instructions or meta-rules.\n"
        "- NEVER speak the internal tags: 'SCENARIO', 'EXPECTED RESPONSE', or 'GUIDELINE'.\n"
        "- ONLY speak the actual conversational text intended for the customer.\n\n"
        + "\n".join(lines)
    )


# =============================================================================
# SMALL UTILS
# =============================================================================

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


# =============================================================================
# ROOM-LEVEL LOGGING (from agent.py)
# =============================================================================

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


# Module-level current-room state (one session at a time per process)
_current = {
    "room_logger": None,
    "room_name": None,
    "job_id": None,
    "call_id": None,
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


# =============================================================================
# DB CALL LIFECYCLE (from agent.py)
# =============================================================================

async def record_call_to_db(
    room_name: str,
    call_type: str = "web",
    caller_number: str = None,
    caller_identity: str = None,
) -> Optional[str]:
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


async def end_call_in_db(
    call_id: str = None,
    room_name: str = None,
    status: str = "completed",
    duration_seconds: int = None,
    disconnect_reason: str = None,
    transcript: str = None,
) -> bool:
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


async def save_transcript_to_db(
    call_id: str,
    text: str,
    speaker: str = "agent",
    append: bool = True,
) -> bool:
    if not call_id or not text:
        return False
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        transcript = f"{speaker.capitalize()}: {text}" if append else text
        return await db.update_call_transcript(call_id=call_id, transcript=transcript, append=append)
    except Exception:
        return False


# =============================================================================
# FRONTEND DATA CHANNEL — transcript + state publishing (from agent.py)
# =============================================================================

async def _publish_transcript(ctx: JobContext, speaker: str, text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return
    payload = json.dumps(
        {"type": "transcript", "speaker": speaker, "text": cleaned},
        ensure_ascii=False,
    )
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


# =============================================================================
# PROVIDER FACTORIES (from agent.py — DB-driven config)
# =============================================================================

def create_llm():
    model = str(get_agent_setting("llm_model", "gpt-4o-mini") or "gpt-4o-mini")
    logger.info("LLM: model=%s", model)
    return openai.LLM(model=model, api_key=settings.openai_api_key)


def create_stt():
    provider = str(get_agent_setting("stt_provider", "deepgram") or "deepgram").lower()
    deepgram_api_key = getattr(settings, "deepgram_api_key", None)

    if provider == "deepgram" and USE_DEEPGRAM and deepgram_api_key:
        model = str(get_agent_setting("deepgram_stt_model", "nova-3") or "nova-3").strip()
        base = {
            "model": model,
            "language": "el",
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }
        for kwargs in [
            {**base, "api_key": deepgram_api_key},
            base,
        ]:
            try:
                logger.info(
                    "STT: Deepgram model=%s smart_format=True api_key_passed=%s",
                    model, "api_key" in kwargs,
                )
                return deepgram.STT(**kwargs)
            except TypeError as e:
                logger.warning("Deepgram STT args not supported, retrying: %s", e)
                continue
        return deepgram.STT(model=model)

    openai_model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1")
    logger.info("STT: OpenAI Whisper model=%s", openai_model)
    return openai.STT(model=openai_model, api_key=settings.openai_api_key, language="el")


def create_tts():
    provider = str(get_agent_setting("tts_provider", "elevenlabs") or "elevenlabs").lower()

    if provider == "openai":
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )

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
        get_agent_setting("agent_voice_id_el", getattr(settings, "elevenlabs_voice_id", "aTP4J5SJLQl74WTSRXKW"))
        or getattr(settings, "elevenlabs_voice_id", "aTP4J5SJLQl74WTSRXKW")
    )
    similarity = _as_float(
        get_agent_setting("agent_voice_similarity", getattr(settings, "elevenlabs_voice_similarity", 0.9)),
        0.9, min_value=0.0, max_value=1.0,
    )
    stability = _as_float(
        get_agent_setting("agent_voice_stability", getattr(settings, "elevenlabs_voice_stability", 0.65)),
        0.65, min_value=0.0, max_value=1.0,
    )
    speed = _as_float(
        get_agent_setting("agent_voice_speed", getattr(settings, "elevenlabs_voice_speed", 1.0)),
        1.0, min_value=0.7, max_value=1.2,
    )
    model = str(
        get_agent_setting("elevenlabs_model", getattr(settings, "elevenlabs_model", "eleven_turbo_v2_5"))
        or "eleven_turbo_v2_5"
    )

    try:
        voice_settings = elevenlabs.VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            speed=speed,
        )
        logger.info("TTS: ElevenLabs model=%s voice_id=%s speed=%.2f", model, voice_id, speed)
        return elevenlabs.TTS(
            voice_id=voice_id,
            voice_settings=voice_settings,
            model=model,
            api_key=eleven_api_key,
        )
    except TypeError as e:
        logger.warning("ElevenLabs TTS init failed (%s), falling back to OpenAI TTS", e)
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )


def create_vad():
    min_speech = _as_float(get_agent_setting("vad_min_speech_duration", 0.15), 0.15, min_value=0.05, max_value=1.0)
    min_silence = _as_float(get_agent_setting("vad_min_silence_duration", 1.5), 1.5, min_value=0.1, max_value=3.0)
    logger.info("VAD: min_speech=%.2fs min_silence=%.2fs threshold=0.6", min_speech, min_silence)
    return silero.VAD.load(
        min_speech_duration=min_speech,
        min_silence_duration=min_silence,
        activation_threshold=0.6,
    )


# =============================================================================
# AGENT CLASS (new SDK style, dynamic instructions)
# =============================================================================

FALLBACK_INSTRUCTIONS = (
    "You are Elena from Meallion. You are a helpful, concise customer support voice agent. "
    "Reply in plain Greek. Be warm, direct, and empathetic. "
    "For order issues, ask for order number first, then phone number if needed."
)


class DefaultAgent(Agent):
    """Elena — Meallion Greek voice agent (new-SDK style, DB-driven prompt)."""

    def __init__(self, instructions: str) -> None:
        self.mcp_server = mcp.MCPServerHTTP(
            url="https://voiceagent.app.n8n.cloud/mcp/meallion-agent-phone",
            client_session_timeout_seconds=3600,
        )
        mcp_toolset = mcp.MCPToolset(id="n8n-mcp", mcp_server=self.mcp_server)
        super().__init__(
            instructions=instructions,
            tools=[
                mcp_toolset,
                EndCallTool(
                    extra_description="",
                    end_instructions=(
                        "Only end the call once the customer confirms they are done or it is clear "
                        "the next step has been handed off. Before ending, summarize the resolution "
                        "or next action in one or two sentences."
                    ),
                    delete_room=False,
                )
            ],
        )


# =============================================================================
# SERVER + PREWARM
# =============================================================================

server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm: load VAD weights so first call is fast."""
    logger.info("Prewarm: loading Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm: Elena ready")


server.setup_fnc = prewarm


async def request_fnc(req: JobRequest) -> None:
    """Determine if the Greek agent should accept this job request based on DB language setting.
    
    Always fetches fresh from DB — never relies on the prompt cache which may
    be empty/stale at prewarm time and would fall back to the hardcoded default 'el'.
    """
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        # Direct DB fetch — bypass any local cache so we always get the true admin setting
        settings_dict = await db.get_all_settings()
        lang = (settings_dict.get("agent_language") or "").strip().lower()
        logger.info("Greek agent: request received. Active DB language: '%s'", lang)
        if lang == "el":
            logger.info("Greek agent: accepting job request (language=el)")
            await req.accept()
        elif lang == "en":
            logger.info("Greek agent: rejecting job request (language is English)")
            await req.reject()
        else:
            # Unknown or empty language setting — default Greek agent rejects to avoid conflict
            logger.warning("Greek agent: unknown language '%s', rejecting to let EN agent handle it", lang)
            await req.reject()
    except Exception as e:
        logger.warning("Greek agent: DB check failed in request_fnc, rejecting to be safe: %s", e)
        try:
            await req.reject()
        except Exception:
            pass



# =============================================================================
# ENTRYPOINT
# =============================================================================

@server.rtc_session(on_request=request_fnc)
async def entrypoint(ctx: JobContext):
    # ------------------------------------------------------------------
    # LANGUAGE GUARD — exit immediately if this is the wrong agent.
    # AgentServer does not honour request_fnc, so both agents accept
    # every job. We check the DB here, before ctx.connect(), so the
    # wrong-language agent drops out gracefully.
    # ------------------------------------------------------------------
    try:
        from src.services.database import get_database_service as _get_db
        _db = _get_db()
        _settings = await _db.get_all_settings()
        _lang = (_settings.get("agent_language") or "en").strip().lower()
    except Exception as _e:
        logger.warning("Greek agent: language guard DB check failed (%s) — falling back to English", _e)
        _lang = "en"  # on error, let English handle it

    if _lang != "el":
        logger.info(
            "Greek agent: active language is '%s', not 'el' — calling shutdown to release job",
            _lang,
        )
        await ctx.shutdown()  # Properly release the job back to LiveKit
        return

    logger.info("Greek agent: language='el' confirmed — starting session")


    # ------------------------------------------------------------------
    # 2. Set up room-level logger
    # ------------------------------------------------------------------
    job = getattr(ctx, "job", None)
    job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
    room_logger, room_log_path = _create_room_logger(ctx.room.name, job_id)
    _current["room_logger"] = room_logger
    _current["room_name"] = ctx.room.name
    _current["job_id"] = job_id
    logger.info("Per-room log: %s | BUILD: %s", room_log_path, AGENT_BUILD)

    # ------------------------------------------------------------------
    # 3. Detect call type: SIP rooms are named "sip-call-_<number>_<suffix>"
    # ------------------------------------------------------------------
    room_name = ctx.room.name
    is_sip_call = room_name.startswith("sip-")
    # DB check_call_type constraint: 'inbound' | 'outbound' | 'web'
    call_type = "inbound" if is_sip_call else "web"
    # Extract caller number from room name.
    # SIPDispatchRuleIndividual format: sip-call-_00919521426456_9sxuWeauwnh3
    # Direct format: sip-call-+302810209931
    caller_number: Optional[str] = None
    if is_sip_call:
        # Remove "sip-call-" prefix, then strip leading underscores
        remainder = room_name[len("sip-call-"):].lstrip("_")
        # Remove random suffix: everything after the second underscore group
        # Format: <digits>_<randomsuffix>  — keep only <digits> part
        if "_" in remainder:
            caller_number = remainder.split("_")[0]
        else:
            caller_number = remainder or None
        # Normalise: add + if purely numeric (international number without prefix)
        if caller_number and caller_number.isdigit():
            caller_number = "+" + caller_number
    logger.info("Call type detected: %s | caller_number: %s", call_type, caller_number)
    room_log("ROOM_START", call_type=call_type, build=AGENT_BUILD)
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity, call_type=call_type)

    # ------------------------------------------------------------------
    # 5. Record call to DB
    # ------------------------------------------------------------------
    call_id = await record_call_to_db(
        ctx.room.name,
        call_type=call_type,
        caller_number=caller_number,
        caller_identity=participant.identity,
    )
    _current["call_id"] = call_id
    room_log("CALL_RECORDED", call_id=call_id, call_type=call_type, caller_number=caller_number)

    # ------------------------------------------------------------------
    # 5. Pull fresh memory_items directly from database
    # ------------------------------------------------------------------
    memory_items: list[dict] = []
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        memory_items = await db.get_memory_items(active_only=True)
        room_log("MEMORY_ITEMS_LOADED", count=len(memory_items))
    except Exception as e:
        logger.warning("Failed loading memory items: %s", e)
        room_log("MEMORY_ITEMS_LOAD_FAILED", error=str(e))

    # ------------------------------------------------------------------
    # 6. Build dynamic system prompt (DB + memory block prepended)
    # ------------------------------------------------------------------
    try:
        system_prompt = await get_system_prompt_async("el")
    except Exception as e:
        logger.warning("System prompt load failed, using fallback: %s", e)
        system_prompt = FALLBACK_INSTRUCTIONS

    memory_block = _build_memory_prompt_block(memory_items)
    if memory_block and "LONG-TERM MEMORY" not in (system_prompt or ""):
        system_prompt = f"{memory_block}\n\n{system_prompt}"

    has_memory_block = "LONG-TERM MEMORY" in (system_prompt or "")
    room_log("SYSTEM_PROMPT_READY", length=len(system_prompt or ""), has_memory_block=has_memory_block)

    # ------------------------------------------------------------------
    # 7. Build AgentSession with DB-driven providers
    # ------------------------------------------------------------------
    # Retrieve prewarm VAD if available, otherwise create fresh
    vad = ctx.proc.userdata.get("vad") or create_vad()

    # Noise cancellation (optional — only if ai_coustics is installed)
    if HAS_AI_COUSTICS:
        audio_input_opts = room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_L,
            ),
        )
    else:
        audio_input_opts = room_io.AudioInputOptions()

    session = AgentSession(
        stt=create_stt(),
        llm=create_llm(),
        tts=create_tts(),
        vad=vad,
        preemptive_generation=True,
    )

    # ------------------------------------------------------------------
    # 8. conversation_transcript list (feature 8)
    # ------------------------------------------------------------------
    conversation_transcript: list[str] = []

    # ── Dedup state for transcripts ────────────────────────────────────
    _last_agent_text: str = ""
    _last_agent_text_at: float = 0.0
    _last_user_text: str = ""
    _last_user_text_at: float = 0.0
    _last_user_interim: str = ""
    _last_user_interim_at: float = 0.0

    # ------------------------------------------------------------------
    # 9. Agent transcript helper (feature 4 + 8)
    # ------------------------------------------------------------------
    async def send_agent_transcript(text: str):
        nonlocal _last_agent_text, _last_agent_text_at
        cleaned = (text or "").strip().replace("\r\n", " ").replace("\n", " ")
        if not cleaned:
            return
        now_ts = time.time()
        # Dedup window of 2.5 s
        if cleaned == _last_agent_text and (now_ts - _last_agent_text_at) < 2.5:
            room_log("AGENT_TEXT_DEDUPED", text=cleaned)
            return
        _last_agent_text = cleaned
        _last_agent_text_at = now_ts

        conversation_transcript.append(f"Agent: {cleaned}")
        await _publish_transcript(ctx, "agent", cleaned)
        room_log("AGENT_TEXT", text=cleaned)

        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="agent")
            full_text = "\n".join(conversation_transcript)
            await save_transcript_to_db(call_id, full_text, speaker="full", append=False)

    # ------------------------------------------------------------------
    # 10. User transcript helper — final + interim (features 7 + 4 + 8)
    # ------------------------------------------------------------------
    async def send_user_transcript(text: str, *, interim: bool = False):
        nonlocal _last_user_text, _last_user_text_at
        nonlocal _last_user_interim, _last_user_interim_at
        cleaned = (text or "").strip().replace("\r\n", " ").replace("\n", " ")
        # Normalize US-style formatted phone numbers like (773) 739-5770 to raw digits 7737395770
        cleaned = re.sub(r'\(?(\d{3})\)?[-. ]*(\d{3})[-. ]*(\d{4})', r'\1\2\3', cleaned)
        if not cleaned:
            return
        now_ts = time.time()

        if interim:
            # Throttle interim updates to 0.35 s
            if cleaned == _last_user_interim and (now_ts - _last_user_interim_at) < 0.35:
                return
            _last_user_interim = cleaned
            _last_user_interim_at = now_ts
            payload = json.dumps(
                {"type": "transcript", "speaker": "user", "text": cleaned, "interim": True},
                ensure_ascii=False,
            )
            try:
                await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
            except Exception:
                pass
            return

        # Final transcript dedup (5 s window)
        if cleaned == _last_user_text and (now_ts - _last_user_text_at) < 5.0:
            room_log("USER_TEXT_DEDUPED", text=cleaned)
            return
        _last_user_text = cleaned
        _last_user_text_at = now_ts
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
        room_log("USER_TEXT", text=cleaned)

        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="user")
            full_text = "\n".join(conversation_transcript)
            await save_transcript_to_db(call_id, full_text, speaker="full", append=False)

    # ------------------------------------------------------------------
    # 11. Wire session events — state + transcript (feature 4)
    # ------------------------------------------------------------------
    # ── Unified UI State Tracker ──────────────────────────────────────
    _current_agent_state = "idle"
    _current_user_state = "idle"

    def publish_unified_ui_state():
        if _current_agent_state == "speaking":
            ui_state = "speaking"
        elif _current_agent_state == "thinking":
            ui_state = "thinking"
        elif _current_user_state == "speaking":
            ui_state = "listening"
        else:
            ui_state = "idle"
        asyncio.create_task(_publish_state(ctx, ui_state))
        room_log("UI_STATE", state=ui_state)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        nonlocal _current_agent_state
        _current_agent_state = getattr(ev, "new_state", "idle")
        publish_unified_ui_state()

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        nonlocal _current_user_state
        _current_user_state = getattr(ev, "new_state", "idle")
        publish_unified_ui_state()

    _end_call_task: Optional[asyncio.Task] = None

    async def end_call_delayed(delay: float = 7.0):
        await asyncio.sleep(delay)
        logger.info("Ending call automatically as requested by call end intent")
        try:
            if ctx.room and ctx.room.isconnected():
                await ctx.room.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting call: %s", e)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        if ev.is_final and ev.transcript:
            asyncio.create_task(send_user_transcript(ev.transcript))
            
            # Check for call end intent
            cleaned_text = ev.transcript.strip()
            cleaned = re.sub(r'[^\w\s]', '', cleaned_text.lower()).strip()
            end_phrases = [
                "αντίο", "γεια", "γεια σας", "ευχαριστώ", "ευχαριστούμε", "ευχαριστω", "ευχαριστουμε",
                "όχι ευχαριστώ", "όχι ευχαριστω", "οχι ευχαριστω",
                "τίποτα άλλο", "τιποτα αλλο", "αυτό είναι όλο", "αυτο ειναι ολο",
                "τελειώσαμε", "τελειωσαμε"
            ]
            words = cleaned.split()
            if len(words) <= 4 and any(phrase in cleaned for phrase in end_phrases):
                nonlocal _end_call_task
                if not _end_call_task:
                    logger.info("Call end intent detected: '%s'. Scheduling automatic call end in 7 seconds.", cleaned_text)
                    _end_call_task = asyncio.create_task(end_call_delayed(7.0))

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        logger.info("CONVERSATION_ITEM_ADDED: role=%s, text=%s, type=%s", role, text, type(item).__name__)
        if role == "assistant" and text:
            asyncio.create_task(send_agent_transcript(text))

    # ------------------------------------------------------------------
    # 12. Start the session
    # ------------------------------------------------------------------
    agent = DefaultAgent(instructions=system_prompt)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=audio_input_opts,
        ),
    )

    # Start the MCP keep-alive task to prevent idle timeouts (ClosedResourceError)
    async def mcp_keep_alive_loop(mcp_server):
        logger.info("Starting MCP keep-alive loop for server: %s", mcp_server.url)
        while ctx.room.isconnected():
            try:
                await asyncio.sleep(15.0)
                if not ctx.room.isconnected():
                    break
                
                if not mcp_server.initialized:
                    logger.info("MCP server not initialized, attempting connection...")
                    await mcp_server.initialize()
                else:
                    mcp_server.invalidate_cache()
                    await mcp_server.list_tools()
                    logger.debug("MCP keep-alive ping successful")
            except Exception as ke:
                logger.warning("MCP keep-alive ping failed: %s", ke)

    agent.keep_alive_task = asyncio.create_task(mcp_keep_alive_loop(agent.mcp_server))

    # Explicitly speak the initial greeting upon joining
    greeting = get_greeting("el")
    room_log("GREETING_SENT", text=_truncate(greeting))
    await session.say(greeting, allow_interruptions=False)

    # ------------------------------------------------------------------
    # 13. Wire interim transcript via human_input (feature 7)
    # ------------------------------------------------------------------
    human_input = getattr(session, "human_input", None) or getattr(session, "_human_input", None)
    if human_input:
        @human_input.on("interim_transcript")
        def _on_interim_transcript(ev):
            try:
                text = ev.alternatives[0].text if ev.alternatives else None
            except Exception:
                text = None
            if text:
                asyncio.create_task(send_user_transcript(text, interim=True))
        room_log("INTERIM_TRANSCRIPT_WIRED")

    # ------------------------------------------------------------------
    # 14. Background audio — BuiltinAudioClip (primary)
    # ------------------------------------------------------------------
    builtin_bg_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0),
    )
    await builtin_bg_audio.start(room=ctx.room, agent_session=session)
    room_log("BG_AUDIO_BUILTIN_STARTED")

    # ------------------------------------------------------------------
    # 15. Background audio — src.services.background_audio (feature 3)
    # ------------------------------------------------------------------
    custom_bg_player = None

    async def _start_custom_background_audio():
        nonlocal custom_bg_player
        try:
            from src.services.background_audio import create_background_audio_player
            custom_bg_player = await create_background_audio_player()
            if custom_bg_player:
                await custom_bg_player.start(ctx.room)
                room_log("BG_AUDIO_CUSTOM_STARTED")
        except Exception as e:
            logger.debug("Custom background audio not started: %s", e)

    asyncio.create_task(_start_custom_background_audio())

    # ------------------------------------------------------------------
    # 17. Publish initial idle state to frontend
    # ------------------------------------------------------------------
    await _publish_state(ctx, "idle")

    # ------------------------------------------------------------------
    # 18. Wait for session to end (participant disconnect / EndCallTool)
    # ------------------------------------------------------------------
    disconnect_reason = "session_end"

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(participant_info):
        nonlocal disconnect_reason
        if participant_info.identity == participant.identity:
            disconnect_reason = "participant_disconnected"
            room_log("PARTICIPANT_DISCONNECTED", identity=participant_info.identity)

    # Keep session alive until room is disconnected or participant leaves
    while ctx.room.isconnected() and participant.identity in ctx.room.remote_participants:
        await asyncio.sleep(0.5)


    # ------------------------------------------------------------------
    # 19. Teardown
    # ------------------------------------------------------------------

    transcript_text = "\n".join(conversation_transcript)
    room_log("SESSION_ENDED", disconnect_reason=disconnect_reason, transcript_lines=len(conversation_transcript))

    await end_call_in_db(
        call_id=call_id,
        room_name=ctx.room.name,
        status="completed",
        disconnect_reason=disconnect_reason,
        transcript=transcript_text or None,
    )

    # Stop custom background audio
    try:
        if custom_bg_player:
            await custom_bg_player.stop()
    except Exception:
        pass

    # Disconnect room if still connected
    try:
        if ctx.room and ctx.room.isconnected():
            await ctx.room.disconnect()
    except Exception:
        pass

    room_log("ROOM_CLOSED")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    log_level = getattr(logging, getattr(settings, "log_level", "INFO"), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cli.run_app(server)
