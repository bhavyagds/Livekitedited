"""
Meallion Voice AI - Elena Voice Agent
Main voice agent implementation using LiveKit Agents SDK (2026 version).
"""

import logging
import asyncio
import time
import os
import re
import json
import threading
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

# Global reference to current session for termination/logging
_current_session: dict = {
    "agent": None,
    "room": None,
    "room_name": None,
    "job_id": None,
    "call_id": None,
    "room_logger": None,
    "should_end": False,
}


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
        return openai.TTS(model=model, voice=voice, speed=speed)

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

    if not elevenlabs_available():
        return create_openai_tts()
    if not elevenlabs_synthesis_available():
        return create_openai_tts()

    agent_lang = get_agent_language()
    auto_language_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)
    
    # CRITICAL: Select the correct model based on language
    # eleven_turbo_v2 is ENGLISH ONLY - Greek requires multilingual model
    configured_model = settings.elevenlabs_model
    if auto_language_switch:
        if configured_model not in {"eleven_multilingual_v2", "eleven_turbo_v2_5"}:
            tts_model = "eleven_multilingual_v2"
            logger.warning(
                "?????? TTS: Overriding %s ??? %s for auto language switching",
                configured_model,
                tts_model,
            )
        else:
            tts_model = configured_model
    elif agent_lang == "el" and configured_model == "eleven_turbo_v2":
        # Override to multilingual for Greek support
        tts_model = "eleven_multilingual_v2"
        logger.warning(f"?????? TTS: Overriding {configured_model} ??? {tts_model} for Greek support")
    elif agent_lang == "el" and "turbo" in configured_model.lower() and "v2_5" not in configured_model:
        # eleven_turbo_v2 doesn't support Greek, v2.5 does
        tts_model = "eleven_turbo_v2_5"
        logger.warning(f"?????? TTS: Overriding {configured_model} ??? {tts_model} for Greek support")
    else:
        tts_model = configured_model
    
    logger.info(f"???? TTS Model: {tts_model} (language: {agent_lang})")
    
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
    voice_stability = _require_float_setting(
        "agent_voice_stability",
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
    return elevenlabs.TTS(
        voice=voice,
        model=tts_model,
        enable_ssml_parsing=tts_use_ssml,
    )


def create_stt():
    """Create the Speech-to-Text instance optimized for speed."""
    agent_lang = get_agent_language()
    stt_lang = get_stt_language(agent_lang)
    auto_language_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)
    
    provider = str(get_agent_setting("stt_provider", "") or "").strip().lower()
    if not provider:
        provider = "deepgram" if USE_DEEPGRAM else "openai"

    if auto_language_switch:
        if provider == "deepgram" and USE_DEEPGRAM:
            logger.warning("Auto language switch enabled; forcing OpenAI Whisper for auto language detection.")
        try:
            logger.info("Using OpenAI Whisper STT - language: auto")
            room_log("STT_PROVIDER", provider="openai", model="whisper-1", language="auto")
            return openai.STT(
                model="whisper-1",
            )
        except TypeError as e:
            logger.warning("OpenAI STT auto language failed (%s); falling back to %s", e, stt_lang)
    
    # Use Deepgram if available (faster than Whisper) and explicitly selected
    if provider == "deepgram" and USE_DEEPGRAM:
        logger.info(f"Using Deepgram STT (fast) - language: {stt_lang}")
        room_log("STT_PROVIDER", provider="deepgram", model="nova-3", language=stt_lang)
        return deepgram.STT(
            model="nova-3",
            language=stt_lang,
            interim_results=True,  # Keep connection active with interim results
            punctuate=True,
            smart_format=True,
        )
    
    # Fallback to OpenAI Whisper
    if provider == "deepgram" and not USE_DEEPGRAM:
        logger.warning("Deepgram requested but not available; falling back to OpenAI Whisper")
    logger.info(f"Using OpenAI Whisper STT - language: {stt_lang}")
    room_log("STT_PROVIDER", provider="openai", model="whisper-1", language=stt_lang)
    return openai.STT(
        model="whisper-1",
        language=stt_lang,
    )


def create_vad():
    """Create Voice Activity Detection tuned for better transcript completeness."""
    min_speech_duration = _as_float(
        get_agent_setting("vad_min_speech_duration", 0.15),
        0.15,
        min_value=0.1,
        max_value=0.8,
    )
    min_silence_duration = _as_float(
        get_agent_setting("vad_min_silence_duration", 0.45),
        0.45,
        min_value=0.2,
        max_value=1.5,
    )

    vad_backend = str(get_agent_setting("vad_backend", "silero") or "").strip().lower()
    if vad_backend in {"energy", "rms", "simple"}:
        energy_threshold = _as_float(
            get_agent_setting("energy_vad_threshold", 0.012),
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
        get_agent_setting("vad_activation_threshold", 0.6),
        0.6,
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

    @llm.ai_callable()
    async def lookup_order(
        self,
        order_number: Annotated[str, llm.TypeInfo(description="The order number (4-5 digits)")],
    ) -> str:
        """Look up an order. Returns brief status first. Use get_order_details for more info."""
        room_log("TOOL_CALL", name="lookup_order", order_number=order_number)
        result = await order_lookup.lookup_order(order_number)
        room_log("TOOL_RESULT", name="lookup_order", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def get_order_details(
        self,
        order_number: Annotated[str, llm.TypeInfo(description="Order number or 'last' for most recent")] = "last",
    ) -> str:
        """Get FULL order details (items, prices, address). Use after lookup_order when customer wants more info."""
        room_log("TOOL_CALL", name="get_order_details", order_number=order_number)
        result = await order_lookup.get_order_details(order_number)
        room_log("TOOL_RESULT", name="get_order_details", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def create_support_ticket(
        self,
        customer_name: Annotated[str, llm.TypeInfo(description="Customer's full name")],
        customer_phone: Annotated[str, llm.TypeInfo(description="Customer's phone number")],
        customer_email: Annotated[str, llm.TypeInfo(description="Customer's email address")],
        issue_description: Annotated[str, llm.TypeInfo(description="Description of the issue")],
    ) -> str:
        """Create a support ticket. Collect ALL 4 fields one by one before calling this."""
        room_log(
            "TOOL_CALL",
            name="create_support_ticket",
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
        )
        result = await support_ticket.create_support_ticket(
            customer_name, customer_phone, customer_email, issue_description
        )
        room_log("TOOL_RESULT", name="create_support_ticket", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def validate_ticket_field(
        self,
        field_name: Annotated[str, llm.TypeInfo(description="Field name: name, phone, email, or issue")],
        value: Annotated[str, llm.TypeInfo(description="Value to validate")],
    ) -> str:
        """Validate a support ticket field value."""
        room_log("TOOL_CALL", name="validate_ticket_field", field=field_name, value=value)
        result = await support_ticket.validate_ticket_field(field_name, value)
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
        result = await support_ticket.log_customer_query(
            customer_question, customer_name, customer_phone
        )
        room_log("TOOL_RESULT", name="log_customer_query", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def search_knowledge_base(
        self,
        query: Annotated[str, llm.TypeInfo(description="The question to search for")],
    ) -> str:
        """Search the knowledge base for answers to common questions."""
        room_log("TOOL_CALL", name="search_knowledge_base", query=query)
        result = await knowledge_base.search_knowledge_base(query)
        room_log("TOOL_RESULT", name="search_knowledge_base", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def get_brand_info(self) -> str:
        """Get information about the Meallion brand."""
        room_log("TOOL_CALL", name="get_brand_info")
        result = await knowledge_base.get_brand_info()
        room_log("TOOL_RESULT", name="get_brand_info", result=_truncate(result))
        return result

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
        goodbye = get_closing(get_agent_language())
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
    agent_lang = get_agent_language()
    
    # Use async version to ensure KB and prompts are loaded from DB
    system_prompt = await get_system_prompt_async(agent_lang)
    
    logger.info(f"Using {agent_lang} system prompt (from database), length: {len(system_prompt)} chars")
    ctx.append(role="system", text=system_prompt)
    return ctx


async def entrypoint(ctx: JobContext):
    """LiveKit Agent entrypoint. Called when a new participant joins the room."""
    global _current_session
    
    startup_time = time.time()
    logger.info(f"Elena agent starting for room: {ctx.room.name}")
    
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
    
    # Reset session state
    _current_session["should_end"] = False
    
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
    
    asyncio.create_task(record_call_async())
    
    # Initialize transcript collection
    conversation_transcript = []
    
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
    base_language = get_agent_language()
    set_runtime_language(base_language)
    session_language = {"value": base_language}
    auto_language_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)

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
        "enabled": True,
    }
    
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
    
    async def send_agent_transcript(text: str):
        """Send spoken agent text to frontend chat in realtime."""
        try:
            cleaned = (text or "").strip()
            if not cleaned:
                return
            if cleaned in _sent_agent_transcripts:
                return
            _sent_agent_transcripts.add(cleaned)
            if len(_sent_agent_transcripts) > 100:
                _sent_agent_transcripts.clear()
                _sent_agent_transcripts.add(cleaned)

            import json
            transcript_data = json.dumps({
                "type": "transcript",
                "speaker": "agent",
                "text": cleaned,
            })
            await ctx.room.local_participant.publish_data(
                transcript_data.encode('utf-8'),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send agent transcript: {e}")

    async def send_agent_info(text: str):
        """Extract and send important information cards without blocking audio."""
        try:
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

    def before_tts_callback(agent_instance, text: str | llm.LLMStream):
        """Callback that fires BEFORE text is sent to TTS - apply prosody."""
        try:
            # text can be a string or LLMStream - handle both
            if isinstance(text, str):
                asyncio.create_task(send_agent_transcript(text))
                logger.info(f"before_tts_cb processing: {text[:50]}...")

                from src.utils import apply_prosody, normalize_time_colons, normalize_punctuation_for_tts
                agent_lang = get_agent_language()
                use_ssml = _as_bool(get_agent_setting("tts_use_ssml", False), default=False)
                tts_text = normalize_time_colons(text)
                tts_text = normalize_punctuation_for_tts(tts_text)
                processed_text = apply_prosody(tts_text, language=agent_lang, use_ssml=use_ssml)
                return processed_text
            else:
                logger.debug("before_tts_cb got LLMStream (will use committed fallback)")
        except Exception as e:
            logger.error(f"Error in before_tts_callback: {e}")
        return text  # Return text unchanged on error
    # Wait for context that was started earlier (should be ready by now)
    initial_ctx = await context_task
    logger.info(f"⏱️ Context ready ({time.time() - startup_time:.1f}s)")
    
    # Create the voice pipeline agent - tuned to avoid clipping user speech.
    min_endpointing_delay = _as_float(
        get_agent_setting("min_endpointing_delay", 0.40),
        0.40,
        min_value=0.2,
        max_value=1.5,
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
    logger.info(f"⏱️ Creating agent ({time.time() - startup_time:.1f}s)")
    agent = VoicePipelineAgent(
        vad=create_vad(),
        stt=create_stt(),
        llm=create_llm(),
        tts=create_tts(),
        chat_ctx=initial_ctx,
        fnc_ctx=ElenaFunctionContext(),
        max_nested_fnc_calls=3,
        # Turn segmentation settings.
        min_endpointing_delay=min_endpointing_delay,
        preemptive_synthesis=True,      # Start TTS immediately as LLM streams
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
        silence_tracker["last_user_speech"] = time.time()
    
    @agent.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        _latency_tracker.user_stopped_speaking()
        schedule_thinking_state()

    async def send_user_transcript(text: str, *, interim: bool = False):
        """Helper to send user transcript to frontend."""
        try:
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
            })
            await ctx.room.local_participant.publish_data(
                transcript_data.encode('utf-8'),
                reliable=True
            )
            logger.debug(f"?? User transcript sent: {cleaned[:50]}...")
            if not interim:
                room_log("USER_TEXT", text=_truncate(cleaned))
        except Exception as e:
            logger.error(f"Failed to send user transcript: {e}")

    @agent.on("user_speech_committed")
    def on_user_speech_committed(message):
        """Send user transcript to frontend and check for abuse."""
        user_text = message.content
        asyncio.create_task(send_user_transcript(user_text))

        if auto_language_switch:
            detected_lang = detect_language(user_text, default=session_language["value"])
            if detected_lang != session_language["value"]:
                session_language["value"] = detected_lang
                set_runtime_language(detected_lang)
                lang_name = "Greek" if detected_lang == "el" else "English"
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "LANGUAGE SWITCH:\n"
                        f"- Respond in {lang_name} for this response and until the caller switches again."
                    ),
                )
                room_log("LANGUAGE_SWITCH", language=detected_lang)
            else:
                set_runtime_language(detected_lang)
        
        # Reset silence timer - user is responding
        reset_silence_timer()
        
        # Add to transcript
        conversation_transcript.append(f"User: {user_text}")
        if abuse_detection_enabled:
            # Check for abusive language
            abuse_detected, abuse_response = check_and_respond_to_abuse(
                user_text,
                language=get_agent_language(),
                tracker=_abuse_tracker,
                use_ssml=True
            )
            
            if abuse_detected:
                logger.warning(f"?????? Abuse detected in: {user_text[:50]}...")
                # The agent will continue normally, but we log the incident
                # The abuse response will be handled by the LLM with special instructions
                # For now, we just track it for escalation purposes
    
    @agent.on("agent_started_speaking")
    def on_agent_started_speaking():
        _latency_tracker.agent_started_speaking()
        cancel_thinking_task()
        logger.info("audio_publish_start: agent_started_speaking")
        asyncio.create_task(send_state_update("speaking"))
    
    @agent.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        asyncio.create_task(send_state_update("idle"))
        # Mark that agent finished speaking - now waiting for user response
        mark_agent_speaking()
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(message):
        """Send committed agent speech to UI (transcript + info cards)."""
        try:
            text = message.content if hasattr(message, 'content') else None
            if text:
                asyncio.create_task(send_agent_transcript(text))
                asyncio.create_task(send_agent_info(text))
                conversation_transcript.append(f"Agent: {text}")
                logger.info(f"agent_speech_committed: {text[:50]}...")
                room_log("AGENT_TEXT", text=_truncate(text))
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
    
    # Store references for session management
    _current_session["agent"] = agent
    _current_session["room"] = ctx.room
    
    # Track if we already handled the call end to avoid duplicate processing
    call_ended = {"value": False}
    
    async def handle_call_end(reason: str = "normal"):
        """Handle call ending - save transcript and clean up."""
        if call_ended["value"]:
            logger.debug("Call end already handled, skipping")
            return
        call_ended["value"] = True
        
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
            set_runtime_language(None)
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
    agent_lang = get_agent_language()
    greeting_enabled = _require_bool_setting("agent_greeting_enabled")
    if greeting_enabled:
        greeting = get_greeting(agent_lang)
        logger.info(f"⏱️ Saying greeting ({time.time() - startup_time:.1f}s): {greeting[:50]}...")
        await agent.say(greeting, allow_interruptions=True)
    else:
        logger.info("Greeting disabled by settings")
    
    total_startup = time.time() - startup_time
    logger.info(f"✅ Elena ready! Total startup: {total_startup:.1f}s, language: {agent_lang}")
    
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
        agent_lang = get_agent_language()
        
        # Silence prompts based on language
        if agent_lang == "el":
            prompts = [
                "Είστε εκεί;",
                "Με ακούτε;",
                "Φαίνεται ότι δεν είστε εκεί. Αντίο!",
            ]
        else:
            prompts = [
                "Are you still there?",
                "Hello? Can you hear me?",
                "It seems like you're not there. Goodbye!",
            ]
        
        try:
            while not _current_session["should_end"] and silence_tracker["enabled"]:
                await asyncio.sleep(1.0)  # Check every second
                
                # Only check silence if we're waiting for a response
                if not silence_tracker["is_waiting_for_response"]:
                    continue
                
                # Calculate time since last activity
                time_since_user = time.time() - silence_tracker["last_user_speech"]
                time_since_agent = time.time() - silence_tracker["last_agent_speech"]
                
                # Only trigger if:
                # 1. User hasn't spoken for silence_timeout seconds
                # 2. Agent finished speaking at least silence_timeout seconds ago
                if time_since_user >= silence_tracker["silence_timeout"] and \
                   time_since_agent >= silence_tracker["silence_timeout"]:
                    
                    prompt_count = silence_tracker["prompt_count"]
                    
                    if prompt_count < silence_tracker["max_prompts"]:
                        # Prompt the user
                        prompt_text = prompts[min(prompt_count, len(prompts) - 1)]
                        logger.info(f"🔇 Silence detected ({time_since_user:.1f}s), prompting user: {prompt_text}")
                        
                        silence_tracker["prompt_count"] += 1
                        silence_tracker["is_waiting_for_response"] = False  # Will be set again after agent speaks
                        
                        # Say the prompt
                        await agent.say(prompt_text, allow_interruptions=True)
                        
                    else:
                        # Max prompts reached - disconnect
                        goodbye_text = prompts[-1]  # Last prompt is goodbye
                        logger.info(f"🔇 Max silence prompts reached, disconnecting: {goodbye_text}")
                        
                        silence_tracker["enabled"] = False
                        await agent.say(goodbye_text, allow_interruptions=False)
                        
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








