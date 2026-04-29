import logging
import json
import time
from typing import Optional
from src.agents.elena.context import _current_session

logger = logging.getLogger(__name__)

def _truncate(text: str, max_len: int = 500) -> str:
    if not text:
        return text
    return text[:max_len] + "..." if len(text) > max_len else text

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



import functools

def log_execution(func):
    """Decorator to log entry and exit points of functions."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger.info(f"Entered async function: {func.__name__}")
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.info(f"Exited async function: {func.__name__} (took {time.time() - start_time:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Error in async function {func.__name__}: {e}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.info(f"Entered function: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exited function: {func.__name__} (took {time.time() - start_time:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {e}")
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
