import logging
import json
import time
import asyncio
import functools
from typing import Optional
from src.agents.elena.context import _current_session

logger = logging.getLogger(__name__)

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

def _create_room_logger(room_name: str, job_id: Optional[str]) -> tuple[logging.Logger, str]:
    """Create a dedicated logger for this room/job session."""
    import os
    log_dir = "logs/rooms"
    os.makedirs(log_dir, exist_ok=True)
    
    # Safe filename from room name
    safe_name = "".join([c if c.isalnum() else "_" for c in room_name])
    log_path = os.path.join(log_dir, f"{safe_name}_{job_id or int(time.time())}.log")
    
    lgr = logging.getLogger(f"room.{room_name}.{job_id}")
    lgr.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if job ID matches
    if not lgr.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        lgr.addHandler(fh)
        lgr.propagate = False
        
    return lgr, log_path

async def log_call_event(event_type: str, **kwargs):
    from src.services.database import get_database_service
    try:
        db = get_database_service()
        await db.create_sip_event(event_type=event_type, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to log call event: {e}")

async def record_call_to_db(**kwargs):
    from src.services.database import get_database_service
    try:
        db = get_database_service()
        return await db.record_call_start(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to record call: {e}")
        return None

async def end_call_in_db(**kwargs):
    from src.services.database import get_database_service
    try:
        db = get_database_service()
        return await db.record_call_end(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to end call: {e}")
        return False

class LatencyTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self._start = time.perf_counter()
    def stt_complete(self, transcript):
        logger.info(f"STT: {transcript}")
