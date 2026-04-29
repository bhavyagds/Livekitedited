from typing import Optional, List, Dict, Any, Annotated
import re
import time
import logging
import time
import logging
from src.agents.elena.context import (
    _current_session, FLOW_IDLE, PHONE_FLOW_STATES
)
from src.agents.elena.logger import room_log

logger = logging.getLogger(__name__)

def _set_support_flow_state(new_state: str, reason: str = "") -> None:
    previous = _current_session.get("support_flow_state") or FLOW_IDLE
    _current_session["support_flow_state"] = new_state
    room_log("SUPPORT_FLOW_STATE", previous=previous, current=new_state, reason=reason)

def _is_phone_flow_active() -> bool:
    flow_state = _current_session.get("support_flow_state") or FLOW_IDLE
    return flow_state in PHONE_FLOW_STATES












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
    _current_session["phone_forced_turn_id"] = 0
    _current_session["phone_forced_pending_turn_id"] = 0



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
    _current_session["forced_response_suppress_llm_until"] = 0.0

    _clear_pending_lookup_wait_phrase("support_session_reset")



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
    tracker["is_waiting_for_response"] = True
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



