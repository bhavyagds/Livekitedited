from typing import Optional
import time
from src.agents.elena.context import _current_session
from src.agents.elena.logger import room_log

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
    combined = current + raw_digits
    max_digits = _as_int(
        get_agent_setting("phone_lookup_max_digits", 15),
        15,
        min_value=10,
        max_value=15,
    )
    if len(combined) > max_digits:
        # Caller likely restarted with a fresh number chunk.
        combined = raw_digits

    _current_session["phone_digit_buffer"] = combined
    _current_session["phone_digit_buffer_updated_at"] = now
    room_log("PHONE_DIGIT_BUFFER_UPDATED", raw_digits=raw_digits, buffer=combined)
    return combined


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


def _repeat_number_prompt_for_mode(mode: str, lang: str) -> str:
    """Recovery prompt when number capture/validation is unclear."""
    is_phone = (mode or "").lower() == "phone"
    if lang == "el":
        if is_phone:
            return "Μπορείτε να επαναλάβετε το κινητό σας ψηφίο προς ψηφίο, παρακαλώ;"
        return "Μπορείτε να επαναλάβετε τον αριθμό παραγγελίας ψηφίο προς ψηφίο, παρακαλώ;"
    if is_phone:
        return "Could you please repeat your mobile number digit by digit?"
    return "Could you please repeat your order number digit by digit?"


def _lookup_progress_prompt() -> str:
    """Progress prompt while deterministic lookup is still in progress."""
    if get_agent_language() == "el":
        return "Ελέγχω ακόμη τα στοιχεία της παραγγελίας σας. Μία στιγμή ακόμη, παρακαλώ."
    return "I’m still checking your order details. One more moment, please."


