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

_current_session: dict = {
    "agent": None,
    "room": None,
    "room_name": None,
    "job_id": None,
    "call_id": None,
    "tts_provider": "unknown",
    "room_logger": None,
    "should_end": False,
    "silence_tracker": None,
    "silence_pause_depth": 0,
    "last_silence_block_reason": None,
    "last_lookup_wait_phrase": None,
    "pending_lookup_wait_phrase": None,
    "pending_lookup_wait_phrase_set_at": 0.0,
    "last_agent_text": "",
    "last_forced_lookup_order": None,
    "last_forced_lookup_at": 0.0,
    "last_user_turn_id": 0,
    "last_lookup_tool_called_at": 0.0,
    "last_lookup_tool_order": None,
    "lookup_progress_prompt_until": 0.0,
    "number_mode_lock": None,
    "number_mode_turn_id": 0,
    "forced_response_manual_say_active": False,
    "forced_response_spoken_turn_id": 0,
    "forced_response_spoken_text": "",
    "forced_response_suppress_llm_until": 0.0,
    "lookup_pending": False,
    "lookup_pending_started_at": 0.0,
    "lookup_pending_order": None,
    "last_lookup_state": "unknown",
    "last_lookup_order": None,
    "pending_phone_candidate": None,
    "phone_digit_buffer": "",
    "phone_digit_buffer_updated_at": 0.0,
    "phone_lookup_inflight": False,
    "phone_forced_turn_id": 0,
    "phone_forced_pending_turn_id": 0,
    "details_lookup_inflight": False,
    "details_forced_turn_id": 0,
    "details_forced_pending_turn_id": 0,
    "details_last_spoken_order": None,
    "details_last_spoken_at": 0.0,
    "details_confirmation_pending": False,
    "details_confirmation_pending_until": 0.0,
    "full_order_details_allowed_until": 0.0,
    "full_details_unlocked_once": False,
    "last_more_details_prompt_at": 0.0,
    "ticket_confirmation_pending": False,
    "ticket_confirmation_pending_until": 0.0,
    "ticket_create_allowed_until": 0.0,
    "pending_ticket_payload": None,
    "ticket_created": False,
    "ticket_reference": None,
    "support_flow_state": FLOW_IDLE,
    "support_issue": None,
}
