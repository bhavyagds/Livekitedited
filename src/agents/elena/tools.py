import os
import logging
import time
import asyncio
from livekit import rtc, agents
from src.agents.elena.logger import log_execution
from livekit.agents import llm
from typing import Annotated, Optional
from src.agents.elena.context import _current_session, room_log, get_agent_language, get_agent_setting, _as_bool, _as_float, _as_int, _expected_order_digits, _normalize_order_id_strict, _set_support_flow_state, FLOW_AWAITING_ORDER_NUMBER, FLOW_CHECKING_ORDER_NUMBER, _set_lookup_pending, _run_tool_with_silence_pause, _classify_lookup_result, FLOW_ORDER_FOUND, FLOW_AWAITING_PHONE_NUMBER, _build_order_voice_summary, _pick_lookup_wait_phrase, FLOW_AWAITING_ORDER_NUMBER, _normalize_phone_for_lookup, FLOW_CHECKING_PHONE_NUMBER, _snooze_silence_prompts, _truncate, _build_phone_lookup_voice_summary, _clear_lookup_pending, _clear_pending_lookup_wait_phrase, _reset_phone_digit_buffer, FLOW_AWAITING_ORDER_NUMBER, FLOW_CHECKING_ORDER_NUMBER, _build_order_details_voice_summary

logger = logging.getLogger(__name__)

from src.agents.tools import order_lookup, support_ticket, knowledge_base

class ElenaFunctionContext(llm.FunctionContext):
    """Function context with all Elena's tools as methods."""

    def _pick_lookup_wait_phrase(self) -> str:
        """Pick a natural, non-repeating wait phrase based on current language."""
        lang = get_agent_language()
        silence_grace_s = _as_float(
            get_agent_setting("order_lookup_silence_grace_seconds", 8.0),
            8.0,
            min_value=2.0,
            max_value=20.0,
        )
        if lang == "en":
            phrases = (
                "Got it. Please give me a moment to check the details for you.",
                "Okay, I have it. One moment while I pull up the details.",
                "Thanks, let me check that for you right away.",
                "Perfect, I’ll quickly look this up for you now.",
            )
        else:
            phrases = (
                "Ωραία, το έχω. Δώστε μου μια στιγμή να το ελέγξω.",
                "Εντάξει, το πήρα. Μια στιγμή να δω τις λεπτομέρειες.",
                "Τέλεια, ευχαριστώ. Το ελέγχω αμέσως για εσάς.",
                "Σας ευχαριστώ, το έχω. Δώστε μου ένα λεπτό να το βρω.",
            )

        last_phrase = _current_session.get("last_lookup_wait_phrase")
        options = [p for p in phrases if p != last_phrase] or list(phrases)
        phrase = random.choice(options)
        _current_session["last_lookup_wait_phrase"] = phrase
        _current_session["pending_lookup_wait_phrase"] = phrase
        _current_session["pending_lookup_wait_phrase_set_at"] = time.time()
        _snooze_silence_prompts(silence_grace_s, reason="lookup_wait_ack")
        room_log("TOOL_WAIT_ACK_SELECTED", language=lang, phrase=_truncate(phrase))
        return phrase

    async def _run_tool_with_silence_pause(self, name: str, coro):
        """Pause silence prompts during tool I/O and resume afterward."""
        _pause_silence_for_tool(name)
        try:
            return await coro
        finally:
            _resume_silence_for_tool(name)

    @llm.ai_callable()
    @log_execution
    async def lookup_order(
        self,
        order_number: Annotated[str, llm.TypeInfo(description="The order number (4-5 digits)")],
    ) -> str:
        """Look up an order. Returns brief status first. Use get_order_details for more info."""
        lang = get_agent_language()
        lock_mode = str(_current_session.get("number_mode_lock") or "")
        lock_turn = int(_current_session.get("number_mode_turn_id") or 0)
        latest_turn = int(_current_session.get("last_user_turn_id") or 0)
        if lock_mode == "phone" and lock_turn == latest_turn:
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order", reason="number_mode_mismatch")
            expected = _expected_order_digits()
            if lang == "el":
                return f"Αυτό μοιάζει με αριθμό τηλεφώνου. Δώστε μου τον {expected}-ψήφιο αριθμό παραγγελίας από την επιβεβαίωσή σας."
            return f"That looks like a phone number. Please share your {expected}-digit order number from the confirmation."

        strict_order = _normalize_order_id_strict(order_number)
        if not strict_order:
            expected = _expected_order_digits()
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order", reason="invalid_order_id_format")
            _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason="invalid_order_format")
            if lang == "el":
                return f"Ο αριθμός παραγγελίας πρέπει να είναι ακριβώς {expected} ψηφία. Μπορείτε να τον πείτε ξανά ψηφίο προς ψηφίο;"
            return f"The order number must be exactly {expected} digits. Could you say it again digit by digit?"

        # We now have a valid order id, so move flow authority back to order lookup.
        _set_support_flow_state(FLOW_CHECKING_ORDER_NUMBER, reason="lookup_order_called")
        _set_lookup_pending(strict_order, reason="lookup_order_called")
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = strict_order
        room_log("TOOL_CALL", name="lookup_order", order_number=strict_order)
        wait_msg = "Μισό λεπτό, ψάχνω την παραγγελία σας." if get_agent_language() == "el" else "Just a moment, I am searching for your order."
        await agent.say(wait_msg, allow_interruptions=False)
        try:
            result = await self._run_tool_with_silence_pause(
                "lookup_order",
                order_lookup.lookup_order(strict_order),
            )
            room_log("TOOL_RESULT", name="lookup_order", result=_truncate(result))
            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            _current_session["last_lookup_order"] = strict_order if lookup_state == "found" else ""
            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason="lookup_order_found")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason=f"lookup_order_{lookup_state}")
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0
                _current_session["number_mode_lock"] = "order"
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("back_to_order_flow")
                tracker = _current_session.get("silence_tracker")
                silence_timeout = 12.0
                if isinstance(tracker, dict):
                    silence_timeout = float(tracker.get("silence_timeout") or 12.0)
                _snooze_silence_prompts(
                    silence_timeout + 5.0,
                    reason="order_lookup_not_found_grace",
                )

            summary = _build_order_voice_summary(result, get_agent_language()) or result
            return summary
        finally:
            _clear_lookup_pending("lookup_order_finished")
            _clear_pending_lookup_wait_phrase("lookup_order_finished")
            _current_session["lookup_progress_prompt_until"] = 0.0
            _snooze_silence_prompts(5.0, reason="lookup_order_finished")

    @llm.ai_callable()
    @log_execution
    async def get_order_details(
        self,
        order_number: Annotated[str, llm.TypeInfo(description="Order number or 'last' for most recent")] = "last",
    ) -> str:
        """Get FULL order details (items, prices, address). Use after lookup_order when customer wants more info."""
        lang = get_agent_language()
        current_turn = int(_current_session.get("last_user_turn_id") or 0)
        forced_turn = int(_current_session.get("details_forced_turn_id") or 0)
        forced_pending_turn = int(_current_session.get("details_forced_pending_turn_id") or 0)
        if current_turn and (forced_turn == current_turn or forced_pending_turn == current_turn):
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="forced_turn_in_progress")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            if lang == "el":
                return "Το ελέγχω ήδη και θα σας πω αμέσως τις λεπτομέρειες."
            return "I am already fetching those details and will share them in a moment."
        if bool(_current_session.get("details_lookup_inflight")):
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="forced_lookup_inflight")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            if lang == "el":
                return "Το ελέγχω ήδη και θα σας πω αμέσως τις λεπτομέρειες."
            return "I am already fetching those details and will share them in a moment."
        now = time.time()
        allowed_until = float(_current_session.get("full_order_details_allowed_until") or 0.0)
        if now > allowed_until:
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="explicit_details_required")
            if lang == "el":
                return "Μπορώ να δώσω όλες τις λεπτομέρειες μόλις μου πείτε ναι. Θέλετε να συνεχίσουμε με πλήρη στοιχεία παραγγελίας;"
            return "I can share full order details as soon as you say yes. Would you like the complete order details now?"

        last_state = str(_current_session.get("last_lookup_state") or "unknown")
        anchor_order = re.sub(r"\D", "", str(_current_session.get("last_lookup_order") or ""))
        expected = _expected_order_digits()
        if last_state != "found" or not re.fullmatch(rf"\d{{{expected}}}", anchor_order):
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="missing_found_lookup_anchor")
            if lang == "el":
                return "Χρειάζομαι πρώτα μια έγκυρη παραγγελία που να έχει βρεθεί για να δώσω πλήρεις λεπτομέρειες."
            return "I first need a valid found order before I can share full details."

        requested_order = anchor_order if str(order_number or "").lower() == "last" else (
            _normalize_order_id_strict(order_number or "") or ""
        )
        if requested_order != anchor_order:
            room_log("TOOL_RESULT_BLOCKED", name="get_order_details", reason="order_anchor_mismatch")
            if lang == "el":
                return "Μπορώ να δώσω λεπτομέρειες μόνο για την τελευταία παραγγελία που βρέθηκε. Δώστε ξανά τον ίδιο αριθμό παραγγελίας."
            return "I can only share details for the last found order. Please provide that same order number again."

        _current_session["full_order_details_allowed_until"] = 0.0
        _current_session["details_confirmation_pending"] = False
        _current_session["details_confirmation_pending_until"] = 0.0
        _set_lookup_pending(requested_order, reason="get_order_details_called")
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = requested_order
        room_log("TOOL_CALL", name="get_order_details", order_number=requested_order)
        result = await self._run_tool_with_silence_pause(
            "get_order_details",
            order_lookup.get_order_details(requested_order),
        )
        room_log("TOOL_RESULT", name="get_order_details", result=_truncate(result))
        spoken_summary = _build_order_details_voice_summary(result, get_agent_language())
        if not spoken_summary:
            spoken_summary = _build_order_voice_summary(result, get_agent_language()) or (
                "Δεν βρήκα λεπτομέρειες για αυτή την παραγγελία."
                if get_agent_language() == "el"
                else "I could not find details for this order."
            )
        room_log("ORDER_DETAILS_FORMATTED", order_number=order_number, result=_truncate(spoken_summary))
        if _as_bool(get_agent_setting("order_lookup_wait_phrase_enabled", True), default=True):
            self._pick_lookup_wait_phrase()
        return spoken_summary

    @llm.ai_callable()
    @log_execution
    async def lookup_order_by_phone(
        self,
        phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
    ) -> str:
        """Look up orders by customer phone number. Use when order number is unknown."""
        lang = get_agent_language()
        current_turn = int(_current_session.get("last_user_turn_id") or 0)
        forced_turn = int(_current_session.get("phone_forced_turn_id") or 0)
        forced_pending_turn = int(_current_session.get("phone_forced_pending_turn_id") or 0)
        if current_turn and (forced_turn == current_turn or forced_pending_turn == current_turn):
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="forced_turn_in_progress")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            if lang == "el":
                return "Το ελέγχω ήδη και θα σας απαντήσω αμέσως."
            return "I am already checking that phone number and will respond in a moment."
        if bool(_current_session.get("phone_lookup_inflight")):
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="forced_lookup_inflight")
            in_flight_text = _strip_markup_for_output(str(_current_session.get("forced_response_spoken_text") or ""))
            if in_flight_text:
                return in_flight_text
            if lang == "el":
                return "Το ελέγχω ήδη και θα σας απαντήσω αμέσως."
            return "I am already checking that phone number and will respond in a moment."
        lock_mode = str(_current_session.get("number_mode_lock") or "")
        lock_turn = int(_current_session.get("number_mode_turn_id") or 0)
        latest_turn = int(_current_session.get("last_user_turn_id") or 0)
        if lock_mode == "order" and lock_turn == latest_turn:
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="number_mode_mismatch")
            if lang == "el":
                return "Αυτό μοιάζει με αριθμό παραγγελίας. Δώστε μου το τηλέφωνό σας ψηφίο προς ψηφίο."
            return "That looks like an order number. Please share your phone number digit by digit."

        normalized_phone = _normalize_phone_for_lookup(phone)
        if not normalized_phone:
            room_log("TOOL_RESULT_BLOCKED", name="lookup_order_by_phone", reason="invalid_phone_pattern")
            _clear_pending_lookup_wait_phrase("invalid_phone_pattern")
            _clear_lookup_pending("invalid_phone_pattern")
            _current_session["phone_lookup_inflight"] = False
            _current_session["pending_phone_candidate"] = None
            _reset_phone_digit_buffer("invalid_phone_pattern")
            _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="invalid_phone_pattern")
            invalid_recovery_grace = _as_float(
                get_agent_setting("invalid_number_recovery_silence_grace_seconds", 12.0),
                12.0,
                min_value=4.0,
                max_value=40.0,
            )
            _snooze_silence_prompts(invalid_recovery_grace, reason="invalid_phone_recovery")
            return _repeat_number_prompt_for_mode("phone", lang)

        pending_phone = str(_current_session.get("pending_phone_candidate") or "").strip()
        awaiting_confirmation = _is_phone_confirmation_pending()
        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        if not awaiting_confirmation and flow_state != FLOW_CHECKING_PHONE_NUMBER:
            _current_session["pending_phone_candidate"] = normalized_phone
            _set_support_flow_state(FLOW_AWAITING_PHONE_CONFIRMATION, reason="lookup_order_by_phone_guard")
            _clear_pending_lookup_wait_phrase("phone_confirmation_required")
            _clear_lookup_pending("phone_confirmation_required")
            _current_session["phone_lookup_inflight"] = False
            room_log(
                "TOOL_RESULT_BLOCKED",
                name="lookup_order_by_phone",
                reason="phone_confirmation_required_hard_gate",
                phone=normalized_phone,
                flow_state=flow_state,
            )
            spoken_phone = _speak_digits(normalized_phone, get_agent_language())
            if lang == "el":
                return f"Για επιβεβαίωση, ο αριθμός τηλεφώνου σας είναι {spoken_phone}. Είναι σωστός;"
            return f"Just to confirm, your phone number is {spoken_phone}. Is that correct?"

        if awaiting_confirmation:
            _clear_lookup_pending("phone_confirmation_required")
            _current_session["phone_lookup_inflight"] = False
            if pending_phone and normalized_phone == pending_phone:
                _clear_pending_lookup_wait_phrase("phone_confirmation_required")
                room_log(
                    "TOOL_RESULT_BLOCKED",
                    name="lookup_order_by_phone",
                    reason="phone_confirmation_required",
                )
                if lang == "el":
                    return (
                        "Πριν το ελέγξω, παρακαλώ επιβεβαιώστε αν αυτός ο αριθμός τηλεφώνου είναι σωστός."
                    )
                return "Before I check that, please confirm whether this phone number is correct."

            room_log(
                "TOOL_RESULT_BLOCKED",
                name="lookup_order_by_phone",
                reason="phone_confirmation_pending_mismatch",
                phone=normalized_phone,
                pending_phone=pending_phone or None,
            )
            _clear_pending_lookup_wait_phrase("phone_confirmation_pending_mismatch")
            _clear_lookup_pending("phone_confirmation_pending_mismatch")
            _current_session["phone_lookup_inflight"] = False
            if lang == "el":
                return "Για να συνεχίσουμε, επιβεβαιώστε πρώτα τον αριθμό τηλεφώνου."
            return "To continue, please confirm the phone number first."

        _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="lookup_order_by_phone_called")
        _current_session["phone_lookup_inflight"] = True
        _set_lookup_pending(normalized_phone, reason="phone_lookup_started")
        _snooze_silence_prompts(45.0, reason="phone_lookup_started")
        _current_session["lookup_progress_prompt_until"] = time.time() + 45.0
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = normalized_phone
        room_log("TOOL_CALL", name="lookup_order_by_phone", phone=normalized_phone)
        wait_msg = "Μισό λεπτό, ψάχνω την παραγγελία σας." if get_agent_language() == "el" else "Just a moment, I am searching for your order."
        await agent.say(wait_msg, allow_interruptions=False)
        try:
            result = await self._run_tool_with_silence_pause(
                "lookup_order_by_phone",
                order_lookup.lookup_order_by_phone(normalized_phone),
            )

            room_log("TOOL_RESULT", name="lookup_order_by_phone", result=_truncate(result))
            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            snapshot = order_lookup.get_last_order_snapshot() or {}
            snapshot_order = str(snapshot.get("order_number") or "")
            strict_snapshot_order = _normalize_order_id_strict(snapshot_order) if snapshot_order else None
            _current_session["last_lookup_order"] = strict_snapshot_order if (lookup_state == "found" and strict_snapshot_order) else ""

            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason="lookup_order_by_phone_found")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason=f"lookup_order_by_phone_{lookup_state}")
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0

            summary = _build_phone_lookup_voice_summary(result, get_agent_language()) or result
            if _as_bool(get_agent_setting("order_lookup_wait_phrase_enabled", True), default=True):
                self._pick_lookup_wait_phrase()
            return summary
        finally:
            _current_session["phone_lookup_inflight"] = False
            _clear_lookup_pending("phone_lookup_finished")
            _clear_pending_lookup_wait_phrase("phone_lookup_finished")
            _reset_phone_digit_buffer("phone_lookup_finished")
            _snooze_silence_prompts(5.0, reason="phone_lookup_finished")

    @llm.ai_callable()
    @log_execution
    async def create_support_ticket(
        self,
        customer_name: Annotated[str, llm.TypeInfo(description="Customer's full name")],
        customer_phone: Annotated[str, llm.TypeInfo(description="Customer's phone number")],
        customer_email: Annotated[str, llm.TypeInfo(description="Customer's email address")],
        issue_description: Annotated[str, llm.TypeInfo(description="Description of the issue")],
    ) -> str:
        """Create a support ticket. Collect ALL 4 fields one by one before calling this."""
        lang = get_agent_language()
        existing_ref = str(_current_session.get("ticket_reference") or "").strip()
        if _current_session.get("ticket_created"):
            room_log("TOOL_RESULT_BLOCKED", name="create_support_ticket", reason="already_created")
            if lang == "el":
                if existing_ref:
                    return (
                        f"Έχουμε ήδη δημιουργήσει αίτημα υποστήριξης με αριθμό αναφοράς {existing_ref}. "
                        "Ένας συνάδελφός μας θα επικοινωνήσει μαζί σας σύντομα."
                    )
                return "Έχουμε ήδη δημιουργήσει αίτημα υποστήριξης. Ένας συνάδελφός μας θα επικοινωνήσει μαζί σας σύντομα."
            if existing_ref:
                return (
                    f"I already created your support request with reference number {existing_ref}. "
                    "One of our colleagues will contact you soon."
                )
            return "I already created your support request. One of our colleagues will contact you soon."

        now = time.time()
        allowed_until = float(_current_session.get("ticket_create_allowed_until") or 0.0)
        pending_payload = {
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "customer_email": customer_email,
            "issue_description": issue_description,
        }
        if now > allowed_until:
            _current_session["pending_ticket_payload"] = pending_payload
            _current_session["ticket_confirmation_pending"] = True
            _current_session["ticket_confirmation_pending_until"] = now + 180.0
            room_log("TOOL_RESULT_BLOCKED", name="create_support_ticket", reason="issue_confirmation_required")
            if lang == "el":
                return (
                    "Πριν δημιουργήσω το αίτημα υποστήριξης, θέλω μια τελική επιβεβαίωση για το πρόβλημα που περιγράψατε. "
                    "Αν είναι σωστό, πείτε ναι και προχωράω αμέσως."
                )
            return (
                "Before I create the support request, I need one final confirmation of the issue you described. "
                "If that's correct, say yes and I will submit it right away."
            )

        _current_session["ticket_create_allowed_until"] = 0.0
        _current_session["ticket_confirmation_pending"] = False
        _current_session["ticket_confirmation_pending_until"] = 0.0
        payload = _current_session.get("pending_ticket_payload") or pending_payload
        _current_session["pending_ticket_payload"] = None
        room_log(
            "TOOL_CALL",
            name="create_support_ticket",
            customer_name=payload.get("customer_name"),
            customer_phone=payload.get("customer_phone"),
            customer_email=payload.get("customer_email"),
        )
        result = await self._run_tool_with_silence_pause(
            "create_support_ticket",
            support_ticket.create_support_ticket(
                payload.get("customer_name"),
                payload.get("customer_phone"),
                payload.get("customer_email"),
                payload.get("issue_description"),
            ),
        )
        room_log("TOOL_RESULT", name="create_support_ticket", result=_truncate(result))
        normalized_result = _normalize_intent_text(result)
        if "cannot create ticket" in normalized_result or "couldn t create the support ticket" in normalized_result:
            return result
        reference = _extract_ticket_reference(result)
        _current_session["ticket_created"] = True
        _current_session["ticket_reference"] = reference
        if lang == "el":
            if reference:
                return (
                    f"Το αίτημα υποστήριξης δημιουργήθηκε με επιτυχία. Ο αριθμός αναφοράς σας είναι {reference}. "
                    "Ένας συνάδελφός μας θα επικοινωνήσει μαζί σας σύντομα."
                )
            return "Το αίτημα υποστήριξης δημιουργήθηκε με επιτυχία. Ένας συνάδελφός μας θα επικοινωνήσει μαζί σας σύντομα."
        if reference:
            return (
                f"Your support request has been created successfully. Your reference number is {reference}. "
                "One of our colleagues will contact you soon."
            )
        return "Your support request has been created successfully. One of our colleagues will contact you soon."

    @llm.ai_callable()
    @log_execution
    async def validate_ticket_field(
        self,
        field_name: Annotated[str, llm.TypeInfo(description="Field name: name, phone, email, or issue")],
        value: Annotated[str, llm.TypeInfo(description="Value to validate")],
    ) -> str:
        """Validate a support ticket field value."""
        room_log("TOOL_CALL", name="validate_ticket_field", field=field_name, value=value)
        result = await self._run_tool_with_silence_pause(
            "validate_ticket_field",
            support_ticket.validate_ticket_field(field_name, value),
        )
        room_log("TOOL_RESULT", name="validate_ticket_field", result=_truncate(result))
        return result

    @llm.ai_callable()
    @log_execution
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
        result = await self._run_tool_with_silence_pause(
            "log_customer_query",
            support_ticket.log_customer_query(
                customer_question, customer_name, customer_phone
            ),
        )
        room_log("TOOL_RESULT", name="log_customer_query", result=_truncate(result))
        return result

    @llm.ai_callable()
    @log_execution
    async def search_knowledge_base(
        self,
        query: Annotated[str, llm.TypeInfo(description="The question to search for")],
    ) -> str:
        """Search the knowledge base for answers to common questions."""
        language = get_agent_language()
        room_log("TOOL_CALL", name="search_knowledge_base", query=query, language=language)
        result = await self._run_tool_with_silence_pause(
            "search_knowledge_base",
            knowledge_base.search_knowledge_base(query, language=language),
        )
        room_log("TOOL_RESULT", name="search_knowledge_base", result=_truncate(result))
        return result

    @llm.ai_callable()
    @log_execution
    async def get_brand_info(self) -> str:
        """Get information about the Meallion brand."""
        room_log("TOOL_CALL", name="get_brand_info")
        result = await self._run_tool_with_silence_pause(
            "get_brand_info",
            knowledge_base.get_brand_info(),
        )
        room_log("TOOL_RESULT", name="get_brand_info", result=_truncate(result))
        return result

    @llm.ai_callable()
    @log_execution
    async def save_agent_memory(
        self,
        question: Annotated[str, llm.TypeInfo(description="The question asked by the user, or topic")],
        answer: Annotated[str, llm.TypeInfo(description="The answer given or the key information")],
        comments: Annotated[str, llm.TypeInfo(description="Any extra comments or feedback provided by user")] = None,
    ) -> str:
        """Save a long-term memory about the user's question, answer, and comments for future training. Use this when you get feedback or an interesting Q/A."""
        lang = get_agent_language()
        try:
            from src.services.database import get_database_service
            db = get_database_service()
            success = await db.add_agent_memory(question, answer, comments, lang)
            if success:
                room_log("TOOL_CALL", name="save_agent_memory", question=question, answer=answer)
                return "Successfully saved to memory."
            return "Failed to save to memory."
        except Exception as e:
            logger.error(f"Error saving memory in tool: {e}")
            return "Failed to save memory due to error."

    @llm.ai_callable()
    @log_execution
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


