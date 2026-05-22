"""
TicketFlow ÔÇö Dedicated support ticket creation state machine.

Encapsulates the entire ticket creation flow (name ÔåÆ phone ÔåÆ email ÔåÆ issue ÔåÆ
confirm ÔåÆ create) with proper email cleaning, timeout recovery, go-back
navigation, full confirmation readback, retry on API failure, and ambiguous
confirmation re-read.

This module is independently testable and receives all dependencies via
constructor callbacks ÔÇö it does NOT import from agent.py.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum
from typing import Awaitable, Callable, Optional

from src.agents.en.tools import (
    clean_email,
    validate_email,
    clean_phone_number,
    validate_phone,
)

logger = logging.getLogger(__name__)


# =============================================================================
# State Enum
# =============================================================================


class TicketFlowState(Enum):
    """States of the ticket creation flow."""

    IDLE = "idle"
    NAME = "ticket_name"
    PHONE = "ticket_phone"
    EMAIL = "ticket_email"
    ISSUE = "ticket_issue"
    CONFIRM = "ticket_confirm"
    CREATING = "creating_ticket"
    RETRY_CONFIRM = "ticket_retry_confirm"


# =============================================================================
# Go-back intent detection
# =============================================================================

_GO_BACK_PATTERN = re.compile(
    r"(go back|previous|change my (name|phone|email)|wait.*(wrong|mistake)|correct.*(name|phone|email))",
    re.IGNORECASE,
)

# =============================================================================
# Yes / No detection (same sets as agent.py)
# =============================================================================

_YES_SET = {"yes", "y", "confirm", "confirmed", "correct", "go ahead", "please do"}
_NO_SET = {"no", "n", "cancel", "stop", "not now"}


def _is_yes(text: str) -> bool:
    return (text or "").strip().lower() in _YES_SET


def _is_no(text: str) -> bool:
    return (text or "").strip().lower() in _NO_SET


# =============================================================================
# TicketFlow class
# =============================================================================


class TicketFlow:
    """
    Encapsulates the support ticket creation state machine.

    All external dependencies are injected via constructor callbacks so the
    class is independently testable without importing agent.py.
    """

    def __init__(
        self,
        *,
        say_fn: Callable[[str], Awaitable[None]],
        suppress_llm_fn: Callable[[float], None],
        room_log_fn: Callable[..., None],
        set_ui_state_fn: Callable[[str], Awaitable[None]],
        snooze_silence_fn: Callable[[float], None],
        create_ticket_fn: Callable[[str, str, str, str], Awaitable[str]],
        timeout_seconds: float = 45.0,
    ) -> None:
        # Callbacks
        self._say = say_fn
        self._suppress_llm = suppress_llm_fn
        self._room_log = room_log_fn
        self._set_ui_state = set_ui_state_fn
        self._snooze_silence = snooze_silence_fn
        self._create_ticket = create_ticket_fn

        # Configuration
        self.timeout_seconds = timeout_seconds

        # Collected fields
        self.name: str = ""
        self.phone: str = ""
        self.email: str = ""
        self.issue: str = ""

        # Internal state
        self._state: TicketFlowState = TicketFlowState.IDLE
        self._timeout_task: Optional[asyncio.Task] = None
        self._timeout_count: int = 0  # 0 = no timeout fired, 1 = first nudge sent

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> TicketFlowState:
        return self._state

    @property
    def state_name(self) -> str:
        """Return string for SessionState sync."""
        return self._state.value

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Begin the ticket flow ÔÇö sets state to NAME and speaks the first prompt."""
        self._reset_fields()
        self._state = TicketFlowState.NAME
        self._suppress_llm(15.0)
        self._room_log("TICKET_FLOW_START", state=self.state_name)
        await self._say(
            "I can help you with that. First, could you please tell me your full name?"
        )
        self._start_timeout()

    async def handle_input(self, user_text: str) -> None:
        """
        Single entry point for user speech during the ticket flow.
        Dispatches to per-state handlers after checking for go-back intent.
        """
        # Cancel any active timeout on user input
        self._cancel_timeout()
        self._timeout_count = 0

        # Check for go-back intent BEFORE field-specific parsing
        if self._state not in (TicketFlowState.IDLE, TicketFlowState.NAME, TicketFlowState.CREATING):
            if await self._handle_go_back(user_text):
                return

        # Dispatch to per-state handler
        handler = {
            TicketFlowState.NAME: self._handle_name,
            TicketFlowState.PHONE: self._handle_phone,
            TicketFlowState.EMAIL: self._handle_email,
            TicketFlowState.ISSUE: self._handle_issue,
            TicketFlowState.CONFIRM: self._handle_confirm,
            TicketFlowState.CREATING: self._handle_creating,
            TicketFlowState.RETRY_CONFIRM: self._handle_retry_confirm,
        }.get(self._state)

        if handler:
            await handler(user_text)

    # -------------------------------------------------------------------------
    # State handlers
    # -------------------------------------------------------------------------

    async def _handle_name(self, user_text: str) -> None:
        """NAME state: store name, advance to PHONE."""
        self.name = user_text.strip()
        self._state = TicketFlowState.PHONE
        self._suppress_llm(10.0)
        self._room_log("TICKET_FIELD_COLLECTED", field="name", value=self.name, state=self.state_name)
        await self._say("Thanks. Please share your phone number.")
        self._start_timeout()

    async def _handle_phone(self, user_text: str) -> None:
        """PHONE state: extract digits, validate 10+, advance to EMAIL."""
        digits = re.sub(r"\D", "", user_text)
        if len(digits) >= 10:
            # Use last 10 digits
            self.phone = digits[-10:]
            self._state = TicketFlowState.EMAIL
            self._suppress_llm(10.0)
            self._room_log(
                "TICKET_FIELD_COLLECTED", field="phone", value=self.phone, state=self.state_name
            )
            await self._say("Got it. Now please share your email address.")
            self._start_timeout()
        else:
            self._suppress_llm(10.0)
            self._room_log("TICKET_FIELD_INVALID", field="phone", raw=user_text, state=self.state_name)
            await self._say("Please share a valid phone number with at least 10 digits.")
            self._start_timeout()

    async def _handle_email(self, user_text: str) -> None:
        """EMAIL state: clean voice artifacts, validate, advance to ISSUE."""
        cleaned = clean_email(user_text)
        if validate_email(cleaned):
            self.email = cleaned
            self._state = TicketFlowState.ISSUE
            self._suppress_llm(10.0)
            self._room_log(
                "TICKET_FIELD_COLLECTED", field="email", value=self.email, state=self.state_name
            )
            await self._say(
                "Thank you. Finally, please describe the issue in one or two sentences."
            )
            self._start_timeout()
        else:
            self._suppress_llm(10.0)
            self._room_log(
                "TICKET_FIELD_INVALID", field="email", raw=user_text, cleaned=cleaned, state=self.state_name
            )
            await self._say("That doesn't look like a valid email address. Please try again.")
            self._start_timeout()

    async def _handle_issue(self, user_text: str) -> None:
        """ISSUE state: store issue, advance to CONFIRM with full readback."""
        self.issue = user_text.strip()
        self._state = TicketFlowState.CONFIRM
        self._suppress_llm(15.0)
        self._room_log(
            "TICKET_FIELD_COLLECTED", field="issue", value=self.issue, state=self.state_name
        )
        await self._say(self._build_confirmation_text())
        self._start_timeout()

    async def _handle_confirm(self, user_text: str) -> None:
        """CONFIRM state: yes ÔåÆ create, no ÔåÆ cancel, else ÔåÆ re-read and ask again."""
        if _is_yes(user_text):
            self._state = TicketFlowState.CREATING
            self._suppress_llm(15.0)
            self._snooze_silence(10.0)
            self._room_log("TICKET_CONFIRM_YES", state=self.state_name)
            await self._set_ui_state("thinking")
            await self._say("Thanks. Creating your support ticket now.")
            await self._do_create_ticket()
        elif _is_no(user_text):
            self._room_log("TICKET_CONFIRM_NO", state=self.state_name)
            await self._say("No problem. I have cancelled the ticket request.")
            self._reset_to_idle()
        else:
            # Ambiguous response ÔÇö re-read all details and ask again
            self._suppress_llm(15.0)
            self._room_log("TICKET_CONFIRM_AMBIGUOUS", text=user_text, state=self.state_name)
            await self._say(self._build_confirmation_text())
            self._start_timeout()

    async def _handle_creating(self, user_text: str) -> None:
        """CREATING state: should not normally receive input, but handle gracefully."""
        self._suppress_llm(10.0)
        await self._say("I'm still creating your ticket. Please wait a moment.")

    async def _handle_retry_confirm(self, user_text: str) -> None:
        """RETRY_CONFIRM state: yes ÔåÆ retry, no ÔåÆ reset to IDLE."""
        if _is_yes(user_text):
            self._state = TicketFlowState.CREATING
            self._suppress_llm(15.0)
            self._snooze_silence(10.0)
            self._room_log("TICKET_RETRY_YES", state=self.state_name)
            await self._set_ui_state("thinking")
            await self._say("Trying again now.")
            await self._do_create_ticket()
        elif _is_no(user_text):
            self._room_log("TICKET_RETRY_NO", state=self.state_name)
            await self._say(
                "No problem. I've cancelled the ticket request. You can start again anytime."
            )
            self._reset_to_idle()
        else:
            self._suppress_llm(10.0)
            self._room_log("TICKET_RETRY_AMBIGUOUS", text=user_text, state=self.state_name)
            await self._say(
                "Would you like me to try creating the ticket again? Please say yes or no."
            )
            self._start_timeout()

    # -------------------------------------------------------------------------
    # Ticket creation
    # -------------------------------------------------------------------------

    async def _do_create_ticket(self) -> None:
        """Call create_ticket_fn; on success reset, on failure offer retry."""
        try:
            result = await self._create_ticket(self.name, self.phone, self.email, self.issue)
            self._room_log("TICKET_CREATED", result=result, state=self.state_name)
            await self._say(result)
            self._reset_to_idle()
        except Exception as exc:
            logger.error("Ticket creation failed: %s", exc, exc_info=True)
            self._state = TicketFlowState.RETRY_CONFIRM
            self._suppress_llm(10.0)
            self._room_log("TICKET_CREATE_FAILED", error=str(exc), state=self.state_name)
            await self._say(
                "Sorry, there was an error creating the ticket. Would you like me to try again?"
            )
            self._start_timeout()

    # -------------------------------------------------------------------------
    # Go-back handling
    # -------------------------------------------------------------------------

    async def _handle_go_back(self, user_text: str) -> bool:
        """
        Detect go-back intent and transition to the appropriate previous state.
        Returns True if go-back was handled, False otherwise.
        """
        match = _GO_BACK_PATTERN.search(user_text.lower())
        if not match:
            return False

        matched_text = match.group(0).lower()

        # Determine target state based on keywords
        target_state = self._determine_go_back_target(matched_text)

        if target_state and target_state != self._state:
            self._state = target_state
            self._suppress_llm(10.0)
            self._room_log(
                "TICKET_GO_BACK", target=self.state_name, trigger=matched_text
            )
            # Speak confirmation and re-prompt for the target field
            await self._speak_go_back_prompt(target_state)
            return True

        return False

    def _determine_go_back_target(self, matched_text: str) -> Optional[TicketFlowState]:
        """Determine which state to go back to based on the matched text."""
        # Explicit field mentions
        if "name" in matched_text:
            return TicketFlowState.NAME
        if "phone" in matched_text:
            return TicketFlowState.PHONE
        if "email" in matched_text:
            return TicketFlowState.EMAIL

        # Generic "go back" / "previous" ÔåÆ go to the previous state
        state_order = [
            TicketFlowState.NAME,
            TicketFlowState.PHONE,
            TicketFlowState.EMAIL,
            TicketFlowState.ISSUE,
            TicketFlowState.CONFIRM,
        ]
        try:
            current_idx = state_order.index(self._state)
            if current_idx > 0:
                return state_order[current_idx - 1]
        except ValueError:
            pass

        return None

    async def _speak_go_back_prompt(self, target_state: TicketFlowState) -> None:
        """Speak the appropriate prompt for the state we're going back to."""
        prompts = {
            TicketFlowState.NAME: "No problem. Please tell me your full name again.",
            TicketFlowState.PHONE: "Sure. Please share your phone number again.",
            TicketFlowState.EMAIL: "Of course. Please share your email address again.",
            TicketFlowState.ISSUE: "Alright. Please describe the issue again in one or two sentences.",
        }
        prompt = prompts.get(target_state, "Please go ahead.")
        await self._say(prompt)
        self._start_timeout()

    # -------------------------------------------------------------------------
    # Timeout handling
    # -------------------------------------------------------------------------

    def _start_timeout(self) -> None:
        """Start (or restart) the inactivity timeout timer."""
        self._cancel_timeout()
        self._timeout_task = asyncio.ensure_future(self._timeout_handler())

    def _cancel_timeout(self) -> None:
        """Cancel the active timeout task if any."""
        if self._timeout_task is not None and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

    async def _timeout_handler(self) -> None:
        """Handle inactivity timeout ÔÇö nudge once, then cancel on second expiry."""
        try:
            await asyncio.sleep(self.timeout_seconds)

            if self._timeout_count == 0:
                # First timeout: nudge the user
                self._timeout_count = 1
                self._room_log("TICKET_TIMEOUT_NUDGE", state=self.state_name)
                await self._say(
                    "Are you still there? If you'd like to continue with the support ticket, please go ahead."
                )
                # Start second timer
                self._timeout_task = asyncio.ensure_future(self._second_timeout_handler())
            else:
                # Should not reach here, but handle gracefully
                await self._timeout_cancel_flow()
        except asyncio.CancelledError:
            pass

    async def _second_timeout_handler(self) -> None:
        """Handle second timeout expiry ÔÇö cancel the flow."""
        try:
            await asyncio.sleep(self.timeout_seconds)
            await self._timeout_cancel_flow()
        except asyncio.CancelledError:
            pass

    async def _timeout_cancel_flow(self) -> None:
        """Cancel the ticket flow due to inactivity."""
        self._room_log("TICKET_TIMEOUT_CANCEL", state=self.state_name)
        await self._say(
            "It seems you're no longer there. I'll cancel the ticket request. "
            "You can start again anytime."
        )
        self._reset_to_idle()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_confirmation_text(self) -> str:
        """Build the full confirmation readback including all four fields."""
        return (
            f"I have your details: name {self.name}, phone {self.phone}, "
            f"email {self.email}, and your issue is: {self.issue}. "
            "Should I create the support ticket now?"
        )

    def _reset_to_idle(self) -> None:
        """Reset the flow to IDLE state."""
        self._cancel_timeout()
        self._state = TicketFlowState.IDLE
        self._timeout_count = 0
        self._room_log("TICKET_FLOW_RESET", state=self.state_name)

    def _reset_fields(self) -> None:
        """Clear all collected fields."""
        self.name = ""
        self.phone = ""
        self.email = ""
        self.issue = ""
        self._timeout_count = 0
