# src/agents/flows/ticket_flow.py
import asyncio
import re
from typing import Optional
from livekit.agents.pipeline import VoicePipelineAgent
from src.agents.flows.base import FlowContext
from src.agents.tools import support_ticket

def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    """Extract and normalize phone numbers for lookup."""
    digits = re.sub(r"\D", "", raw_text)
    if not digits:
        return None
    if len(digits) >= 10:
        return digits[-10:]
    return None

def _looks_like_email(text: str) -> bool:
    """Basic sanity check for email intent."""
    return "@" in text and "." in text

def _is_yes(text: str) -> bool:
    """Check for affirmative user responses."""
    return bool(re.search(r"\b(yes|yeah|sure|okay|ok|correct|right|yep|yes please|do it|create)\b", text.lower()))

def _is_no(text: str) -> bool:
    """Check for negative user responses."""
    return bool(re.search(r"\b(no|nay|nope|cancel|stop|don't|incorrect|wrong|wait)\b", text.lower()))

async def _run_create_ticket(ctx: FlowContext):
    """Call the support ticket tool and speak the result."""
    state = ctx.state
    if state.ticket_inflight:
        return
    state.ticket_inflight = True
    state.support_state = "creating_ticket"
    ctx.room_log("TICKET_CREATE_STARTED")
    try:
        result = await support_ticket.create_support_ticket(
            state.ticket_name or "Customer",
            state.ticket_phone,
            state.ticket_email,
            state.ticket_issue,
        )
        ctx.room_log("TICKET_CREATE_RESULT", result=result)
        await ctx.agent.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.ticket_name = ""
        state.ticket_phone = ""
        state.ticket_email = ""
        state.ticket_issue = ""
    finally:
        state.ticket_inflight = False

async def handle(ctx: FlowContext, user_text: str) -> bool:
    """Handle the support ticket creation flow."""
    state = ctx.state
    user_text_lower = user_text.lower()
    
    # 1) Escape: user wants to raise a ticket (from any state)
    ticket_intent = bool(re.search(r"\b(human|representative|call me|callback|support ticket|open ticket|create ticket|raise.*ticket|open.*ticket|make.*ticket|want.*ticket|need.*ticket|complaint)\b", user_text_lower))
    
    _in_ticket_flow = (state.support_state or "").startswith("ticket_") or state.support_state == "creating_ticket"
    
    if ticket_intent and not _in_ticket_flow:
        ctx.room_log("FLOW_TRANSITION", from_state=state.support_state, to_state="ticket_name", reason="ticket_escape")
        state.support_state = "ticket_name"
        ctx.suppress_llm(15.0)
        ctx.cancel_thinking_task()
        asyncio.create_task(ctx.agent.say(
            "I can help you with that. First, could you please tell me your full name?",
            allow_interruptions=True
        ))
        return ctx.handled()

    # 2) Sub-flow processing
    if not state.support_state.startswith("ticket_"):
        return False

    # Stronger LLM suppression while we are in a deterministic sub-flow
    ctx.suppress_llm(15.0)
    ctx.cancel_thinking_task()

    if state.support_state == "ticket_name":
        state.ticket_name = user_text
        state.support_state = "ticket_phone"
        ctx.suppress_llm(10.0)
        asyncio.create_task(ctx.agent.say("Thanks. Please share your phone number.", allow_interruptions=True))
        return ctx.handled()

    if state.support_state == "ticket_phone":
        ticket_phone = _normalize_phone_for_lookup(user_text)
        if not ticket_phone:
            prompt = "Please share a valid phone number."
            if not ctx.should_suppress_clarification(prompt, 5.0):
                ctx.suppress_llm(10.0)
                asyncio.create_task(ctx.agent.say(prompt, allow_interruptions=True))
            return ctx.handled()
        state.ticket_phone = ticket_phone
        state.support_state = "ticket_email"
        ctx.suppress_llm(10.0)
        asyncio.create_task(ctx.agent.say("Got it. Now please share your email address.", allow_interruptions=True))
        return ctx.handled()

    if state.support_state == "ticket_email":
        # Clean email from voice artifacts
        cleaned_email = user_text_lower.strip()
        cleaned_email = re.sub(r'\s+at\s+', '@', cleaned_email)
        cleaned_email = re.sub(r'\s+dot\s+', '.', cleaned_email)
        cleaned_email = cleaned_email.replace(" ", "")
        
        if not _looks_like_email(cleaned_email):
            prompt = "Please share a valid email address."
            if not ctx.should_suppress_clarification(prompt, 5.0):
                ctx.suppress_llm(10.0)
                asyncio.create_task(ctx.agent.say(prompt, allow_interruptions=True))
            return ctx.handled()
        state.ticket_email = cleaned_email
        state.support_state = "ticket_issue"
        ctx.suppress_llm(10.0)
        asyncio.create_task(ctx.agent.say("Thank you. Finally, please describe the issue in one or two sentences.", allow_interruptions=True))
        return ctx.handled()

    if state.support_state == "ticket_issue":
        state.ticket_issue = user_text
        state.support_state = "ticket_confirm"
        confirm_text = (
            f"I have your details as name {state.ticket_name}, phone {state.ticket_phone}, and email {state.ticket_email}. "
            "Should I create the support ticket now?"
        )
        ctx.suppress_llm(10.0)
        asyncio.create_task(ctx.agent.say(confirm_text, allow_interruptions=True))
        return ctx.handled()

    if state.support_state == "ticket_confirm":
        if _is_yes(user_text):
            ctx.suppress_llm(15.0)
            asyncio.create_task(ctx.set_ui_state("thinking"))
            ctx.snooze_silence(10.0)
            asyncio.create_task(ctx.agent.say("Thanks. Creating your support ticket now.", allow_interruptions=True))
            asyncio.create_task(_run_create_ticket(ctx))
            return ctx.handled()
        if _is_no(user_text):
            state.support_state = "idle"
            state.ticket_name = ""
            state.ticket_phone = ""
            state.ticket_email = ""
            state.ticket_issue = ""
            ctx.suppress_llm(10.0)
            asyncio.create_task(ctx.agent.say("No problem. I have cancelled the ticket request.", allow_interruptions=True))
            return ctx.handled()
        
        prompt = "Please say yes to create the ticket, or no to cancel."
        ctx.suppress_llm(10.0)
        asyncio.create_task(ctx.agent.say(prompt, allow_interruptions=True))
        return ctx.handled()

    return False
