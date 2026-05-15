# src/agents/flows/order_flow.py
import asyncio
import re
from typing import Optional
from src.agents.flows.base import FlowContext
from src.agents.tools import order_lookup

# Spoken digit words → numeric characters
_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Stopwords that should not be converted when parsing an order number
_ORDER_STOPWORDS = {
    "my", "is", "number", "order", "the", "a", "an", "it", "with",
    "please", "check", "hi", "i", "have", "got", "given",
}

def _normalize_order_id_strict(raw_text: str) -> Optional[str]:
    """Extract and normalize order IDs, supporting both numeric and spoken-word digits.
    
    Handles:
    - Pure digits: '12345' -> '12345'
    - Spoken words: 'one two three four five six' -> '123456'
    - Mixed: 'order 1-2-three' -> '123'
    - With prefix: '#1234', 'order number 1234'
    """
    text = (raw_text or "").lower()
    # Strip common prefixes
    text = re.sub(r"\b(order|number|#)\b", " ", text)
    
    # First: try to extract a pure digit sequence (most reliable)
    digit_matches = re.findall(r"\d{3,}", text)  # at least 3 digits in a row
    if digit_matches:
        return max(digit_matches, key=len)  # return the longest match
    
    # Second: convert spoken digit words to numerals
    tokens = re.findall(r"[a-z0-9]+", text)
    parts: list[str] = []
    for token in tokens:
        if token in _ORDER_STOPWORDS:
            continue  # skip non-digit words
        if token in _WORD_TO_DIGIT:
            parts.append(_WORD_TO_DIGIT[token])
        elif token.isdigit():
            parts.append(token)
        else:
            # If we hit a real word that's not a digit word, stop collecting
            if parts:
                break
    
    if len(parts) >= 3:  # require at least 3 digits to be a valid order number
        return "".join(parts)
    
    return None

def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    """Extract and normalize phone numbers for lookup."""
    digits = re.sub(r"\D", "", raw_text)
    if not digits:
        return None
    if len(digits) >= 10:
        return digits[-10:]
    return None

def _mentions_no_order_number(text: str) -> bool:
    """Check if user explicitly says they don't have the order number."""
    return bool(re.search(r"\b(don't have|do not have|no order number|don't know|no idea)\b", text.lower()))

def _mentions_phone_lookup_intent(text: str) -> bool:
    """Check if user wants to use phone number for lookup."""
    return bool(re.search(r"\b(phone number|mobile number|telephone)\b", text.lower()))

def _is_order_relevant(text: str) -> bool:
    """Check if the user turn is about the order at all."""
    return bool(re.search(r"(order|check|status|track|package|where is)", text.lower()))

async def _run_order_lookup(ctx: FlowContext, order_number: str):
    """Run the order lookup tool and speak the result."""
    state = ctx.state
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_order"
    # Keep silence monitor snoozed for the full duration of the API call
    ctx.snooze_silence(45.0)
    ctx.suppress_llm(45.0)
    ctx.room_log("ORDER_LOOKUP_STARTED", order_number=order_number)
    try:
        result = await order_lookup.lookup_order(order_number)
        ctx.room_log("ORDER_LOOKUP_RESULT", result=result)
        await ctx.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_order_number = order_number
    finally:
        state.lookup_inflight = False
        # Reset snooze after lookup finishes so silence monitor can resume normally
        ctx.snooze_silence(8.0)  # short grace period after speaking result

async def _run_phone_lookup(ctx: FlowContext, phone_number: str):
    """Run the phone-based order lookup tool and speak the result."""
    state = ctx.state
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_phone"
    # Keep silence monitor snoozed for the full duration of the API call
    ctx.snooze_silence(45.0)
    ctx.suppress_llm(45.0)
    ctx.room_log("PHONE_LOOKUP_STARTED", phone_number=phone_number)
    try:
        result = await order_lookup.lookup_order_by_phone(phone_number)
        ctx.room_log("PHONE_LOOKUP_RESULT", result=result)
        await ctx.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_phone_number = phone_number
    finally:
        state.lookup_inflight = False
        ctx.snooze_silence(8.0)  # short grace period after speaking result

async def handle(ctx: FlowContext, user_text: str) -> bool:
    """Handle the order lookup flow."""
    state = ctx.state
    
    if state.support_state not in {"awaiting_order", "checking_order"}:
        # Also check for general support intent to enter this flow
        support_intent = bool(re.search(r"(problem|issue|complaint|order problem|wrong order|late order|my order)", user_text.lower()))
        if support_intent and state.support_state == "idle":
            state.support_state = "awaiting_order"
            ctx.room_log("FLOW_TRANSITION", from_state="idle", to_state="awaiting_order", reason="support_intent")
            
            # Check if an Order ID or Phone was already provided in this same sentence.
            order_id = _normalize_order_id_strict(user_text)
            if order_id:
                ctx.suppress_llm(15.0)
                await ctx.set_ui_state("thinking")
                ctx.snooze_silence(15.0)
                asyncio.create_task(_run_order_lookup(ctx, order_id))
                return ctx.handled()
            
            phone = _normalize_phone_for_lookup(user_text)
            if phone:
                state.support_state = "awaiting_phone"
                ctx.suppress_llm(15.0)
                await ctx.set_ui_state("thinking")
                ctx.snooze_silence(15.0)
                asyncio.create_task(_run_phone_lookup(ctx, phone))
                return ctx.handled()
            
            # Let the LLM handle the first response (asking for order number)
            return False
        return False

    # 1) Busy state
    if state.lookup_inflight:
        await ctx.set_ui_state("thinking")
        ctx.snooze_silence(8.0)
        asyncio.create_task(ctx.say("I am still checking that now. One moment please.", allow_interruptions=True))
        return ctx.handled()

    # 2) Process input
    if _mentions_no_order_number(user_text) or _mentions_phone_lookup_intent(user_text):
        state.support_state = "awaiting_phone"
        ctx.room_log("FLOW_TRANSITION", from_state="awaiting_order", to_state="awaiting_phone", reason="no_order_or_phone_intent")
        return False # Let LLM handle transition

    # Phone escape
    phone_candidate = _normalize_phone_for_lookup(user_text)
    if phone_candidate:
        state.support_state = "awaiting_phone"
        ctx.suppress_llm(15.0)
        await ctx.set_ui_state("thinking")
        ctx.snooze_silence(15.0)
        asyncio.create_task(_run_phone_lookup(ctx, phone_candidate))
        return ctx.handled()

    order_id = _normalize_order_id_strict(user_text)
    if order_id:
        ctx.suppress_llm(15.0)
        await ctx.set_ui_state("thinking")
        ctx.snooze_silence(15.0)
        asyncio.create_task(_run_order_lookup(ctx, order_id))
        return ctx.handled()

    # Remove deterministic clarification to avoid double-response with LLM.
    # LLM will naturally ask for the order number if it sees the support intent.
    return False
