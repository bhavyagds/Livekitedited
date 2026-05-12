# src/agents/flows/phone_flow.py
import asyncio
import re
from typing import Optional
from src.agents.flows.base import FlowContext
from src.agents.tools import order_lookup

def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    """Extract and normalize phone numbers for lookup."""
    digits = re.sub(r"\D", "", raw_text)
    if not digits:
        return None
    if len(digits) >= 10:
        return digits[-10:]
    return None

def _normalize_order_id_strict(raw_text: str) -> Optional[str]:
    """Extract and normalize order IDs."""
    text = raw_text.lower().replace("#", "").replace("order", "").strip()
    digits = re.findall(r"\d+", text)
    if digits:
        return digits[0]
    return None

def _mentions_phone_lookup_intent(text: str) -> bool:
    """Check if user wants to use phone number."""
    return bool(re.search(r"\b(phone number|mobile number|telephone)\b", text.lower()))

def _is_order_relevant(text: str) -> bool:
    """Check if the user turn is about the order."""
    return bool(re.search(r"(order|check|status|track|package|where is)", text.lower()))

async def _run_order_lookup(ctx: FlowContext, order_number: str):
    """Run order lookup."""
    state = ctx.state
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_order"
    ctx.room_log("ORDER_LOOKUP_STARTED", order_number=order_number)
    try:
        result = await order_lookup.lookup_order(order_number)
        ctx.room_log("ORDER_LOOKUP_RESULT", result=result)
        await ctx.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_order_number = order_number
    finally:
        state.lookup_inflight = False

async def _run_phone_lookup(ctx: FlowContext, phone_number: str):
    """Run phone lookup."""
    state = ctx.state
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_phone"
    ctx.room_log("PHONE_LOOKUP_STARTED", phone_number=phone_number)
    try:
        result = await order_lookup.lookup_order_by_phone(phone_number)
        ctx.room_log("PHONE_LOOKUP_RESULT", result=result)
        await ctx.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_phone_number = phone_number
    finally:
        state.lookup_inflight = False

async def handle(ctx: FlowContext, user_text: str) -> bool:
    """Handle the phone lookup flow."""
    state = ctx.state
    
    if state.support_state not in {"awaiting_phone", "checking_phone"}:
        return False

    # 1) Busy state
    if state.lookup_inflight:
        await ctx.set_ui_state("thinking")
        ctx.snooze_silence(8.0)
        asyncio.create_task(ctx.say("I am still checking that now. One moment please.", allow_interruptions=True))
        return ctx.handled()

    # 2) Process input
    if _mentions_phone_lookup_intent(user_text):
        prompt = "Sure. Please provide the full phone number used for the order."
        if not ctx.should_suppress_clarification(prompt, 5.0):
            ctx.suppress_llm(10.0)
            asyncio.create_task(ctx.say(prompt, allow_interruptions=True))
        return ctx.handled()

    phone = _normalize_phone_for_lookup(user_text)
    if phone:
        ctx.suppress_llm(15.0)
        await ctx.set_ui_state("thinking")
        ctx.snooze_silence(15.0)
        asyncio.create_task(_run_phone_lookup(ctx, phone))
        return ctx.handled()

    # Order ID escape
    order_id_escape = _normalize_order_id_strict(user_text)
    if order_id_escape:
        ctx.room_log("FLOW_TRANSITION", from_state="awaiting_phone", to_state="awaiting_order", reason="order_id_given_in_phone_flow")
        state.support_state = "awaiting_order"
        ctx.suppress_llm(15.0)
        await ctx.set_ui_state("thinking")
        ctx.snooze_silence(15.0)
        asyncio.create_task(_run_order_lookup(ctx, order_id_escape))
        return ctx.handled()

    if _is_order_relevant(user_text):
        prompt = "I need the full phone number to check the order. Please share it once."
        if not ctx.should_suppress_clarification(prompt, 5.0):
            ctx.suppress_llm(10.0)
            asyncio.create_task(ctx.say(prompt, allow_interruptions=True))
        return ctx.handled()

    return False
