# src/agents/flows/order_flow.py
import asyncio
import re
from typing import Optional
from src.agents.flows.base import FlowContext
from src.agents.tools import order_lookup

def _normalize_order_id_strict(raw_text: str) -> Optional[str]:
    """Extract and normalize order IDs (e.g. #1234, 1234, order 1234)."""
    text = raw_text.lower().replace("#", "").replace("order", "").strip()
    digits = re.findall(r"\d+", text)
    if digits:
        return digits[0]
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
    ctx.room_log("ORDER_LOOKUP_STARTED", order_number=order_number)
    try:
        result = await order_lookup.lookup_order(order_number)
        ctx.room_log("ORDER_LOOKUP_RESULT", result=result)
        await ctx.agent.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_order_number = order_number
    finally:
        state.lookup_inflight = False

async def _run_phone_lookup(ctx: FlowContext, phone_number: str):
    """Run the phone-based order lookup tool and speak the result."""
    state = ctx.state
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_phone"
    ctx.room_log("PHONE_LOOKUP_STARTED", phone_number=phone_number)
    try:
        result = await order_lookup.lookup_order_by_phone(phone_number)
        ctx.room_log("PHONE_LOOKUP_RESULT", result=result)
        await ctx.agent.say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_phone_number = phone_number
    finally:
        state.lookup_inflight = False

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
        asyncio.create_task(ctx.agent.say("I am still checking that now. One moment please.", allow_interruptions=True))
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

    if _is_order_relevant(user_text):
        prompt = "Whenever you are ready, please share your order number. If you do not have it, say that and I will check by phone number."
        if not ctx.should_suppress_clarification(prompt, 5.0):
            ctx.suppress_llm(10.0)
            asyncio.create_task(ctx.agent.say(prompt, allow_interruptions=True))
        return ctx.handled()

    return False
