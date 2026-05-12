# src/agents/flows/farewell.py
import asyncio
import re
from src.agents.flows.base import FlowContext
from src.agents.prompts import get_closing

async def handle(ctx: FlowContext, user_text: str) -> bool:
    """Check for goodbye/farewell intent and handle graceful teardown."""
    state = ctx.state
    _t = user_text.lower().strip()
    
    # 1) Core farewell regex
    _farewell_intent = bool(re.search(
        r"\b(bye|by\b|goodbye|good bye|good night|see you|take care|"
        r"thanks? bye|that.?s all|no thank|nothing else|"
        r"i.?m done|all good|that will be all|have a good|have a great|"
        r"no more|no further|no other)\b",
        _t
    ))
    
    # 2) Composite detection for Greek if lang is el
    if ctx.lang == "el":
        _farewell_intent = _farewell_intent or bool(re.search(
            r"\b(γεια|αντίο|ευχαριστώ γεια|αυτά είναι όλα|τίποτα άλλο|τελείωσα|όλα καλά)\b",
            _t
        ))

    # 3) Composite: short sentences that combine "thank you" with a clear close signal
    if not _farewell_intent:
        _has_thanks = bool(re.search(r"\b(thank|ευχαριστώ)\b", _t))
        _has_close = bool(re.search(r"\b(no|okay|ok|alright|all right|done|that.?s it|enough|όχι|εντάξει|φτάνει)\b", _t))
        _is_short = len(_t.split()) <= 10
        if _has_thanks and _has_close and _is_short:
            _farewell_intent = True
            
    # 4) Shield: do not treat digit-heavy strings (order/phone numbers) as farewells.
    if _farewell_intent and len(re.findall(r"\d", user_text)) >= 3:
        _farewell_intent = False
        ctx.room_log("FAREWELL_SHIELDED", text=user_text)

    if _farewell_intent:
        ctx.room_log("FAREWELL_DETECTED", text=user_text)
        state.silence_enabled = False
        
        ctx.suppress_llm(15.0)
        
        goodbye_msg = get_closing(ctx.lang)
        asyncio.create_task(ctx.agent.say(goodbye_msg, allow_interruptions=True))
        
        async def _delayed_end_farewell():
            # Allow time for the goodbye message to play
            await asyncio.sleep(7.0 if ctx.lang == "el" else 5.0)
            state.should_end = True
            state.disconnect_reason = "farewell"
            
        asyncio.create_task(_delayed_end_farewell())
        return ctx.handled()
        
    return False
