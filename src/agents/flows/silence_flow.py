# src/agents/flows/silence_flow.py
import asyncio
import time
from src.agents.flows.base import FlowContext

def get_contextual_silence_prompt(ctx: FlowContext) -> str:
    """Choose a prompt based on the current support state."""
    state = ctx.state
    if state.support_state in {"awaiting_order", "checking_order"}:
        if ctx.lang == "el":
            return "Είστε ακόμα εκεί; Χρειάζομαι τον αριθμό της παραγγελίας σας για να ελέγξω την κατάσταση."
        return "Are you still there? I need your order number to check the status."
    
    if state.support_state in {"awaiting_phone", "checking_phone"}:
        if ctx.lang == "el":
            return "Παρακαλώ δώστε τον αριθμό τηλεφώνου σας για να βρω την παραγγελία."
        return "Please provide your phone number so I can find your order."
        
    if state.support_state.startswith("ticket_"):
        if ctx.lang == "el":
            return "Θα θέλατε να συνεχίσετε με το αίτημα υποστήριξης;"
        return "Would you like to continue with the support ticket?"
        
    if ctx.lang == "el":
        return "Είστε ακόμα εκεί; Πώς μπορώ να σας βοηθήσω;"
    return "Are you still there? How can I help you further?"

async def monitor_iteration(ctx: FlowContext) -> bool:
    """One iteration of the silence monitor. Returns True if should break/stop."""
    state = ctx.state
    
    if not state.silence_enabled or not state.waiting_for_user or state.lookup_inflight:
        return False
        
    now = time.time()
    if now < state.silence_snooze_until:
        return False
        
    # Disconnect if timeout reached
    if (now - state.last_user_activity) > state.silence_timeout_s and (now - state.last_agent_activity) > state.silence_timeout_s:
        # If max_prompts is 0 or we've reached the limit, disconnect.
        if state.silence_max_prompts <= 0 or state.silence_prompt_count >= state.silence_max_prompts:
            ctx.room_log("SILENCE_TERMINATION", count=state.silence_prompt_count)
            state.silence_enabled = False # Stop further monitor checks
            
            message = "I haven't heard from you. I will end the call now. Goodbye!"
            if ctx.lang == "el":
                message = "Δεν σας ακούω. Θα κλείσω την κλήση τώρα. Γεια σας!"
                
            await ctx.say(message, allow_interruptions=True)
            await asyncio.sleep(5.0) # Wait for audio to reach user
            state.should_end = True
            state.disconnect_reason = "silence_termination"
            return True

        text = get_contextual_silence_prompt(ctx)
        state.silence_prompt_count += 1
        # Snooze for 15s to allow the agent to finish speaking and the user to react.
        state.silence_snooze_until = time.time() + 15.0
        ctx.room_log("SILENCE_PROMPT", count=state.silence_prompt_count, text=text)
        await ctx.say(text, allow_interruptions=True)
        
    return False
