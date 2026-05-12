# src/agents/flows/termination.py
import asyncio
import re
from src.agents.flows.base import FlowContext

async def handle(ctx: FlowContext, user_text: str) -> bool:
    """Check for user repeating the same sentence and disconnect if needed."""
    state = ctx.state
    
    # Normalize: remove non-alphanumeric and convert to lowercase
    # Supports both English and Greek characters
    norm_text = re.sub(r"[^a-z0-9α-ωά-ώ]", "", user_text.lower())
    
    if norm_text and norm_text == state.last_user_text_norm:
        state.user_repetition_count += 1
        if state.user_repetition_count >= 1: # 1 repetition = 2 times total
            ctx.room_log("REPETITION_TERMINATION", text=user_text)
            state.silence_enabled = False # Stop silence monitor immediately
            
            ctx.suppress_llm(15.0)
            
            message = "I've heard that already. I will end the call now. Goodbye!"
            if ctx.lang == "el":
                message = "Το έχω ακούσει ήδη αυτό. Θα κλείσω την κλήση τώρα. Γεια σας!"
                
            asyncio.create_task(ctx.agent.say(message, allow_interruptions=True))
            
            async def _end_rep():
                await asyncio.sleep(5.0) # Wait for agent to start/finish speaking
                state.should_end = True
                state.disconnect_reason = "repetition_termination"
            
            asyncio.create_task(_end_rep())
            return ctx.handled()
    else:
        state.last_user_text_norm = norm_text
        state.user_repetition_count = 0
        
    return False
