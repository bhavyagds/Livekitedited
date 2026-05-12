# src/agents/flows/greeting_flow.py
import asyncio
from src.agents.flows.base import FlowContext
from src.agents.prompts import get_greeting

async def handle(ctx: FlowContext) -> bool:
    """Send the initial greeting message."""
    greeting = get_greeting(ctx.lang)
    await ctx.say(greeting, allow_interruptions=True)
    return True
