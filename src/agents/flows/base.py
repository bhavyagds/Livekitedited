# src/agents/flows/base.py
import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Any, Optional
from livekit.agents.pipeline import VoicePipelineAgent

@dataclass
class FlowContext:
    state: Any                    # SessionState
    agent: VoicePipelineAgent
    suppress_llm: Callable[[float], None]
    snooze_silence: Callable[[float], None]
    set_ui_state: Callable[[str], Awaitable[None]]
    room_log: Callable[[str, Any], None]
    should_suppress_clarification: Callable[[str, float], bool]
    cancel_thinking_task: Callable[[], None]
    lang: str = "en"

    def handled(self) -> bool:
        """Mark this turn as handled by a deterministic flow to suppress the LLM."""
        if hasattr(self.state, "deterministic_replied"):
            self.state.deterministic_replied = True
        return True
