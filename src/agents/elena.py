"""
Meallion Voice AI - Elena Voice Agent Router
Dispatches calls to language-specific agents (English or Greek)
based on the configured language setting.
"""

import logging
import asyncio
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from src.agents.prompts import get_agent_language
from src.config import settings

# Configure logging for the router
logger = logging.getLogger("src.agents.elena_router")

async def entrypoint(ctx: JobContext):
    """
    Router entrypoint: determine language and hand off to the specific agent.
    """
    # DO NOT wait for participant here; the sub-agents handle room connection and participants.
    # Hand off as quickly as possible to the correct language module.
    
    # Ensure settings are loaded before routing
    from src.agents.prompts import _fetch_from_db
    await _fetch_from_db()
    
    # Get the language from the database settings
    lang = get_agent_language()
    logger.info(f"Elena Router: dispatching call {ctx.job.id} to {lang.upper()} agent")
    
    if lang == "el":
        from src.agents.elena_el import entrypoint as el_entrypoint
        await el_entrypoint(ctx)
    else:
        from src.agents.elena_en import entrypoint as en_entrypoint
        await en_entrypoint(ctx)

def prewarm(proc: JobProcess):
    """
    Prewarm the specific agent process based on language.
    """
    # NOTE: We avoid full database fetch here to prevent connection pool issues
    # during process initialization. Each job will fetch its own settings.
    
    lang = get_agent_language()
    logger.info(f"Elena Router: prewarming {lang.upper()} agent process")
    
    if lang == "el":
        from src.agents.elena_el import prewarm as el_prewarm
        el_prewarm(proc)
    else:
        from src.agents.elena_en import prewarm as en_prewarm
        en_prewarm(proc)

def run_agent():
    """
    Run the Elena voice agent as a LiveKit worker.
    """
    # The worker itself doesn't need to be language-specific at boot time,
    # because the language can change in the database.
    # The entrypoint will handle the dynamic routing per job.
    
    import os
    
    # Configure logging
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info(f"Elena Router Worker starting (log_level={settings.log_level})")

    def _get_float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _get_int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    initialize_timeout = _get_float_env("LIVEKIT_AGENTS_INITIALIZE_TIMEOUT", 60.0)
    shutdown_timeout = _get_float_env("LIVEKIT_AGENTS_SHUTDOWN_TIMEOUT", 60.0)
    num_idle_processes = _get_int_env("LIVEKIT_AGENTS_NUM_IDLE_PROCESSES", 1)
    load_threshold = _get_float_env("LIVEKIT_AGENTS_LOAD_THRESHOLD", 0.90)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
            initialize_process_timeout=initialize_timeout,
            shutdown_process_timeout=shutdown_timeout,
            num_idle_processes=num_idle_processes,
            load_threshold=load_threshold,
        ),
    )

if __name__ == "__main__":
    run_agent()
