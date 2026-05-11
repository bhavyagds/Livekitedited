"""
Meallion Voice AI - System Prompts for Elena
Loads ALL content from database for real-time updates without restarts.
No hardcoded instructions - everything comes from DB.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, Any, List

logger = logging.getLogger(__name__)

# Runtime language override (set per active call/session).
_runtime_language: Optional[str] = None


def _as_bool(value: object, default: bool = False) -> bool:
    """Safely coerce string/number/bool values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def set_runtime_language(language: Optional[str]) -> None:
    """Set the runtime language for the active call/session."""
    global _runtime_language
    if language:
        _runtime_language = str(language).strip().lower()
    else:
        _runtime_language = None


def get_runtime_language() -> Optional[str]:
    """Get the runtime language override if set."""
    return _runtime_language

# Cache for database content with TTL
_cache = {
    "kb_content": {},  # language -> content
    "prompts_content": {},  # language -> content
    "settings": {},  # key -> value
    "long_term_memory": "",  # formatted context string
    "last_fetch": 0,
    "ttl": 10,  # Refresh frequently
}
_defaults_initialized = False
_fetch_task: Optional[asyncio.Task] = None


async def _fetch_from_db(force: bool = False):
    """Fetch KB, Prompts, and Settings from database."""
    global _fetch_task
    
    if _fetch_task and not _fetch_task.done():
        await _fetch_task
        # If a forced refresh is requested while another fetch was in-flight,
        # run one more pass to guarantee fresh data.
        if force:
            _fetch_task = None
        else:
            return

    import time
    current_time = time.time()
    
    cache_populated = bool(_cache["kb_content"] or _cache["prompts_content"] or _cache["settings"])
    ttl_valid = (current_time - _cache["last_fetch"]) < _cache["ttl"]
    
    if cache_populated and ttl_valid and not force:
        return
    
    _fetch_task = asyncio.create_task(_actual_fetch_from_db(force))
    try:
        await _fetch_task
    except Exception as e:
        logger.error(f"Fetch task failed: {e}")
    finally:
        # Prevent stale completed task objects from blocking future fetch decisions.
        if _fetch_task and _fetch_task.done():
            _fetch_task = None


async def _actual_fetch_from_db(force: bool = False):
    """Internal implementation of database fetch."""
    import time
    fetch_start = time.time()
    current_time = fetch_start
    
    try:
        from src.services.database import get_database_service
        db = get_database_service()

        global _defaults_initialized
        if not _defaults_initialized:
            try:
                await db.init_default_settings()
                _defaults_initialized = True
            except Exception as e:
                logger.warning(f"Default settings init failed: {e}")
        
        # Fetch all in parallel
        kb_task = asyncio.create_task(db.get_all_kb_content())
        prompts_task = asyncio.create_task(db.get_all_prompts_content())
        settings_task = asyncio.create_task(db.get_all_settings())
        memory_task = asyncio.create_task(db.get_active_memory_context())
        
        kb_items, prompts_items, settings, memory_context = await asyncio.gather(
            kb_task, prompts_task, settings_task, memory_task,
            return_exceptions=True
        )
        
        if isinstance(kb_items, Exception):
            logger.warning(f"KB fetch failed: {kb_items}")
        else:
            for item in kb_items:
                _cache["kb_content"][item["language"]] = item["content"]
        
        if isinstance(prompts_items, Exception):
            logger.warning(f"Prompts fetch failed: {prompts_items}")
        else:
            for item in prompts_items:
                _cache["prompts_content"][item["language"]] = item["content"]
        
        if isinstance(settings, Exception):
            logger.warning(f"Settings fetch failed: {settings}")
        elif settings:
            _cache["settings"] = settings
            
        if isinstance(memory_context, Exception):
            logger.warning(f"Memory fetch failed: {memory_context}")
        else:
            _cache["long_term_memory"] = memory_context if isinstance(memory_context, str) else ""
            
        _cache["last_fetch"] = current_time
        logger.info(f"✅ DB Refresh: KB={len(_cache['kb_content'])}, Prompts={len(_cache['prompts_content'])}, Memory={len(_cache['long_term_memory'])} chars")
        
    except Exception as e:
        logger.error(f"❌ Database fetch failed: {e}")


def _sync_fetch_from_db():
    """Synchronous wrapper for database fetch."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if not _fetch_task or _fetch_task.done():
                asyncio.create_task(_fetch_from_db())
        else:
            loop.run_until_complete(_fetch_from_db())
    except Exception:
        try:
            asyncio.run(_fetch_from_db())
        except:
            pass


def load_knowledge_base(language: str = "el") -> str:
    if not _cache["kb_content"]: _sync_fetch_from_db()
    return _cache["kb_content"].get(language) or next(iter(_cache["kb_content"].values()), "")


async def load_knowledge_base_async(language: str = "el") -> str:
    await _fetch_from_db(force=not _cache["kb_content"])
    return _cache["kb_content"].get(language) or next(iter(_cache["kb_content"].values()), "")


def get_prompts_content(language: str = "el") -> Optional[str]:
    if not _cache["prompts_content"]: _sync_fetch_from_db()
    return _cache["prompts_content"].get(language) or next(iter(_cache["prompts_content"].values()), "")


async def get_prompts_content_async(language: str = "el") -> Optional[str]:
    await _fetch_from_db(force=not _cache["prompts_content"])
    return _cache["prompts_content"].get(language) or next(iter(_cache["prompts_content"].values()), "")


def get_agent_language() -> str:
    runtime_lang = get_runtime_language()
    if runtime_lang: return runtime_lang
    if not _cache["settings"]: _sync_fetch_from_db()
    return _cache["settings"].get("agent_language", "en")


def get_agent_setting(key: str, default: Any = None) -> Any:
    if not _cache["settings"]: _sync_fetch_from_db()
    return _cache["settings"].get(key, default)


def build_system_prompt(language: str = "el") -> str:
    kb_content = load_knowledge_base(language)
    prompts_content = get_prompts_content(language)
    memory_context = _cache.get("long_term_memory", "")
    
    parts = []
    parts.append(_get_response_language_instruction(language))
    
    # PRIORITY 1: LONG-TERM MEMORY (Specific scenarios from admin)
    if memory_context:
        parts.append(
            "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
            "When user intent matches any memory scenario, respond using that memory response first.\n"
            "Treat scenario matching as semantic/intention-based (not exact wording).\n"
            "If a memory entry gives an explicit response sentence, prefer that exact response text.\n\n"
            "INTERNAL INSTRUCTIONS WARNING:\n"
            "- The tags 'SCENARIO', 'KEY CONCEPTS', and 'GUIDELINE' are for your internal logic ONLY.\n"
            "- NEVER speak these tags or any meta-instructions (e.g., 'Never say...', 'Always say...', 'Avoid...').\n"
            "- ONLY speak the text intended for the customer.\n\n"
            + memory_context
        )
    
    # PRIORITY 2: KNOWLEDGE BASE
    if kb_content:
        parts.append("### KNOWLEDGE BASE\n" + kb_content)

    # PRIORITY 3: SYSTEM INSTRUCTIONS
    if prompts_content:
        parts.append("### CORE BEHAVIOR & SYSTEM INSTRUCTIONS\n" + prompts_content)
    else:
        parts.append(MINIMAL_FALLBACK_PROMPT)
    
    parts.append(
        "FACT VS BEHAVIOR PRECEDENCE (CRITICAL):\n"
        "1. MEMORY FIRST: Matching long-term memory scenario overrides generic phrasing.\n"
        "2. KB SECOND: If no memory scenario matches, answer from knowledge base facts.\n"
        "3. SYSTEM THIRD: Apply general behavior instructions after Memory/KB.\n"
        "4. NO HALLUCINATION: If missing from all sources, say you don't have that info.\n"
    )

    parts.append("""
TOOL USAGE GUARDRAIL:
- Report findings EXACTLY as provided by tools.
- Never use emojis.
- Be precise and concise.
""")
    return "\n\n".join(parts)


async def build_system_prompt_async(language: str = "el") -> str:
    kb_content = await load_knowledge_base_async(language)
    prompts_content = await get_prompts_content_async(language)
    memory_context = _cache.get("long_term_memory", "")
    
    parts = []
    parts.append(_get_response_language_instruction(language))
    
    if memory_context:
        parts.append(
            "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
            "When user intent matches any memory scenario, respond using that memory response first.\n"
            "Treat scenario matching as semantic/intention-based (not exact wording).\n"
            "If a memory entry gives an explicit response sentence, prefer that exact response text.\n\n"
            "INTERNAL INSTRUCTIONS WARNING:\n"
            "- The tags 'SCENARIO', 'KEY CONCEPTS', and 'GUIDELINE' are for your internal logic ONLY.\n"
            "- NEVER speak these tags or any meta-instructions (e.g., 'Never say...', 'Always say...', 'Avoid...').\n"
            "- ONLY speak the text intended for the customer.\n\n"
            + memory_context
        )
    
    if kb_content:
        parts.append("### KNOWLEDGE BASE\n" + kb_content)

    if prompts_content:
        parts.append("### CORE BEHAVIOR & SYSTEM INSTRUCTIONS\n" + prompts_content)
    else:
        parts.append(MINIMAL_FALLBACK_PROMPT)
    
    parts.append(
        "FACT VS BEHAVIOR PRECEDENCE (CRITICAL):\n"
        "1. MEMORY FIRST: Matching long-term memory scenario overrides generic phrasing.\n"
        "2. KB SECOND: If no memory scenario matches, answer from knowledge base facts.\n"
        "3. SYSTEM THIRD: Apply general behavior instructions after Memory/KB.\n"
        "4. NO HALLUCINATION: If missing from all sources, say you don't have that info.\n"
    )
    return "\n\n".join(parts)


def get_system_prompt(language: str = "el") -> str:
    return build_system_prompt(language)

async def get_system_prompt_async(language: str = "el") -> str:
    return await build_system_prompt_async(language)

def get_greeting(language: str = "el") -> str:
    prompts_content = get_prompts_content(language)
    if prompts_content:
        import re
        match = re.search(r'##\s*Greeting\s*\n(.+?)(?:\n##|\Z)', prompts_content, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
    return "Hello! How can I help you today?" if language.lower() in ("en", "english") else "Γεια σας! Πώς μπορώ να σας βοηθήσω σήμερα;"

def get_closing(language: str = "el") -> str:
    prompts_content = get_prompts_content(language)
    if prompts_content:
        import re
        match = re.search(r'##\s*Closing\s*\n(.+?)(?:\n##|\Z)', prompts_content, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
    return "Thank you! Goodbye!" if language.lower() in ("en", "english") else "Σας ευχαριστώ πολύ. Να έχετε μια όμορφη μέρα!"

def get_stt_language(language: str = "el") -> str:
    return "en" if language.lower() in ("en", "english") else "el"

def _get_response_language_instruction(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang == "el": return "RESPONSE LANGUAGE: Greek (Ελληνικά). GENDER: Female (Elena)."
    return "RESPONSE LANGUAGE: English. GENDER: Female (Elena)."

MINIMAL_FALLBACK_PROMPT = "You are Elena, a female customer service assistant. Be helpful."

async def refresh_cache():
    _cache["last_fetch"] = 0
    await _fetch_from_db(force=True)
