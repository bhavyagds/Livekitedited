"""
Meallion Voice AI - System Prompts for Elena (English Agent)
Loads ALL content from database for real-time updates without restarts.
No hardcoded instructions - everything comes from DB.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, Any, List

logger = logging.getLogger(__name__)

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
        logger.info(f"✅ English DB Refresh: KB={len(_cache['kb_content'])}, Prompts={len(_cache['prompts_content'])}, Memory={len(_cache['long_term_memory'])} chars")
        
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


def load_knowledge_base(language: str = "en") -> str:
    if not _cache["kb_content"]: _sync_fetch_from_db()
    return _cache["kb_content"].get(language) or next(iter(_cache["kb_content"].values()), "")


async def load_knowledge_base_async(language: str = "en") -> str:
    await _fetch_from_db(force=not _cache["kb_content"])
    return _cache["kb_content"].get(language) or next(iter(_cache["kb_content"].values()), "")


def get_prompts_content(language: str = "en") -> Optional[str]:
    if not _cache["prompts_content"]: _sync_fetch_from_db()
    return _cache["prompts_content"].get(language) or next(iter(_cache["prompts_content"].values()), "")


async def get_prompts_content_async(language: str = "en") -> Optional[str]:
    await _fetch_from_db(force=not _cache["prompts_content"])
    return _cache["prompts_content"].get(language) or next(iter(_cache["prompts_content"].values()), "")


def get_agent_language() -> str:
    return "en"


def get_agent_setting(key: str, default: Any = None) -> Any:
    if not _cache["settings"]: _sync_fetch_from_db()
    return _cache["settings"].get(key, default)


def build_system_prompt(language: str = "en") -> str:
    kb_content = load_knowledge_base(language)
    prompts_content = get_prompts_content(language)
    memory_context = _cache.get("long_term_memory", "")
    
    parts = []
    parts.append(_get_response_language_instruction(language))
    
    # PRIORITY 0: GLOBAL VERBAL GUARDRAILS (CRITICAL)
    parts.append(
        "### ABSOLUTE VERBAL GUARDRAILS (INTERNAL ONLY)\n"
        "- NEVER speak section headers (e.g., '## Closing', '### CORE BEHAVIOR', 'GUIDELINE').\n"
        "- NEVER speak behavioral instructions or meta-rules (e.g., phrases starting with 'Never say...', 'Always say...', 'Avoid...', 'Close naturally...').\n"
        "- Bullet points in the 'CORE BEHAVIOR' or 'Closing' sections are for your logic reasoning ONLY.\n"
        "- NEVER speak the internal tags: 'SCENARIO', 'EXPECTED RESPONSE', or 'GUIDELINE'.\n"
        "- ONLY speak the actual conversational text intended for the customer.\n"
    )

    # PRIORITY 1: LONG-TERM MEMORY
    if memory_context:
        parts.append(
            "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
            "When user intent matches any memory scenario, respond using that memory response first.\n"
            "Treat scenario matching as semantic/intention-based (not exact wording).\n"
            "If a memory entry gives an explicit response sentence, prefer that exact response text.\n\n"
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

    # Always inject ticket instructions so the LLM knows the 7-step sequence
    parts.append(TICKET_INSTRUCTIONS)

    parts.append(TOOL_USAGE_GUARDRAIL)
    return "\n\n".join(parts)


async def build_system_prompt_async(language: str = "en") -> str:
    kb_content = await load_knowledge_base_async(language)
    prompts_content = await get_prompts_content_async(language)
    memory_context = _cache.get("long_term_memory", "")
    
    parts = []
    parts.append(_get_response_language_instruction(language))
    
    # PRIORITY 0: GLOBAL VERBAL GUARDRAILS (CRITICAL)
    parts.append(
        "### ABSOLUTE VERBAL GUARDRAILS (INTERNAL ONLY)\n"
        "- NEVER speak section headers (e.g., '## Closing', '### CORE BEHAVIOR', 'GUIDELINE').\n"
        "- NEVER speak behavioral instructions or meta-rules (e.g., phrases starting with 'Never say...', 'Always say...', 'Avoid...', 'Close naturally...').\n"
        "- Bullet points in the 'CORE BEHAVIOR' or 'Closing' sections are for your logic reasoning ONLY.\n"
        "- NEVER speak the internal tags: 'SCENARIO', 'EXPECTED RESPONSE', or 'GUIDELINE'.\n"
        "- ONLY speak the actual conversational text intended for the customer.\n"
    )

    # PRIORITY 1: LONG-TERM MEMORY
    if memory_context:
        parts.append(
            "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
            "When user intent matches any memory scenario, respond using that memory response first.\n"
            "Treat scenario matching as semantic/intention-based (not exact wording).\n"
            "If a memory entry gives an explicit response sentence, prefer that exact response text.\n\n"
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

    # Always inject ticket instructions so the LLM knows the 7-step sequence
    parts.append(TICKET_INSTRUCTIONS)

    parts.append(TOOL_USAGE_GUARDRAIL)
    return "\n\n".join(parts)


def get_system_prompt(language: str = "en") -> str:
    return build_system_prompt(language)


async def get_system_prompt_async(language: str = "en") -> str:
    return await build_system_prompt_async(language)


def get_greeting(language: str = "en") -> str:
    prompts_content = get_prompts_content(language)
    if prompts_content:
        import re
        match = re.search(r'##\s*Greeting\s*\n(.+?)(?:\n##|\Z)', prompts_content, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
    return "Hello! How can I help you today?"


def get_closing(language: str = "en") -> str:
    prompts_content = get_prompts_content(language)
    if prompts_content:
        import re
        match = re.search(r'##\s*Closing\s*\n(.+?)(?:\n##|\Z)', prompts_content, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
    return "Thank you! Goodbye!"


def get_stt_language(language: str = "en") -> str:
    return "en"


def _get_response_language_instruction(language: str) -> str:
    return "RESPONSE LANGUAGE: English. GENDER: Female (Elena)."


MINIMAL_FALLBACK_PROMPT = "You are Elena, a female customer service assistant. Be helpful."


# ---------------------------------------------------------------------------
# Support ticket collection instructions injected into every system prompt
# ---------------------------------------------------------------------------

TICKET_INSTRUCTIONS = """
## SUPPORT TICKET CREATION — FOLLOW THIS EXACT SEQUENCE

When a customer has an unresolvable issue, create a support ticket
by following these steps IN ORDER. Never skip a step. Never ask for
two pieces of information in the same message.

STEP 1: Call initiate_ticket_creation()
STEP 2: Customer gives name  → call collect_ticket_name(name="...")
STEP 3: Customer gives email → call collect_ticket_email(email="...")
STEP 4: Customer gives phone → call collect_ticket_phone(phone="...")
STEP 5: Customer describes issue → call collect_ticket_issue(issue="...")
STEP 6: Read back ALL details (name, email, phone, issue) and ask for confirmation
STEP 7: Customer says yes → call confirm_and_submit_ticket(confirmed=True)
        Customer says no  → call confirm_and_submit_ticket(confirmed=False)

RULES:
- If customer provides multiple details at once (e.g. name and email
  in same sentence), still call each tool separately in order.
- If customer says "cancel", "never mind", or "forget it" at any
  point during collection → call cancel_ticket_creation()
- If customer says details are wrong at confirmation step →
  call confirm_and_submit_ticket(confirmed=False) to restart
- Never invent or assume any customer details
- Always read back ALL details before calling submit
"""

TOOL_USAGE_GUARDRAIL = """
## TOOL USAGE GUARDRAIL (CRITICAL)
- Report findings EXACTLY as provided by tools.
- Never use emojis.
- Be precise and concise.
- ALWAYS speak order numbers digit-by-digit (e.g., read order number '1234' or '2345' as 'one two three four' or 'two three four five', and NEVER as 'one thousand...' or 'two thousand...').
- NEVER speak, output, or share web links, URLs, or authentication keys (such as order status links, checkout URLs, or authenticate?key=... tokens) in your responses. These contain security secrets and sound extremely awkward when spoken over the phone. Just summarize the details verbally (e.g., 'Your order is unfulfilled.').
"""


async def refresh_cache():
    _cache["last_fetch"] = 0
    await _fetch_from_db(force=True)
