"""
Meallion Voice AI - System Prompts for Elena
Loads ALL content from database for real-time updates without restarts.
No hardcoded instructions - everything comes from DB.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Optional

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
    "last_fetch": 0,
    "ttl": 10,  # Refresh frequently so admin changes apply quickly across containers
}
_defaults_initialized = False


def _get_response_language_instruction(language: str) -> str:
    """Hard guardrail so the agent responds in the selected language."""
    auto_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)
    lang = (language or "").strip().lower()
    if auto_switch:
        fallback = "Greek" if lang == "el" else "English"
        return (
            "RESPONSE LANGUAGE REQUIREMENT:\n"
            "- Always reply in the same language as the caller's most recent message.\n"
            "- If the caller switches language, switch with them immediately.\n"
            f"- If the caller's language is unclear, default to {fallback}."
        )
    if lang == "el":
        return (
            "RESPONSE LANGUAGE REQUIREMENT:\n"
            "- You must reply in Greek (??????????????) for all normal responses.\n"
            "- Use English only if the caller explicitly asks for English."
        )
    return (
        "RESPONSE LANGUAGE REQUIREMENT:\n"
        "- You must reply in English for all normal responses.\n"
        "- Use Greek only if the caller explicitly asks for Greek."
    )


async def _fetch_from_db(force: bool = False):
    """Fetch KB, Prompts, and Settings from database.
    
    Optimized for speed:
    - Runs all database queries in PARALLEL using asyncio.gather
    - Uses connection pooling via DatabaseService singleton
    - Only fetches when cache is empty or TTL expired
    """
    import time
    
    current_time = time.time()
    
    # Only skip if cache is populated AND TTL is valid
    cache_populated = bool(_cache["kb_content"] or _cache["prompts_content"] or _cache["settings"])
    ttl_valid = (current_time - _cache["last_fetch"]) < _cache["ttl"]
    
    if cache_populated and ttl_valid and not force:
        logger.debug(f"Cache valid (TTL: {_cache['ttl'] - (current_time - _cache['last_fetch']):.0f}s remaining)")
        return  # Cache still valid
    
    fetch_start = time.time()
    logger.info(f"📥 Fetching from database (cache_populated={cache_populated}, ttl_valid={ttl_valid}, force={force})")
    
    try:
        from src.services.database import get_database_service
        db = get_database_service()  # Use singleton for connection pooling

        global _defaults_initialized
        if not _defaults_initialized:
            try:
                await db.init_default_settings()
                _defaults_initialized = True
            except Exception as e:
                logger.warning(f"Default settings init failed: {e}")
        
        # Run ALL database queries in PARALLEL for faster startup
        kb_task = asyncio.create_task(db.get_all_kb_content())
        prompts_task = asyncio.create_task(db.get_all_prompts_content())
        settings_task = asyncio.create_task(db.get_all_settings())
        
        # Wait for all queries to complete simultaneously
        kb_items, prompts_items, settings = await asyncio.gather(
            kb_task, prompts_task, settings_task,
            return_exceptions=True
        )
        
        # Process results (handle potential exceptions from individual tasks)
        if isinstance(kb_items, Exception):
            logger.warning(f"KB fetch failed: {kb_items}")
            kb_items = []
        if isinstance(prompts_items, Exception):
            logger.warning(f"Prompts fetch failed: {prompts_items}")
            prompts_items = []
        if isinstance(settings, Exception):
            logger.warning(f"Settings fetch failed: {settings}")
            settings = {}
        
        # Update cache
        for item in kb_items:
            _cache["kb_content"][item["language"]] = item["content"]
        
        for item in prompts_items:
            _cache["prompts_content"][item["language"]] = item["content"]
        
        _cache["settings"] = settings
        _cache["last_fetch"] = current_time
        
        fetch_duration = time.time() - fetch_start
        logger.info(f"✅ Refreshed in {fetch_duration:.1f}s: KB={list(_cache['kb_content'].keys())}, Prompts={list(_cache['prompts_content'].keys())}, Settings={list(_cache['settings'].keys())}")
        
    except Exception as e:
        logger.error(f"❌ Database fetch failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _sync_fetch_from_db():
    """Synchronous wrapper for database fetch."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a task if we're already in an async context
            asyncio.create_task(_fetch_from_db())
        else:
            loop.run_until_complete(_fetch_from_db())
    except RuntimeError:
        # No event loop, create one
        asyncio.run(_fetch_from_db())


# ============================================================================
# KNOWLEDGE BASE LOADER - From Database Only
# ============================================================================
def load_knowledge_base(language: str = "el") -> str:
    """Load knowledge base content from database (sync version - use cache)."""
    # Only try to fetch if cache is empty (don't call _sync_fetch_from_db in async context)
    if not _cache["kb_content"] and not _cache["prompts_content"]:
        _sync_fetch_from_db()
    
    # Check cache for requested language
    if language in _cache["kb_content"] and _cache["kb_content"][language]:
        logger.info(f"📚 Loaded KB from database for language: {language}")
        return _cache["kb_content"][language]
    
    # Fallback: try other language if main not found
    for lang, content in _cache["kb_content"].items():
        if content:
            logger.info(f"📚 KB not found for {language}, using {lang} instead")
            return content
    
    logger.info(f"📚 No KB content in database for any language")
    return ""


async def load_knowledge_base_async(language: str = "el") -> str:
    """Load knowledge base content from database (async version - ensures cache is populated)."""
    # Force fetch if cache is empty
    force = not bool(_cache["kb_content"])
    await _fetch_from_db(force=force)
    
    # Check cache for requested language
    if language in _cache["kb_content"] and _cache["kb_content"][language]:
        kb_len = len(_cache["kb_content"][language])
        logger.info(f"📚 Loaded KB from database for language: {language} ({kb_len} chars)")
        return _cache["kb_content"][language]
    
    # Fallback: try other language if main not found
    for lang, content in _cache["kb_content"].items():
        if content:
            logger.info(f"📚 KB not found for {language}, using {lang} instead ({len(content)} chars)")
            return content
    
    logger.warning(f"📚 No KB content in database for any language")
    return ""


def get_prompts_content(language: str = "el") -> Optional[str]:
    """Get prompts content from database (sync version - use cache)."""
    # Only try to fetch if cache is empty
    if not _cache["kb_content"] and not _cache["prompts_content"]:
        _sync_fetch_from_db()
    
    content = _cache["prompts_content"].get(language)
    if content:
        logger.info(f"📝 Loaded prompts from database for language: {language} ({len(content)} chars)")
    else:
        # Fallback: try other language
        for lang, c in _cache["prompts_content"].items():
            if c:
                logger.info(f"📝 Prompts not found for {language}, using {lang} instead")
                return c
        logger.info(f"📝 No prompts in database for any language")
    return content


async def get_prompts_content_async(language: str = "el") -> Optional[str]:
    """Get prompts content from database (async version - ensures cache is populated)."""
    # Force fetch if cache is empty
    force = not bool(_cache["prompts_content"])
    await _fetch_from_db(force=force)
    
    content = _cache["prompts_content"].get(language)
    if content:
        logger.info(f"📝 Loaded prompts from database for language: {language} ({len(content)} chars)")
    else:
        # Fallback: try other language
        for lang, c in _cache["prompts_content"].items():
            if c:
                logger.info(f"📝 Prompts not found for {language}, using {lang} instead ({len(c)} chars)")
                return c
        logger.warning(f"📝 No prompts in database for any language")
    return content


def get_agent_language() -> str:
    """Get the agent language from database settings."""
    runtime_lang = get_runtime_language()
    if runtime_lang:
        return runtime_lang
    _sync_fetch_from_db()
    
    # Try database first
    db_lang = _cache["settings"].get("agent_language")
    if db_lang:
        logger.debug(f"Using agent_language from database: {db_lang}")
        return db_lang

    raise RuntimeError("Missing required setting: agent_language")


def get_agent_setting(key: str, default: any = None) -> any:
    """Get an agent setting from database."""
    _sync_fetch_from_db()
    return _cache["settings"].get(key, default)


# ============================================================================
# MINIMAL FALLBACK - Only used if DB is completely empty
# ============================================================================
MINIMAL_FALLBACK_PROMPT = """You are Elena, a customer service assistant.
Keep responses short and helpful.
Available tools: lookup_order, get_order_details, create_support_ticket, search_knowledge_base, end_session.
When user says goodbye, call end_session."""


# ============================================================================
# SYSTEM PROMPT BUILDER - All content from Database
# ============================================================================
def build_system_prompt(language: str = "el") -> str:
    """
    Build the complete system prompt from database content (sync version).
    ALL instructions come from the database - no hardcoded rules.
    """
    kb_content = load_knowledge_base(language)
    prompts_content = get_prompts_content(language)
    
    # Build system prompt from DB content only
    parts = []

    # Always enforce response language according to selected agent language.
    parts.append(_get_response_language_instruction(language))
    
    # Add prompts content (this should contain ALL instructions from DB)
    if prompts_content:
        parts.append(prompts_content)
        logger.info(f"✅ Using DATABASE prompts for language: {language}")
    else:
        # Only use minimal fallback if DB has nothing
        parts.append(MINIMAL_FALLBACK_PROMPT)
        logger.warning(f"⚠️ No prompts in DB for {language}, using minimal fallback")
    
    # Add knowledge base content if available
    if kb_content:
        parts.append("\n\n" + "="*60)
        parts.append("""KNOWLEDGE BASE - CRITICAL: USE THIS TO ANSWER QUESTIONS!

When users ask about company info, founders, owner, chef, menu, meals, delivery, pricing, 
or any general questions - LOOK HERE FIRST and answer from this information.
DO NOT say "I don't have information" if the answer is below.""")
        parts.append("="*60)
        parts.append(kb_content)
        parts.append("="*60 + "\n")
    
    system_prompt = "\n\n".join(parts)
    logger.info(f"📋 Built system prompt: {len(system_prompt)} chars, KB: {'yes' if kb_content else 'no'}")
    
    return system_prompt


async def build_system_prompt_async(language: str = "el") -> str:
    """
    Build the complete system prompt from database content (async version).
    This ensures the cache is properly populated before building.
    ALL instructions come from the database - no hardcoded rules.
    """
    kb_content = await load_knowledge_base_async(language)
    prompts_content = await get_prompts_content_async(language)
    
    # Build system prompt from DB content only
    parts = []

    # Always enforce response language according to selected agent language.
    parts.append(_get_response_language_instruction(language))
    
    # Add prompts content (this should contain ALL instructions from DB)
    if prompts_content:
        parts.append(prompts_content)
        logger.info(f"✅ Using DATABASE prompts for language: {language}")
    else:
        # Only use minimal fallback if DB has nothing
        parts.append(MINIMAL_FALLBACK_PROMPT)
        logger.warning(f"⚠️ No prompts in DB for {language}, using minimal fallback")
    
    # Add knowledge base content if available
    if kb_content:
        parts.append("\n\n" + "="*60)
        parts.append("""KNOWLEDGE BASE - CRITICAL: USE THIS TO ANSWER QUESTIONS!

When users ask about company info, founders, owner, chef, menu, meals, delivery, pricing, 
or any general questions - LOOK HERE FIRST and answer from this information.
DO NOT say "I don't have information" if the answer is below.""")
        parts.append("="*60)
        parts.append(kb_content)
        parts.append("="*60 + "\n")
    
    system_prompt = "\n\n".join(parts)
    logger.info(f"📋 Built system prompt (async): {len(system_prompt)} chars, KB: {'yes' if kb_content else 'no'}")
    
    return system_prompt


# ============================================================================
# PUBLIC API - These functions are called by the agent
# ============================================================================
def get_system_prompt(language: str = "el") -> str:
    """Get the system prompt for the specified language (from database) - sync version."""
    return build_system_prompt(language)


async def get_system_prompt_async(language: str = "el") -> str:
    """Get the system prompt for the specified language (from database) - async version.
    This version ensures the cache is properly populated.
    """
    return await build_system_prompt_async(language)


def _get_prompts_content_exact(language: str = "el") -> Optional[str]:
    """Return prompts only for the requested language (no cross-language fallback)."""
    if not _cache["prompts_content"]:
        _sync_fetch_from_db()
    return _cache["prompts_content"].get(language)


def get_greeting(language: str = "el") -> str:
    """Get the greeting for the specified language from database."""
    prompts_content = _get_prompts_content_exact(language)
    
    if prompts_content:
        import re
        # Look for ## Greeting section - use English headers for all languages
        match = re.search(r'##\s*Greeting\s*\n(.+?)(?:\n##|\Z)', prompts_content, re.DOTALL | re.IGNORECASE)
        if match:
            greeting = match.group(1).strip()
            if greeting:
                logger.info(f"🎤 Using greeting from database: {greeting[:50]}...")
                return greeting
    
    # Minimal fallback
    logger.warning(f"⚠️ No greeting in DB for {language}, using fallback")
    if language.lower() in ("en", "english"):
        return "Hello! How can I help you today?"
    return "Γεια σας! Πώς μπορώ να σας βοηθήσω;"


def get_closing(language: str = "el") -> str:
    """Get the closing for the specified language from database."""
    prompts_content = _get_prompts_content_exact(language)
    
    if prompts_content:
        import re
        # Look for ## Closing section - use English headers for all languages
        match = re.search(r'##\s*Closing\s*\n(.+?)(?:\n##|\Z)', prompts_content, re.DOTALL | re.IGNORECASE)
        if match:
            closing = match.group(1).strip()
            if closing:
                logger.info(f"👋 Using closing from database: {closing[:50]}...")
                return closing
    
    # Minimal fallback
    logger.warning(f"⚠️ No closing in DB for {language}, using fallback")
    if language.lower() in ("en", "english"):
        return "Thank you! Goodbye!"
    return "Ευχαριστώ! Γεια σας!"


def get_stt_language(language: str = "el") -> str:
    """Get the STT language code."""
    if language.lower() in ("en", "english"):
        return "en"
    return "el"


# ============================================================================
# BACKWARD COMPATIBILITY - Lazy loaded functions
# ============================================================================
def _get_elena_system_prompt():
    return get_system_prompt("el")

def _get_elena_system_prompt_greek():
    return get_system_prompt("el")

def _get_elena_system_prompt_english():
    return get_system_prompt("en")

def _get_elena_greeting_greek():
    return get_greeting("el")

def _get_elena_greeting_english():
    return get_greeting("en")

def _get_elena_closing_greek():
    return get_closing("el")

def _get_elena_closing_english():
    return get_closing("en")


# Lazy property access - only fetches from DB when actually accessed
class _LazyPrompts:
    @property
    def ELENA_SYSTEM_PROMPT(self):
        return _get_elena_system_prompt()
    
    @property
    def ELENA_SYSTEM_PROMPT_GREEK(self):
        return _get_elena_system_prompt_greek()
    
    @property
    def ELENA_SYSTEM_PROMPT_ENGLISH(self):
        return _get_elena_system_prompt_english()
    
    @property
    def ELENA_GREETING_GREEK(self):
        return _get_elena_greeting_greek()
    
    @property
    def ELENA_GREETING_ENGLISH(self):
        return _get_elena_greeting_english()
    
    @property
    def ELENA_CLOSING_GREEK(self):
        return _get_elena_closing_greek()
    
    @property
    def ELENA_CLOSING_ENGLISH(self):
        return _get_elena_closing_english()


# Intent detection keywords (both languages)
INTENT_KEYWORDS = {
    "order": ["παραγγελία", "order", "αριθμός", "number", "status", "κατάσταση"],
    "complaint": ["πρόβλημα", "problem", "issue", "παράπονο", "complaint", "λάθος", "wrong"],
    "support": ["βοήθεια", "help", "support", "υποστήριξη"],
    "human": ["πραγματικό άνθρωπο", "real human", "real person", "speak to someone", "μιλήσω με κάποιον", "άνθρωπο"],
}


async def refresh_cache():
    """Force refresh the cache from database. Call this after admin updates."""
    _cache["last_fetch"] = 0
    await _fetch_from_db()
    logger.info("🔄 Cache refreshed from database")
