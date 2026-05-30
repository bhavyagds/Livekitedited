"""
Meallion Voice AI - Unified Prompts Router
==========================================
This module re-exports the prompts API that the API server (admin.py, health.py,
background_audio.py) imports via `from src.agents.prompts import ...`.

Prompts actually live in language-specific sub-packages:
  - src/agents/en/prompts.py  (English)
  - src/agents/el/prompts.py  (Greek)

This shim provides a language-aware facade:
  - Sync functions (get_system_prompt, get_greeting, etc.) read the active
    language from the shared en/prompts.py _cache (which is populated from DB),
    then delegate to the correct sub-package.
  - refresh_cache() refreshes BOTH sub-package caches so the API stays in sync.
  - _cache is aliased to the en cache for the /verify-config endpoint.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Import both language prompt modules directly (bypass __init__.py) ─────────
import importlib
_en = importlib.import_module("src.agents.en.prompts")
_el = importlib.import_module("src.agents.el.prompts")

# Alias for the /verify-config endpoint which accesses _cache directly
_cache = _en._cache


# ── Helper: resolve current language ────────────────────────────────────────

def _active_language() -> str:
    """Return the active agent language from the DB settings cache.

    Falls back to 'en' if the cache hasn't been populated yet.
    """
    lang = (_en._cache.get("settings") or {}).get("agent_language", "en")
    return (str(lang) or "en").strip().lower()


def _prompts_for(language: Optional[str] = None):
    """Return the correct prompts module for the given (or current) language."""
    lang = (language or _active_language()).strip().lower()
    return _el if lang == "el" else _en


# ── Sync API ────────────────────────────────────────────────────────────────

def get_agent_language(language: Optional[str] = None) -> str:
    return _prompts_for(language).get_agent_language()


def get_agent_setting(key: str, default: Any = None) -> Any:
    return _en.get_agent_setting(key, default)


def get_prompts_content(language: Optional[str] = None) -> Optional[str]:
    return _prompts_for(language).get_prompts_content(language or _active_language())


def load_knowledge_base(language: Optional[str] = None) -> str:
    return _prompts_for(language).load_knowledge_base(language or _active_language())


def get_system_prompt(language: Optional[str] = None) -> str:
    return _prompts_for(language).get_system_prompt(language or _active_language())


def get_greeting(language: Optional[str] = None) -> str:
    return _prompts_for(language).get_greeting(language or _active_language())


def get_closing(language: Optional[str] = None) -> str:
    return _prompts_for(language).get_closing(language or _active_language())


# ── Async API ────────────────────────────────────────────────────────────────

async def get_system_prompt_async(language: Optional[str] = None) -> str:
    return await _prompts_for(language).get_system_prompt_async(language or _active_language())


async def load_knowledge_base_async(language: Optional[str] = None) -> str:
    return await _prompts_for(language).load_knowledge_base_async(language or _active_language())


async def get_prompts_content_async(language: Optional[str] = None) -> Optional[str]:
    return await _prompts_for(language).get_prompts_content_async(language or _active_language())


# ── Cache refresh — refreshes BOTH sub-packages ──────────────────────────────

async def refresh_cache() -> None:
    """Refresh the prompt/KB cache for all languages."""
    await asyncio.gather(
        _en.refresh_cache(),
        _el.refresh_cache(),
        return_exceptions=True,
    )


__all__ = [
    "get_agent_language",
    "get_agent_setting",
    "get_prompts_content",
    "get_prompts_content_async",
    "load_knowledge_base",
    "load_knowledge_base_async",
    "get_system_prompt",
    "get_system_prompt_async",
    "get_greeting",
    "get_closing",
    "refresh_cache",
    "_cache",
]
