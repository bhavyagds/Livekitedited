# src/agents/flows/memory_flow.py
import logging
from src.services.database import get_database_service

logger = logging.getLogger(__name__)

async def fetch_memory_items() -> list[dict]:
    """Fetch active memory items from the database."""
    try:
        db = get_database_service()
        items = await db.get_memory_items(active_only=True)
        return items
    except Exception as e:
        logger.warning("Failed loading memory items for direct matcher: %s", e)
        return []

def build_memory_prompt_block(memory_items: list[dict]) -> str:
    """Format memory items into a system prompt block."""
    lines: list[str] = []
    for item in memory_items:
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        c = str(item.get("comment") or item.get("comments") or "").strip()
        if not q or not a:
            continue
        lines.append(f'SCENARIO (match by intent, not exact words): "{q}"')
        lines.append(f'EXPECTED RESPONSE: "{a}"')
        if c:
            lines.append(f"GUIDELINE: {c}")
        lines.append("-" * 20)
    
    if not lines:
        return ""
        
    return (
        "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
        "When user intent matches any memory scenario, respond using that memory response first.\n"
        "Treat scenario matching as semantic/intention-based (not exact wording).\n\n"
        "### ABSOLUTE VERBAL GUARDRAILS (INTERNAL ONLY)\n"
        "- NEVER speak section headers (e.g., '## Closing', '### CORE BEHAVIOR', 'GUIDELINE').\n"
        "- NEVER speak behavioral instructions or meta-rules (e.g., phrases starting with 'Never say...', 'Always say...', 'Avoid...', 'Close naturally...').\n"
        "- Bullet points in the 'CORE BEHAVIOR' or 'Closing' sections are for your logic reasoning ONLY.\n"
        "- NEVER speak the internal tags: 'SCENARIO', 'EXPECTED RESPONSE', or 'GUIDELINE'.\n"
        "- ONLY speak the actual conversational text intended for the customer.\n\n"
        + "\n".join(lines)
    )
