"""
Meallion Voice AI - English Tools
Combines order lookup, support ticket creation, and knowledge base query.
Imports settings and language from src.agents.en.prompts.
"""

import logging
import re
import json
from pathlib import Path
from typing import Annotated, Optional, Any, List
from contextvars import ContextVar
from livekit.agents import llm

from src.services.shopify import get_shopify_service, ShopifyService
from src.services.n8n import (
    create_draft_ticket,
    update_ticket_field,
    submit_ticket,
    abandon_ticket,
)
from src.agents.en.prompts import get_agent_language, get_agent_setting

logger = logging.getLogger(__name__)

# =============================================================================
# ORDER LOOKUP STUFF
# =============================================================================

_last_order_cache_var: ContextVar[dict] = ContextVar("order_lookup_last_order_cache", default={})


def _get_last_order_cache() -> dict:
    cache = _last_order_cache_var.get()
    if isinstance(cache, dict):
        return cache
    return {}


def _set_last_order_cache(order) -> None:
    _last_order_cache_var.set(
        {
            "last": order,
            "number": getattr(order, "order_number", ""),
        }
    )


def _order_digit_range() -> tuple[int, int]:
    try:
        min_d = int(get_agent_setting("order_id_min_digits", 3) or 3)
        max_d = int(get_agent_setting("order_id_max_digits", 6) or 6)
        min_d = max(3, min(min_d, 9))
        max_d = max(min_d, min(max_d, 9))
        return min_d, max_d
    except Exception:
        return 3, 6


def _phone_digit_bounds() -> tuple[int, int]:
    try:
        min_digits = int(get_agent_setting("phone_lookup_min_digits", 10) or 10)
        max_digits = int(get_agent_setting("phone_lookup_max_digits", 15) or 15)
        if min_digits < 7: min_digits = 7
        if max_digits > 15: max_digits = 15
        if max_digits < min_digits: max_digits = min_digits
        return min_digits, max_digits
    except Exception:
        return 10, 15


async def prefetch_orders():
    shopify = get_shopify_service()
    count = await shopify.prefetch_recent_orders()
    logger.info(f"Prefetched {count} orders")
    return count


async def lookup_order(
    order_number: Annotated[str, "The order number to look up"],
) -> str:
    """
    Look up an order and return BRIEF status info.
    Returns brief summary: status, delivery date, total.
    If customer asks for more details, use get_order_details function.
    """
    shopify = get_shopify_service()
    cleaned = shopify.clean_order_number(order_number)
    min_digits, max_digits = _order_digit_range()

    if not cleaned or not cleaned.isdigit() or not (min_digits <= len(cleaned) <= max_digits):
        return (
            f"The order number must be between {min_digits} and {max_digits} digits. "
            "Could you please repeat it?"
        )

    logger.info(f"Looking up order: {cleaned}")
    order = await shopify.lookup_order_cached(cleaned)

    if order is None:
        return (
            "I could not find that order. "
            "Please double-check the order number from your confirmation email."
        )

    _set_last_order_cache(order)
    await shopify.localize_order(order, "en")
    return shopify.format_order_brief(order, language="en")


async def get_order_details(
    order_number: Annotated[str, "The order number, or 'last' for the last looked up order"] = "last",
) -> str:
    """
    Get FULL details about an order.
    Use this when customer asks for more information after initial lookup.
    """
    shopify = get_shopify_service()
    cache = _get_last_order_cache()

    if order_number.lower() == "last" and "last" in cache:
        order = cache["last"]
    else:
        cleaned = shopify.clean_order_number(order_number)
        order = await shopify.lookup_order_cached(cleaned)

        if order is None:
            return f"I couldn't find order {cleaned}."

    await shopify.localize_order(order, "en")
    return shopify.format_order_for_voice(order, include_details=True, language="en")


async def lookup_order_by_phone(
    phone: Annotated[str, "The customer's phone number to look up orders for"],
) -> str:
    """
    Look up orders by customer phone number.
    Use this when the customer doesn't have their order number.
    """
    shopify = get_shopify_service()
    cleaned = shopify.clean_phone_number(phone)
    min_digits, max_digits = _phone_digit_bounds()

    if not cleaned or not cleaned.isdigit() or not (min_digits <= len(cleaned) <= max_digits):
        return (
            "That does not look like a complete phone number. "
            f"Please repeat the full number, at least {min_digits} digits."
        )

    logger.info(f"Looking up orders for phone: {cleaned}")
    orders = await shopify.lookup_order_by_phone(cleaned)

    if not orders:
        return (
            "No order was found for this phone number. "
            "Please check the number and repeat it."
        )

    for order in orders:
        await shopify.localize_order(order, "en")

    _set_last_order_cache(orders[0])

    if len(orders) == 1:
        summary = shopify.format_order_brief(orders[0], language="en")
        return f"I found one order for you. {summary}"

    count = len(orders)
    summary = shopify.format_order_brief(orders[0], language="en")
    return (
        f"I found {count} orders for this phone number. "
        f"The most recent one is this: {summary}"
    )


def get_last_order_snapshot() -> dict | None:
    order = _get_last_order_cache().get("last")
    if not order:
        return None
    return {
        "order_number": order.order_number,
        "status": order.status,
    }


# =============================================================================
# SUPPORT TICKET STUFF  (n8n progressive collection)
# =============================================================================

# ---------------------------------------------------------------------------
# Helpers shared by ticket tools
# ---------------------------------------------------------------------------

def clean_phone_number(phone: str) -> str:
    return re.sub(r'[^0-9+]', '', phone)


def clean_email(email: str) -> str:
    cleaned = email.lower().strip()
    cleaned = re.sub(r'\s+at\s+', '@', cleaned)
    cleaned = re.sub(r'\s+dot\s+', '.', cleaned)
    cleaned = cleaned.replace(" ", "")
    return cleaned


def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    cleaned = clean_phone_number(phone)
    return len(cleaned) >= 10


# ── Lightweight state container used by the session ─────────────────────────

from dataclasses import dataclass, field as dc_field


@dataclass
class TicketData:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    issue: Optional[str] = None
    ticket_id: Optional[str] = None  # n8n/ClickUp draft ID


# ── TOOL 1 ──────────────────────────────────────────────────────────────────

async def initiate_ticket_creation(call_id: str, ticket_data: TicketData) -> str:
    """
    Call this when a customer has an issue that needs a support ticket.
    This starts the ticket creation flow. Do NOT call any other ticket
    tool before this one.
    """
    # Reset to a fresh TicketData object (in-place so callers retain their reference)
    ticket_data.name = None
    ticket_data.email = None
    ticket_data.phone = None
    ticket_data.issue = None
    ticket_data.ticket_id = None

    return (
        "I'll create a support ticket for you right away. "
        "Could I get your full name please?"
    )


# ── TOOL 2 ──────────────────────────────────────────────────────────────────

async def collect_ticket_name(
    call_id: str,
    name: str,
    ticket_data: TicketData,
) -> str:
    """
    Call this when the customer provides their name during ticket creation.
    Args:
        name: The customer's full name as spoken
    """
    ticket_data.name = name

    try:
        ticket_id = await create_draft_ticket(
            call_id=call_id,
            name=name,
            language="en",
        )
        ticket_data.ticket_id = ticket_id
        logger.info(f"Draft ticket created: {ticket_id}")
    except Exception as e:
        logger.error(f"n8n create_draft_ticket failed: {e}")
        # Continue conversation even if n8n fails

    return f"Thank you {name}. What's the best email address to reach you at?"


# ── TOOL 3 ──────────────────────────────────────────────────────────────────

async def collect_ticket_email(
    email: str,
    ticket_data: TicketData,
) -> str:
    """
    Call this when the customer provides their email address.
    Args:
        email: The customer's email address
    """
    cleaned = clean_email(email)
    if not validate_email(cleaned):
        return (
            "That doesn't quite look like a valid email address. "
            "Could you spell it out for me?"
        )

    ticket_data.email = cleaned

    if ticket_data.ticket_id:
        try:
            await update_ticket_field(
                ticket_id=ticket_data.ticket_id,
                field="email",
                value=cleaned,
            )
        except Exception as e:
            logger.error(f"n8n update email failed: {e}")

    return "Got it. And your phone number please?"


# ── TOOL 4 ──────────────────────────────────────────────────────────────────

async def collect_ticket_phone(
    phone: str,
    ticket_data: TicketData,
) -> str:
    """
    Call this when the customer provides their phone number.
    Args:
        phone: The customer's phone number
    """
    ticket_data.phone = phone

    if ticket_data.ticket_id:
        try:
            await update_ticket_field(
                ticket_id=ticket_data.ticket_id,
                field="phone",
                value=phone,
            )
        except Exception as e:
            logger.error(f"n8n update phone failed: {e}")

    return (
        "Perfect. Now please describe your issue "
        "in as much detail as you can."
    )


# ── TOOL 5 ──────────────────────────────────────────────────────────────────

async def collect_ticket_issue(
    issue: str,
    ticket_data: TicketData,
) -> str:
    """
    Call this when the customer describes their issue or problem.
    Args:
        issue: Full description of the customer's issue
    """
    ticket_data.issue = issue

    if ticket_data.ticket_id:
        try:
            await update_ticket_field(
                ticket_id=ticket_data.ticket_id,
                field="issue",
                value=issue,
            )
        except Exception as e:
            logger.error(f"n8n update issue failed: {e}")

    return (
        f"Let me confirm your details before I submit. "
        f"Name: {ticket_data.name}. "
        f"Email: {ticket_data.email}. "
        f"Phone: {ticket_data.phone}. "
        f"Issue: {issue}. "
        f"Is all of that correct?"
    )


# ── TOOL 6 ──────────────────────────────────────────────────────────────────

async def confirm_and_submit_ticket(
    confirmed: bool,
    ticket_data: TicketData,
) -> str:
    """
    Call this when the customer confirms or rejects their ticket details.
    Args:
        confirmed: True if customer said yes/correct, False if they want changes
    """
    if not confirmed:
        # Restart collection — keep same ticket_id, just re-collect fields
        ticket_data.name = None
        ticket_data.email = None
        ticket_data.phone = None
        ticket_data.issue = None
        return (
            "No problem, let's go through it again. "
            "What's your full name?"
        )

    try:
        if ticket_data.ticket_id:
            await submit_ticket(ticket_id=ticket_data.ticket_id)
        ticket_id = ticket_data.ticket_id
        email = ticket_data.email

        # Clear out data so the session is clean
        ticket_data.name = None
        ticket_data.email = None
        ticket_data.phone = None
        ticket_data.issue = None
        ticket_data.ticket_id = None

        return (
            f"Your support ticket has been submitted successfully. "
            f"Your reference number is {ticket_id}. "
            f"Our team will get back to you at {email} within 24 hours. "
            f"Is there anything else I can help you with?"
        )

    except Exception as e:
        logger.error(f"n8n submit_ticket failed: {e}")
        ticket_data.name = None
        ticket_data.email = None
        ticket_data.phone = None
        ticket_data.issue = None
        ticket_data.ticket_id = None
        return (
            "I'm sorry, I had a technical issue submitting your ticket. "
            "Please call us back or email support at hello@meallion.com "
            "and we'll take care of it."
        )


# ── TOOL 7 ──────────────────────────────────────────────────────────────────

async def cancel_ticket_creation(ticket_data: TicketData) -> str:
    """
    Call this if the customer wants to cancel the ticket creation process.
    """
    if ticket_data.ticket_id:
        try:
            await abandon_ticket(ticket_id=ticket_data.ticket_id)
        except Exception as e:
            logger.error(f"n8n abandon failed: {e}")

    ticket_data.name = None
    ticket_data.email = None
    ticket_data.phone = None
    ticket_data.issue = None
    ticket_data.ticket_id = None

    return (
        "No problem, I've cancelled the ticket. "
        "Is there anything else I can help you with?"
    )


# Legacy thin wrapper kept so agent.py's _run_create_ticket still works
async def create_ticket_without_phone(
    customer_name: str,
    customer_email: str,
    issue_description: str,
    order_number: str = None,
) -> dict:
    """
    Legacy wrapper — used by the old deterministic ticket handler in agent.py.
    Calls n8n in a single shot: create draft → update email/issue → submit.
    """
    try:
        ticket_id = await create_draft_ticket(
            call_id="legacy",
            name=customer_name,
            language="en",
        )
        if ticket_id:
            cleaned_email = clean_email(customer_email)
            await update_ticket_field(ticket_id=ticket_id, field="email", value=cleaned_email)
            await update_ticket_field(ticket_id=ticket_id, field="issue", value=issue_description)
            await submit_ticket(ticket_id=ticket_id)
        return {
            "success": True,
            "task_id": ticket_id or "n/a",
            "message": (
                f"Your support ticket has been created. "
                f"Your reference number is {ticket_id or 'pending'}. "
                "One of our colleagues will contact you via email soon."
            ),
        }
    except Exception as e:
        logger.error(f"create_ticket_without_phone (legacy n8n) failed: {e}")
        return {
            "success": False,
            "message": "Sorry, I couldn't create the support ticket. Please try again.",
        }


# =============================================================================
# KNOWLEDGE BASE STUFF
# =============================================================================

class KnowledgeBaseTool:
    def __init__(self):
        self.kb_data = None
        self.db_items = []
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        kb_path = Path(__file__).parent.parent.parent.parent / "knowledge" / "meallion_faq.json"
        try:
            if kb_path.exists():
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.kb_data = json.load(f)
            else:
                self.kb_data = self._get_default_kb()
        except Exception:
            self.kb_data = self._get_default_kb()

    async def load_db_items(self) -> None:
        try:
            from src.services.database import DatabaseService
            db = DatabaseService()
            self.db_items = await db.get_kb_items(active_only=True)
        except Exception:
            self.db_items = []

    def search_db_items(self, query: str, language: str = "en") -> Optional[str]:
        if not self.db_items: return None
        query_lower = query.lower()
        for item in self.db_items:
            if item.get("language", "en") != language and language != "all": continue
            for keyword in item.get("keywords", []):
                if keyword.lower() in query_lower: return item["answer"]
        for item in self.db_items:
            if item.get("language", "en") != language and language != "all": continue
            question_words = item["question"].lower().split()
            if sum(1 for word in question_words if word in query_lower) >= 2: return item["answer"]
        return None

    def _get_default_kb(self) -> dict:
        return {
            "brand": {"name": "Meallion", "founder": "Chef Lambros Vakiaros", "description": "Premium Greek food delivery"},
            "greeting": {"english": "Hello! I'm Elena from Meallion. How can I help you today?"},
            "closing": {"english": "Thank you for contacting Meallion. Have a great day!"}
        }


_kb_instance = None

def get_kb_instance():
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBaseTool()
    return _kb_instance


async def search_knowledge_base(
    query: Annotated[str, "The customer's question to search for"],
    language: Annotated[str, "Language: 'en' or 'el'"] = "en",
) -> str:
    """
    Search the Meallion knowledge base for answers to common questions.
    """
    kb = get_kb_instance()
    query_lower = query.lower()
    
    try:
        if not kb.db_items: await kb.load_db_items()
        db_result = kb.search_db_items(query, "en")
        if db_result: return db_result
    except Exception:
        pass
        
    if not kb.kb_data:
        return "I couldn't access the knowledge base. How else can I help you?"
        
    brand = kb.kb_data.get("brand", {})
    return brand.get("description", "Meallion is high-quality ready-to-eat food designed for eating well consistently.")


async def get_brand_info(
    aspect: Annotated[str, "What aspect: 'name', 'founder', 'description'"] = "description",
) -> str:
    """Get info about the brand."""
    kb = get_kb_instance()
    brand = kb.kb_data.get("brand", {}) if kb.kb_data else {}
    if aspect == "name": return "Our name is Meallion, pronounced like Million."
    elif aspect == "founder": return "Meallion was founded by Chef Lambros Vakiaros."
    return "Meallion is premium Greek food delivery service."
