"""
Meallion Voice AI - Greek Tools
Combines order lookup, support ticket creation, and knowledge base query.
Imports settings and language from src.agents.el.prompts.
"""

import logging
import re
import json
from pathlib import Path
from typing import Annotated, Optional, Any, List
from contextvars import ContextVar
from livekit.agents import llm

from src.services.shopify import get_shopify_service, ShopifyService
from src.services.clickup import clickup_service
from src.agents.el.prompts import get_agent_language, get_agent_setting

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
            f"Ο αριθμός παραγγελίας πρέπει να έχει από {min_digits} έως {max_digits} ψηφία. "
            "Παρακαλώ επαναλάβετε το;"
        )

    logger.info(f"Looking up order: {cleaned}")
    order = await shopify.lookup_order_cached(cleaned)

    if order is None:
        return (
            "Δεν μπόρεσα να βρω αυτή την παραγγελία. "
            "Παρακαλώ ελέγξτε ξανά τον αριθμό παραγγελίας στο email επιβεβαίωσης που λάβατε."
        )

    _set_last_order_cache(order)
    await shopify.localize_order(order, "el")
    return shopify.format_order_brief(order, language="el")


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
            return f"Δεν μπόρεσα να βρω την παραγγελία {cleaned}."

    await shopify.localize_order(order, "el")
    return shopify.format_order_for_voice(order, include_details=True, language="el")


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
            "Αυτό δεν φαίνεται να είναι πλήρης αριθμός τηλεφώνου. "
            f"Παρακαλώ επαναλάβετε ολόκληρο τον αριθμό, τουλάχιστον {min_digits} ψηφία."
        )

    logger.info(f"Looking up orders for phone: {cleaned}")
    orders = await shopify.lookup_order_by_phone(cleaned)

    if not orders:
        return (
            "Δεν μπόρεσα να βρω κάποια παραγγελία με αυτόν τον αριθμό τηλεφώνου. "
            "Μπορείτε να ελέγξετε τον αριθμό και να το πείτε ξανά, παρακαλώ;"
        )

    for order in orders:
        await shopify.localize_order(order, "el")

    _set_last_order_cache(orders[0])

    if len(orders) == 1:
        summary = shopify.format_order_brief(orders[0], language="el")
        return f"Βρήκα μία παραγγελία για εσάς. {summary}"

    count = len(orders)
    summary = shopify.format_order_brief(orders[0], language="el")
    return (
        f"Βρήκα {count} παραγγελίες για αυτόν τον αριθμό τηλεφώνου. "
        f"Η πιο πρόσφατη είναι η εξής: {summary}"
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
# SUPPORT TICKET STUFF
# =============================================================================

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


async def create_support_ticket(
    customer_name: Annotated[str, "The customer's full name"],
    customer_phone: Annotated[str, "The customer's phone number"],
    customer_email: Annotated[str, "The customer's email address"],
    issue_description: Annotated[str, "A clear description of the customer's issue or complaint"],
    order_number: Annotated[Optional[str], "Related order number if applicable"] = None,
) -> str:
    """
    Create a support ticket in ClickUp.
    All fields MUST have been collected and verified ONE BY ONE first.
    """
    cleaned_phone = clean_phone_number(customer_phone)
    cleaned_email = clean_email(customer_email)
    
    errors = []
    if not customer_name or len(customer_name.strip()) < 2:
        errors.append("Invalid name")
    if not validate_phone(cleaned_phone):
        errors.append("Invalid phone number")
    if not validate_email(cleaned_email):
        errors.append("Invalid email address")
    if not issue_description or len(issue_description.strip()) < 10:
        errors.append("Issue description too short")
    
    if errors:
        return f"Δεν μπορώ να δημιουργήσω το αίτημα: {', '.join(errors)}. Παρακαλώ διορθώστε τα στοιχεία."
    
    result = await clickup_service.create_support_ticket(
        customer_name=customer_name.strip(),
        customer_phone=cleaned_phone,
        customer_email=cleaned_email,
        issue_description=issue_description.strip(),
        order_number=order_number,
    )
    
    if not result["success"]:
        return (
            "Συγγνώμη, δεν μπόρεσα να δημιουργήσω το αίτημα υποστήριξης. "
            "Παρακαλώ δοκιμάστε ξανά ή επικοινωνήστε μαζί μας απευθείας μέσω email."
        )
    
    return (
        f"Το αίτημα υποστήριξης δημιουργήθηκε με επιτυχία. "
        f"Ο αριθμός αναφοράς σας είναι {result['task_id']}. "
        "Η ομάδα υποστήριξής μας θα επικοινωνήσει μαζί σας σύντομα στο τηλέφωνο ή το email που δώσατε."
    )


async def log_customer_query(
    customer_question: Annotated[str, "The customer's question or issue that you cannot answer"],
    customer_name: Annotated[Optional[str], "Customer name if known"] = None,
    customer_phone: Annotated[Optional[str], "Customer phone if known"] = None,
) -> str:
    """
    Log a customer query that you cannot answer for follow-up by the team.
    """
    logger.info(f"Logging customer query: {customer_question[:100]}")
    result = await clickup_service.create_support_ticket(
        customer_name=customer_name or "Unknown Caller",
        customer_phone=customer_phone or "Not provided",
        customer_email="callback-needed@meallion.gr",
        issue_description=f"[CALLBACK NEEDED] Customer asked: {customer_question}",
        order_number=None,
        tags=["callback-needed", "ai-escalation"],
    )
    if result["success"]:
        return "Έγινε! Σημείωσα την ερώτησή σας και κάποιος από την ομάδα μας θα σας απαντήσει σύντομα."
    return "Κράτησα μια σημείωση - η ομάδα μας θα επικοινωνήσει μαζί σας σύντομα."


async def create_ticket_without_phone(
    customer_name: str,
    customer_email: str,
    issue_description: str,
    order_number: str = None,
) -> dict:
    """
    Δημιουργία αιτήματος υποστήριξης χωρίς αριθμό τηλεφώνου (συλλέγεται μετά).
    Returns a dict: {success, task_id, message} or {success: False, message}.
    """
    cleaned_email = clean_email(customer_email)

    errors = []
    if not customer_name or len(customer_name.strip()) < 2:
        errors.append("το όνομα είναι πολύ μικρό")
    if not validate_email(cleaned_email):
        errors.append("η διεύθυνση email δεν είναι έγκυρη")
    if not issue_description or len(issue_description.strip()) < 5:
        errors.append("η περιγραφή προβλήματος είναι πολύ μικρή")

    if errors:
        return {
            "success": False,
            "message": f"Δεν μπορώ να δημιουργήσω το αίτημα: {', '.join(errors)}. Παρακαλώ διορθώστε τα στοιχεία.",
        }

    result = await clickup_service.create_support_ticket(
        customer_name=customer_name.strip(),
        customer_phone="Pending",
        customer_email=cleaned_email,
        issue_description=issue_description.strip(),
        order_number=order_number,
    )

    if result["success"]:
        return {
            "success": True,
            "task_id": result["task_id"],
            "message": (
                f"Το αίτημα υποστήριξης δημιουργήθηκε με επιτυχία. "
                f"Ο αριθμός αναφοράς σας είναι {result['task_id']}."
            ),
        }
    return {
        "success": False,
        "message": "Συγγνώμη, δεν μπόρεσα να δημιουργήσω το αίτημα υποστήριξης. Παρακαλώ δοκιμάστε ξανά.",
    }


async def update_ticket_with_phone(task_id: str, phone: str) -> dict:
    """
    Ενημέρωση υπάρχοντος αιτήματος με τον αριθμό τηλεφώνου του πελάτη.
    Returns {success, message}.
    """
    cleaned = clean_phone_number(phone)
    if not validate_phone(cleaned):
        return {"success": False, "message": "Μη έγκυρος αριθμός τηλεφώνου."}

    result = await clickup_service.update_ticket_phone(task_id, cleaned)
    if result.get("success"):
        return {"success": True, "message": f"Ο αριθμός τηλεφώνου {cleaned} προστέθηκε στο αίτημα."}
    return {"success": False, "message": "Δεν ήταν δυνατή η ενημέρωση του αιτήματος με τον αριθμό τηλεφώνου."}



async def validate_ticket_field(
    field_name: Annotated[str, "The field being validated: 'name', 'phone', 'email', or 'issue'"],
    field_value: Annotated[str, "The value provided by the customer"],
) -> str:
    """
    Validate and format a support ticket field before final submission.
    """
    field_name = field_name.lower()
    
    if field_name == "name":
        cleaned = field_value.strip()
        if len(cleaned) < 2:
            return "Το όνομα είναι πολύ μικρό. Μπορείτε να μου πείτε το πλήρες όνομά σας;"
        return f"Έχω το εξής: {cleaned}. Είναι σωστό;"
    
    elif field_name == "phone":
        cleaned = clean_phone_number(field_value)
        if not validate_phone(cleaned):
            return "Ο αριθμός τηλεφώνου δεν φαίνεται σωστός. Μπορείτε να τον επαναλάβετε;"
        return f"Έχω τον αριθμό τηλεφώνου: {cleaned}. Είναι σωστό;"
    
    elif field_name == "email":
        cleaned = clean_email(field_value)
        if not validate_email(cleaned):
            return "Η διεύθυνση email δεν φαίνεται σωστή. Μπορείτε να μου την πείτε γράμμα-γράμμα;"
        return f"Έχω το email: {cleaned}. Είναι σωστό;"
    
    elif field_name == "issue":
        cleaned = field_value.strip()
        if len(cleaned) < 10:
            return "Μπορείτε να μου δώσετε περισσότερες λεπτομέρειες για το πρόβλημα;"
        summary = cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
        return f"Κατάλαβα ότι το πρόβλημα είναι: {summary}. Είναι σωστό;"
    
    return f"Άγνωστο πεδίο: {field_name}"


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

    def search_db_items(self, query: str, language: str = "el") -> Optional[str]:
        if not self.db_items: return None
        query_lower = query.lower()
        for item in self.db_items:
            if item.get("language", "el") != language and language != "all": continue
            for keyword in item.get("keywords", []):
                if keyword.lower() in query_lower: return item["answer"]
        for item in self.db_items:
            if item.get("language", "el") != language and language != "all": continue
            question_words = item["question"].lower().split()
            if sum(1 for word in question_words if word in query_lower) >= 2: return item["answer"]
        return None

    def _get_default_kb(self) -> dict:
        return {
            "brand": {"name": "Meallion", "founder": "Chef Lambros Vakiaros", "description": "Premium Greek food delivery"},
            "greeting": {"default": "Γεια σας! Είμαι η Έλενα από το Meallion. Πώς μπορώ να σας βοηθήσω;"},
            "closing": {"default": "Ευχαριστώ που επικοινωνήσατε με το Meallion. Καλή σας ημέρα!"}
        }


_kb_instance = None

def get_kb_instance():
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBaseTool()
    return _kb_instance


async def search_knowledge_base(
    query: Annotated[str, "The customer's question to search for"],
    language: Annotated[str, "Language: 'en' or 'el'"] = "el",
) -> str:
    """
    Search the Meallion knowledge base for answers to common questions.
    """
    kb = get_kb_instance()
    query_lower = query.lower()
    
    try:
        if not kb.db_items: await kb.load_db_items()
        db_result = kb.search_db_items(query, "el")
        if db_result: return db_result
    except Exception:
        pass
        
    if not kb.kb_data:
        return "Δεν μπόρεσα να αποκτήσω πρόσβαση στη βάση γνώσεων."
        
    brand = kb.kb_data.get("brand", {})
    return brand.get("description", "Meallion είναι premium υπηρεσία παράδοσης ελληνικού φαγητού.")


async def get_brand_info(
    aspect: Annotated[str, "What aspect: 'name', 'founder', 'description'"] = "description",
) -> str:
    """Get info about the brand."""
    kb = get_kb_instance()
    brand = kb.kb_data.get("brand", {}) if kb.kb_data else {}
    if aspect == "name": return "Το όνομά μας είναι Meallion."
    elif aspect == "founder": return "Το Meallion ιδρύθηκε από τον Chef Λάμπρο Βακιάρο."
    return "Το Meallion είναι μια premium υπηρεσία παράδοσης ελληνικού φαγητού."
