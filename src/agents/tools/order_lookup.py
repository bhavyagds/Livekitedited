"""
Meallion Voice AI - Order Lookup Tool
Handles order status lookups via Shopify API with caching for fast responses.
"""

import logging
from contextvars import ContextVar
from typing import Annotated

from livekit.agents import llm

from src.services.shopify import get_shopify_service, ShopifyService
from src.agents.prompts import get_agent_language, get_agent_setting
from src.config import settings

logger = logging.getLogger(__name__)

# Store last looked up order per async context to reduce cross-call leakage.
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
    """Return (min_digits, max_digits) for order ID validation. Defaults to 3-6."""
    try:
        min_d = int(get_agent_setting("order_id_min_digits", 3) or 3)
        max_d = int(get_agent_setting("order_id_max_digits", 6) or 6)
        min_d = max(3, min(min_d, 9))
        max_d = max(min_d, min(max_d, 9))
        return min_d, max_d
    except Exception:
        return 3, 6


def _expected_order_digits() -> int:
    """Kept for backward compatibility — returns min digit length."""
    min_d, _ = _order_digit_range()
    return min_d


def _phone_digit_bounds() -> tuple[int, int]:
    try:
        min_digits = int(get_agent_setting("phone_lookup_min_digits", 10) or 10)
        max_digits = int(get_agent_setting("phone_lookup_max_digits", 15) or 15)
        if min_digits < 7:
            min_digits = 7
        if max_digits > 15:
            max_digits = 15
        if max_digits < min_digits:
            max_digits = min_digits
        return min_digits, max_digits
    except Exception:
        return 10, 15


class OrderLookupTool:
    """
    Order lookup tool for Elena voice agent.

    Features:
    - Order prefetching for instant responses
    - Brief response first, full details on request
    - Voice input artifact cleaning
    """

    def __init__(self):
        self.shopify: ShopifyService = get_shopify_service()

    def get_tools(self) -> list:
        """Get the list of function tools for this module."""
        return [
            lookup_order,
            lookup_order_by_phone,
            get_order_details,
        ]


async def prefetch_orders():
    """
    Prefetch recent orders into cache.
    Call this when session starts for instant lookups.
    """
    shopify = get_shopify_service()
    count = await shopify.prefetch_recent_orders()
    logger.info(f"Prefetched {count} orders")
    return count


async def lookup_order(
    order_number: Annotated[str, "The order number to look up (exact configured length)"],
) -> str:
    """
    Look up an order and return BRIEF status info.
    Uses cache for instant responses on prefetched orders.

    Returns brief summary: status, delivery date, total.
    If customer asks for more details, use get_order_details function.

    Args:
        order_number: The order number

    Returns:
        Brief order status, asks if they want more details
    """
    shopify = get_shopify_service()
    agent_lang = get_agent_language()

    cleaned = shopify.clean_order_number(order_number)
    min_digits, max_digits = _order_digit_range()

    if not cleaned or not cleaned.isdigit() or not (min_digits <= len(cleaned) <= max_digits):
        logger.warning(f"Invalid order number: {order_number} -> {cleaned} (expected {min_digits}-{max_digits} digits)")
        if agent_lang == "el":
            return (
                f"Ο αριθμός παραγγελίας πρέπει να έχει από {min_digits} έως {max_digits} ψηφία. "
                "Μπορείτε να τον επαναλάβετε ψηφίο προς ψηφίο;"
            )
        return (
            f"The order number must be between {min_digits} and {max_digits} digits. "
            "Could you repeat it digit by digit?"
        )

    logger.info(f"Looking up order: {cleaned}")
    order = await shopify.lookup_order_cached(cleaned)

    if order is None:
        logger.info(f"Order not found: {cleaned}")
        if agent_lang == "el":
            return (
                "Δεν μπόρεσα να βρω αυτή την παραγγελία. "
                "Παρακαλώ ελέγξτε ξανά τον αριθμό παραγγελίας από την επιβεβαίωση που λάβατε."
            )
        return (
            "I could not find that order. "
            "Please double-check the order number from your confirmation email."
        )

    _set_last_order_cache(order)

    logger.info(f"Order lookup using language: {agent_lang}")
    await shopify.localize_order(order, agent_lang)

    response = shopify.format_order_brief(order, language=agent_lang)
    logger.info(f"Order {cleaned} found (brief): {order.status}")

    return response


async def get_order_details(
    order_number: Annotated[str, "The order number, or 'last' for the last looked up order"] = "last",
) -> str:
    """
    Get FULL details about an order.
    Use this when customer asks for more information after initial lookup.

    Includes: all items ordered, prices, delivery address, customer info, refund status.

    Args:
        order_number: Order number or 'last' for most recent lookup

    Returns:
        Complete order details
    """
    shopify = get_shopify_service()
    cache = _get_last_order_cache()

    if order_number.lower() == "last" and "last" in cache:
        order = cache["last"]
        logger.info(f"Returning full details for last order: {order.order_number}")
    else:
        cleaned = shopify.clean_order_number(order_number)
        order = await shopify.lookup_order_cached(cleaned)

        if order is None:
            return f"I couldn't find order {cleaned}."

    agent_lang = get_agent_language()
    logger.info(f"Order details using language: {agent_lang}")

    await shopify.localize_order(order, agent_lang)

    response = shopify.format_order_for_voice(order, include_details=True, language=agent_lang)
    return response


async def lookup_order_by_phone(
    phone: Annotated[str, "The customer's phone number to look up orders for"],
) -> str:
    """
    Look up orders by customer phone number.
    Use this when the customer doesn't have their order number.

    Args:
        phone: The phone number to search for

    Returns:
        Summary of orders found for this phone number
    """
    shopify = get_shopify_service()
    agent_lang = get_agent_language()

    cleaned = shopify.clean_phone_number(phone)
    min_digits, max_digits = _phone_digit_bounds()

    if not cleaned or not cleaned.isdigit() or not (min_digits <= len(cleaned) <= max_digits):
        logger.warning(f"Invalid phone number: {phone} -> {cleaned}")
        if agent_lang == "el":
            return (
                "Αυτό δεν φαίνεται να είναι πλήρης αριθμός τηλεφώνου. "
                f"Παρακαλώ επαναλάβετε ολόκληρο τον αριθμό, τουλάχιστον {min_digits} ψηφία, ψηφίο προς ψηφίο."
            )
        return (
            "That does not look like a complete phone number. "
            f"Please repeat the full number, at least {min_digits} digits, digit by digit."
        )

    logger.info(f"Looking up orders for phone: {cleaned}")
    orders = await shopify.lookup_order_by_phone(cleaned)

    if not orders:
        if agent_lang == "el":
            return (
                "Δεν βρέθηκε παραγγελία με αυτόν τον αριθμό τηλεφώνου. "
                "Μπορείτε να ελέγξετε τον αριθμό και να τον επαναλάβετε ψηφίο προς ψηφίο;"
            )
        return (
            "No order was found for this phone number. "
            "Please check the number and repeat it digit by digit."
        )

    for order in orders:
        await shopify.localize_order(order, agent_lang)

    _set_last_order_cache(orders[0])

    if len(orders) == 1:
        summary = shopify.format_order_brief(orders[0], language=agent_lang)
        if agent_lang == "el":
            return f"Βρήκα μία παραγγελία για εσάς. {summary}"
        return f"I found one order for you. {summary}"

    count = len(orders)
    most_recent = orders[0]
    summary = shopify.format_order_brief(most_recent, language=agent_lang)
    if agent_lang == "el":
        return (
            f"Βρήκα {count} παραγγελίες για αυτόν τον αριθμό τηλεφώνου. "
            f"Η πιο πρόσφατη είναι η εξής: {summary}"
        )
    return (
        f"I found {count} orders for this phone number. "
        f"The most recent one is this: {summary}"
    )


def get_last_order_snapshot() -> dict | None:
    """Return last looked-up order status info for deterministic responses."""
    order = _get_last_order_cache().get("last")
    if not order:
        return None
    return {
        "order_number": order.order_number,
        "status": order.status,
    }
