"""
Meallion Voice AI - Order Lookup Tool
Handles order status lookups via Shopify API with caching for fast responses.
"""

import logging
import re
import time
from typing import Annotated, Optional

from livekit.agents import llm

from src.services.shopify import get_shopify_service, ShopifyService
from src.agents.prompts import get_agent_language
from src.config import settings

logger = logging.getLogger(__name__)

# Store last looked up order for "more details" requests
_last_order_cache = {}


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
    order_number: Annotated[str, "The order number to look up (typically 4-5 digits)"],
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
    
    # Clean the order number
    cleaned = shopify.clean_order_number(order_number)
    
    # Validate format
    if not shopify.validate_order_number(cleaned):
        logger.warning(f"Invalid order number format: {order_number} -> {cleaned}")
        if agent_lang == "el":
            return f"Αυτό δεν μοιάζει με έγκυρο αριθμό παραγγελίας. Μπορείτε να μου δώσετε τον 4-ψήφιο ή 5-ψήφιο αριθμό από την επιβεβαίωσή σας;"
        return f"That doesn't look like a valid order number. Can you give me the 4 or 5 digit number from your confirmation?"
    
    # Look up order (uses cache if available - instant!)
    logger.info(f"Looking up order: {cleaned}")
    order = await shopify.lookup_order_cached(cleaned)
    
    if order is None:
        logger.info(f"Order not found: {cleaned}")
        if agent_lang == "el":
            return (
                f"Λυπάμαι, αλλά δεν βρήκα την παραγγελία {cleaned}. "
                "Θέλετε να δοκιμάσω να την βρω με τον αριθμό του τηλεφώνου σας;"
            )
        return (
            f"I'm sorry, but I couldn't find order {cleaned}. "
            "Would you like me to try looking it up with your phone number instead?"
        )
    
    # Store for "more details" requests
    _last_order_cache["last"] = order
    _last_order_cache["number"] = cleaned
    
    # Localize order fields to match the user's language
    await shopify.localize_order(order, agent_lang)
    
    # Return BRIEF response in the configured language
    response = shopify.format_order_brief(order, language=agent_lang)
    logger.info(f"Order {cleaned} found (brief): {order.status}")
    
    return response


async def get_order_details(
    order_number: Annotated[str, "The order number, or 'last' for the last looked up order"] = "last",
) -> str:
    """
    Get FULL details about an order.
    Use this when customer asks for more information after initial lookup.
    """
    shopify = get_shopify_service()
    agent_lang = get_agent_language()
    
    # Check if asking about last order
    if order_number.lower() == "last" and "last" in _last_order_cache:
        order = _last_order_cache["last"]
    else:
        # Look up the specific order
        cleaned = shopify.clean_order_number(order_number)
        order = await shopify.lookup_order_cached(cleaned)
        
        if order is None:
            if agent_lang == "el":
                return f"Δεν βρέθηκαν λεπτομέρειες για την παραγγελία {cleaned}."
            return f"I couldn't find details for order {cleaned}."
    
    # Localize order fields to match the user's language
    await shopify.localize_order(order, agent_lang)
    
    # Return FULL details in the configured language
    response = shopify.format_order_for_voice(order, include_details=True, language=agent_lang)
    return response


async def lookup_order_by_phone(
    phone: Annotated[str, "The customer's phone number to look up orders for"],
) -> str:
    """
    Look up orders by customer phone number.
    Use this when the customer doesn't have their order number.
    """
    shopify = get_shopify_service()
    agent_lang = get_agent_language()
    
    cleaned = shopify.clean_phone_number(phone)
    if not cleaned or len(cleaned) < 8:
        if agent_lang == "el":
            return "Λυπάμαι, δεν κατάλαβα αυτόν τον αριθμό τηλεφώνου. Μπορείτε να τον πείτε ξανά ψηφίο προς ψηφίο;"
        return "I'm sorry, I couldn't understand that phone number. Could you please say it again digit by digit?"
        
    logger.info(f"Looking up orders for phone: {cleaned}")
    orders = await shopify.lookup_order_by_phone(cleaned)
    
    if not orders:
        if agent_lang == "el":
            return (
                f"Δεν βρέθηκε καμία παραγγελία συνδεδεμένη με τον αριθμό {phone}. "
                "Θέλετε να δοκιμάσετε με έναν άλλον αριθμό ή να σας συνδέσω με έναν εκπρόσωπο;"
            )
        return (
            f"I couldn't find any orders attached to the phone number {phone}. "
            "Would you like to try another number or should I connect you with a representative?"
        )
    
    # Store the first/most recent order as "last"
    _last_order_cache["last"] = orders[0]
    _last_order_cache["number"] = orders[0].order_number
    
    # Localize first order
    await shopify.localize_order(orders[0], agent_lang)
    
    # Format response
    if len(orders) == 1:
        summary = shopify.format_order_brief(orders[0], language=agent_lang)
        if agent_lang == "el":
            return f"Βρήκα μία παραγγελία για εσάς. {summary}"
        return f"I found one order for you. {summary}"
    else:
        count = len(orders)
        most_recent = orders[0]
        status = most_recent.status
        if agent_lang == "el":
            return f"Βρήκα {count} παραγγελίες για αυτό το τηλέφωνο. Η πιο πρόσφατη με αριθμό {most_recent.order_number} είναι σε κατάσταση {status}. Θέλετε περισσότερες λεπτομέρειες;"
        return f"I found {count} orders for this phone number. Your most recent order {most_recent.order_number} is currently {status}. Would you like more details?"


def get_last_order_snapshot() -> dict | None:
    """Return last looked-up order status info for deterministic responses."""
    order = _last_order_cache.get("last")
    if not order:
        return None
    return {
        "order_number": order.order_number,
        "status": order.status,
    }
