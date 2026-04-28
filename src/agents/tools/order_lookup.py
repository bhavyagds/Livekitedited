"""
Meallion Voice AI - Order Lookup Tool
Handles order status lookups via Shopify API with caching for fast responses.
"""

import logging
from typing import Annotated

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
    
    # Clean the order number
    cleaned = shopify.clean_order_number(order_number)
    
    # Validate format
    if not shopify.validate_order_number(cleaned):
        logger.warning(f"Invalid order number: {order_number} -> {cleaned}")
        return f"That doesn't look like a valid order number. Can you give me the 4 or 5 digit number from your confirmation?"
    
    # Look up order (uses cache if available - instant!)
    logger.info(f"Looking up order: {cleaned}")
    order = await shopify.lookup_order_cached(cleaned)
    
    if order is None:
        logger.info(f"Order not found: {cleaned}")
        return f"I couldn't find order {cleaned}. Could you double-check the number?"
    
    # Store for "more details" requests
    _last_order_cache["last"] = order
    _last_order_cache["number"] = cleaned
    
    # Get language from database settings (not env)
    agent_lang = get_agent_language()
    logger.info(f"Order lookup using language: {agent_lang}")

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
    
    Includes: all items ordered, prices, delivery address, customer info, refund status.
    
    Args:
        order_number: Order number or 'last' for most recent lookup
        
    Returns:
        Complete order details
    """
    shopify = get_shopify_service()
    
    # Check if asking about last order
    if order_number.lower() == "last" and "last" in _last_order_cache:
        order = _last_order_cache["last"]
        logger.info(f"Returning full details for last order: {order.order_number}")
    else:
        # Look up the specific order
        cleaned = shopify.clean_order_number(order_number)
        order = await shopify.lookup_order_cached(cleaned)
        
        if order is None:
            return f"I couldn't find order {cleaned}."
    
    # Get language from database settings (not env)
    agent_lang = get_agent_language()
    logger.info(f"Order details using language: {agent_lang}")

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
    
    Args:
        phone: The phone number to search for
        
    Returns:
        Summary of orders found for this phone number
    """
    shopify = get_shopify_service()
    agent_lang = get_agent_language()
    
    cleaned = shopify.clean_phone_number(phone)
    if not cleaned or len(cleaned) < 8:
        return f"I'm sorry, I couldn't understand that phone number. Could you please say it again?"
        
    logger.info(f"Looking up orders for phone: {cleaned}")
    orders = await shopify.lookup_order_by_phone(cleaned)
    
    if not orders:
        if agent_lang == "el":
            return (
                f"Δεν βρέθηκε καμία παραγγελία συνδεδεμένη με αυτόν τον αριθμό τηλεφώνου: {phone}. "
                "Μπορείτε να ελέγξετε ξανά τον αριθμό και να τον πείτε ψηφίο προς ψηφίο;"
            )
        return (
            f"I couldn't find any orders attached to this phone number: {phone}. "
            "Please double-check the number and say it again digit by digit."
        )
    
    # Store the first/most recent order as "last" for potential detailed lookup
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
        # Multiple orders
        count = len(orders)
        most_recent = orders[0]
        status = most_recent.status
        if agent_lang == "el":
            return f"Βρήκα {count} παραγγελίες για αυτόν τον αριθμό τηλεφώνου. Η πιο πρόσφατη παραγγελία σας με αριθμό {most_recent.order_number} είναι σε κατάσταση {status}. Θέλετε περισσότερες λεπτομέρειες;"
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
