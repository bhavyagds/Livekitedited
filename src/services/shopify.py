"""
Meallion Voice AI - Shopify API Service
Handles order lookups and customer data retrieval from Shopify Admin API.
Includes order caching for faster responses.
"""

import re
import logging
import asyncio
from typing import Optional, Dict
from datetime import datetime
from dataclasses import dataclass

import httpx

from src.config import settings
from src.utils.greek_numbers import number_to_greek, format_price_greek, format_order_number_greek

logger = logging.getLogger(__name__)


@dataclass
class OrderInfo:
    """Structured order information."""
    order_number: str
    status: str
    fulfillment_status: str
    financial_status: str
    created_at: datetime
    estimated_delivery: Optional[str]
    customer_name: Optional[str]
    total_price: str
    currency: str
    item_count: int
    raw_data: dict
    # Subscription info
    has_subscription: bool = False
    subscription_name: Optional[str] = None
    subscription_frequency: Optional[str] = None
    subscription_bundle_id: Optional[str] = None


# =============================================================================
# ORDER CACHE - Prefetched orders for instant lookups
# =============================================================================
class OrderCache:
    """In-memory cache for prefetched orders."""
    
    def __init__(self):
        self._orders: Dict[str, OrderInfo] = {}
        self._loaded = False
        self._loading = False
    
    def get(self, order_number: str) -> Optional[OrderInfo]:
        """Get order from cache."""
        return self._orders.get(order_number)
    
    def set(self, order_number: str, order: OrderInfo):
        """Add order to cache."""
        self._orders[order_number] = order
    
    def has(self, order_number: str) -> bool:
        """Check if order is in cache."""
        return order_number in self._orders
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def count(self) -> int:
        return len(self._orders)


# Global cache instance
order_cache = OrderCache()


class ShopifyService:
    """
    Shopify Admin API client for order lookups.
    
    Features:
    - Order prefetching (loads recent orders on startup)
    - In-memory caching for instant responses
    - Order lookup by order number
    - Data cleaning for voice input artifacts
    """

    # Number of orders to prefetch
    PREFETCH_COUNT = 50

    def __init__(self):
        self.store_url = settings.shopify_store_url
        self.access_token = settings.shopify_access_token
        self.api_version = "2024-01"
        self._client: Optional[httpx.AsyncClient] = None
        self.cache = order_cache

    @property
    def base_url(self) -> str:
        """Get the Shopify Admin API base URL."""
        store = self.store_url.replace("https://", "").replace("http://", "")
        return f"https://{store}/admin/api/{self.api_version}"

    @property
    def headers(self) -> dict:
        """Get API request headers."""
        return {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def prefetch_recent_orders(self) -> int:
        """
        Prefetch recent orders into cache for instant lookups.
        Called when agent starts a session.
        Returns number of orders cached.
        """
        if self.cache._loading:
            return self.cache.count
        
        self.cache._loading = True
        
        try:
            client = await self.get_client()
            url = f"{self.base_url}/orders.json?limit={self.PREFETCH_COUNT}&status=any"
            
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            orders = data.get("orders", [])
            
            for order_data in orders:
                order_info = self._parse_order(order_data)
                if order_info:
                    # Always use string key for consistency
                    cache_key = str(order_info.order_number)
                    self.cache.set(cache_key, order_info)
                    logger.debug(f"Cached order: {cache_key}")
            
            self.cache._loaded = True
            # Log the range of cached orders for debugging
            if self.cache.count > 0:
                order_nums = list(self.cache._orders.keys())
                logger.info(f"Prefetched {self.cache.count} orders into cache. Range: {min(order_nums)} to {max(order_nums)}")
            else:
                logger.info("Prefetched 0 orders into cache")
            return self.cache.count
            
        except Exception as e:
            logger.error(f"Failed to prefetch orders: {e}")
            return 0
        finally:
            self.cache._loading = False

    async def lookup_order_cached(self, order_number: str) -> Optional[OrderInfo]:
        """
        Look up order - checks cache first, then API.
        Much faster for prefetched orders.
        """
        cleaned = self.clean_order_number(order_number)
        
        # Check cache first (instant)
        if self.cache.has(cleaned):
            logger.info(f"Order {cleaned} found in cache (instant)")
            return self.cache.get(cleaned)
        
        # Not in cache, fetch from API
        logger.info(f"Order {cleaned} not in cache, fetching from API...")
        order = await self.lookup_order_by_number(cleaned)
        
        # Add to cache for future lookups
        if order:
            self.cache.set(cleaned, order)
        
        return order

    @staticmethod
    def clean_order_number(raw_input: str) -> str:
        """
        Clean order number from voice input artifacts.
        
        Handles:
        - Spaces: "1 2 3 4 5" -> "12345"
        - Dashes: "1-2-3-4-5" -> "12345"
        - Words: "one two three four five" -> "12345"
        - Prefixes: "order 12345" -> "12345"
        - Hash: "#12345" -> "12345"
        """
        # Convert word numbers to digits
        word_to_digit = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "oh": "0", "o": "0",
        }
        
        cleaned = raw_input.lower().strip()
        
        # Replace word numbers
        for word, digit in word_to_digit.items():
            cleaned = re.sub(rf'\b{word}\b', digit, cleaned)
        
        # Remove common prefixes
        prefixes = ["order", "number", "order number", "#", "no", "no."]
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        
        # Keep only digits
        cleaned = re.sub(r'[^0-9]', '', cleaned)
        
        return cleaned

    @staticmethod
    def clean_phone_number(raw_input: str) -> str:
        """Clean phone number from voice input."""
        # Remove all non-digits except + for country code
        cleaned = re.sub(r'[^0-9+]', '', raw_input)
        return cleaned

    @staticmethod
    def clean_email(raw_input: str) -> str:
        """Clean email from voice input artifacts."""
        cleaned = raw_input.lower().strip()
        # Handle "at" spoken as word
        cleaned = re.sub(r'\s+at\s+', '@', cleaned)
        # Handle "dot" spoken as word
        cleaned = re.sub(r'\s+dot\s+', '.', cleaned)
        # Remove spaces
        cleaned = cleaned.replace(" ", "")
        return cleaned

    @staticmethod
    def validate_order_number(order_number: str) -> bool:
        """Validate that order number is 3-6 digits."""
        return bool(re.match(r'^\d{3,6}$', order_number))

    async def lookup_order_by_number(self, order_number: str) -> Optional[OrderInfo]:
        """
        Look up order by order number.
        
        Args:
            order_number: The 5-digit order number (raw or cleaned)
            
        Returns:
            OrderInfo if found, None otherwise
        """
        # Clean the input
        cleaned_number = self.clean_order_number(order_number)
        
        if not self.validate_order_number(cleaned_number):
            logger.warning(f"Invalid order number format: {cleaned_number}")
            return None

        try:
            client = await self.get_client()
            
            # Query orders by name (Shopify uses #number format)
            response = await client.get(
                f"{self.base_url}/orders.json",
                params={
                    "name": cleaned_number,
                    "status": "any",
                    "limit": 1,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            orders = data.get("orders", [])
            if not orders:
                # Try with # prefix
                response = await client.get(
                    f"{self.base_url}/orders.json",
                    params={
                        "name": f"#{cleaned_number}",
                        "status": "any",
                        "limit": 1,
                    }
                )
                response.raise_for_status()
                data = response.json()
                orders = data.get("orders", [])
            
            if orders:
                return self._parse_order(orders[0])
            
            logger.info(f"No order found for number: {cleaned_number}")
            return None

        except httpx.HTTPError as e:
            logger.error(f"Shopify API error: {e}")
            return None

    async def lookup_order_by_email(self, email: str) -> list[OrderInfo]:
        """Look up orders by customer email."""
        cleaned_email = self.clean_email(email)
        
        try:
            client = await self.get_client()
            
            response = await client.get(
                f"{self.base_url}/orders.json",
                params={
                    "email": cleaned_email,
                    "status": "any",
                    "limit": 5,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return [self._parse_order(order) for order in data.get("orders", [])]

        except httpx.HTTPError as e:
            logger.error(f"Shopify API error: {e}")
            return []

    async def lookup_order_by_phone(self, phone: str) -> list[OrderInfo]:
        """Look up orders by customer phone number."""
        cleaned_phone = self.clean_phone_number(phone)
        
        try:
            client = await self.get_client()
            
            # Search customers first
            response = await client.get(
                f"{self.base_url}/customers/search.json",
                params={"query": f"phone:{cleaned_phone}"}
            )
            response.raise_for_status()
            customers = response.json().get("customers", [])
            
            if not customers:
                return []
            
            # Get orders for the first matching customer
            customer_id = customers[0]["id"]
            response = await client.get(
                f"{self.base_url}/customers/{customer_id}/orders.json",
                params={"status": "any", "limit": 5}
            )
            response.raise_for_status()
            data = response.json()
            
            return [self._parse_order(order) for order in data.get("orders", [])]

        except httpx.HTTPError as e:
            logger.error(f"Shopify API error: {e}")
            return []

    def _parse_order(self, order_data: dict) -> OrderInfo:
        """Parse Shopify order data into OrderInfo."""
        # Get fulfillment status
        fulfillments = order_data.get("fulfillments", [])
        if fulfillments:
            fulfillment_status = fulfillments[-1].get("status", "unfulfilled")
            # Try to get tracking/estimated delivery
            tracking = fulfillments[-1].get("tracking_info", {})
            estimated_delivery = tracking.get("estimated_delivery_at")
        else:
            fulfillment_status = order_data.get("fulfillment_status") or "unfulfilled"
            estimated_delivery = None

        # Get customer name
        customer = order_data.get("customer", {})
        customer_name = None
        if customer:
            first = customer.get("first_name", "")
            last = customer.get("last_name", "")
            customer_name = f"{first} {last}".strip() or None

        # Count items and detect Loop Subscriptions
        line_items = order_data.get("line_items", [])
        item_count = sum(item.get("quantity", 1) for item in line_items)
        
        # Detect Loop Subscription from line item properties
        has_subscription = False
        subscription_name = None
        subscription_frequency = None
        subscription_bundle_id = None
        
        for item in line_items:
            properties = item.get("properties", [])
            for prop in properties:
                prop_name = prop.get("name", "")
                prop_value = prop.get("value", "")
                
                # Loop Subscriptions uses these property names
                if prop_name == "_bundleName":
                    has_subscription = True
                    subscription_name = prop_value
                elif prop_name == "_bundleId":
                    subscription_bundle_id = prop_value
                # Detect frequency from Greek/English text
                elif "εβδομάδα" in prop_value.lower() or "week" in prop_value.lower():
                    subscription_frequency = prop_value
                elif "μήνα" in prop_value.lower() or "month" in prop_value.lower():
                    subscription_frequency = prop_value
                # Also check for subscription selling plan
                if item.get("selling_plan_allocation"):
                    has_subscription = True
                    plan = item.get("selling_plan_allocation", {}).get("selling_plan", {})
                    if not subscription_name:
                        subscription_name = plan.get("name", "Subscription")

        # Determine overall status
        if order_data.get("cancelled_at"):
            status = "cancelled"
        elif order_data.get("closed_at"):
            status = "completed"
        elif fulfillment_status == "fulfilled":
            status = "delivered"
        elif fulfillment_status in ["partial", "in_transit"]:
            status = "in_transit"
        else:
            status = "processing"

        # Use 'name' field (e.g., "#12614") for order number - this matches what users see in Shopify
        # The 'order_number' field is just the sequential position, not the display number
        order_name = order_data.get("name", "")
        # Strip # and any other prefixes to get just the number
        order_number_str = order_name.replace("#", "").strip() if order_name else str(order_data.get("order_number", ""))
        
        return OrderInfo(
            order_number=order_number_str,
            status=status,
            fulfillment_status=fulfillment_status,
            financial_status=order_data.get("financial_status", "unknown"),
            created_at=datetime.fromisoformat(
                order_data.get("created_at", "").replace("Z", "+00:00")
            ) if order_data.get("created_at") else datetime.now(),
            estimated_delivery=estimated_delivery,
            customer_name=customer_name,
            total_price=order_data.get("total_price", "0.00"),
            currency=order_data.get("currency", "EUR"),
            item_count=item_count,
            raw_data=order_data,
            has_subscription=has_subscription,
            subscription_name=subscription_name,
            subscription_frequency=subscription_frequency,
            subscription_bundle_id=subscription_bundle_id,
        )

    def format_order_brief(self, order: OrderInfo, language: str = None) -> str:
        """
        Format brief order summary for initial response.
        Includes: status, delivery date, total, subscription, customer, address.
        Supports Greek number pronunciation when language is 'el'.
        """
        raw = order.raw_data
        lang = language or settings.agent_language
        is_greek = lang == "el"
        
        # Status text in Greek or English
        if is_greek:
            status_text = {
                "processing": "ετοιμάζεται",
                "in_transit": "σε μεταφορά",
                "delivered": "παραδόθηκε",
                "completed": "ολοκληρώθηκε",
                "cancelled": "ακυρώθηκε",
            }
        else:
            status_text = {
                "processing": "being prepared",
                "in_transit": "on the way",
                "delivered": "delivered",
                "completed": "completed",
                "cancelled": "cancelled",
            }
        status = status_text.get(order.status, order.status)
        
        # Get delivery date from note_attributes
        delivery_date = None
        for attr in raw.get("note_attributes", []):
            if attr.get("name") == "Delivery-Date":
                delivery_date = attr.get("value")
                break
        
        # Customer info
        customer = raw.get("customer", {})
        customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip() or ("Άγνωστο" if is_greek else "Unknown")
        
        # Shipping address
        shipping = raw.get("shipping_address", {})
        if shipping:
            address_parts = [
                shipping.get('address1', ''),
                shipping.get('city', ''),
                shipping.get('zip', '')
            ]
            shipping_address = ", ".join(part for part in address_parts if part)
        else:
            shipping_address = "Δεν δόθηκε" if is_greek else "Not provided"
        
        # Format order number and price for Greek pronunciation
        if is_greek:
            order_num_spoken = format_order_number_greek(order.order_number)
            price_spoken = format_price_greek(float(order.total_price), order.currency)
            
            response = f"Η παραγγελία {order_num_spoken} {status}."
            if delivery_date:
                response += f" Προγραμματισμένη παράδοση στις {delivery_date}."
            response += f" Σύνολο: {price_spoken}."
            response += f" Πελάτης: {customer_name}."
            response += f" Διεύθυνση παράδοσης: {shipping_address}."
            
            # Subscription info in Greek
            if order.has_subscription:
                response += f" Αυτή είναι παραγγελία ΣΥΝΔΡΟΜΗΣ"
                if order.subscription_name:
                    response += f" ({order.subscription_name})"
                if order.subscription_frequency:
                    response += f" - {order.subscription_frequency}"
                response += "."
            
            response += " Θέλετε περισσότερες λεπτομέρειες για αυτή την παραγγελία;"
        else:
            response = f"Order {order.order_number} is {status}."
            if delivery_date:
                response += f" Scheduled for delivery on {delivery_date}."
            response += f" Total: {order.total_price} {order.currency}."
            response += f" Customer: {customer_name}."
            response += f" Delivery address: {shipping_address}."
            
            # Add subscription info
            if order.has_subscription:
                response += f" This is a SUBSCRIPTION order"
                if order.subscription_name:
                    response += f" ({order.subscription_name})"
                if order.subscription_frequency:
                    response += f" - {order.subscription_frequency}"
                response += "."
            
            response += " Would you like more details about this order?"
        
        return response

    def format_order_for_voice(self, order: OrderInfo, include_details: bool = True, language: str = None) -> str:
        """
        Format COMPLETE order information for voice response.
        Returns all details so agent can answer any question.
        """
        raw = order.raw_data
        lang = language or settings.agent_language
        is_greek = lang == "el"
        
        # Extract all items with details - formatted naturally for voice
        items_list = []
        line_items = raw.get("line_items", [])
        for item in line_items:
            item_name = item.get("title", "Unknown item")
            quantity = item.get("quantity", 1)
            price = item.get("price", "0")
            # Format naturally for voice
            # Note: Item names are from Shopify, keep them as-is (may be in Greek)
            if is_greek:
                if quantity == 1:
                    items_list.append(f"- {item_name}, {price} {order.currency}")
                else:
                    items_list.append(f"- {quantity} τεμάχια {item_name}, {price} {order.currency} το ένα")
            else:
                if quantity == 1:
                    items_list.append(f"- {item_name}, {price} {order.currency}")
                else:
                    items_list.append(f"- {quantity} of {item_name}, {price} {order.currency} each")
        
        items_text = "\n".join(items_list) if items_list else "No items"
        
        # Customer info
        customer = raw.get("customer", {})
        customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip() or "Unknown"
        customer_email = customer.get("email", "Not provided")
        customer_phone = raw.get("phone") or customer.get("phone", "Not provided")
        
        # Shipping address
        shipping = raw.get("shipping_address", {})
        shipping_address = f"{shipping.get('address1', '')}, {shipping.get('city', '')}, {shipping.get('zip', '')}" if shipping else "Not provided"
        
        # Financial info
        financial_status = raw.get("financial_status", "unknown")
        refund_status = "No refunds"
        refunds = raw.get("refunds", [])
        if refunds:
            total_refunded = sum(float(r.get("transactions", [{}])[0].get("amount", 0)) for r in refunds if r.get("transactions"))
            refund_status = f"Refunded: {total_refunded} {order.currency}"
        
        # Delivery info from note_attributes
        delivery_date = "Not scheduled"
        delivery_time = ""
        for attr in raw.get("note_attributes", []):
            if attr.get("name") == "Delivery-Date":
                delivery_date = attr.get("value", "Not scheduled")
            if attr.get("name") == "Delivery-Time":
                delivery_time = attr.get("value", "")
        
        # Status
        status_text = {
            "processing": "being prepared",
            "in_transit": "on the way",
            "delivered": "delivered",
            "completed": "completed",
            "cancelled": "cancelled",
        }
        status = status_text.get(order.status, order.status)
        
        # Subscription info
        subscription_info = "No subscription"
        if order.has_subscription:
            subscription_info = "YES - This is a SUBSCRIPTION order"
            if order.subscription_name:
                subscription_info += f"\n  - Plan: {order.subscription_name}"
            if order.subscription_frequency:
                subscription_info += f"\n  - Frequency: {order.subscription_frequency}"
            if order.subscription_bundle_id:
                subscription_info += f"\n  - Bundle ID: {order.subscription_bundle_id}"
        
        # Build comprehensive response
        response = f"""ORDER DETAILS FOR #{order.order_number}:
- Status: {status}
- Customer: {customer_name}
- Email: {customer_email}
- Phone: {customer_phone}
- Delivery Address: {shipping_address}
- Delivery Date: {delivery_date} {delivery_time}
- Payment: {financial_status}
- Refund Status: {refund_status}
- Total: {order.total_price} {order.currency}
- SUBSCRIPTION: {subscription_info}
- Items ({order.item_count}):
{items_text}

Use this information to answer any customer questions about this order."""
        
        return response


# Singleton instance
_shopify_service: Optional[ShopifyService] = None


def get_shopify_service() -> ShopifyService:
    """Get or create Shopify service instance."""
    global _shopify_service
    if _shopify_service is None:
        _shopify_service = ShopifyService()
    return _shopify_service
