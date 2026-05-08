"""
Meallion Voice AI - Shopify API Service
Handles order lookups and customer data retrieval from Shopify Admin API.
Includes order caching for faster responses.
"""

import re
import logging
import asyncio
import json
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from src.config import settings
from src.utils.greek_numbers import number_to_greek, format_price_greek, format_order_number_greek
from src.utils.numbers import extract_digits
from src.utils.voice_formatting import format_date_for_voice, format_currency_for_voice, format_order_number_for_voice

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
        self._translator: Optional[AsyncOpenAI] = None
        self._translation_cache: Dict[Tuple[str, str], str] = {}
        self._greek_re = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
        self._latin_re = re.compile(r"[A-Za-z]")

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

    def _get_translator(self) -> Optional[AsyncOpenAI]:
        if self._translator is not None:
            return self._translator
        if not settings.openai_api_key:
            logger.warning("OpenAI API key missing; skipping translation.")
            return None
        self._translator = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._translator

    def _contains_greek(self, text: str) -> bool:
        return bool(self._greek_re.search(text))

    def _contains_latin(self, text: str) -> bool:
        return bool(self._latin_re.search(text))

    def _needs_translation(self, text: str, target_lang: str) -> bool:
        if not text:
            return False
        if target_lang == "en":
            return self._contains_greek(text)
        if target_lang == "el":
            return self._contains_latin(text)
        return False

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", content).strip()
            content = re.sub(r"\n?```$", "", content).strip()
        return content

    async def _translate_texts(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate a list of strings into target_lang using OpenAI."""
        if not texts:
            return []

        client = self._get_translator()
        if client is None:
            return texts

        target_name = "English" if target_lang == "en" else "Greek"
        system_prompt = (
            "You are a translation engine for a female customer support assistant named Elena. "
            f"Translate every string into {target_name}. "
            "Use feminine gender forms for self-references in Greek. "
            "Preserve numbers, order IDs, emails, phone numbers, and currency amounts exactly. "
            "For names/addresses, translate or transliterate into the target script. "
            "Keep brand names (e.g., Meallion) unchanged. "
            "Return ONLY a JSON array of strings, same length and order as input."
        )
        user_payload = json.dumps(texts, ensure_ascii=False)

        try:
            response = await client.chat.completions.create(
                model=settings.openai_model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            content = self._strip_json_fence(content)
            translated = json.loads(content)
            if not isinstance(translated, list) or len(translated) != len(texts):
                logger.warning("Translation returned unexpected format; using original texts.")
                return texts
            return [str(item) for item in translated]
        except Exception as e:
            logger.warning(f"Translation failed; using original texts. Error: {e}")
            return texts

    def _get_localized(self, order: OrderInfo, lang: str, key: str, default: str):
        localized = order.raw_data.get("_localized", {})
        return localized.get(lang, {}).get(key) or default

    async def localize_order(self, order: OrderInfo, language: str) -> OrderInfo:
        """Translate order fields to the target language and cache results on the order."""
        target_lang = language or settings.agent_language
        if target_lang not in {"en", "el"}:
            return order

        localized_bucket = order.raw_data.setdefault("_localized", {})
        if target_lang in localized_bucket:
            return order

        raw = order.raw_data

        customer_name = order.customer_name or ""
        shipping = raw.get("shipping_address", {})
        shipping_address = ""
        if shipping:
            address_parts = [
                shipping.get("address1", ""),
                shipping.get("city", ""),
                shipping.get("zip", ""),
            ]
            shipping_address = ", ".join(part for part in address_parts if part)

        delivery_date = ""
        delivery_time = ""
        for attr in raw.get("note_attributes", []):
            if attr.get("name") == "Delivery-Date":
                delivery_date = attr.get("value", "")
            if attr.get("name") == "Delivery-Time":
                delivery_time = attr.get("value", "")

        line_items = raw.get("line_items", [])
        item_titles = [item.get("title", "") for item in line_items]

        subscription_name = order.subscription_name or ""
        subscription_frequency = order.subscription_frequency or ""

        fields = [
            customer_name,
            shipping_address,
            delivery_date,
            delivery_time,
            subscription_name,
            subscription_frequency,
            *item_titles,
        ]

        results: List[str] = list(fields)
        to_translate: List[str] = []
        translate_indices: List[int] = []

        for idx, text in enumerate(fields):
            key = (target_lang, text)
            if text and key in self._translation_cache:
                results[idx] = self._translation_cache[key]
                continue
            if self._needs_translation(text, target_lang):
                to_translate.append(text)
                translate_indices.append(idx)

        if to_translate:
            translated = await self._translate_texts(to_translate, target_lang)
            for idx, translated_text in zip(translate_indices, translated):
                original_text = fields[idx]
                self._translation_cache[(target_lang, original_text)] = translated_text
                results[idx] = translated_text

        localized_bucket[target_lang] = {
            "customer_name": results[0],
            "shipping_address": results[1],
            "delivery_date": results[2],
            "delivery_time": results[3],
            "subscription_name": results[4],
            "subscription_frequency": results[5],
            "item_titles": results[6:],
        }
        return order

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
            logger.info(f"[SHOPIFY_API] Order {cleaned} found in cache (instant hit)")
            return self.cache.get(cleaned)
        
        # Not in cache, fetch from API
        logger.info(f"[SHOPIFY_API] Order {cleaned} NOT in cache, fetching from live API...")
        order = await self.lookup_order_by_number(cleaned)
        
        # Add to cache for future lookups
        if order:
            self.cache.set(cleaned, order)
        
        return order

    @staticmethod
    def clean_order_number(raw_input: str) -> str:
        """Clean order number from voice input artifacts using robust extraction."""
        cleaned = extract_digits(raw_input)
        return cleaned

    @staticmethod
    def clean_phone_number(raw_input: str) -> str:
        """Clean phone number from voice input using robust digit extraction."""
        return extract_digits(raw_input)

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
        """Validate that order number is 3-6 digits (covers all Shopify store ranges)."""
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
            logger.warning(f"[SHOPIFY_API] Invalid order number format: {cleaned_number}")
            return None

        try:
            client = await self.get_client()
            url = f"{self.base_url}/orders.json"
            
            logger.info(f"[SHOPIFY_API] Calling Shopify API: {url} for order {cleaned_number}")
            # Query orders by name (Shopify uses #number format)
            response = await client.get(
                url,
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
                logger.info(f"[SHOPIFY_API] No match for {cleaned_number}, trying with # prefix")
                # Try with # prefix
                response = await client.get(
                    url,
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
                order_info = self._parse_order(orders[0])
                logger.info(f"[SHOPIFY_API] SUCCESS: Found order {cleaned_number} (Shopify Name: {orders[0].get('name')})")
                return order_info
            
            logger.info(f"[SHOPIFY_API] NOT_FOUND: No order matching {cleaned_number} in Shopify")
            return None

        except httpx.HTTPError as e:
            logger.error(f"[SHOPIFY_API] Shopify API request failed: {e}")
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
        """
        Look up orders by customer phone number.

        Shopify stores phone numbers in many formats (+country-code + local, local-only, etc.).
        This method tries several variants so a missed country-code prefix never causes a false
        "not found" result.  It short-circuits as soon as a customer record is found.
        """
        cleaned_phone = self.clean_phone_number(phone)

        # Build candidate query strings in priority order:
        # 1. Plain digits as spoken (e.g. 6912345678)
        # 2. International format with + prefix (e.g. +6912345678)
        # 3. Greek country code (+30) – most common for this deployment
        # 4. US/Canada country code (+1) – common if store has international customers
        # 5. Digits without leading country-code zero stripping
        candidates: list[str] = [cleaned_phone]
        if not cleaned_phone.startswith("+"):
            candidates.append(f"+{cleaned_phone}")
            # Greek (+30): local numbers are 10 digits; stored as +30XXXXXXXXXX
            if len(cleaned_phone) == 10 and not cleaned_phone.startswith("30"):
                candidates.append(f"+30{cleaned_phone}")
            # If caller already included country code 30, also try without it
            if cleaned_phone.startswith("30") and len(cleaned_phone) > 10:
                candidates.append(f"+{cleaned_phone}")
                candidates.append(cleaned_phone[2:])  # strip 30 prefix
            # US/Canada (+1): local numbers are 10 digits
            if len(cleaned_phone) == 10 and not cleaned_phone.startswith("1"):
                candidates.append(f"+1{cleaned_phone}")

        # De-duplicate while preserving order
        seen: set[str] = set()
        unique_candidates: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        try:
            client = await self.get_client()

            customer_id: int | None = None
            matched_format: str | None = None

            for fmt in unique_candidates:
                logger.info(f"[SHOPIFY_API] PHONE_SEARCH: Searching for phone pattern {fmt!r}")
                response = await client.get(
                    f"{self.base_url}/customers/search.json",
                    params={"query": f"phone:{fmt}", "limit": 5},
                )
                response.raise_for_status()
                customers = response.json().get("customers", [])
                if customers:
                    customer_id = customers[0]["id"]
                    matched_format = fmt
                    logger.info(
                        f"[SHOPIFY_API] CUSTOMER_FOUND: Found customer ID {customer_id} for pattern {fmt!r} "
                        f"({customers[0].get('first_name')} {customers[0].get('last_name')})"
                    )
                    break

            if customer_id is None:
                logger.info(
                    f"[SHOPIFY_API] CUSTOMER_NOT_FOUND: No customer matched {cleaned_phone!r} "
                    f"after trying {len(unique_candidates)} pattern(s)"
                )
                return []

            # Retrieve the customer's recent orders
            logger.info(f"[SHOPIFY_API] FETCHING_ORDERS: Getting orders for customer ID {customer_id}")
            response = await client.get(
                f"{self.base_url}/customers/{customer_id}/orders.json",
                params={"status": "any", "limit": 5},
            )
            response.raise_for_status()
            data = response.json()
            orders_data = data.get("orders", [])
            orders = [self._parse_order(o) for o in orders_data]
            
            order_summary = ", ".join([f"#{o.order_number} ({o.status})" for o in orders])
            logger.info(
                f"[SHOPIFY_API] PHONE_LOOKUP_SUCCESS: Found {len(orders)} order(s) for customer {customer_id}: {order_summary}"
            )
            return orders

        except httpx.HTTPError as e:
            logger.error(f"Shopify API error during phone lookup: {e}")
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

        # Apply localized fields when available
        customer_name = self._get_localized(order, lang, "customer_name", customer_name)
        shipping_address = self._get_localized(order, lang, "shipping_address", shipping_address)
        if delivery_date:
            delivery_date = self._get_localized(order, lang, "delivery_date", delivery_date)
        
        # Format order number, price and date for natural speech
        order_num_spoken = format_order_number_for_voice(order.order_number, lang)
        price_spoken = format_currency_for_voice(float(order.total_price), order.currency, lang)
        date_spoken = format_date_for_voice(delivery_date, lang) if delivery_date else None

        if is_greek:
            response = f"Η παραγγελία {order_num_spoken} {status}."
            if date_spoken:
                response += f" Προγραμματισμένη παράδοση στις {date_spoken}."
            response += f" Σύνολο: {price_spoken}."
            response += f" Πελάτης: {customer_name}."
            response += f" Διεύθυνση παράδοσης: {shipping_address}."
            
            # Subscription info in Greek
            if order.has_subscription:
                response += f" Αυτή είναι παραγγελία ΣΥΝΔΡΟΜΗΣ"
                if order.subscription_name:
                    subscription_name = self._get_localized(order, lang, "subscription_name", order.subscription_name)
                    response += f" ({subscription_name})"
                if order.subscription_frequency:
                    subscription_frequency = self._get_localized(order, lang, "subscription_frequency", order.subscription_frequency)
                    response += f" - {subscription_frequency}"
                response += "."
            
            response += " Θέλετε περισσότερες λεπτομέρειες για αυτή την παραγγελία;"
        else:
            response = f"Order {order_num_spoken} is {status}."
            if date_spoken:
                response += f" Scheduled for delivery on {date_spoken}."
            response += f" Total: {price_spoken}."
            response += f" Customer: {customer_name}."
            response += f" Delivery address: {shipping_address}."
            
            # Add subscription info
            if order.has_subscription:
                response += f" This is a SUBSCRIPTION order"
                if order.subscription_name:
                    subscription_name = self._get_localized(order, lang, "subscription_name", order.subscription_name)
                    response += f" ({subscription_name})"
                if order.subscription_frequency:
                    subscription_frequency = self._get_localized(order, lang, "subscription_frequency", order.subscription_frequency)
                    response += f" - {subscription_frequency}"
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
        localized_titles = self._get_localized(order, lang, "item_titles", None)
        for idx, item in enumerate(line_items):
            fallback_title = item.get("title", "Unknown item")
            if isinstance(localized_titles, list) and idx < len(localized_titles):
                item_name = localized_titles[idx] or fallback_title
            else:
                item_name = fallback_title
            quantity = item.get("quantity", 1)
            price_raw = item.get("price", "0")
            price_val = float(price_raw)
            price_spoken = format_currency_for_voice(price_val, order.currency, lang)
            # Format naturally for voice
            # Note: Item names are from Shopify, keep them as-is (may be in Greek)
            if is_greek:
                if quantity == 1:
                    items_list.append(f"- {item_name}, {price_spoken}")
                else:
                    items_list.append(f"- {quantity} τεμάχια {item_name}, {price_spoken} το ένα")
            else:
                if quantity == 1:
                    items_list.append(f"- {item_name}, {price_spoken}")
                else:
                    items_list.append(f"- {quantity} of {item_name}, {price_spoken} each")
        
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
            refund_spoken = format_currency_for_voice(total_refunded, order.currency, lang)
            refund_status = f"Refunded: {refund_spoken}"
        
        # Delivery info from note_attributes
        delivery_date = "Not scheduled"
        delivery_time = ""
        for attr in raw.get("note_attributes", []):
            if attr.get("name") == "Delivery-Date":
                delivery_date = attr.get("value", "Not scheduled")
            if attr.get("name") == "Delivery-Time":
                delivery_time = attr.get("value", "")

        # Apply localized fields when available
        customer_name = self._get_localized(order, lang, "customer_name", customer_name)
        shipping_address = self._get_localized(order, lang, "shipping_address", shipping_address)
        delivery_date = self._get_localized(order, lang, "delivery_date", delivery_date)
        if delivery_time:
            delivery_time = self._get_localized(order, lang, "delivery_time", delivery_time)
        
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
                subscription_name = self._get_localized(order, lang, "subscription_name", order.subscription_name)
                subscription_info += f"\n  - Plan: {subscription_name}"
            if order.subscription_frequency:
                subscription_frequency = self._get_localized(order, lang, "subscription_frequency", order.subscription_frequency)
                subscription_info += f"\n  - Frequency: {subscription_frequency}"
            if order.subscription_bundle_id:
                subscription_info += f"\n  - Bundle ID: {order.subscription_bundle_id}"
        
        # Spoken versions for the summary
        order_num_spoken = format_order_number_for_voice(order.order_number, lang)
        date_spoken = format_date_for_voice(delivery_date, lang) if delivery_date != "Not scheduled" else delivery_date
        total_spoken = format_currency_for_voice(float(order.total_price), order.currency, lang)

        # Build comprehensive response
        response = f"""ORDER DETAILS FOR #{order_num_spoken}:
- Status: {status}
- Customer: {customer_name}
- Email: {customer_email}
- Phone: {customer_phone}
- Delivery Address: {shipping_address}
- Delivery Date: {date_spoken} {delivery_time}
- Payment: {financial_status}
- Refund Status: {refund_status}
- Total: {total_spoken}
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
