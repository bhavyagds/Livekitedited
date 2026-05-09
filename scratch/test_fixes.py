
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.tools.order_lookup import lookup_order_by_phone
from src.agents.tools.support_ticket import create_support_ticket

async def test_order_lookup_details():
    print("\n--- Testing Order Lookup Details ---")
    # Mock shopify service
    from src.services.shopify import ShopifyService
    mock_shopify = MagicMock(spec=ShopifyService)
    mock_shopify.clean_phone_number.return_value = "6987654321"
    
    # Create a mock order
    mock_order = MagicMock()
    mock_order.order_number = "1234"
    mock_order.status = "Completed"
    
    mock_shopify.lookup_order_by_phone = AsyncMock(return_value=[mock_order])
    mock_shopify.localize_order = AsyncMock()
    mock_shopify.format_order_for_voice.return_value = "Order #1234 is Completed, total 50 Euros, shipping to Athens."
    
    # Patch the get_shopify_service
    import src.agents.tools.order_lookup
    src.agents.tools.order_lookup.get_shopify_service = lambda: mock_shopify
    
    result = await lookup_order_by_phone("6987654321")
    print(f"Result (Should contain full details): {result}")
    if "Athens" in result or "Completed" in result:
        print("SUCCESS: Full details returned.")
    else:
        print("FAILURE: Only brief summary or error returned.")

async def test_support_ticket_guardrails():
    print("\n--- Testing Support Ticket Guardrails ---")
    
    # 1. Missing name
    result = await create_support_ticket(
        customer_name="",
        customer_phone="6987654321",
        customer_email="test@example.com",
        issue_description="Order is delayed"
    )
    print(f"Result (Missing Name): {result}")
    if "full name" in result.lower():
        print("SUCCESS: Missing name guardrail triggered.")
    
    # 2. Missing email
    result = await create_support_ticket(
        customer_name="Naitik Parikh",
        customer_phone="6987654321",
        customer_email="",
        issue_description="Order is delayed"
    )
    print(f"Result (Missing Email): {result}")
    if "email address" in result.lower():
        print("SUCCESS: Missing email guardrail triggered.")

async def main():
    await test_order_lookup_details()
    await test_support_ticket_guardrails()

if __name__ == "__main__":
    asyncio.run(main())
