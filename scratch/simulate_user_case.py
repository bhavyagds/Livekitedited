
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.tools.support_ticket import create_support_ticket

async def run_specific_simulation():
    print("="*50)
    print("ELENA SPECIFIC CASE SIMULATION")
    print("="*50)

    # USER DATA
    name = "Naitik"
    phone = "8239507535"
    email = "naitiik@gmail.com"
    issue = "bad food or not good packaging"

    print(f"\n[INPUT DATA]")
    print(f"Name: {name}")
    print(f"Phone: {phone}")
    print(f"Email: {email}")
    print(f"Issue: {issue}")

    # Mock ClickUp Service
    from src.services.clickup import ClickUpService
    mock_clickup = MagicMock(spec=ClickUpService)
    mock_clickup.create_support_ticket = AsyncMock(return_value={"success": True, "task_id": "86c9q1abc"})
    
    import src.agents.tools.support_ticket
    src.agents.tools.support_ticket.clickup_service = mock_clickup

    print("\n[SIMULATING TOOL CALL...]")
    response = await create_support_ticket(
        customer_name=name,
        customer_phone=phone,
        customer_email=email,
        issue_description=issue
    )
    
    print(f"\n[ELENA RESPONSE]")
    print(response)

    if "86c9q1abc" in response:
        print("\nSUCCESS: Ticket created with all user details!")
    else:
        print("\nFAILURE: Ticket creation failed or blocked.")

    print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(run_specific_simulation())
