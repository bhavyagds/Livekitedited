# src/services/n8n.py
"""
Meallion Voice AI - n8n Ticket Service
Handles progressive ticket creation via n8n webhook.
Each step fires a separate call so tickets are persisted incrementally.
"""

import httpx
import logging
from src.config import settings

logger = logging.getLogger(__name__)


async def _call_n8n(payload: dict) -> dict:
    """Base function — all n8n calls go through here."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            settings.n8n_webhook_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {settings.n8n_webhook_secret}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()


async def create_draft_ticket(
    call_id: str,
    name: str,
    language: str = "en",
) -> str:
    """
    Step 1 — Customer gives name.
    Creates a DRAFT ticket in n8n/ClickUp.
    Returns ticket_id which must be stored in session state.
    """
    result = await _call_n8n(
        {
            "action": "create",
            "call_id": call_id,
            "field": "name",
            "value": name,
            "language": language,
            "source": "voice_agent",
        }
    )
    ticket_id = result.get("ticket_id", "")
    logger.info(f"Draft ticket created: {ticket_id}")
    return ticket_id


async def update_ticket_field(
    ticket_id: str,
    field: str,  # "email" | "phone" | "issue"
    value: str,
) -> dict:
    """
    Step 2/3/4 — Customer gives email, phone, or issue.
    Updates a single field on the existing draft ticket.
    """
    result = await _call_n8n(
        {
            "action": "update",
            "ticket_id": ticket_id,
            "field": field,
            "value": value,
        }
    )
    logger.info(f"Ticket {ticket_id} updated: {field}")
    return result


async def submit_ticket(ticket_id: str) -> dict:
    """
    Final step — Customer confirms all details.
    Changes ticket status from DRAFT → OPEN in ClickUp.
    Triggers email + Slack notification in n8n.
    """
    result = await _call_n8n(
        {
            "action": "submit",
            "ticket_id": ticket_id,
        }
    )
    logger.info(f"Ticket {ticket_id} submitted successfully")
    return result


async def abandon_ticket(ticket_id: str) -> dict:
    """
    Cleanup — Customer hung up before completing.
    Archives/deletes the draft ticket in ClickUp.
    """
    result = await _call_n8n(
        {
            "action": "abandon",
            "ticket_id": ticket_id,
        }
    )
    logger.info(f"Ticket {ticket_id} abandoned and cleaned up")
    return result
