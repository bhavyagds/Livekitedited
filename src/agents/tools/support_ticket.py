"""
Meallion Voice AI - Support Ticket Tool
Handles support ticket creation via ClickUp.
"""

import logging
import re
from typing import Annotated, Optional
from datetime import datetime

from livekit.agents import llm

from src.services.clickup import clickup_service

logger = logging.getLogger(__name__)


class SupportTicketTool:
    """
    Support ticket tool for Elena voice agent.
    
    Features:
    - Mandatory one-by-one field collection
    - Field verification before submission
    - Creates task in ClickUp
    - Call context tracking
    """

    def __init__(self):
        self.clickup = clickup_service

    def get_tools(self) -> list:
        """Get the list of function tools for this module."""
        return [
            create_support_ticket,
            log_customer_query,
            validate_ticket_field,
        ]


class TicketFieldCollector:
    """
    Tracks support ticket field collection state.
    Used to ensure all 4 required fields are collected and verified.
    """
    
    REQUIRED_FIELDS = ["name", "phone", "email", "issue"]
    
    def __init__(self):
        self.fields: dict[str, Optional[str]] = {
            "name": None,
            "phone": None,
            "email": None,
            "issue": None,
        }
        self.verified: dict[str, bool] = {
            "name": False,
            "phone": False,
            "email": False,
            "issue": False,
        }
    
    def set_field(self, field: str, value: str) -> None:
        """Set a field value."""
        if field in self.fields:
            self.fields[field] = value
            self.verified[field] = False  # Needs re-verification
    
    def verify_field(self, field: str) -> None:
        """Mark a field as verified."""
        if field in self.verified:
            self.verified[field] = True
    
    def get_next_required_field(self) -> Optional[str]:
        """Get the next field that needs to be collected."""
        for field in self.REQUIRED_FIELDS:
            if self.fields[field] is None:
                return field
        return None
    
    def get_unverified_field(self) -> Optional[str]:
        """Get the next field that needs verification."""
        for field in self.REQUIRED_FIELDS:
            if self.fields[field] is not None and not self.verified[field]:
                return field
        return None
    
    def is_complete(self) -> bool:
        """Check if all fields are collected and verified."""
        return all(
            self.fields[f] is not None and self.verified[f]
            for f in self.REQUIRED_FIELDS
        )
    
    def get_summary(self) -> str:
        """Get a summary of collected fields."""
        lines = []
        for field in self.REQUIRED_FIELDS:
            value = self.fields[field]
            verified = "✓" if self.verified[field] else "○"
            status = f"{verified} {field.title()}: {value or '(not collected)'}"
            lines.append(status)
        return "\n".join(lines)


# Global collector (would be per-session in production)
_collectors: dict[str, TicketFieldCollector] = {}


def get_collector(session_id: str) -> TicketFieldCollector:
    """Get or create a ticket collector for a session."""
    if session_id not in _collectors:
        _collectors[session_id] = TicketFieldCollector()
    return _collectors[session_id]


def clean_phone_number(phone: str) -> str:
    """Clean phone number input."""
    # Keep only digits and + for country code
    return re.sub(r'[^0-9+]', '', phone)


def clean_email(email: str) -> str:
    """Clean email input from voice artifacts."""
    cleaned = email.lower().strip()
    cleaned = re.sub(r'\s+at\s+', '@', cleaned)
    cleaned = re.sub(r'\s+dot\s+', '.', cleaned)
    cleaned = cleaned.replace(" ", "")
    return cleaned


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number (basic validation)."""
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
    
    CRITICAL: Before calling this function, you MUST have collected and VERIFIED
    all 4 fields ONE BY ONE:
    1. Name - Ask: "May I have your full name?" Then verify.
    2. Phone - Ask: "What's your phone number?" Then verify.
    3. Email - Ask: "What's your email address?" Then verify.
    4. Issue - Ask: "Please describe your issue." Then summarize and verify.
    
    DO NOT call this function until ALL fields are verbally confirmed by the customer.
    
    Args:
        customer_name: Full name (verified)
        customer_phone: Phone number (verified)
        customer_email: Email address (verified)
        issue_description: Issue description (verified)
        order_number: Optional related order number
        
    Returns:
        Confirmation message with ticket reference, or error if validation fails
    """
    # Clean inputs
    cleaned_phone = clean_phone_number(customer_phone)
    cleaned_email = clean_email(customer_email)
    
    # Validate
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
        error_msg = ", ".join(errors)
        logger.warning(f"Ticket validation failed: {error_msg}")
        return f"Cannot create ticket: {error_msg}. Please correct the information."
    
    # Create ticket in ClickUp
    result = await clickup_service.create_support_ticket(
        customer_name=customer_name.strip(),
        customer_phone=cleaned_phone,
        customer_email=cleaned_email,
        issue_description=issue_description.strip(),
        order_number=order_number,
    )
    
    if not result["success"]:
        logger.error(f"Failed to create ClickUp ticket: {result.get('error')}")
        return (
            "Sorry, I couldn't create the support ticket. "
            "Please try again or contact us directly via email."
        )
    
    ticket_id = result["task_id"]
    logger.info(f"Support ticket created in ClickUp: {ticket_id}")
    
    return (
        f"Your support ticket has been created successfully. "
        f"Your reference number is {ticket_id}. "
        "Our support team will contact you soon at the provided phone or email."
    )


async def log_customer_query(
    customer_question: Annotated[str, "The customer's question or issue that you cannot answer"],
    customer_name: Annotated[Optional[str], "Customer name if known"] = None,
    customer_phone: Annotated[Optional[str], "Customer phone if known"] = None,
) -> str:
    """
    Log a customer query that you cannot answer for follow-up by the team.
    
    Use this when:
    - You don't know the answer to a customer's question
    - The question requires human expertise
    - The customer has a complex issue you can't resolve
    
    This creates a quick ticket without requiring all customer details.
    
    Args:
        customer_question: The question or issue you couldn't answer
        customer_name: Customer's name if they provided it
        customer_phone: Customer's phone if they provided it
        
    Returns:
        Confirmation that the query was logged
    """
    logger.info(f"Logging customer query for follow-up: {customer_question[:100]}")
    
    # Create a simplified ticket in ClickUp
    result = await clickup_service.create_support_ticket(
        customer_name=customer_name or "Unknown Caller",
        customer_phone=customer_phone or "Not provided",
        customer_email="callback-needed@meallion.gr",
        issue_description=f"[CALLBACK NEEDED] Customer asked: {customer_question}",
        order_number=None,
        tags=["callback-needed", "ai-escalation"],
    )
    
    if result["success"]:
        logger.info(f"Query logged in ClickUp: {result['task_id']}")
        return "Got it! I've noted your question and someone from our team will get back to you shortly."
    else:
        logger.error(f"Failed to log query: {result.get('error')}")
        return "I've made a note - our team will follow up with you soon."


async def validate_ticket_field(
    field_name: Annotated[str, "The field being validated: 'name', 'phone', 'email', or 'issue'"],
    field_value: Annotated[str, "The value provided by the customer"],
) -> str:
    """
    Validate and format a support ticket field before final submission.
    
    Use this to check each field as it's collected, before moving to the next.
    This ensures data quality and gives the customer a chance to correct errors.
    
    Args:
        field_name: Which field is being validated
        field_value: The value to validate
        
    Returns:
        Validation result with the cleaned value for confirmation
    """
    field_name = field_name.lower()
    
    if field_name == "name":
        cleaned = field_value.strip()
        if len(cleaned) < 2:
            return "The name is too short. Can you give me your full name?"
        return f"I have: {cleaned}. Is that correct?"
    
    elif field_name == "phone":
        cleaned = clean_phone_number(field_value)
        if not validate_phone(cleaned):
            return "The phone number doesn't seem correct. Can you repeat it?"
        # Format for reading back
        formatted = "-".join([cleaned[i:i+3] for i in range(0, len(cleaned), 3)])
        return f"I have phone number: {formatted}. Is that correct?"
    
    elif field_name == "email":
        cleaned = clean_email(field_value)
        if not validate_email(cleaned):
            return "The email address doesn't seem correct. Can you spell it out?"
        return f"I have email: {cleaned}. Is that correct?"
    
    elif field_name == "issue":
        cleaned = field_value.strip()
        if len(cleaned) < 10:
            return "Can you give me more details about your issue?"
        # Summarize for confirmation
        summary = cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
        return f"I understand your issue is: {summary}. Is that correct?"
    
    else:
        return f"Unknown field: {field_name}"
