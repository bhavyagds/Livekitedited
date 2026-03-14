"""
ClickUp service for creating support tickets as tasks.
Uses ClickUp's markdown_description for proper formatting.
"""
import httpx
import logging
from datetime import datetime
from typing import Optional
from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# ISSUE CATEGORIES - Maps keywords to (label, tag, priority)
# Priority: 1=urgent, 2=high, 3=normal, 4=low
# =============================================================================
ISSUE_CATEGORIES = {
    "Need Refund": {
        "keywords": ["refund", "money back", "return money", "get refund", "want refund"],
        "tag": "refund",
        "priority": 2,
    },
    "Delivery Issue": {
        "keywords": ["not delivered", "didn't arrive", "never arrived", "missing", "lost", "where is my order"],
        "tag": "delivery",
        "priority": 2,
    },
    "Late Delivery": {
        "keywords": ["late", "delay", "delayed", "taking too long", "slow"],
        "tag": "late-delivery",
        "priority": 3,
    },
    "Wrong Order": {
        "keywords": ["wrong item", "wrong order", "incorrect", "not what i ordered", "mistake"],
        "tag": "wrong-order",
        "priority": 2,
    },
    "Damaged Item": {
        "keywords": ["damaged", "broken", "spoiled", "bad condition", "ruined"],
        "tag": "damaged",
        "priority": 2,
    },
    "Cancel Order": {
        "keywords": ["cancel", "cancellation", "don't want", "stop order"],
        "tag": "cancellation",
        "priority": 3,
    },
    "Payment Issue": {
        "keywords": ["payment", "charged", "billing", "double charge", "overcharged"],
        "tag": "payment",
        "priority": 2,
    },
    "Quality Issue": {
        "keywords": ["quality", "taste", "bad taste", "not fresh", "expired"],
        "tag": "quality",
        "priority": 3,
    },
    "Account Issue": {
        "keywords": ["account", "login", "password", "can't access"],
        "tag": "account",
        "priority": 3,
    },
    "General Inquiry": {
        "keywords": ["question", "info", "information", "how to", "help"],
        "tag": "inquiry",
        "priority": 4,
    },
}


def categorize_issue(issue_description: str) -> dict:
    """
    Categorize an issue based on keywords in the description.
    Returns dict with: category name, tag, and priority.
    """
    issue_lower = issue_description.lower()
    
    for category, config in ISSUE_CATEGORIES.items():
        for keyword in config["keywords"]:
            if keyword in issue_lower:
                return {
                    "name": category,
                    "tag": config["tag"],
                    "priority": config["priority"],
                }
    
    # Default
    return {
        "name": "Support Request",
        "tag": "general",
        "priority": 3,
    }


# =============================================================================
# STANDARD TICKET TEMPLATE - Edit this to change the format for all tickets
# =============================================================================
TICKET_TEMPLATE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 VOICE SUPPORT TICKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Date: {date}
🤖 Source: Elena Voice Agent

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 CUSTOMER DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Name:    {customer_name}
   Phone:   {customer_phone}
   Email:   {customer_email}
   Order:   {order_number}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 ISSUE DESCRIPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{issue_description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class ClickUpService:
    """
    ClickUp API client for creating support tickets as tasks.
    """
    
    BASE_URL = "https://api.clickup.com/api/v2"
    
    def __init__(self):
        self.api_token = settings.clickup_api_token
        self.list_id = settings.clickup_list_id
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict:
        """Get API request headers."""
        return {
            "Authorization": self.api_token,
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
    
    async def create_task(
        self,
        name: str,
        description: str,
        priority: int = 3,  # 1=urgent, 2=high, 3=normal, 4=low
        tags: list[str] = None,
    ) -> dict:
        """
        Create a task in ClickUp.
        
        Args:
            name: Task title
            description: Task description (plain text with emoji formatting)
            priority: 1=urgent, 2=high, 3=normal, 4=low
            tags: List of tag names
            
        Returns:
            Dict with task details including id and url
        """
        client = await self.get_client()
        
        payload = {
            "name": name,
            "description": description,
            "priority": priority,
            "tags": tags or ["support-ticket"],
        }
        
        try:
            response = await client.post(
                f"{self.BASE_URL}/list/{self.list_id}/task",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Created ClickUp task: {data.get('id')} - {name}")
            return {
                "success": True,
                "task_id": data.get("id"),
                "task_url": data.get("url"),
                "task_name": data.get("name"),
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"ClickUp API error: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "error": f"API error: {e.response.status_code}",
            }
        except Exception as e:
            logger.error(f"Failed to create ClickUp task: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def create_support_ticket(
        self,
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        issue_description: str,
        order_number: str = None,
        tags: list[str] = None,
    ) -> dict:
        """
        Create a support ticket as a ClickUp task using standard template.
        
        Args:
            customer_name: Customer's full name
            customer_phone: Customer's phone number
            customer_email: Customer's email address
            issue_description: Description of the issue
            order_number: Optional order number
            tags: Optional custom tags (overrides auto-categorization)
            
        Returns:
            Dict with ticket/task details
        """
        # Categorize the issue - returns {name, tag, priority}
        category = categorize_issue(issue_description)
        
        # Check if this is a callback/escalation request
        is_callback = tags and "callback-needed" in tags
        
        # Build task title with category for easy sorting
        if is_callback:
            title = f"📞 [Callback Needed] {customer_name}"
        else:
            title = f"🎫 [{category['name']}] {customer_name}"
            
        if order_number:
            title += f" | #{order_number}"
        
        # Format description using standard template
        description = TICKET_TEMPLATE.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            order_number=f"#{order_number}" if order_number else "N/A",
            issue_description=issue_description,
        )
        
        # Use custom tags if provided, otherwise auto-categorize
        if tags:
            final_tags = ["voice-support"] + tags
            priority = 2  # High priority for callback requests
        else:
            final_tags = ["voice-support", category["tag"]]
            priority = category["priority"]
        
        # Create the task with category tag and priority
        result = await self.create_task(
            name=title,
            description=description,
            priority=priority,
            tags=final_tags,
        )
        
        if result["success"]:
            result["ticket_id"] = result["task_id"]
            result["message"] = f"Support ticket created successfully. Reference: {result['task_id']}"
        
        return result


# Singleton instance
clickup_service = ClickUpService()
