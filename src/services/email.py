"""
Meallion Voice AI - Email Service
Handles support ticket email notifications.
"""

import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiosmtplib

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SupportTicket:
    """Support ticket data structure."""
    ticket_id: str
    customer_name: str
    customer_phone: str
    customer_email: str
    issue_description: str
    created_at: datetime
    call_id: Optional[str] = None
    priority: str = "normal"


class EmailService:
    """
    Email service for sending support ticket notifications.
    
    Features:
    - Async SMTP email sending
    - HTML and plain text support
    - Support ticket formatting
    """

    def __init__(self):
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_user = settings.smtp_user
        self.smtp_pass = settings.smtp_pass
        self.use_tls = settings.smtp_use_tls
        self.support_email = settings.support_email

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
    ) -> bool:
        """
        Send an email asynchronously.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body_text: Plain text body
            body_html: Optional HTML body
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["From"] = self.smtp_user
            message["To"] = to_email
            message["Subject"] = subject

            # Add plain text part
            part1 = MIMEText(body_text, "plain", "utf-8")
            message.attach(part1)

            # Add HTML part if provided
            if body_html:
                part2 = MIMEText(body_html, "html", "utf-8")
                message.attach(part2)

            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_pass,
                start_tls=self.use_tls,
            )

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def send_support_ticket(self, ticket: SupportTicket) -> bool:
        """
        Send a support ticket notification to the support team.
        
        Args:
            ticket: The support ticket to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        subject = f"[Meallion Support] New Ticket #{ticket.ticket_id} - {ticket.priority.upper()}"
        
        # Plain text version
        body_text = f"""
New Support Ticket from Meallion Voice AI

Ticket ID: {ticket.ticket_id}
Priority: {ticket.priority.upper()}
Created: {ticket.created_at.strftime("%Y-%m-%d %H:%M:%S")}

Customer Information:
- Name: {ticket.customer_name}
- Phone: {ticket.customer_phone}
- Email: {ticket.customer_email}

Issue Description:
{ticket.issue_description}

---
This ticket was created automatically by Elena, the Meallion Voice AI assistant.
Call ID: {ticket.call_id or 'N/A'}
"""

        # HTML version
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #1a1a2e; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
        .footer {{ background-color: #eee; padding: 15px; border-radius: 0 0 8px 8px; font-size: 12px; color: #666; }}
        .field {{ margin-bottom: 10px; }}
        .label {{ font-weight: bold; color: #1a1a2e; }}
        .priority-high {{ color: #e74c3c; }}
        .priority-normal {{ color: #f39c12; }}
        .priority-low {{ color: #27ae60; }}
        .issue-box {{ background-color: white; border-left: 4px solid #1a1a2e; padding: 15px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 style="margin: 0;">🎫 New Support Ticket</h2>
            <p style="margin: 5px 0 0 0;">Ticket #{ticket.ticket_id}</p>
        </div>
        <div class="content">
            <div class="field">
                <span class="label">Priority:</span>
                <span class="priority-{ticket.priority}">{ticket.priority.upper()}</span>
            </div>
            <div class="field">
                <span class="label">Created:</span>
                {ticket.created_at.strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            
            <h3>👤 Customer Information</h3>
            <div class="field">
                <span class="label">Name:</span> {ticket.customer_name}
            </div>
            <div class="field">
                <span class="label">Phone:</span> {ticket.customer_phone}
            </div>
            <div class="field">
                <span class="label">Email:</span> {ticket.customer_email}
            </div>
            
            <h3>📝 Issue Description</h3>
            <div class="issue-box">
                {ticket.issue_description.replace(chr(10), '<br>')}
            </div>
        </div>
        <div class="footer">
            <p>This ticket was created automatically by <strong>Elena</strong>, the Meallion Voice AI assistant.</p>
            <p>Call ID: {ticket.call_id or 'N/A'}</p>
        </div>
    </div>
</body>
</html>
"""

        return await self.send_email(
            to_email=self.support_email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
        )

    async def send_ticket_confirmation(
        self,
        customer_email: str,
        ticket: SupportTicket,
    ) -> bool:
        """
        Send confirmation email to the customer.
        
        Args:
            customer_email: Customer's email address
            ticket: The support ticket
            
        Returns:
            True if sent successfully, False otherwise
        """
        subject = f"Meallion - Your Support Request #{ticket.ticket_id}"
        
        body_text = f"""
Dear {ticket.customer_name},

Thank you for contacting Meallion support.

We have received your support request and created ticket #{ticket.ticket_id}.

Your Issue:
{ticket.issue_description}

Our support team will review your request and get back to you as soon as possible.

Best regards,
The Meallion Team

---
This is an automated message. Please do not reply directly to this email.
"""

        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #1a1a2e; color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background-color: #f9f9f9; padding: 30px; border: 1px solid #ddd; }}
        .footer {{ background-color: #eee; padding: 20px; border-radius: 0 0 8px 8px; text-align: center; font-size: 12px; color: #666; }}
        .ticket-number {{ background-color: #e8e8e8; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: bold; }}
        .issue-box {{ background-color: white; border: 1px solid #ddd; padding: 15px; margin: 20px 0; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0;">Meallion</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Support Request Received</p>
        </div>
        <div class="content">
            <p>Dear <strong>{ticket.customer_name}</strong>,</p>
            <p>Thank you for contacting Meallion support. We have received your request.</p>
            
            <p style="text-align: center; margin: 25px 0;">
                <span class="ticket-number">Ticket #{ticket.ticket_id}</span>
            </p>
            
            <p><strong>Your Issue:</strong></p>
            <div class="issue-box">
                {ticket.issue_description.replace(chr(10), '<br>')}
            </div>
            
            <p>Our support team will review your request and get back to you as soon as possible.</p>
            
            <p>Best regards,<br><strong>The Meallion Team</strong></p>
        </div>
        <div class="footer">
            <p>This is an automated message. Please do not reply directly to this email.</p>
            <p>© 2024 Meallion - Premium Greek Food Delivery</p>
        </div>
    </div>
</body>
</html>
"""

        return await self.send_email(
            to_email=customer_email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
        )


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
