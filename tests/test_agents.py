"""
Meallion Voice AI - Test Suite
Basic tests for agent tools and services.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

# Test Shopify service
class TestShopifyService:
    """Tests for Shopify order lookup service."""

    def test_clean_order_number_digits(self):
        """Test cleaning order number with just digits."""
        from src.services.shopify import ShopifyService
        
        assert ShopifyService.clean_order_number("12345") == "12345"
        assert ShopifyService.clean_order_number("1 2 3 4 5") == "12345"
        assert ShopifyService.clean_order_number("1-2-3-4-5") == "12345"
        assert ShopifyService.clean_order_number("#12345") == "12345"

    def test_clean_order_number_words(self):
        """Test cleaning order number with word numbers."""
        from src.services.shopify import ShopifyService
        
        assert ShopifyService.clean_order_number("one two three four five") == "12345"
        assert ShopifyService.clean_order_number("one 2 three 4 five") == "12345"

    def test_validate_order_number(self):
        """Test order number validation."""
        from src.services.shopify import ShopifyService
        
        assert ShopifyService.validate_order_number("12345") is True
        assert ShopifyService.validate_order_number("1234") is False
        assert ShopifyService.validate_order_number("123456") is False
        assert ShopifyService.validate_order_number("abcde") is False

    def test_clean_email(self):
        """Test email cleaning from voice input."""
        from src.services.shopify import ShopifyService
        
        assert ShopifyService.clean_email("test at example dot com") == "test@example.com"
        assert ShopifyService.clean_email("TEST@EXAMPLE.COM") == "test@example.com"

    def test_clean_phone(self):
        """Test phone number cleaning."""
        from src.services.shopify import ShopifyService
        
        assert ShopifyService.clean_phone_number("+30 123 456 7890") == "+301234567890"
        assert ShopifyService.clean_phone_number("123-456-7890") == "1234567890"


# Test Support Ticket validation
class TestSupportTicket:
    """Tests for support ticket tool."""

    def test_clean_phone_number(self):
        """Test phone number cleaning."""
        from src.agents.tools.support_ticket import clean_phone_number
        
        assert clean_phone_number("+30 123 456 7890") == "+301234567890"
        assert clean_phone_number("(123) 456-7890") == "1234567890"

    def test_clean_email(self):
        """Test email cleaning."""
        from src.agents.tools.support_ticket import clean_email
        
        assert clean_email("test at example dot com") == "test@example.com"

    def test_validate_email(self):
        """Test email validation."""
        from src.agents.tools.support_ticket import validate_email
        
        assert validate_email("test@example.com") is True
        assert validate_email("invalid-email") is False
        assert validate_email("test@.com") is False

    def test_validate_phone(self):
        """Test phone validation."""
        from src.agents.tools.support_ticket import validate_phone
        
        assert validate_phone("1234567890") is True
        assert validate_phone("123") is False


# Test Knowledge Base
class TestKnowledgeBase:
    """Tests for knowledge base tool."""

    def test_load_knowledge_base(self):
        """Test knowledge base loading."""
        from src.agents.tools.knowledge_base import get_kb_instance
        
        kb = get_kb_instance()
        assert kb.kb_data is not None
        assert "brand" in kb.kb_data
        assert kb.kb_data["brand"]["name"] == "Meallion"


# Test Configuration
class TestConfig:
    """Tests for configuration module."""

    def test_settings_defaults(self):
        """Test default settings values."""
        from src.config import Settings
        
        settings = Settings()
        assert settings.port == 8000
        assert settings.elevenlabs_voice_stability == 0.45
        assert settings.elevenlabs_voice_similarity == 0.80

    def test_voice_settings(self):
        """Test ElevenLabs voice settings generation."""
        from src.config import Settings
        
        settings = Settings()
        voice_settings = settings.get_elevenlabs_voice_settings()
        
        assert "stability" in voice_settings
        assert "similarity_boost" in voice_settings
        assert voice_settings["stability"] == 0.45


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
