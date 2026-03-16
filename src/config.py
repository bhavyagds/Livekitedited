"""
Meallion Voice AI - Configuration Management
Uses Pydantic Settings for type-safe environment variable handling.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Agent Language Configuration
    agent_language: str = Field(
        default="el",
        description="Agent language: 'el' for Greek, 'en' for English"
    )

    # LiveKit Configuration
    livekit_url: str = Field(
        default="ws://localhost:7880",
        description="LiveKit server URL (internal)"
    )
    livekit_public_url: str = Field(
        default="ws://localhost:7880",
        description="LiveKit server URL for web clients"
    )
    livekit_api_key: str = Field(
        default="devkey",
        description="LiveKit API key"
    )
    livekit_api_secret: str = Field(
        default="secret",
        description="LiveKit API secret"
    )

    # ElevenLabs Configuration
    elevenlabs_api_key: str = Field(
        default="",
        description="ElevenLabs API key"
    )
    elevenlabs_voice_id: str = Field(
        default="aTP4J5SJLQl74WTSRXKW",
        description="ElevenLabs voice ID for Elena"
    )
    elevenlabs_model: str = Field(
        default="eleven_multilingual_v2",
        description="ElevenLabs model: eleven_multilingual_v2 for Greek, eleven_turbo_v2 for English-only"
    )
    elevenlabs_voice_stability: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Voice stability (0-1)"
    )
    elevenlabs_voice_similarity: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Voice similarity boost (0-1)"
    )
    elevenlabs_voice_speed: float = Field(
        default=0.60,
        ge=0.0,
        le=2.0,
        description="Voice speed multiplier"
    )

    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for conversation"
    )

    # Shopify Configuration
    shopify_store_url: str = Field(
        default="",
        description="Shopify store URL (e.g., your-store.myshopify.com)"
    )
    shopify_access_token: str = Field(
        default="",
        description="Shopify Admin API access token"
    )

    # Yuboto SIP Configuration (Greek VoIP)
    yuboto_sip_server: str = Field(
        default="sip.yuboto-telephony.gr",
        description="Yuboto SIP server address"
    )
    yuboto_sip_username: str = Field(
        default="",
        description="Yuboto SIP username"
    )
    yuboto_sip_password: str = Field(
        default="",
        description="Yuboto SIP password"
    )
    yuboto_phone_number: str = Field(
        default="",
        description="Yuboto Greek phone number"
    )

    # ClickUp Configuration (for support tickets)
    clickup_api_token: str = Field(
        default="",
        description="ClickUp API token"
    )
    clickup_list_id: str = Field(
        default="",
        description="ClickUp list ID for support tickets"
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/conversations.db",
        description="Database connection URL"
    )

    # PostgreSQL Configuration (for admin dashboard)
    postgres_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/meallion",
        description="PostgreSQL connection URL for admin dashboard"
    )

    # Admin Dashboard Configuration
    admin_email: str = Field(
        default="admin@vakmedia.co",
        description="Admin user email"
    )
    admin_password: str = Field(
        default="admin123",
        description="Admin user password (change in production!)"
    )
    admin_jwt_secret: str = Field(
        default="change-me-admin-jwt-secret-key",
        description="JWT secret for admin authentication"
    )
    admin_jwt_expiry_hours: int = Field(
        default=24,
        description="JWT token expiry in hours"
    )

    # Security
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for signing"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v

    @property
    def livekit_ws_url(self) -> str:
        """Get WebSocket URL for LiveKit."""
        return self.livekit_url

    @property
    def shopify_api_url(self) -> str:
        """Get Shopify Admin API base URL."""
        store = self.shopify_store_url.replace("https://", "").replace("http://", "")
        return f"https://{store}/admin/api/2024-01"

    def get_elevenlabs_voice_settings(self) -> dict:
        """Get ElevenLabs voice settings as dict."""
        return {
            "stability": self.elevenlabs_voice_stability,
            "similarity_boost": self.elevenlabs_voice_similarity,
            "style": 0.0,
            "use_speaker_boost": True,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses LRU cache to avoid re-reading env vars on every call.
    """
    return Settings()


# Convenience alias
settings = get_settings()
