"""
Meallion Admin Dashboard - Database Models
"""

import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    String,
    Text,
    Boolean,
    Integer,
    Float,
    DateTime,
    Date,
    ForeignKey,
    JSON,
    Index,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.models.base import Base


class AdminUser(Base):
    """Admin users table."""
    __tablename__ = "admin_users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    audit_logs: Mapped[list["AuditLog"]] = relationship("AuditLog", back_populates="user")


class Call(Base):
    """Call history table."""
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_sid: Mapped[Optional[str]] = mapped_column(String(255))
    room_name: Mapped[Optional[str]] = mapped_column(String(255))
    caller_number: Mapped[Optional[str]] = mapped_column(String(50))
    caller_name: Mapped[Optional[str]] = mapped_column(String(255))
    call_type: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[Optional[str]] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    disconnect_reason: Mapped[Optional[str]] = mapped_column(String(255))
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    transcript: Mapped[Optional[str]] = mapped_column(Text)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    sessions: Mapped[list["AgentSession"]] = relationship("AgentSession", back_populates="call")

    __table_args__ = (
        CheckConstraint("call_type IN ('inbound', 'outbound', 'web')", name="check_call_type"),
        CheckConstraint("status IN ('active', 'completed', 'failed', 'missed', 'busy')", name="check_status"),
        Index("idx_calls_started_at", "started_at"),
        Index("idx_calls_status", "status"),
        Index("idx_calls_call_type", "call_type"),
        Index("idx_calls_caller_number", "caller_number"),
    )


class KBVersion(Base):
    """Knowledge base versions table."""
    __tablename__ = "kb_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_number: Mapped[int] = mapped_column(Integer, autoincrement=True)
    content: Mapped[dict] = mapped_column(JSON, nullable=False)
    file_name: Mapped[Optional[str]] = mapped_column(String(255))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    changed_by: Mapped[Optional[str]] = mapped_column(String(255))
    change_summary: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_kb_versions_active", "is_active", postgresql_where=(is_active == True)),
        Index("idx_kb_versions_created", "created_at"),
    )


class KBItem(Base):
    """Individual knowledge base FAQ items."""
    __tablename__ = "kb_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="General")
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    language: Mapped[str] = mapped_column(String(5), default="el")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    updated_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_kb_items_category", "category"),
        Index("idx_kb_items_active", "is_active"),
        Index("idx_kb_items_language", "language"),
    )


class PromptsVersion(Base):
    """Prompts versions table."""
    __tablename__ = "prompts_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_number: Mapped[int] = mapped_column(Integer, autoincrement=True)
    language: Mapped[str] = mapped_column(String(5), nullable=False)
    prompt_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    changed_by: Mapped[Optional[str]] = mapped_column(String(255))
    change_summary: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("language IN ('en', 'el')", name="check_language"),
        CheckConstraint("prompt_type IN ('system', 'greeting', 'closing')", name="check_prompt_type"),
        Index("idx_prompts_active", "is_active", "language", "prompt_type"),
        Index("idx_prompts_created", "created_at"),
    )


class SIPConfigVersion(Base):
    """SIP configuration versions table."""
    __tablename__ = "sip_config_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_number: Mapped[int] = mapped_column(Integer, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    changed_by: Mapped[Optional[str]] = mapped_column(String(255))
    change_summary: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_sip_config_active", "is_active", postgresql_where=(is_active == True)),
    )


class AuditLog(Base):
    """Audit logs table."""
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("admin_users.id"))
    user_email: Mapped[Optional[str]] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50))
    resource_id: Mapped[Optional[str]] = mapped_column(String(255))
    old_value: Mapped[Optional[dict]] = mapped_column(JSON)
    new_value: Mapped[Optional[dict]] = mapped_column(JSON)
    ip_address: Mapped[Optional[str]] = mapped_column(String(50))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user: Mapped[Optional["AdminUser"]] = relationship("AdminUser", back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_logs_created", "created_at"),
        Index("idx_audit_logs_user", "user_id"),
        Index("idx_audit_logs_action", "action"),
    )


class CallAnalytics(Base):
    """Daily call analytics table."""
    __tablename__ = "call_analytics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    successful_calls: Mapped[int] = mapped_column(Integer, default=0)
    failed_calls: Mapped[int] = mapped_column(Integer, default=0)
    missed_calls: Mapped[int] = mapped_column(Integer, default=0)
    web_calls: Mapped[int] = mapped_column(Integer, default=0)
    sip_calls: Mapped[int] = mapped_column(Integer, default=0)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, default=0)
    total_duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    hourly_breakdown: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_analytics_date", "date"),
    )


class ErrorLog(Base):
    """Error logs table."""
    __tablename__ = "error_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service: Mapped[Optional[str]] = mapped_column(String(50))
    level: Mapped[Optional[str]] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    stack_trace: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name="check_level"),
        Index("idx_error_logs_service", "service", "created_at"),
        Index("idx_error_logs_level", "level"),
        Index("idx_error_logs_created", "created_at"),
    )


class AgentSession(Base):
    """Agent conversation sessions table."""
    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id"))
    session_id: Mapped[Optional[str]] = mapped_column(String(255))
    room_name: Mapped[Optional[str]] = mapped_column(String(255))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    messages_count: Mapped[int] = mapped_column(Integer, default=0)
    tools_used: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    topics_discussed: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    resolution_status: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    call: Mapped[Optional["Call"]] = relationship("Call", back_populates="sessions")

    __table_args__ = (
        Index("idx_agent_sessions_call", "call_id"),
        Index("idx_agent_sessions_created", "created_at"),
    )


class SystemSetting(Base):
    """System settings table."""
    __tablename__ = "system_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Language(Base):
    """Supported languages table."""
    __tablename__ = "languages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(5), unique=True, nullable=False)  # e.g., 'en', 'el', 'de'
    name: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'English', 'Greek'
    native_name: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'English', 'Ελληνικά'
    flag_emoji: Mapped[Optional[str]] = mapped_column(String(10))  # e.g., '🇬🇧', '🇬🇷'
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_languages_code", "code"),
        Index("idx_languages_active", "is_active"),
    )


class KBContent(Base):
    """Knowledge base content per language - simple text storage."""
    __tablename__ = "kb_content"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    language: Mapped[str] = mapped_column(String(5), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_kb_content_language", "language"),
    )


class PromptsContent(Base):
    """Agent prompts per language - simple text storage."""
    __tablename__ = "prompts_content"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    language: Mapped[str] = mapped_column(String(5), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_prompts_content_language", "language"),
    )


class SIPEvent(Base):
    """SIP events and connection logs."""
    __tablename__ = "sip_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # call_incoming, call_connected, call_ended, auth_failed, etc.
    trunk_id: Mapped[Optional[str]] = mapped_column(String(100))
    trunk_name: Mapped[Optional[str]] = mapped_column(String(255))
    call_id: Mapped[Optional[str]] = mapped_column(String(255))
    room_name: Mapped[Optional[str]] = mapped_column(String(255))
    from_uri: Mapped[Optional[str]] = mapped_column(String(500))  # SIP URI of caller
    to_uri: Mapped[Optional[str]] = mapped_column(String(500))  # SIP URI of destination
    caller_number: Mapped[Optional[str]] = mapped_column(String(50))
    status_code: Mapped[Optional[int]] = mapped_column(Integer)  # SIP status code (200, 401, 404, etc.)
    status_message: Mapped[Optional[str]] = mapped_column(String(255))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    source_ip: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_sip_events_type", "event_type"),
        Index("idx_sip_events_trunk", "trunk_id"),
        Index("idx_sip_events_created", "created_at"),
        Index("idx_sip_events_caller", "caller_number"),
    )


class SIPTrunkStatus(Base):
    """SIP trunk connection status tracking."""
    __tablename__ = "sip_trunk_status"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trunk_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    trunk_name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_name: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="unknown")  # connected, disconnected, error, unknown
    last_call_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    successful_calls: Mapped[int] = mapped_column(Integer, default=0)
    failed_calls: Mapped[int] = mapped_column(Integer, default=0)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    last_error_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_sip_trunk_status_trunk", "trunk_id"),
        Index("idx_sip_trunk_status_status", "status"),
    )


class SIPProvider(Base):
    """SIP provider configurations - persisted for auto-sync on startup."""
    __tablename__ = "sip_providers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    server: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[str] = mapped_column(String(255), nullable=False, default="")  # Optional for inbound-only
    password_encrypted: Mapped[str] = mapped_column(String(500), nullable=False, default="")  # Encrypted password
    phone_numbers: Mapped[list] = mapped_column(JSON, default=list)  # List of E.164 phone numbers
    allowed_ips: Mapped[list] = mapped_column(JSON, default=list)  # List of allowed IP ranges (CIDR notation)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # LiveKit IDs (populated after sync)
    livekit_trunk_id: Mapped[Optional[str]] = mapped_column(String(100))
    livekit_rule_id: Mapped[Optional[str]] = mapped_column(String(100))
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    sync_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, synced, failed
    sync_error: Mapped[Optional[str]] = mapped_column(Text)
    
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_sip_providers_name", "name"),
        Index("idx_sip_providers_active", "is_active"),
        Index("idx_sip_providers_sync_status", "sync_status"),
    )
