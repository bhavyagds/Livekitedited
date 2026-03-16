"""Initial schema for admin dashboard

Revision ID: 001
Revises: 
Create Date: 2026-01-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create admin_users table
    op.create_table(
        'admin_users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255)),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_login', sa.DateTime(timezone=True)),
    )

    # Create calls table
    op.create_table(
        'calls',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('call_sid', sa.String(255)),
        sa.Column('room_name', sa.String(255)),
        sa.Column('caller_number', sa.String(50)),
        sa.Column('caller_name', sa.String(255)),
        sa.Column('call_type', sa.String(20)),
        sa.Column('status', sa.String(20)),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime(timezone=True)),
        sa.Column('duration_seconds', sa.Integer()),
        sa.Column('disconnect_reason', sa.String(255)),
        sa.Column('metadata_json', postgresql.JSON(), default={}),
        sa.Column('transcript', sa.Text()),
        sa.Column('sentiment_score', sa.Float()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("call_type IN ('inbound', 'outbound', 'web')", name='check_call_type'),
        sa.CheckConstraint("status IN ('active', 'completed', 'failed', 'missed', 'busy')", name='check_status'),
    )
    op.create_index('idx_calls_started_at', 'calls', ['started_at'])
    op.create_index('idx_calls_status', 'calls', ['status'])
    op.create_index('idx_calls_call_type', 'calls', ['call_type'])
    op.create_index('idx_calls_caller_number', 'calls', ['caller_number'])

    # Create kb_versions table
    op.create_table(
        'kb_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('version_number', sa.Integer()),
        sa.Column('content', postgresql.JSON(), nullable=False),
        sa.Column('file_name', sa.String(255)),
        sa.Column('file_size', sa.Integer()),
        sa.Column('changed_by', sa.String(255)),
        sa.Column('change_summary', sa.Text()),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_kb_versions_created', 'kb_versions', ['created_at'])

    # Create prompts_versions table
    op.create_table(
        'prompts_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('version_number', sa.Integer()),
        sa.Column('language', sa.String(5), nullable=False),
        sa.Column('prompt_type', sa.String(50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('changed_by', sa.String(255)),
        sa.Column('change_summary', sa.Text()),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("language IN ('en', 'el')", name='check_language'),
        sa.CheckConstraint("prompt_type IN ('system', 'greeting', 'closing')", name='check_prompt_type'),
    )
    op.create_index('idx_prompts_created', 'prompts_versions', ['created_at'])

    # Create sip_config_versions table
    op.create_table(
        'sip_config_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('version_number', sa.Integer()),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('changed_by', sa.String(255)),
        sa.Column('change_summary', sa.Text()),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('admin_users.id')),
        sa.Column('user_email', sa.String(255)),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50)),
        sa.Column('resource_id', sa.String(255)),
        sa.Column('old_value', postgresql.JSON()),
        sa.Column('new_value', postgresql.JSON()),
        sa.Column('ip_address', sa.String(50)),
        sa.Column('user_agent', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_audit_logs_created', 'audit_logs', ['created_at'])
    op.create_index('idx_audit_logs_user', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_logs_action', 'audit_logs', ['action'])

    # Create call_analytics table
    op.create_table(
        'call_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('date', sa.Date(), unique=True, nullable=False),
        sa.Column('total_calls', sa.Integer(), default=0),
        sa.Column('successful_calls', sa.Integer(), default=0),
        sa.Column('failed_calls', sa.Integer(), default=0),
        sa.Column('missed_calls', sa.Integer(), default=0),
        sa.Column('web_calls', sa.Integer(), default=0),
        sa.Column('sip_calls', sa.Integer(), default=0),
        sa.Column('avg_duration_seconds', sa.Float(), default=0),
        sa.Column('total_duration_seconds', sa.Integer(), default=0),
        sa.Column('hourly_breakdown', postgresql.JSON(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_analytics_date', 'call_analytics', ['date'])

    # Create error_logs table
    op.create_table(
        'error_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('service', sa.String(50)),
        sa.Column('level', sa.String(20)),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('context', postgresql.JSON(), default={}),
        sa.Column('stack_trace', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name='check_level'),
    )
    op.create_index('idx_error_logs_service', 'error_logs', ['service', 'created_at'])
    op.create_index('idx_error_logs_level', 'error_logs', ['level'])
    op.create_index('idx_error_logs_created', 'error_logs', ['created_at'])

    # Create agent_sessions table
    op.create_table(
        'agent_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('call_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('calls.id')),
        sa.Column('session_id', sa.String(255)),
        sa.Column('room_name', sa.String(255)),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime(timezone=True)),
        sa.Column('messages_count', sa.Integer(), default=0),
        sa.Column('tools_used', postgresql.JSON(), default=[]),
        sa.Column('sentiment_score', sa.Float()),
        sa.Column('topics_discussed', postgresql.JSON(), default=[]),
        sa.Column('resolution_status', sa.String(50)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_agent_sessions_call', 'agent_sessions', ['call_id'])
    op.create_index('idx_agent_sessions_created', 'agent_sessions', ['created_at'])

    # Create system_settings table
    op.create_table(
        'system_settings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('key', sa.String(100), unique=True, nullable=False),
        sa.Column('value', postgresql.JSON(), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('updated_by', sa.String(255)),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('system_settings')
    op.drop_table('agent_sessions')
    op.drop_table('error_logs')
    op.drop_table('call_analytics')
    op.drop_table('audit_logs')
    op.drop_table('sip_config_versions')
    op.drop_table('prompts_versions')
    op.drop_table('kb_versions')
    op.drop_table('calls')
    op.drop_table('admin_users')
