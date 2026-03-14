"""Add KB items table for direct FAQ management

Revision ID: 20260116_000001
Revises: 20260116_000000
Create Date: 2026-01-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260116_000001'
down_revision = '20260116_000000'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'kb_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('category', sa.String(100), nullable=False, default='General'),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('keywords', postgresql.JSON(), nullable=True),
        sa.Column('language', sa.String(5), nullable=False, default='el'),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('display_order', sa.Integer(), nullable=False, default=0),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('updated_by', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_kb_items_category', 'kb_items', ['category'])
    op.create_index('idx_kb_items_active', 'kb_items', ['is_active'])
    op.create_index('idx_kb_items_language', 'kb_items', ['language'])


def downgrade() -> None:
    op.drop_index('idx_kb_items_language', table_name='kb_items')
    op.drop_index('idx_kb_items_active', table_name='kb_items')
    op.drop_index('idx_kb_items_category', table_name='kb_items')
    op.drop_table('kb_items')
