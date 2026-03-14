"""
Meallion Admin Dashboard - Database Service
Uses SQLAlchemy async with PostgreSQL.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uuid

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, func, and_, desc
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.models.admin import (
    AdminUser,
    Call,
    KBVersion,
    KBItem,
    KBContent,
    PromptsVersion,
    PromptsContent,
    SIPConfigVersion,
    AuditLog,
    CallAnalytics,
    ErrorLog,
    AgentSession,
    Language,
    SystemSetting,
    SIPEvent,
    SIPTrunkStatus,
    SIPProvider,
)
from src.models.base import Base

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.postgres_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create async session factory
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def get_session() -> AsyncSession:
    """Get an async database session."""
    async with async_session() as session:
        yield session


@asynccontextmanager
async def get_db():
    """Context manager for database sessions."""
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


class DatabaseService:
    """Service for database operations."""

    # =========================================================================
    # ADMIN USERS
    # =========================================================================

    async def get_admin_by_email(self, email: str) -> Optional[Dict]:
        """Get admin user by email."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(AdminUser).where(AdminUser.email == email)
                )
                user = result.scalar_one_or_none()
                if user:
                    return {
                        "id": str(user.id),
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "name": user.name,
                        "is_active": user.is_active,
                        "last_login": user.last_login.isoformat() if user.last_login else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting admin user by email: {e}")
            return None

    async def update_admin_last_login(self, user_id: str) -> bool:
        """Update admin's last login timestamp."""
        try:
            async with get_db() as session:
                await session.execute(
                    update(AdminUser)
                    .where(AdminUser.id == uuid.UUID(user_id))
                    .values(last_login=datetime.utcnow())
                )
            return True
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
            return False

    async def create_admin_user(self, email: str, password_hash: str, name: str = "Admin") -> Optional[Dict]:
        """Create a new admin user."""
        try:
            async with get_db() as session:
                user = AdminUser(email=email, password_hash=password_hash, name=name)
                session.add(user)
                await session.flush()
                return {
                    "id": str(user.id),
                    "email": user.email,
                    "name": user.name,
                }
        except Exception as e:
            logger.error(f"Error creating admin user: {e}")
            return None

    # =========================================================================
    # CALLS
    # =========================================================================

    async def create_call(self, call_data: Dict) -> Optional[Dict]:
        """Create a new call record."""
        try:
            async with get_db() as session:
                call = Call(**call_data)
                session.add(call)
                await session.flush()
                return {"id": str(call.id)}
        except Exception as e:
            logger.error(f"Error creating call: {e}")
            return None

    async def update_call(self, call_id: str, data: Dict) -> bool:
        """Update a call record."""
        try:
            async with get_db() as session:
                await session.execute(
                    update(Call)
                    .where(Call.id == uuid.UUID(call_id))
                    .values(**data)
                )
            return True
        except Exception as e:
            logger.error(f"Error updating call: {e}")
            return False

    async def get_calls(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        call_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict]:
        """Get calls with optional filtering."""
        try:
            async with get_db() as session:
                query = select(Call).order_by(desc(Call.started_at))

                conditions = []
                if status:
                    conditions.append(Call.status == status)
                if call_type:
                    conditions.append(Call.call_type == call_type)
                if start_date:
                    conditions.append(Call.started_at >= start_date)
                if end_date:
                    conditions.append(Call.started_at <= end_date)

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.offset(offset).limit(limit)
                result = await session.execute(query)
                calls = result.scalars().all()

                return [
                    {
                        "id": str(c.id),
                        "call_sid": c.call_sid,
                        "room_name": c.room_name,
                        "caller_number": c.caller_number,
                        "caller_name": c.caller_name,
                        "call_type": c.call_type,
                        "status": c.status,
                        "started_at": c.started_at.isoformat() if c.started_at else None,
                        "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                        "duration_seconds": c.duration_seconds,
                        "disconnect_reason": c.disconnect_reason,
                    }
                    for c in calls
                ]
        except Exception as e:
            logger.error(f"Error getting calls: {e}")
            return []

    async def get_call_by_id(self, call_id: str) -> Optional[Dict]:
        """Get a single call by ID."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(Call).where(Call.id == uuid.UUID(call_id))
                )
                call = result.scalar_one_or_none()
                if call:
                    return {
                        "id": str(call.id),
                        "call_sid": call.call_sid,
                        "room_name": call.room_name,
                        "caller_number": call.caller_number,
                        "caller_name": call.caller_name,
                        "call_type": call.call_type,
                        "status": call.status,
                        "started_at": call.started_at.isoformat() if call.started_at else None,
                        "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                        "duration_seconds": call.duration_seconds,
                        "disconnect_reason": call.disconnect_reason,
                        "transcript": call.transcript,
                        "sentiment_score": call.sentiment_score,
                        "metadata": call.metadata_json,
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting call: {e}")
            return None

    async def get_calls_count(
        self,
        status: Optional[str] = None,
        call_type: Optional[str] = None,
    ) -> int:
        """Get total count of calls."""
        try:
            async with get_db() as session:
                query = select(func.count(Call.id))
                conditions = []
                if status:
                    conditions.append(Call.status == status)
                if call_type:
                    conditions.append(Call.call_type == call_type)
                if conditions:
                    query = query.where(and_(*conditions))
                result = await session.execute(query)
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error getting calls count: {e}")
            return 0

    # =========================================================================
    # KNOWLEDGE BASE VERSIONS
    # =========================================================================

    async def create_kb_version(
        self,
        content: Dict,
        changed_by: str,
        change_summary: str = "",
        file_name: str = "",
        file_size: int = 0,
    ) -> Optional[Dict]:
        """Create a new knowledge base version."""
        try:
            async with get_db() as session:
                # Deactivate all existing versions
                await session.execute(
                    update(KBVersion).where(KBVersion.is_active == True).values(is_active=False)
                )

                # Get next version number
                result = await session.execute(
                    select(func.coalesce(func.max(KBVersion.version_number), 0))
                )
                next_version = result.scalar() + 1

                # Insert new version
                version = KBVersion(
                    version_number=next_version,
                    content=content,
                    changed_by=changed_by,
                    change_summary=change_summary,
                    file_name=file_name,
                    file_size=file_size,
                    is_active=True,
                )
                session.add(version)
                await session.flush()

                return {
                    "id": str(version.id),
                    "version_number": version.version_number,
                }
        except Exception as e:
            logger.error(f"Error creating KB version: {e}")
            return None

    async def get_active_kb(self) -> Optional[Dict]:
        """Get the currently active knowledge base."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBVersion).where(KBVersion.is_active == True)
                )
                kb = result.scalar_one_or_none()
                if kb:
                    return {
                        "id": str(kb.id),
                        "version_number": kb.version_number,
                        "content": kb.content,
                        "file_name": kb.file_name,
                        "file_size": kb.file_size,
                        "changed_by": kb.changed_by,
                        "created_at": kb.created_at.isoformat() if kb.created_at else None,
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting active KB: {e}")
            return None

    async def get_kb_versions(self, limit: int = 20) -> List[Dict]:
        """Get knowledge base version history."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBVersion)
                    .order_by(desc(KBVersion.created_at))
                    .limit(limit)
                )
                versions = result.scalars().all()
                return [
                    {
                        "id": str(v.id),
                        "version_number": v.version_number,
                        "changed_by": v.changed_by,
                        "change_summary": v.change_summary,
                        "file_name": v.file_name,
                        "file_size": v.file_size,
                        "is_active": v.is_active,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                    }
                    for v in versions
                ]
        except Exception as e:
            logger.error(f"Error getting KB versions: {e}")
            return []

    async def get_kb_version_by_id(self, version_id: str) -> Optional[Dict]:
        """Get a specific KB version."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBVersion).where(KBVersion.id == uuid.UUID(version_id))
                )
                kb = result.scalar_one_or_none()
                if kb:
                    return {
                        "id": str(kb.id),
                        "version_number": kb.version_number,
                        "content": kb.content,
                        "file_name": kb.file_name,
                        "file_size": kb.file_size,
                        "changed_by": kb.changed_by,
                        "change_summary": kb.change_summary,
                        "is_active": kb.is_active,
                        "created_at": kb.created_at.isoformat() if kb.created_at else None,
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting KB version: {e}")
            return None

    async def rollback_kb_version(self, version_id: str, changed_by: str) -> bool:
        """Rollback to a specific KB version."""
        try:
            async with get_db() as session:
                # Deactivate all
                await session.execute(
                    update(KBVersion).where(KBVersion.is_active == True).values(is_active=False)
                )
                # Activate the specified version
                await session.execute(
                    update(KBVersion)
                    .where(KBVersion.id == uuid.UUID(version_id))
                    .values(is_active=True)
                )
            return True
        except Exception as e:
            logger.error(f"Error rolling back KB version: {e}")
            return False

    # =========================================================================
    # KB ITEMS (Individual FAQ entries)
    # =========================================================================

    async def get_kb_items(
        self,
        category: str = None,
        language: str = None,
        active_only: bool = True,
    ) -> List[Dict]:
        """Get knowledge base FAQ items."""
        try:
            async with get_db() as session:
                query = select(KBItem).order_by(KBItem.category, KBItem.display_order)
                conditions = []
                if active_only:
                    conditions.append(KBItem.is_active == True)
                if category:
                    conditions.append(KBItem.category == category)
                if language:
                    conditions.append(KBItem.language == language)
                if conditions:
                    query = query.where(and_(*conditions))
                result = await session.execute(query)
                items = result.scalars().all()
                return [
                    {
                        "id": str(item.id),
                        "category": item.category,
                        "question": item.question,
                        "answer": item.answer,
                        "keywords": item.keywords or [],
                        "language": item.language,
                        "is_active": item.is_active,
                        "display_order": item.display_order,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                    }
                    for item in items
                ]
        except Exception as e:
            logger.error(f"Error getting KB items: {e}")
            return []

    async def get_kb_item(self, item_id: str) -> Optional[Dict]:
        """Get a specific KB item."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBItem).where(KBItem.id == uuid.UUID(item_id))
                )
                item = result.scalar_one_or_none()
                if item:
                    return {
                        "id": str(item.id),
                        "category": item.category,
                        "question": item.question,
                        "answer": item.answer,
                        "keywords": item.keywords or [],
                        "language": item.language,
                        "is_active": item.is_active,
                        "display_order": item.display_order,
                        "created_by": item.created_by,
                        "updated_by": item.updated_by,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting KB item: {e}")
            return None

    async def create_kb_item(
        self,
        category: str,
        question: str,
        answer: str,
        created_by: str,
        keywords: List[str] = None,
        language: str = "el",
        display_order: int = 0,
    ) -> Optional[Dict]:
        """Create a new KB item."""
        try:
            async with get_db() as session:
                item = KBItem(
                    category=category,
                    question=question,
                    answer=answer,
                    keywords=keywords or [],
                    language=language,
                    display_order=display_order,
                    created_by=created_by,
                    updated_by=created_by,
                    is_active=True,
                )
                session.add(item)
                await session.flush()
                return {
                    "id": str(item.id),
                    "category": item.category,
                    "question": item.question,
                    "answer": item.answer,
                }
        except Exception as e:
            logger.error(f"Error creating KB item: {e}")
            return None

    async def update_kb_item(
        self,
        item_id: str,
        updated_by: str,
        category: str = None,
        question: str = None,
        answer: str = None,
        keywords: List[str] = None,
        language: str = None,
        is_active: bool = None,
        display_order: int = None,
    ) -> bool:
        """Update a KB item."""
        try:
            async with get_db() as session:
                values = {"updated_by": updated_by}
                if category is not None:
                    values["category"] = category
                if question is not None:
                    values["question"] = question
                if answer is not None:
                    values["answer"] = answer
                if keywords is not None:
                    values["keywords"] = keywords
                if language is not None:
                    values["language"] = language
                if is_active is not None:
                    values["is_active"] = is_active
                if display_order is not None:
                    values["display_order"] = display_order

                await session.execute(
                    update(KBItem)
                    .where(KBItem.id == uuid.UUID(item_id))
                    .values(**values)
                )
            return True
        except Exception as e:
            logger.error(f"Error updating KB item: {e}")
            return False

    async def delete_kb_item(self, item_id: str) -> bool:
        """Delete a KB item (soft delete by setting is_active=False)."""
        try:
            async with get_db() as session:
                await session.execute(
                    update(KBItem)
                    .where(KBItem.id == uuid.UUID(item_id))
                    .values(is_active=False)
                )
            return True
        except Exception as e:
            logger.error(f"Error deleting KB item: {e}")
            return False

    async def get_kb_categories(self) -> List[str]:
        """Get all unique KB categories."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBItem.category)
                    .where(KBItem.is_active == True)
                    .distinct()
                    .order_by(KBItem.category)
                )
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error getting KB categories: {e}")
            return []

    async def import_kb_items_from_json(self, data: dict, imported_by: str) -> int:
        """Import KB items from JSON structure (for migration from file-based KB)."""
        count = 0
        items_to_add = []
        
        def get_bilingual(obj, key_en="en", key_el="el"):
            """Extract bilingual content from object."""
            if isinstance(obj, dict):
                return obj.get(key_el, obj.get(key_en, str(obj)))
            return str(obj) if obj else ""

        def add_item(category: str, question: str, answer_en: str, answer_el: str, keywords: list = None):
            """Add bilingual FAQ items."""
            if answer_en:
                items_to_add.append({
                    "category": category,
                    "question": question,
                    "answer": answer_en,
                    "keywords": keywords or [],
                    "language": "en",
                })
            if answer_el:
                items_to_add.append({
                    "category": category,
                    "question": question,
                    "answer": answer_el,
                    "keywords": keywords or [],
                    "language": "el",
                })

        try:
            # Extract from "about" section
            about = data.get("about", {})
            if about:
                what_is = about.get("what_is_meallion", {})
                if what_is:
                    add_item("About", "What is Meallion?", what_is.get("en"), what_is.get("el"), ["meallion", "about", "company"])
                
                promise = about.get("promise", {})
                if promise:
                    add_item("About", "What is Meallion's promise?", promise.get("en"), promise.get("el"), ["promise", "quality"])
                
                target = about.get("target_audience", {})
                if target:
                    add_item("About", "Who is Meallion for?", target.get("en"), target.get("el"), ["target", "audience", "customers"])

            # Extract from "brand" section
            brand = data.get("brand", {})
            if brand:
                one_liner = brand.get("one_liner", {})
                if one_liner:
                    add_item("About", "Describe Meallion in one sentence", one_liner.get("en"), one_liner.get("el"), ["description", "summary"])

            # Extract meal categories
            categories = data.get("meal_categories", {})
            for cat_key, cat_data in categories.items():
                if isinstance(cat_data, dict):
                    name = cat_data.get("name", cat_key)
                    desc = cat_data.get("description", {})
                    if desc:
                        add_item("Meals", f"What is {name}?", desc.get("en"), desc.get("el"), [cat_key, "category", "meals"])

            # Extract product info
            product = data.get("product_info", {})
            if product:
                heating = product.get("heating_instructions", {})
                if heating:
                    micro = heating.get("microwave", {})
                    if micro:
                        add_item("Product", "How to heat in microwave?", micro.get("en"), micro.get("el"), ["microwave", "heating", "reheat"])
                    oven = heating.get("oven", {})
                    if oven:
                        add_item("Product", "How to heat in oven?", oven.get("en"), oven.get("el"), ["oven", "heating", "reheat"])
                
                storage = product.get("storage", {})
                if storage:
                    fridge = storage.get("fridge", {})
                    if fridge:
                        add_item("Product", "How long can I store meals?", fridge.get("en"), fridge.get("el"), ["storage", "fridge", "shelf life"])

            # Extract ordering info
            ordering = data.get("ordering", {})
            if ordering:
                how_it_works = ordering.get("how_it_works", {})
                if how_it_works:
                    add_item("Ordering", "How does ordering work?", how_it_works.get("en"), how_it_works.get("el"), ["order", "how to", "process"])
                
                min_order = ordering.get("minimum_order", {})
                if min_order:
                    add_item("Ordering", "What is the minimum order?", 
                             min_order.get("en") if isinstance(min_order, dict) else min_order,
                             min_order.get("el") if isinstance(min_order, dict) else min_order,
                             ["minimum", "order"])
                
                delivery_areas = ordering.get("delivery_areas", {})
                if delivery_areas:
                    add_item("Delivery", "Where do you deliver?", delivery_areas.get("en"), delivery_areas.get("el"), ["delivery", "areas", "location"])

            # Extract call scripts as FAQs
            scripts = data.get("call_scripts", {})
            for script_key, script_data in scripts.items():
                if isinstance(script_data, dict) and ("en" in script_data or "el" in script_data):
                    question = script_key.replace("_", " ").title()
                    add_item("Support", question, script_data.get("en"), script_data.get("el"), [script_key])

            # Extract contact info
            contact = data.get("contact", {})
            if contact:
                support = contact.get("support", {})
                if support:
                    add_item("Contact", "How can I contact support?", support.get("en"), support.get("el"), ["contact", "support", "phone", "email"])

            # Now save all items to database
            async with get_db() as session:
                for i, item_data in enumerate(items_to_add):
                    item = KBItem(
                        category=item_data["category"],
                        question=item_data["question"],
                        answer=item_data["answer"],
                        keywords=item_data["keywords"],
                        language=item_data["language"],
                        display_order=i,
                        created_by=imported_by,
                        updated_by=imported_by,
                        is_active=True,
                    )
                    session.add(item)
                    count += 1
            
            return count
        except Exception as e:
            logger.error(f"Error importing KB items: {e}")
            return count

    # =========================================================================
    # KB CONTENT (Simple text per language)
    # =========================================================================

    async def get_kb_content(self, language: str) -> Optional[Dict]:
        """Get knowledge base content for a language."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBContent).where(KBContent.language == language)
                )
                kb = result.scalar_one_or_none()
                if kb:
                    return {
                        "id": str(kb.id),
                        "language": kb.language,
                        "content": kb.content,
                        "updated_by": kb.updated_by,
                        "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting KB content: {e}")
            return None

    async def save_kb_content(self, language: str, content: str, updated_by: str) -> bool:
        """Save or update knowledge base content for a language."""
        try:
            async with get_db() as session:
                # Check if exists
                result = await session.execute(
                    select(KBContent).where(KBContent.language == language)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    await session.execute(
                        update(KBContent)
                        .where(KBContent.language == language)
                        .values(content=content, updated_by=updated_by)
                    )
                else:
                    # Create
                    kb = KBContent(
                        language=language,
                        content=content,
                        updated_by=updated_by,
                    )
                    session.add(kb)
            return True
        except Exception as e:
            logger.error(f"Error saving KB content: {e}")
            return False

    async def get_all_kb_content(self) -> List[Dict]:
        """Get all KB content for all languages."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(KBContent).order_by(KBContent.language)
                )
                items = result.scalars().all()
                return [
                    {
                        "language": kb.language,
                        "content": kb.content,
                        "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
                    }
                    for kb in items
                ]
        except Exception as e:
            logger.error(f"Error getting all KB content: {e}")
            return []

    # =========================================================================
    # PROMPTS CONTENT (Simple text per language)
    # =========================================================================

    async def get_prompts_content(self, language: str) -> Optional[Dict]:
        """Get prompts content for a language."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(PromptsContent).where(PromptsContent.language == language)
                )
                prompts = result.scalar_one_or_none()
                if prompts:
                    return {
                        "id": str(prompts.id),
                        "language": prompts.language,
                        "content": prompts.content,
                        "updated_by": prompts.updated_by,
                        "updated_at": prompts.updated_at.isoformat() if prompts.updated_at else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting prompts content: {e}")
            return None

    async def save_prompts_content(self, language: str, content: str, updated_by: str) -> bool:
        """Save or update prompts content for a language."""
        try:
            async with get_db() as session:
                # Check if exists
                result = await session.execute(
                    select(PromptsContent).where(PromptsContent.language == language)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    await session.execute(
                        update(PromptsContent)
                        .where(PromptsContent.language == language)
                        .values(content=content, updated_by=updated_by)
                    )
                else:
                    # Create
                    prompts = PromptsContent(
                        language=language,
                        content=content,
                        updated_by=updated_by,
                    )
                    session.add(prompts)
            return True
        except Exception as e:
            logger.error(f"Error saving prompts content: {e}")
            return False

    async def get_all_prompts_content(self) -> List[Dict]:
        """Get all prompts content for all languages."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(PromptsContent).order_by(PromptsContent.language)
                )
                items = result.scalars().all()
                return [
                    {
                        "language": p.language,
                        "content": p.content,
                        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                    }
                    for p in items
                ]
        except Exception as e:
            logger.error(f"Error getting all prompts content: {e}")
            return []

    # =========================================================================
    # SYSTEM SETTINGS (Runtime configuration)
    # =========================================================================

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a system setting value."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(SystemSetting).where(SystemSetting.key == key)
                )
                setting = result.scalar_one_or_none()
                if setting:
                    return setting.value
                return default
        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return default

    async def set_setting(self, key: str, value: Any, description: str = None, updated_by: str = None) -> bool:
        """Set a system setting value."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(SystemSetting).where(SystemSetting.key == key)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    await session.execute(
                        update(SystemSetting)
                        .where(SystemSetting.key == key)
                        .values(value=value, updated_by=updated_by)
                    )
                else:
                    setting = SystemSetting(
                        key=key,
                        value=value,
                        description=description,
                        updated_by=updated_by,
                    )
                    session.add(setting)
            return True
        except Exception as e:
            logger.error(f"Error setting {key}: {e}")
            return False

    async def get_all_settings(self) -> Dict[str, Any]:
        """Get all system settings as a dictionary."""
        try:
            async with get_db() as session:
                result = await session.execute(select(SystemSetting))
                settings = result.scalars().all()
                return {s.key: s.value for s in settings}
        except Exception as e:
            logger.error(f"Error getting all settings: {e}")
            return {}

    async def init_default_settings(self) -> None:
        """Initialize default system settings if not present."""
        defaults = {
            "agent_language": {"value": "el", "description": "Default agent language (el/en)"},
            "agent_voice_id": {"value": "aTP4J5SJLQl74WTSRXKW", "description": "ElevenLabs voice ID"},
            "agent_voice_speed": {"value": 0.6, "description": "Voice speed multiplier"},
            "agent_greeting_enabled": {"value": True, "description": "Enable greeting on call start"},
            "abuse_detection_enabled": {"value": True, "description": "Enable abuse detection"},
        }
        
        for key, data in defaults.items():
            existing = await self.get_setting(key)
            if existing is None:
                await self.set_setting(key, data["value"], data["description"], "system")
                logger.info(f"Initialized default setting: {key}")

    # =========================================================================
    # LANGUAGES
    # =========================================================================

    async def get_languages(self, active_only: bool = True) -> List[Dict]:
        """Get all supported languages."""
        try:
            async with get_db() as session:
                query = select(Language).order_by(Language.is_default.desc(), Language.name)
                if active_only:
                    query = query.where(Language.is_active == True)
                result = await session.execute(query)
                languages = result.scalars().all()
                return [
                    {
                        "id": str(lang.id),
                        "code": lang.code,
                        "name": lang.name,
                        "native_name": lang.native_name,
                        "flag_emoji": lang.flag_emoji,
                        "is_default": lang.is_default,
                        "is_active": lang.is_active,
                    }
                    for lang in languages
                ]
        except Exception as e:
            logger.error(f"Error getting languages: {e}")
            return []

    async def create_language(
        self,
        code: str,
        name: str,
        native_name: str,
        flag_emoji: str = None,
        is_default: bool = False,
    ) -> Optional[Dict]:
        """Create a new language."""
        try:
            async with get_db() as session:
                # If setting as default, unset other defaults
                if is_default:
                    await session.execute(
                        update(Language).where(Language.is_default == True).values(is_default=False)
                    )
                
                lang = Language(
                    code=code.lower(),
                    name=name,
                    native_name=native_name,
                    flag_emoji=flag_emoji,
                    is_default=is_default,
                    is_active=True,
                )
                session.add(lang)
                await session.flush()
                return {
                    "id": str(lang.id),
                    "code": lang.code,
                    "name": lang.name,
                }
        except Exception as e:
            logger.error(f"Error creating language: {e}")
            return None

    async def update_language(
        self,
        language_id: str,
        code: str = None,
        name: str = None,
        native_name: str = None,
        flag_emoji: str = None,
        is_default: bool = None,
        is_active: bool = None,
    ) -> bool:
        """Update a language."""
        try:
            async with get_db() as session:
                values = {}
                if code is not None:
                    values["code"] = code.lower()
                if name is not None:
                    values["name"] = name
                if native_name is not None:
                    values["native_name"] = native_name
                if flag_emoji is not None:
                    values["flag_emoji"] = flag_emoji
                if is_default is not None:
                    if is_default:
                        # Unset other defaults
                        await session.execute(
                            update(Language).where(Language.is_default == True).values(is_default=False)
                        )
                    values["is_default"] = is_default
                if is_active is not None:
                    values["is_active"] = is_active

                if values:
                    await session.execute(
                        update(Language)
                        .where(Language.id == uuid.UUID(language_id))
                        .values(**values)
                    )
            return True
        except Exception as e:
            logger.error(f"Error updating language: {e}")
            return False

    async def delete_language(self, language_id: str) -> bool:
        """Delete a language (soft delete)."""
        try:
            async with get_db() as session:
                await session.execute(
                    update(Language)
                    .where(Language.id == uuid.UUID(language_id))
                    .values(is_active=False)
                )
            return True
        except Exception as e:
            logger.error(f"Error deleting language: {e}")
            return False

    async def init_default_languages(self) -> int:
        """Initialize default languages if none exist."""
        count = 0
        try:
            existing = await self.get_languages(active_only=False)
            if existing:
                return 0  # Already have languages
            
            default_languages = [
                {"code": "el", "name": "Greek", "native_name": "Ελληνικά", "flag_emoji": "🇬🇷", "is_default": True},
                {"code": "en", "name": "English", "native_name": "English", "flag_emoji": "🇬🇧", "is_default": False},
            ]
            
            for lang_data in default_languages:
                result = await self.create_language(**lang_data)
                if result:
                    count += 1
            
            return count
        except Exception as e:
            logger.error(f"Error initializing default languages: {e}")
            return count

    # =========================================================================
    # PROMPTS VERSIONS
    # =========================================================================

    async def create_prompt_version(
        self,
        language: str,
        prompt_type: str,
        content: str,
        changed_by: str,
        change_summary: str = "",
    ) -> Optional[Dict]:
        """Create a new prompt version."""
        try:
            async with get_db() as session:
                # Deactivate existing versions for this language/type
                await session.execute(
                    update(PromptsVersion)
                    .where(
                        and_(
                            PromptsVersion.language == language,
                            PromptsVersion.prompt_type == prompt_type,
                            PromptsVersion.is_active == True,
                        )
                    )
                    .values(is_active=False)
                )

                # Get next version number
                result = await session.execute(
                    select(func.coalesce(func.max(PromptsVersion.version_number), 0))
                )
                next_version = result.scalar() + 1

                version = PromptsVersion(
                    version_number=next_version,
                    language=language,
                    prompt_type=prompt_type,
                    content=content,
                    changed_by=changed_by,
                    change_summary=change_summary,
                    is_active=True,
                )
                session.add(version)
                await session.flush()

                return {
                    "id": str(version.id),
                    "version_number": version.version_number,
                }
        except Exception as e:
            logger.error(f"Error creating prompt version: {e}")
            return None

    async def get_active_prompts(self) -> List[Dict]:
        """Get all active prompts."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(PromptsVersion).where(PromptsVersion.is_active == True)
                )
                prompts = result.scalars().all()
                return [
                    {
                        "id": str(p.id),
                        "language": p.language,
                        "prompt_type": p.prompt_type,
                        "content": p.content,
                        "version_number": p.version_number,
                    }
                    for p in prompts
                ]
        except Exception as e:
            logger.error(f"Error getting active prompts: {e}")
            return []

    async def get_prompt_versions(
        self,
        language: str = None,
        prompt_type: str = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Get prompt version history."""
        try:
            async with get_db() as session:
                query = select(PromptsVersion).order_by(desc(PromptsVersion.created_at))
                conditions = []
                if language:
                    conditions.append(PromptsVersion.language == language)
                if prompt_type:
                    conditions.append(PromptsVersion.prompt_type == prompt_type)
                if conditions:
                    query = query.where(and_(*conditions))
                query = query.limit(limit)
                result = await session.execute(query)
                versions = result.scalars().all()
                return [
                    {
                        "id": str(v.id),
                        "version_number": v.version_number,
                        "language": v.language,
                        "prompt_type": v.prompt_type,
                        "changed_by": v.changed_by,
                        "change_summary": v.change_summary,
                        "is_active": v.is_active,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                    }
                    for v in versions
                ]
        except Exception as e:
            logger.error(f"Error getting prompt versions: {e}")
            return []

    # =========================================================================
    # SIP CONFIG VERSIONS
    # =========================================================================

    async def create_sip_config_version(
        self,
        content: str,
        changed_by: str,
        change_summary: str = "",
    ) -> Optional[Dict]:
        """Create a new SIP config version."""
        try:
            async with get_db() as session:
                # Deactivate all existing
                await session.execute(
                    update(SIPConfigVersion)
                    .where(SIPConfigVersion.is_active == True)
                    .values(is_active=False)
                )

                # Get next version number
                result = await session.execute(
                    select(func.coalesce(func.max(SIPConfigVersion.version_number), 0))
                )
                next_version = result.scalar() + 1

                version = SIPConfigVersion(
                    version_number=next_version,
                    content=content,
                    changed_by=changed_by,
                    change_summary=change_summary,
                    is_active=True,
                )
                session.add(version)
                await session.flush()

                return {
                    "id": str(version.id),
                    "version_number": version.version_number,
                }
        except Exception as e:
            logger.error(f"Error creating SIP config version: {e}")
            return None

    async def get_active_sip_config(self) -> Optional[Dict]:
        """Get active SIP config."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(SIPConfigVersion).where(SIPConfigVersion.is_active == True)
                )
                config = result.scalar_one_or_none()
                if config:
                    return {
                        "id": str(config.id),
                        "version_number": config.version_number,
                        "content": config.content,
                        "changed_by": config.changed_by,
                        "created_at": config.created_at.isoformat() if config.created_at else None,
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting active SIP config: {e}")
            return None

    async def get_sip_config_versions(self, limit: int = 20) -> List[Dict]:
        """Get SIP config version history."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(SIPConfigVersion)
                    .order_by(desc(SIPConfigVersion.created_at))
                    .limit(limit)
                )
                versions = result.scalars().all()
                return [
                    {
                        "id": str(v.id),
                        "version_number": v.version_number,
                        "changed_by": v.changed_by,
                        "change_summary": v.change_summary,
                        "is_active": v.is_active,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                    }
                    for v in versions
                ]
        except Exception as e:
            logger.error(f"Error getting SIP config versions: {e}")
            return []

    # =========================================================================
    # AUDIT LOGS
    # =========================================================================

    async def create_audit_log(
        self,
        user_id: Optional[str],
        user_email: str,
        action: str,
        resource_type: str = None,
        resource_id: str = None,
        old_value: Any = None,
        new_value: Any = None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> Optional[Dict]:
        """Create an audit log entry."""
        try:
            async with get_db() as session:
                log = AuditLog(
                    user_id=uuid.UUID(user_id) if user_id and user_id != "env-admin" else None,
                    user_email=user_email,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    old_value=old_value,
                    new_value=new_value,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                session.add(log)
                await session.flush()
                return {"id": str(log.id)}
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
            return None

    async def get_audit_logs(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get audit logs."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(AuditLog)
                    .order_by(desc(AuditLog.created_at))
                    .offset(offset)
                    .limit(limit)
                )
                logs = result.scalars().all()
                return [
                    {
                        "id": str(l.id),
                        "user_id": str(l.user_id) if l.user_id else None,
                        "user_email": l.user_email,
                        "action": l.action,
                        "resource_type": l.resource_type,
                        "resource_id": l.resource_id,
                        "old_value": l.old_value,
                        "new_value": l.new_value,
                        "ip_address": l.ip_address,
                        "created_at": l.created_at.isoformat() if l.created_at else None,
                    }
                    for l in logs
                ]
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return []

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    async def get_analytics_summary(self, days: int = 30) -> Dict:
        """Get analytics summary for dashboard."""
        try:
            from_date = datetime.utcnow().date() - timedelta(days=days)

            async with get_db() as session:
                result = await session.execute(
                    select(CallAnalytics)
                    .where(CallAnalytics.date >= from_date)
                    .order_by(desc(CallAnalytics.date))
                )
                data = result.scalars().all()

                total_calls = sum(d.total_calls for d in data)
                successful = sum(d.successful_calls for d in data)
                failed = sum(d.failed_calls for d in data)
                total_duration = sum(d.total_duration_seconds for d in data)

                return {
                    "total_calls": total_calls,
                    "successful_calls": successful,
                    "failed_calls": failed,
                    "success_rate": (successful / total_calls * 100) if total_calls > 0 else 0,
                    "avg_duration_seconds": (total_duration / successful) if successful > 0 else 0,
                    "daily_data": [
                        {
                            "date": d.date.isoformat(),
                            "total_calls": d.total_calls,
                            "successful_calls": d.successful_calls,
                            "failed_calls": d.failed_calls,
                        }
                        for d in data
                    ],
                }
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}

    async def get_today_stats(self) -> Dict:
        """Get today's call statistics."""
        try:
            today = datetime.utcnow().date()
            async with get_db() as session:
                result = await session.execute(
                    select(CallAnalytics).where(CallAnalytics.date == today)
                )
                stats = result.scalar_one_or_none()
                if stats:
                    return {
                        "total_calls": stats.total_calls,
                        "successful_calls": stats.successful_calls,
                        "failed_calls": stats.failed_calls,
                        "missed_calls": stats.missed_calls,
                        "web_calls": stats.web_calls,
                        "sip_calls": stats.sip_calls,
                    }
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "missed_calls": 0,
                "web_calls": 0,
                "sip_calls": 0,
            }
        except Exception as e:
            logger.error(f"Error getting today stats: {e}")
            return {}

    async def update_analytics_for_call(self, call: Dict) -> bool:
        """Update analytics when a call is recorded."""
        try:
            today = datetime.utcnow().date()
            async with get_db() as session:
                # Try to update existing record
                result = await session.execute(
                    select(CallAnalytics).where(CallAnalytics.date == today)
                )
                analytics = result.scalar_one_or_none()

                if analytics:
                    analytics.total_calls += 1
                    if call.get("status") == "completed":
                        analytics.successful_calls += 1
                    elif call.get("status") == "failed":
                        analytics.failed_calls += 1
                    elif call.get("status") == "missed":
                        analytics.missed_calls += 1
                    if call.get("call_type") == "web":
                        analytics.web_calls += 1
                    elif call.get("call_type") in ("inbound", "outbound", "sip"):
                        analytics.sip_calls += 1
                else:
                    # Create new record
                    analytics = CallAnalytics(
                        date=today,
                        total_calls=1,
                        successful_calls=1 if call.get("status") == "completed" else 0,
                        failed_calls=1 if call.get("status") == "failed" else 0,
                        missed_calls=1 if call.get("status") == "missed" else 0,
                        web_calls=1 if call.get("call_type") == "web" else 0,
                        sip_calls=1 if call.get("call_type") in ("inbound", "outbound", "sip") else 0,
                    )
                    session.add(analytics)
            return True
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
            return False

    async def record_call_start(
        self,
        room_name: str,
        call_type: str = "web",
        caller_number: str = None,
        caller_identity: str = None,
    ) -> Optional[str]:
        """Record when a call starts and return the call ID."""
        try:
            async with get_db() as session:
                call = Call(
                    room_name=room_name,
                    call_type=call_type,
                    caller_number=caller_number,
                    caller_name=caller_identity,
                    status="active",
                    started_at=datetime.utcnow(),
                )
                session.add(call)
                await session.flush()
                call_id = str(call.id)
                await session.commit()
                logger.info(f"Recorded call start: {call_id} ({call_type})")
                return call_id
        except Exception as e:
            logger.error(f"Error recording call start: {e}")
            return None

    async def record_call_end(
        self,
        call_id: str = None,
        room_name: str = None,
        status: str = "completed",
        duration_seconds: int = None,
        disconnect_reason: str = None,
        transcript: str = None,
    ) -> bool:
        """Record when a call ends and update analytics."""
        try:
            async with get_db() as session:
                # Find the call by ID or room name
                if call_id:
                    query = select(Call).where(Call.id == uuid.UUID(call_id))
                elif room_name:
                    query = select(Call).where(
                        and_(Call.room_name == room_name, Call.status == "active")
                    ).order_by(desc(Call.started_at)).limit(1)
                else:
                    return False
                
                result = await session.execute(query)
                call = result.scalar_one_or_none()
                
                if call:
                    call.status = status
                    call.ended_at = datetime.utcnow()
                    call.duration_seconds = duration_seconds
                    call.disconnect_reason = disconnect_reason
                    if transcript:
                        call.transcript = transcript
                    
                    # Calculate duration if not provided
                    if not duration_seconds and call.started_at:
                        call.duration_seconds = int((datetime.utcnow() - call.started_at).total_seconds())
                    
                    await session.commit()
                    
                    # Update analytics
                    await self.update_analytics_for_call({
                        "status": status,
                        "call_type": call.call_type,
                        "duration_seconds": call.duration_seconds,
                    })
                    
                    # Also update duration in analytics
                    await self._update_analytics_duration(call.duration_seconds or 0)
                    
                    logger.info(f"Recorded call end: {call.id} ({status}, {call.duration_seconds}s)")
                    return True
                else:
                    logger.warning(f"Call not found for end recording: {call_id or room_name}")
                    return False
        except Exception as e:
            logger.error(f"Error recording call end: {e}")
            return False

    async def update_call_transcript(
        self,
        call_id: str = None,
        room_name: str = None,
        transcript: str = None,
        append: bool = True,
    ) -> bool:
        """Update the transcript for a call."""
        try:
            async with get_db() as session:
                # Find the call by ID or room name
                if call_id:
                    query = select(Call).where(Call.id == uuid.UUID(call_id))
                elif room_name:
                    query = select(Call).where(Call.room_name == room_name).order_by(desc(Call.started_at)).limit(1)
                else:
                    return False
                
                result = await session.execute(query)
                call = result.scalar_one_or_none()
                
                if call:
                    if append and call.transcript:
                        call.transcript = call.transcript + "\n" + transcript
                    else:
                        call.transcript = transcript
                    await session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating call transcript: {e}")
            return False

    async def get_call_transcript(self, call_id: str) -> Optional[str]:
        """Get the transcript for a call."""
        try:
            async with get_db() as session:
                query = select(Call).where(Call.id == uuid.UUID(call_id))
                result = await session.execute(query)
                call = result.scalar_one_or_none()
                return call.transcript if call else None
        except Exception as e:
            logger.error(f"Error getting call transcript: {e}")
            return None

    async def cleanup_orphaned_calls(self, max_age_minutes: int = 30) -> int:
        """
        Clean up calls that are stuck in 'active' status.
        This can happen if the agent crashes or restarts unexpectedly.
        """
        try:
            async with get_db() as session:
                cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
                
                # Find orphaned active calls
                query = select(Call).where(
                    and_(
                        Call.status == "active",
                        Call.started_at < cutoff_time
                    )
                )
                result = await session.execute(query)
                orphaned_calls = result.scalars().all()
                
                count = 0
                for call in orphaned_calls:
                    call.status = "completed"
                    call.ended_at = datetime.utcnow()
                    call.disconnect_reason = "orphaned_cleanup"
                    if call.started_at:
                        call.duration_seconds = int((datetime.utcnow() - call.started_at).total_seconds())
                    count += 1
                
                if count > 0:
                    await session.commit()
                    logger.info(f"Cleaned up {count} orphaned calls")
                
                return count
        except Exception as e:
            logger.error(f"Error cleaning up orphaned calls: {e}")
            return 0

    async def sync_calls_with_livekit(self, active_room_names: list) -> int:
        """
        Sync call status with LiveKit rooms.
        Mark calls as completed if their room no longer exists.
        """
        try:
            async with get_db() as session:
                # Get all active calls
                query = select(Call).where(Call.status == "active")
                result = await session.execute(query)
                active_calls = result.scalars().all()
                
                count = 0
                for call in active_calls:
                    if call.room_name and call.room_name not in active_room_names:
                        call.status = "completed"
                        call.ended_at = datetime.utcnow()
                        call.disconnect_reason = "room_closed"
                        if call.started_at:
                            call.duration_seconds = int((datetime.utcnow() - call.started_at).total_seconds())
                        count += 1
                        logger.info(f"Marked call {call.id} as completed (room {call.room_name} no longer exists)")
                
                if count > 0:
                    await session.commit()
                
                return count
        except Exception as e:
            logger.error(f"Error syncing calls with LiveKit: {e}")
            return 0

    async def _update_analytics_duration(self, duration: int) -> None:
        """Update the average duration in today's analytics."""
        try:
            today = datetime.utcnow().date()
            async with get_db() as session:
                result = await session.execute(
                    select(CallAnalytics).where(CallAnalytics.date == today)
                )
                analytics = result.scalar_one_or_none()
                
                if analytics and analytics.total_calls > 0:
                    analytics.total_duration_seconds += duration
                    analytics.avg_duration_seconds = (
                        analytics.total_duration_seconds / analytics.successful_calls
                        if analytics.successful_calls > 0 else 0
                    )
                    await session.commit()
        except Exception as e:
            logger.error(f"Error updating analytics duration: {e}")

    # =========================================================================
    # ERROR LOGS
    # =========================================================================

    async def create_error_log(
        self,
        service: str,
        level: str,
        message: str,
        context: Dict = None,
        stack_trace: str = None,
    ) -> Optional[Dict]:
        """Create an error log entry."""
        try:
            async with get_db() as session:
                log = ErrorLog(
                    service=service,
                    level=level,
                    message=message,
                    context=context or {},
                    stack_trace=stack_trace,
                )
                session.add(log)
                await session.flush()
                return {"id": str(log.id)}
        except Exception as e:
            logger.error(f"Error creating error log: {e}")
            return None

    async def get_error_logs(
        self,
        service: str = None,
        level: str = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get error logs."""
        try:
            async with get_db() as session:
                query = select(ErrorLog).order_by(desc(ErrorLog.created_at))
                conditions = []
                if service:
                    conditions.append(ErrorLog.service == service)
                if level:
                    conditions.append(ErrorLog.level == level)
                if conditions:
                    query = query.where(and_(*conditions))
                query = query.limit(limit)
                result = await session.execute(query)
                logs = result.scalars().all()
                return [
                    {
                        "id": str(l.id),
                        "service": l.service,
                        "level": l.level,
                        "message": l.message,
                        "context": l.context,
                        "stack_trace": l.stack_trace,
                        "created_at": l.created_at.isoformat() if l.created_at else None,
                    }
                    for l in logs
                ]
        except Exception as e:
            logger.error(f"Error getting error logs: {e}")
            return []

    # =========================================================================
    # SIP EVENTS AND STATUS
    # =========================================================================

    async def create_sip_event(
        self,
        event_type: str,
        trunk_id: str = None,
        trunk_name: str = None,
        call_id: str = None,
        room_name: str = None,
        from_uri: str = None,
        to_uri: str = None,
        caller_number: str = None,
        status_code: int = None,
        status_message: str = None,
        duration_seconds: int = None,
        error_message: str = None,
        metadata: dict = None,
        source_ip: str = None,
    ) -> Optional[str]:
        """Log a SIP event."""
        try:
            async with get_db() as session:
                event = SIPEvent(
                    event_type=event_type,
                    trunk_id=trunk_id,
                    trunk_name=trunk_name,
                    call_id=call_id,
                    room_name=room_name,
                    from_uri=from_uri,
                    to_uri=to_uri,
                    caller_number=caller_number,
                    status_code=status_code,
                    status_message=status_message,
                    duration_seconds=duration_seconds,
                    error_message=error_message,
                    metadata_json=metadata or {},
                    source_ip=source_ip,
                )
                session.add(event)
                await session.commit()
                return str(event.id)
        except Exception as e:
            logger.error(f"Error creating SIP event: {e}")
            return None

    async def get_sip_events(
        self,
        event_type: str = None,
        trunk_id: str = None,
        caller_number: str = None,
        limit: int = 100,
        offset: int = 0,
        from_date: datetime = None,
        to_date: datetime = None,
    ) -> List[Dict]:
        """Get SIP events with filtering."""
        try:
            async with get_db() as session:
                query = select(SIPEvent).order_by(desc(SIPEvent.created_at))
                conditions = []
                
                if event_type:
                    conditions.append(SIPEvent.event_type == event_type)
                if trunk_id:
                    conditions.append(SIPEvent.trunk_id == trunk_id)
                if caller_number:
                    conditions.append(SIPEvent.caller_number.ilike(f"%{caller_number}%"))
                if from_date:
                    conditions.append(SIPEvent.created_at >= from_date)
                if to_date:
                    conditions.append(SIPEvent.created_at <= to_date)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                query = query.offset(offset).limit(limit)
                result = await session.execute(query)
                events = result.scalars().all()
                
                return [
                    {
                        "id": str(e.id),
                        "event_type": e.event_type,
                        "trunk_id": e.trunk_id,
                        "trunk_name": e.trunk_name,
                        "call_id": e.call_id,
                        "room_name": e.room_name,
                        "from_uri": e.from_uri,
                        "to_uri": e.to_uri,
                        "caller_number": e.caller_number,
                        "status_code": e.status_code,
                        "status_message": e.status_message,
                        "duration_seconds": e.duration_seconds,
                        "error_message": e.error_message,
                        "metadata": e.metadata_json,
                        "source_ip": e.source_ip,
                        "created_at": e.created_at.isoformat() if e.created_at else None,
                    }
                    for e in events
                ]
        except Exception as e:
            logger.error(f"Error getting SIP events: {e}")
            return []

    async def get_sip_event_stats(
        self,
        from_date: datetime = None,
        to_date: datetime = None,
    ) -> Dict:
        """Get SIP event statistics."""
        try:
            async with get_db() as session:
                # Default to last 24 hours if no date range
                if not from_date:
                    from_date = datetime.utcnow() - timedelta(hours=24)
                if not to_date:
                    to_date = datetime.utcnow()
                
                conditions = [
                    SIPEvent.created_at >= from_date,
                    SIPEvent.created_at <= to_date,
                ]
                
                # Get counts by event type
                query = select(
                    SIPEvent.event_type,
                    func.count(SIPEvent.id).label("count")
                ).where(and_(*conditions)).group_by(SIPEvent.event_type)
                
                result = await session.execute(query)
                type_counts = {row.event_type: row.count for row in result.fetchall()}
                
                # Get counts by trunk
                query = select(
                    SIPEvent.trunk_id,
                    SIPEvent.trunk_name,
                    func.count(SIPEvent.id).label("count")
                ).where(
                    and_(*conditions, SIPEvent.trunk_id.isnot(None))
                ).group_by(SIPEvent.trunk_id, SIPEvent.trunk_name)
                
                result = await session.execute(query)
                trunk_counts = [
                    {"trunk_id": row.trunk_id, "trunk_name": row.trunk_name, "count": row.count}
                    for row in result.fetchall()
                ]
                
                # Get total events
                query = select(func.count(SIPEvent.id)).where(and_(*conditions))
                result = await session.execute(query)
                total_events = result.scalar() or 0
                
                # Get average duration for completed calls
                query = select(func.avg(SIPEvent.duration_seconds)).where(
                    and_(*conditions, SIPEvent.duration_seconds.isnot(None))
                )
                result = await session.execute(query)
                avg_duration = result.scalar() or 0
                
                return {
                    "total_events": total_events,
                    "by_type": type_counts,
                    "by_trunk": trunk_counts,
                    "avg_call_duration": round(avg_duration, 1) if avg_duration else 0,
                    "period": {
                        "from": from_date.isoformat(),
                        "to": to_date.isoformat(),
                    }
                }
        except Exception as e:
            logger.error(f"Error getting SIP event stats: {e}")
            return {"total_events": 0, "by_type": {}, "by_trunk": [], "avg_call_duration": 0}

    async def update_trunk_status(
        self,
        trunk_id: str,
        trunk_name: str,
        provider_name: str = None,
        status: str = "unknown",
        last_call_at: datetime = None,
        increment_total: bool = False,
        increment_success: bool = False,
        increment_failed: bool = False,
        duration_seconds: int = None,
        error: str = None,
    ) -> bool:
        """Update or create SIP trunk status."""
        try:
            async with get_db() as session:
                # Check if trunk status exists
                query = select(SIPTrunkStatus).where(SIPTrunkStatus.trunk_id == trunk_id)
                result = await session.execute(query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing
                    existing.trunk_name = trunk_name
                    existing.status = status
                    if provider_name:
                        existing.provider_name = provider_name
                    if last_call_at:
                        existing.last_call_at = last_call_at
                    if increment_total:
                        existing.total_calls += 1
                    if increment_success:
                        existing.successful_calls += 1
                    if increment_failed:
                        existing.failed_calls += 1
                    if duration_seconds and existing.total_calls > 0:
                        # Update rolling average
                        total_duration = existing.avg_duration_seconds * (existing.total_calls - 1) + duration_seconds
                        existing.avg_duration_seconds = total_duration / existing.total_calls
                    if error:
                        existing.last_error = error
                        existing.last_error_at = datetime.utcnow()
                else:
                    # Create new
                    trunk_status = SIPTrunkStatus(
                        trunk_id=trunk_id,
                        trunk_name=trunk_name,
                        provider_name=provider_name,
                        status=status,
                        last_call_at=last_call_at,
                        total_calls=1 if increment_total else 0,
                        successful_calls=1 if increment_success else 0,
                        failed_calls=1 if increment_failed else 0,
                        avg_duration_seconds=float(duration_seconds) if duration_seconds else 0,
                        last_error=error,
                        last_error_at=datetime.utcnow() if error else None,
                    )
                    session.add(trunk_status)
                
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating trunk status: {e}")
            return False

    async def get_trunk_statuses(self) -> List[Dict]:
        """Get all trunk statuses."""
        try:
            async with get_db() as session:
                query = select(SIPTrunkStatus).order_by(desc(SIPTrunkStatus.updated_at))
                result = await session.execute(query)
                statuses = result.scalars().all()
                
                return [
                    {
                        "id": str(s.id),
                        "trunk_id": s.trunk_id,
                        "trunk_name": s.trunk_name,
                        "provider_name": s.provider_name,
                        "status": s.status,
                        "last_call_at": s.last_call_at.isoformat() if s.last_call_at else None,
                        "total_calls": s.total_calls,
                        "successful_calls": s.successful_calls,
                        "failed_calls": s.failed_calls,
                        "success_rate": round(s.successful_calls / s.total_calls * 100, 1) if s.total_calls > 0 else 0,
                        "avg_duration_seconds": round(s.avg_duration_seconds, 1),
                        "last_error": s.last_error,
                        "last_error_at": s.last_error_at.isoformat() if s.last_error_at else None,
                        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    }
                    for s in statuses
                ]
        except Exception as e:
            logger.error(f"Error getting trunk statuses: {e}")
            return []

    # =========================================================================
    # SIP PROVIDERS (Persistent configuration with auto-sync)
    # =========================================================================

    async def create_sip_provider(
        self,
        name: str,
        server: str,
        username: str = "",
        password: str = "",
        phone_numbers: List[str] = None,
        allowed_ips: List[str] = None,
        created_by: str = None,
    ) -> Optional[Dict]:
        """Create a new SIP provider configuration."""
        try:
            # Simple encryption for password (in production, use proper encryption)
            import base64
            password_encrypted = base64.b64encode((password or "").encode()).decode()
            
            # Default allowed IPs if not provided
            if not allowed_ips:
                allowed_ips = ["0.0.0.0/0"]  # Allow all by default
            
            if not phone_numbers:
                phone_numbers = []
            
            async with get_db() as session:
                provider = SIPProvider(
                    name=name,
                    server=server,
                    username=username,
                    password_encrypted=password_encrypted,
                    phone_numbers=phone_numbers,
                    allowed_ips=allowed_ips,
                    is_active=True,
                    sync_status="pending",
                    created_by=created_by,
                )
                session.add(provider)
                await session.flush()
                
                return {
                    "id": str(provider.id),
                    "name": provider.name,
                    "server": provider.server,
                    "phone_numbers": provider.phone_numbers,
                    "allowed_ips": provider.allowed_ips,
                }
        except Exception as e:
            logger.error(f"Error creating SIP provider: {e}")
            return None

    async def get_sip_providers(self, active_only: bool = True) -> List[Dict]:
        """Get all SIP providers."""
        try:
            async with get_db() as session:
                query = select(SIPProvider).order_by(SIPProvider.created_at)
                if active_only:
                    query = query.where(SIPProvider.is_active == True)
                result = await session.execute(query)
                providers = result.scalars().all()
                
                return [
                    {
                        "id": str(p.id),
                        "name": p.name,
                        "server": p.server,
                        "username": p.username,
                        "phone_numbers": p.phone_numbers or [],
                        "allowed_ips": p.allowed_ips or [],
                        "is_active": p.is_active,
                        "livekit_trunk_id": p.livekit_trunk_id,
                        "livekit_rule_id": p.livekit_rule_id,
                        "sync_status": p.sync_status,
                        "sync_error": p.sync_error,
                        "last_sync_at": p.last_sync_at.isoformat() if p.last_sync_at else None,
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                    }
                    for p in providers
                ]
        except Exception as e:
            logger.error(f"Error getting SIP providers: {e}")
            return []

    async def get_sip_provider(self, provider_id: str) -> Optional[Dict]:
        """Get a specific SIP provider with decrypted password."""
        try:
            import base64
            async with get_db() as session:
                result = await session.execute(
                    select(SIPProvider).where(SIPProvider.id == uuid.UUID(provider_id))
                )
                p = result.scalar_one_or_none()
                if p:
                    # Decrypt password
                    password = base64.b64decode(p.password_encrypted.encode()).decode()
                    return {
                        "id": str(p.id),
                        "name": p.name,
                        "server": p.server,
                        "username": p.username,
                        "password": password,
                        "phone_numbers": p.phone_numbers or [],
                        "allowed_ips": p.allowed_ips or [],
                        "is_active": p.is_active,
                        "livekit_trunk_id": p.livekit_trunk_id,
                        "livekit_rule_id": p.livekit_rule_id,
                        "sync_status": p.sync_status,
                        "sync_error": p.sync_error,
                        "last_sync_at": p.last_sync_at.isoformat() if p.last_sync_at else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting SIP provider: {e}")
            return None

    async def get_all_sip_providers_with_credentials(self) -> List[Dict]:
        """Get all active SIP providers with decrypted passwords for sync."""
        try:
            import base64
            async with get_db() as session:
                result = await session.execute(
                    select(SIPProvider).where(SIPProvider.is_active == True)
                )
                providers = result.scalars().all()
                
                return [
                    {
                        "id": str(p.id),
                        "name": p.name,
                        "server": p.server,
                        "username": p.username,
                        "password": base64.b64decode(p.password_encrypted.encode()).decode(),
                        "phone_numbers": p.phone_numbers or [],
                        "allowed_ips": p.allowed_ips or [],
                        "livekit_trunk_id": p.livekit_trunk_id,
                        "livekit_rule_id": p.livekit_rule_id,
                    }
                    for p in providers
                ]
        except Exception as e:
            logger.error(f"Error getting SIP providers with credentials: {e}")
            return []

    async def update_sip_provider_sync(
        self,
        provider_id: str,
        livekit_trunk_id: str = None,
        livekit_rule_id: str = None,
        sync_status: str = "synced",
        sync_error: str = None,
    ) -> bool:
        """Update SIP provider sync status after LiveKit sync."""
        try:
            async with get_db() as session:
                values = {
                    "sync_status": sync_status,
                    "last_sync_at": datetime.utcnow(),
                }
                if livekit_trunk_id is not None:
                    values["livekit_trunk_id"] = livekit_trunk_id
                if livekit_rule_id is not None:
                    values["livekit_rule_id"] = livekit_rule_id
                if sync_error is not None:
                    values["sync_error"] = sync_error
                
                await session.execute(
                    update(SIPProvider)
                    .where(SIPProvider.id == uuid.UUID(provider_id))
                    .values(**values)
                )
            return True
        except Exception as e:
            logger.error(f"Error updating SIP provider sync: {e}")
            return False

    async def delete_sip_provider(self, provider_id: str) -> bool:
        """Delete (deactivate) a SIP provider."""
        try:
            async with get_db() as session:
                await session.execute(
                    update(SIPProvider)
                    .where(SIPProvider.id == uuid.UUID(provider_id))
                    .values(is_active=False, sync_status="deleted")
                )
            return True
        except Exception as e:
            logger.error(f"Error deleting SIP provider: {e}")
            return False

    async def get_sip_analytics(
        self,
        days: int = 7,
    ) -> Dict:
        """Get SIP analytics for the specified period."""
        try:
            async with get_db() as session:
                from_date = datetime.utcnow() - timedelta(days=days)
                
                # Get daily call counts using cast for PostgreSQL
                from sqlalchemy import cast, Date, case
                
                query = select(
                    cast(SIPEvent.created_at, Date).label("date"),
                    func.count(SIPEvent.id).label("total"),
                    func.sum(
                        case((SIPEvent.event_type == "call_connected", 1), else_=0)
                    ).label("connected"),
                    func.sum(
                        case((SIPEvent.event_type == "call_failed", 1), else_=0)
                    ).label("failed"),
                ).where(
                    SIPEvent.created_at >= from_date
                ).group_by(
                    cast(SIPEvent.created_at, Date)
                ).order_by(
                    cast(SIPEvent.created_at, Date)
                )
                
                result = await session.execute(query)
                rows = result.fetchall()
                daily_data = [
                    {
                        "date": str(row.date) if row.date else None,
                        "total": row.total or 0,
                        "connected": int(row.connected or 0),
                        "failed": int(row.failed or 0),
                    }
                    for row in rows
                ]
                
                # Get hourly distribution (for today)
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                query = select(
                    func.extract('hour', SIPEvent.created_at).label("hour"),
                    func.count(SIPEvent.id).label("count"),
                ).where(
                    SIPEvent.created_at >= today_start
                ).group_by(
                    func.extract('hour', SIPEvent.created_at)
                ).order_by(
                    func.extract('hour', SIPEvent.created_at)
                )
                
                result = await session.execute(query)
                rows = result.fetchall()
                hourly_data = {int(row.hour): row.count for row in rows}
                
                # Get top callers
                query = select(
                    SIPEvent.caller_number,
                    func.count(SIPEvent.id).label("count"),
                ).where(
                    and_(
                        SIPEvent.created_at >= from_date,
                        SIPEvent.caller_number.isnot(None),
                    )
                ).group_by(
                    SIPEvent.caller_number
                ).order_by(
                    desc(func.count(SIPEvent.id))
                ).limit(10)
                
                result = await session.execute(query)
                rows = result.fetchall()
                top_callers = [
                    {"number": row.caller_number, "count": row.count}
                    for row in rows
                ]
                
                return {
                    "daily": daily_data,
                    "hourly_today": hourly_data,
                    "top_callers": top_callers,
                    "period_days": days,
                }
        except Exception as e:
            logger.error(f"Error getting SIP analytics: {e}")
            import traceback
            traceback.print_exc()
            return {"daily": [], "hourly_today": {}, "top_callers": [], "period_days": days}


# Singleton instance
db = DatabaseService()


def get_database_service() -> DatabaseService:
    """Get the database service instance."""
    return db
