"""Script to append LongTermMemory CRUD methods to database.py"""

METHODS = '''

    # =========================================================================
    # LONG TERM MEMORY
    # =========================================================================

    async def get_memory_items(
        self,
        active_only: bool = False,
        limit: int = 200,
        offset: int = 0,
    ):
        """Get all long-term memory entries."""
        try:
            async with get_db() as session:
                query = select(LongTermMemory).order_by(desc(LongTermMemory.created_at))
                if active_only:
                    query = query.where(LongTermMemory.is_active == True)
                query = query.offset(offset).limit(limit)
                result = await session.execute(query)
                items = result.scalars().all()
                return [
                    {
                        "id": str(item.id),
                        "question": item.question,
                        "answer": item.answer,
                        "comment": item.comment,
                        "is_active": item.is_active,
                        "created_by": item.created_by,
                        "updated_by": item.updated_by,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                    }
                    for item in items
                ]
        except Exception as e:
            logger.error(f"Error getting memory items: {e}")
            return []

    async def get_memory_item(self, item_id: str):
        """Get a single memory entry by ID."""
        try:
            async with get_db() as session:
                result = await session.execute(
                    select(LongTermMemory).where(LongTermMemory.id == uuid.UUID(item_id))
                )
                item = result.scalar_one_or_none()
                if item:
                    return {
                        "id": str(item.id),
                        "question": item.question,
                        "answer": item.answer,
                        "comment": item.comment,
                        "is_active": item.is_active,
                        "created_by": item.created_by,
                        "updated_by": item.updated_by,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting memory item: {e}")
            return None

    async def create_memory_item(self, question: str, answer: str, created_by: str, comment=None):
        """Create a new long-term memory entry."""
        try:
            async with get_db() as session:
                item = LongTermMemory(
                    question=question,
                    answer=answer,
                    comment=comment,
                    is_active=True,
                    created_by=created_by,
                    updated_by=created_by,
                )
                session.add(item)
                await session.flush()
                return {
                    "id": str(item.id),
                    "question": item.question,
                    "answer": item.answer,
                    "comment": item.comment,
                    "is_active": item.is_active,
                }
        except Exception as e:
            logger.error(f"Error creating memory item: {e}")
            return None

    async def update_memory_item(
        self, item_id: str, updated_by: str,
        question=None, answer=None, comment=None, is_active=None,
    ) -> bool:
        """Update a long-term memory entry."""
        try:
            async with get_db() as session:
                values = {"updated_by": updated_by}
                if question is not None:
                    values["question"] = question
                if answer is not None:
                    values["answer"] = answer
                if comment is not None:
                    values["comment"] = comment
                if is_active is not None:
                    values["is_active"] = is_active
                await session.execute(
                    update(LongTermMemory)
                    .where(LongTermMemory.id == uuid.UUID(item_id))
                    .values(**values)
                )
            return True
        except Exception as e:
            logger.error(f"Error updating memory item: {e}")
            return False

    async def delete_memory_item(self, item_id: str) -> bool:
        """Hard-delete a long-term memory entry."""
        try:
            from sqlalchemy import delete as sa_delete
            async with get_db() as session:
                await session.execute(
                    sa_delete(LongTermMemory).where(LongTermMemory.id == uuid.UUID(item_id))
                )
            return True
        except Exception as e:
            logger.error(f"Error deleting memory item: {e}")
            return False

    async def get_active_memory_context(self) -> str:
        """Get all active memory items as formatted context for the agent."""
        try:
            items = await self.get_memory_items(active_only=True)
            if not items:
                return ""
            lines = []
            for item in items:
                lines.append(f"Q: {item[\'question\']}")
                lines.append(f"A: {item[\'answer\']}")
                lines.append("")
            return "\\n".join(lines).strip()
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            return ""
'''

SINGLETON = """

# Singleton instance
db = DatabaseService()


def get_database_service() -> DatabaseService:
    \"\"\"Get the database service instance.\"\"\"
    return db
"""

filepath = "src/services/database.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find insertion point: just before "# Singleton instance"
marker = "\n# Singleton instance"
idx = content.rfind(marker)
if idx == -1:
    print("ERROR: marker not found")
    exit(1)

new_content = content[:idx] + METHODS + SINGLETON

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)

print("SUCCESS: Memory CRUD methods added to database.py")
