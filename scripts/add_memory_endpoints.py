"""Script to append Long-term Memory API endpoints to admin.py"""

ENDPOINTS = '''

# =============================================================================
# LONG TERM MEMORY ENDPOINTS
# =============================================================================

class MemoryItemCreate(BaseModel):
    question: str
    answer: str
    comment: Optional[str] = None


class MemoryItemUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    comment: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("/memory")
async def get_memory_items(
    active_only: bool = False,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service),
):
    """Get all long-term memory entries."""
    items = await db.get_memory_items(active_only=active_only)
    return {"items": items, "total": len(items)}


@router.get("/memory/{item_id}")
async def get_memory_item(
    item_id: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service),
):
    """Get a single memory entry."""
    item = await db.get_memory_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return item


@router.post("/memory")
async def create_memory_item(
    data: MemoryItemCreate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service),
):
    """Create a new long-term memory entry."""
    result = await db.create_memory_item(
        question=data.question,
        answer=data.answer,
        comment=data.comment,
        created_by=current_user["email"],
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create memory item")

    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="memory_create",
        resource_type="long_term_memory",
        resource_id=result["id"],
        new_value={"question": data.question},
        ip_address=req.client.host if req.client else None,
    )

    # Refresh agent cache so new memory takes effect immediately
    try:
        from src.agents.prompts import refresh_cache
        await refresh_cache()
    except Exception as e:
        logger.debug(f"Could not refresh agent cache: {e}")

    return {"success": True, "item": result}


@router.put("/memory/{item_id}")
async def update_memory_item(
    item_id: str,
    data: MemoryItemUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service),
):
    """Update a long-term memory entry."""
    existing = await db.get_memory_item(item_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Memory item not found")

    success = await db.update_memory_item(
        item_id=item_id,
        updated_by=current_user["email"],
        question=data.question,
        answer=data.answer,
        comment=data.comment,
        is_active=data.is_active,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update memory item")

    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="memory_update",
        resource_type="long_term_memory",
        resource_id=item_id,
        old_value={"question": existing["question"]},
        new_value={"question": data.question or existing["question"]},
        ip_address=req.client.host if req.client else None,
    )

    # Refresh agent cache
    try:
        from src.agents.prompts import refresh_cache
        await refresh_cache()
    except Exception as e:
        logger.debug(f"Could not refresh agent cache: {e}")

    return {"success": True, "message": "Memory item updated"}


@router.delete("/memory/{item_id}")
async def delete_memory_item(
    item_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service),
):
    """Delete a long-term memory entry."""
    existing = await db.get_memory_item(item_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Memory item not found")

    success = await db.delete_memory_item(item_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete memory item")

    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="memory_delete",
        resource_type="long_term_memory",
        resource_id=item_id,
        old_value={"question": existing["question"]},
        ip_address=req.client.host if req.client else None,
    )

    # Refresh agent cache
    try:
        from src.agents.prompts import refresh_cache
        await refresh_cache()
    except Exception as e:
        logger.debug(f"Could not refresh agent cache: {e}")

    return {"success": True, "message": "Memory item deleted"}
'''

filepath = "src/api/admin.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

content = content.rstrip() + "\n" + ENDPOINTS + "\n"

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: Memory API endpoints added to admin.py")
