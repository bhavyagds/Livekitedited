"""
Meallion Admin Dashboard - Admin API Endpoints
Handles authentication, knowledge base, prompts, calls, and system management.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Any

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from src.config import settings
from src.services.database import get_database_service, DatabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])
security = HTTPBearer()


# =============================================================================
# MODELS
# =============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


class AdminUser(BaseModel):
    id: str
    email: str
    name: Optional[str]


class KBUploadResponse(BaseModel):
    success: bool
    version_id: str
    message: str


class PromptUpdate(BaseModel):
    language: str = Field(..., pattern="^(en|el)$")
    prompt_type: str = Field(..., pattern="^(system|greeting|closing)$")
    content: str
    change_summary: Optional[str] = ""


class SIPConfigUpdate(BaseModel):
    content: str
    change_summary: Optional[str] = ""


class CallsResponse(BaseModel):
    calls: List[dict]
    total: int
    page: int
    page_size: int


class AnalyticsResponse(BaseModel):
    summary: dict
    today: dict


# =============================================================================
# AUTH HELPERS
# =============================================================================

def create_jwt_token(user_id: str, email: str) -> str:
    """Create a JWT token for admin authentication."""
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=settings.admin_jwt_expiry_hours)
    }
    return jwt.encode(payload, settings.admin_jwt_secret, algorithm="HS256")


def verify_jwt_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, settings.admin_jwt_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseService = Depends(get_database_service)
) -> dict:
    """Dependency to get current authenticated admin user."""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    email = payload.get("email")
    user_id = payload.get("sub")
    
    # Handle env-admin user (fallback authentication)
    if user_id == "env-admin" or email == settings.admin_email:
        return {
            "id": "env-admin",
            "email": email,
            "name": "Admin"
        }
    
    # Try database lookup, but handle errors gracefully
    try:
        user = await db.get_admin_by_email(email)
        if user:
            return user
    except Exception as e:
        logger.warning(f"Database error in get_current_admin: {e}")
        # If database fails but we have a valid token, allow access with minimal user info
        return {
            "id": user_id,
            "email": email,
            "name": "Admin"
        }
    
    # If no user found in database and not env-admin, reject
    raise HTTPException(status_code=401, detail="User not found")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def hash_password(password: str) -> str:
    """Hash a password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# =============================================================================
# AUTH ENDPOINTS
# =============================================================================

@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    req: Request,
    db: DatabaseService = Depends(get_database_service)
):
    """Authenticate admin user and return JWT token."""
    user = None
    
    # Try database first, but handle errors gracefully
    try:
        user = await db.get_admin_by_email(request.email)
    except Exception as e:
        logger.warning(f"Database error during login lookup: {e}")
        # Continue to fallback authentication
    
    if user:
        # Verify against database
        if not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    else:
        # Fallback to env credentials for initial setup
        if request.email != settings.admin_email or request.password != settings.admin_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create user dict for env-based auth
        user = {
            "id": "env-admin",
            "email": settings.admin_email,
            "name": "Admin"
        }
    
    # Update last login (ignore errors if database is unavailable)
    if user.get("id") and user["id"] != "env-admin":
        try:
            await db.update_admin_last_login(user["id"])
        except Exception as e:
            logger.warning(f"Failed to update last login: {e}")
    
    # Create audit log (ignore errors if database is unavailable)
    try:
        await db.create_audit_log(
            user_id=user.get("id"),
            user_email=user["email"],
            action="login",
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("user-agent")
        )
    except Exception as e:
        logger.warning(f"Failed to create audit log: {e}")
    
    # Generate token
    token = create_jwt_token(user.get("id", "env-admin"), user["email"])
    
    return LoginResponse(
        token=token,
        user={
            "id": user.get("id", "env-admin"),
            "email": user["email"],
            "name": user.get("name", "Admin")
        }
    )


@router.get("/me")
async def get_current_user(current_user: dict = Depends(get_current_admin)):
    """Get current authenticated admin user info."""
    return {
        "id": current_user.get("id"),
        "email": current_user.get("email"),
        "name": current_user.get("name")
    }


@router.post("/logout")
async def logout(
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Logout and create audit log."""
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="logout",
        ip_address=req.client.host if req.client else None
    )
    return {"success": True, "message": "Logged out successfully"}


# =============================================================================
# KNOWLEDGE BASE ENDPOINTS
# =============================================================================

@router.get("/kb")
async def get_knowledge_base(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get current active knowledge base."""
    # Try database first
    kb = await db.get_active_kb()
    
    if kb:
        return {
            "source": "database",
            "version_id": kb["id"],
            "version_number": kb["version_number"],
            "content": kb["content"],
            "file_name": kb.get("file_name"),
            "updated_at": kb["created_at"],
            "updated_by": kb.get("changed_by")
        }
    
    # Fallback to file
    kb_path = Path(__file__).parent.parent.parent / "knowledge" / "meallion_faq.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return {
            "source": "file",
            "content": content,
            "file_name": "meallion_faq.json",
            "updated_at": datetime.fromtimestamp(kb_path.stat().st_mtime).isoformat()
        }
    
    raise HTTPException(status_code=404, detail="Knowledge base not found")


@router.post("/kb/upload", response_model=KBUploadResponse)
async def upload_knowledge_base(
    file: UploadFile = File(...),
    change_summary: str = Query(default=""),
    req: Request = None,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Upload a new knowledge base JSON file."""
    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are allowed")
    
    # Read and parse content
    try:
        content = await file.read()
        kb_data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Save to database
    version = await db.create_kb_version(
        content=kb_data,
        changed_by=current_user["email"],
        change_summary=change_summary or f"Uploaded {file.filename}",
        file_name=file.filename,
        file_size=len(content)
    )
    
    if not version:
        raise HTTPException(status_code=500, detail="Failed to save knowledge base")
    
    # Also save to file for the agent to use
    kb_path = Path(__file__).parent.parent.parent / "knowledge" / "meallion_faq.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_upload",
        resource_type="knowledge_base",
        resource_id=version["id"],
        new_value={"file_name": file.filename, "size": len(content)},
        ip_address=req.client.host if req and req.client else None
    )
    
    return KBUploadResponse(
        success=True,
        version_id=version["id"],
        message=f"Knowledge base updated successfully (version {version['version_number']})"
    )


@router.get("/kb/versions")
async def get_kb_versions(
    limit: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get knowledge base version history."""
    versions = await db.get_kb_versions(limit=limit)
    return {"versions": versions}


@router.get("/kb/versions/{version_id}")
async def get_kb_version(
    version_id: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get a specific knowledge base version."""
    version = await db.get_kb_version_by_id(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@router.post("/kb/rollback/{version_id}")
async def rollback_kb_version(
    version_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Rollback to a specific knowledge base version."""
    # Get the version to rollback to
    version = await db.get_kb_version_by_id(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    # Rollback in database
    success = await db.rollback_kb_version(version_id, current_user["email"])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to rollback")
    
    # Update file
    kb_path = Path(__file__).parent.parent.parent / "knowledge" / "meallion_faq.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(version["content"], f, ensure_ascii=False, indent=2)
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_rollback",
        resource_type="knowledge_base",
        resource_id=version_id,
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "message": f"Rolled back to version {version['version_number']}"}


@router.get("/kb/download")
async def download_knowledge_base(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Download current knowledge base as JSON."""
    from fastapi.responses import JSONResponse
    
    kb = await db.get_active_kb()
    if kb:
        return JSONResponse(
            content=kb["content"],
            headers={
                "Content-Disposition": f'attachment; filename="{kb.get("file_name", "knowledge_base.json")}"'
            }
        )
    
    # Fallback to file
    kb_path = Path(__file__).parent.parent.parent / "knowledge" / "meallion_faq.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return JSONResponse(
            content=content,
            headers={"Content-Disposition": 'attachment; filename="meallion_faq.json"'}
        )
    
    raise HTTPException(status_code=404, detail="Knowledge base not found")


# =============================================================================
# KB CONTENT ENDPOINTS (Simple text per language)
# =============================================================================

class KBContentUpdate(BaseModel):
    content: str


@router.get("/kb/content/{language}")
async def get_kb_content(
    language: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get knowledge base content for a specific language."""
    content = await db.get_kb_content(language)
    if content:
        return content
    # Return empty content if not found
    return {"language": language, "content": "", "updated_at": None}


@router.put("/kb/content/{language}")
async def save_kb_content(
    language: str,
    data: KBContentUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Save knowledge base content for a specific language."""
    success = await db.save_kb_content(
        language=language,
        content=data.content,
        updated_by=current_user["email"]
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save content")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_content_update",
        resource_type="kb_content",
        resource_id=language,
        new_value={"language": language, "content_length": len(data.content)},
        ip_address=req.client.host if req.client else None
    )
    
    # Trigger cache refresh in agent (if running)
    try:
        from src.agents.prompts import refresh_cache
        await refresh_cache()
    except Exception as e:
        logger.debug(f"Could not refresh agent cache: {e}")
    
    return {"success": True, "message": f"Knowledge base for {language} saved"}


@router.get("/kb/content")
async def get_all_kb_content(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all knowledge base content for all languages."""
    content = await db.get_all_kb_content()
    return {"content": content}


# =============================================================================
# PROMPTS CONTENT ENDPOINTS (Simple text per language)
# =============================================================================

class PromptsContentUpdate(BaseModel):
    content: str


@router.get("/prompts/content/{language}")
async def get_prompts_content(
    language: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get prompts content for a specific language."""
    content = await db.get_prompts_content(language)
    if content:
        return content
    # Return empty content if not found
    return {"language": language, "content": "", "updated_at": None}


@router.put("/prompts/content/{language}")
async def save_prompts_content(
    language: str,
    data: PromptsContentUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Save prompts content for a specific language."""
    success = await db.save_prompts_content(
        language=language,
        content=data.content,
        updated_by=current_user["email"]
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save prompts")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="prompts_content_update",
        resource_type="prompts_content",
        resource_id=language,
        new_value={"language": language, "content_length": len(data.content)},
        ip_address=req.client.host if req.client else None
    )
    
    # Trigger cache refresh in agent (if running)
    try:
        from src.agents.prompts import refresh_cache
        await refresh_cache()
    except Exception as e:
        logger.debug(f"Could not refresh agent cache: {e}")
    
    return {"success": True, "message": f"Prompts for {language} saved"}


@router.get("/prompts/content")
async def get_all_prompts_content(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all prompts content for all languages."""
    content = await db.get_all_prompts_content()
    return {"content": content}


# =============================================================================
# KB ITEMS ENDPOINTS (Direct FAQ Management)
# =============================================================================

class KBItemCreate(BaseModel):
    category: str = "General"
    question: str
    answer: str
    keywords: Optional[List[str]] = []
    language: str = "el"
    display_order: int = 0


class KBItemUpdate(BaseModel):
    category: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    keywords: Optional[List[str]] = None
    language: Optional[str] = None
    is_active: Optional[bool] = None
    display_order: Optional[int] = None


@router.get("/kb/items")
async def get_kb_items(
    category: Optional[str] = None,
    language: Optional[str] = None,
    include_inactive: bool = False,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all FAQ items from the knowledge base."""
    items = await db.get_kb_items(
        category=category,
        language=language,
        active_only=not include_inactive,
    )
    categories = await db.get_kb_categories()
    return {
        "items": items,
        "categories": categories,
        "total": len(items),
    }


@router.get("/kb/items/{item_id}")
async def get_kb_item(
    item_id: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get a specific FAQ item."""
    item = await db.get_kb_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.post("/kb/items")
async def create_kb_item(
    item: KBItemCreate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Create a new FAQ item."""
    result = await db.create_kb_item(
        category=item.category,
        question=item.question,
        answer=item.answer,
        keywords=item.keywords,
        language=item.language,
        display_order=item.display_order,
        created_by=current_user["email"],
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create item")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_item_create",
        resource_type="kb_item",
        resource_id=result["id"],
        new_value={"question": item.question, "category": item.category},
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "item": result}


@router.put("/kb/items/{item_id}")
async def update_kb_item(
    item_id: str,
    item: KBItemUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Update a FAQ item."""
    # Get existing item for audit
    existing = await db.get_kb_item(item_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Item not found")
    
    success = await db.update_kb_item(
        item_id=item_id,
        updated_by=current_user["email"],
        category=item.category,
        question=item.question,
        answer=item.answer,
        keywords=item.keywords,
        language=item.language,
        is_active=item.is_active,
        display_order=item.display_order,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update item")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_item_update",
        resource_type="kb_item",
        resource_id=item_id,
        old_value={"question": existing["question"]},
        new_value={"question": item.question or existing["question"]},
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "message": "Item updated"}


@router.delete("/kb/items/{item_id}")
async def delete_kb_item(
    item_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Delete a FAQ item (soft delete)."""
    existing = await db.get_kb_item(item_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Item not found")
    
    success = await db.delete_kb_item(item_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete item")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_item_delete",
        resource_type="kb_item",
        resource_id=item_id,
        old_value={"question": existing["question"]},
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "message": "Item deleted"}


@router.post("/kb/import")
async def import_kb_from_file(
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Import FAQ items from the existing JSON file into the database."""
    kb_path = Path(__file__).parent.parent.parent / "knowledge" / "meallion_faq.json"
    
    if not kb_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge base file not found")
    
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    count = await db.import_kb_items_from_json(data, current_user["email"])
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="kb_import",
        resource_type="kb_items",
        new_value={"imported_count": count},
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "message": f"Imported {count} FAQ items", "count": count}


# =============================================================================
# LANGUAGES ENDPOINTS
# =============================================================================

class LanguageCreate(BaseModel):
    code: str
    name: str
    native_name: str
    flag_emoji: Optional[str] = None
    is_default: bool = False


class LanguageUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    native_name: Optional[str] = None
    flag_emoji: Optional[str] = None
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None


@router.get("/languages")
async def get_languages(
    include_inactive: bool = False,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all supported languages."""
    languages = await db.get_languages(active_only=not include_inactive)
    
    # If no languages exist, initialize defaults
    if not languages:
        count = await db.init_default_languages()
        if count > 0:
            languages = await db.get_languages(active_only=not include_inactive)
    
    return {"languages": languages}


@router.post("/languages")
async def create_language(
    lang: LanguageCreate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Create a new language."""
    result = await db.create_language(
        code=lang.code,
        name=lang.name,
        native_name=lang.native_name,
        flag_emoji=lang.flag_emoji,
        is_default=lang.is_default,
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create language")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="language_create",
        resource_type="language",
        resource_id=result["id"],
        new_value={"code": lang.code, "name": lang.name},
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "language": result}


@router.put("/languages/{language_id}")
async def update_language(
    language_id: str,
    lang: LanguageUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Update a language."""
    success = await db.update_language(
        language_id=language_id,
        code=lang.code,
        name=lang.name,
        native_name=lang.native_name,
        flag_emoji=lang.flag_emoji,
        is_default=lang.is_default,
        is_active=lang.is_active,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update language")

    # If this language is marked as default, sync agent_language setting too.
    # This keeps "Languages default" and "Default Agent Language" behavior aligned.
    if lang.is_default:
        language_code = (lang.code or "").lower().strip()
        if not language_code:
            all_languages = await db.get_languages(active_only=False)
            selected = next((l for l in all_languages if l.get("id") == language_id), None)
            language_code = (selected or {}).get("code", "")

        if language_code:
            await db.set_setting(
                key="agent_language",
                value=language_code,
                description="Default agent language (synced from Languages)",
                updated_by=current_user["email"],
            )
    
    return {"success": True, "message": "Language updated"}


@router.delete("/languages/{language_id}")
async def delete_language(
    language_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Delete a language (soft delete)."""
    success = await db.delete_language(language_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete language")
    
    return {"success": True, "message": "Language deleted"}


# =============================================================================
# PROMPTS ENDPOINTS
# =============================================================================

@router.get("/prompts")
async def get_prompts(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get current prompts from file."""
    prompts_path = Path(__file__).parent.parent / "agents" / "prompts.py"
    
    if not prompts_path.exists():
        raise HTTPException(status_code=404, detail="Prompts file not found")
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Also get active prompts from DB if any
    db_prompts = await db.get_active_prompts()
    
    return {
        "file_content": content,
        "db_prompts": db_prompts,
        "file_path": str(prompts_path)
    }


@router.get("/prompts/versions")
async def get_prompt_versions(
    language: Optional[str] = None,
    prompt_type: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get prompt version history."""
    versions = await db.get_prompt_versions(language=language, prompt_type=prompt_type, limit=limit)
    return {"versions": versions}


# =============================================================================
# CALLS ENDPOINTS
# =============================================================================

@router.get("/calls", response_model=CallsResponse)
async def get_calls(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, le=100),
    status: Optional[str] = None,
    call_type: Optional[str] = None,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get paginated call history."""
    offset = (page - 1) * page_size
    
    calls = await db.get_calls(
        limit=page_size,
        offset=offset,
        status=status,
        call_type=call_type
    )
    
    total = await db.get_calls_count(status=status, call_type=call_type)
    
    return CallsResponse(
        calls=calls,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/calls/{call_id}")
async def get_call(
    call_id: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get a single call with full details."""
    call = await db.get_call_by_id(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@router.get("/calls/{call_id}/transcript")
async def get_call_transcript(
    call_id: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get the transcript for a specific call."""
    transcript = await db.get_call_transcript(call_id)
    if transcript is None:
        call = await db.get_call_by_id(call_id)
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        return {"call_id": call_id, "transcript": None, "message": "No transcript available"}
    return {"call_id": call_id, "transcript": transcript}


# =============================================================================
# LIVE SESSIONS ENDPOINTS
# =============================================================================

@router.get("/sessions")
async def get_active_sessions(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service),
):
    """Get all active LiveKit sessions (rooms)."""
    try:
        from src.services.livekit_rooms import get_room_service
        room_service = get_room_service()
        
        sessions = await room_service.get_active_sessions()
        
        # Sync database with actual LiveKit rooms (cleanup orphaned calls)
        active_room_names = [s["room_name"] for s in sessions]
        synced = await db.sync_calls_with_livekit(active_room_names)
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "synced_orphaned_calls": synced,
        }
    except Exception as e:
        logger.error(f"Failed to get active sessions: {e}")
        return {"sessions": [], "count": 0, "error": str(e)}


@router.get("/sessions/{room_name}")
async def get_session_details(
    room_name: str,
    current_user: dict = Depends(get_current_admin),
):
    """Get details of a specific session."""
    try:
        from src.services.livekit_rooms import get_room_service
        room_service = get_room_service()
        
        room = await room_service.get_room(room_name)
        if not room:
            raise HTTPException(status_code=404, detail="Session not found")
        
        participants = await room_service.list_participants(room_name)
        
        return {
            "room": room,
            "participants": participants,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{room_name}/participants")
async def get_session_participants(
    room_name: str,
    current_user: dict = Depends(get_current_admin),
):
    """Get participants in a session."""
    try:
        from src.services.livekit_rooms import get_room_service
        room_service = get_room_service()
        
        participants = await room_service.list_participants(room_name)
        return {"participants": participants, "count": len(participants)}
    except Exception as e:
        logger.error(f"Failed to get session participants: {e}")
        return {"participants": [], "count": 0, "error": str(e)}


@router.delete("/sessions/{room_name}")
async def terminate_session(
    room_name: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Terminate a session (end the call)."""
    try:
        from src.services.livekit_rooms import get_room_service
        room_service = get_room_service()
        
        success = await room_service.delete_room(room_name)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to terminate session")
        
        # Update the call record if it exists
        await db.record_call_end(
            room_name=room_name,
            status="completed",
            disconnect_reason="terminated_by_admin",
        )
        
        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="session_terminate",
            resource_type="session",
            resource_id=room_name,
            ip_address=req.client.host if req.client else None
        )
        
        return {"success": True, "message": f"Session {room_name} terminated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{room_name}/participants/{identity}")
async def remove_participant(
    room_name: str,
    identity: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Remove a participant from a session."""
    try:
        from src.services.livekit_rooms import get_room_service
        room_service = get_room_service()
        
        success = await room_service.remove_participant(room_name, identity)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove participant")
        
        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="participant_remove",
            resource_type="participant",
            resource_id=f"{room_name}/{identity}",
            ip_address=req.client.host if req.client else None
        )
        
        return {"success": True, "message": f"Participant {identity} removed from {room_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove participant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = Query(default=30, le=90),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get analytics summary."""
    summary = await db.get_analytics_summary(days=days)
    today = await db.get_today_stats()
    
    return AnalyticsResponse(summary=summary, today=today)


# =============================================================================
# SYSTEM SETTINGS ENDPOINTS
# =============================================================================

class SettingUpdate(BaseModel):
    value: Any
    description: Optional[str] = None


@router.get("/settings")
async def get_all_settings(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all system settings."""
    settings = await db.get_all_settings()
    return {"settings": settings}


@router.get("/settings/{key}")
async def get_setting(
    key: str,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get a specific system setting."""
    value = await db.get_setting(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return {"key": key, "value": value}


@router.put("/settings/{key}")
async def update_setting(
    key: str,
    data: SettingUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Update a system setting."""
    success = await db.set_setting(
        key=key,
        value=data.value,
        description=data.description,
        updated_by=current_user["email"]
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update setting")
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="setting_update",
        resource_type="system_setting",
        resource_id=key,
        new_value={"key": key, "value": data.value},
        ip_address=req.client.host if req.client else None
    )
    
    # Refresh local cache view after any setting update.
    # (Agent container reads from DB independently.)
    try:
        from src.agents.prompts import refresh_cache
        await refresh_cache()
    except Exception as e:
        logger.debug(f"Could not refresh agent cache: {e}")
    
    return {"success": True, "key": key, "value": data.value}


@router.post("/settings/init")
async def init_default_settings(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Initialize default system settings."""
    await db.init_default_settings()
    settings = await db.get_all_settings()
    return {"success": True, "settings": settings}


# =============================================================================
# SIP CONFIG ENDPOINTS
# =============================================================================

@router.get("/sip-config")
async def get_sip_config(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get current SIP configuration."""
    # Get from file
    sip_path = Path(__file__).parent.parent.parent / "livekit" / "sip-config.yaml"
    
    content = ""
    if sip_path.exists():
        with open(sip_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # Get versions from DB
    versions = await db.get_sip_config_versions(limit=10)
    
    return {
        "content": content,
        "file_path": str(sip_path),
        "versions": versions
    }


@router.put("/sip-config")
async def update_sip_config(
    update: SIPConfigUpdate,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Update SIP configuration."""
    # Save to database
    version = await db.create_sip_config_version(
        content=update.content,
        changed_by=current_user["email"],
        change_summary=update.change_summary
    )
    
    # Save to file
    sip_path = Path(__file__).parent.parent.parent / "livekit" / "sip-config.yaml"
    with open(sip_path, "w", encoding="utf-8") as f:
        f.write(update.content)
    
    # Audit log
    await db.create_audit_log(
        user_id=current_user.get("id"),
        user_email=current_user["email"],
        action="sip_config_update",
        resource_type="sip_config",
        ip_address=req.client.host if req.client else None
    )
    
    return {"success": True, "message": "SIP configuration updated", "version_id": version["id"] if version else None}


@router.get("/sip-config/versions")
async def get_sip_config_versions(
    limit: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get SIP config version history."""
    versions = await db.get_sip_config_versions(limit=limit)
    return {"versions": versions}


# =============================================================================
# LIVEKIT SIP API ENDPOINTS (Hot-reload without restart)
# =============================================================================

class SIPProviderConfig(BaseModel):
    provider_name: str
    server: str
    username: str = ""  # Optional - leave empty for IP-based auth (inbound only)
    password: str = ""  # Optional - leave empty for IP-based auth (inbound only)
    phone_numbers: List[str] = []
    allowed_ips: List[str] = []  # List of allowed IP ranges in CIDR notation


@router.get("/sip/status")
async def get_sip_status(
    current_user: dict = Depends(get_current_admin),
):
    """Get current SIP configuration status from LiveKit."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        status = await sip_service.get_sip_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get SIP status: {e}")
        return {"status": "error", "error": str(e), "trunks": [], "rules": []}


@router.get("/sip/trunks")
async def list_sip_trunks(
    current_user: dict = Depends(get_current_admin),
):
    """List all SIP inbound trunks."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        trunks = await sip_service.list_inbound_trunks()
        return {"trunks": trunks}
    except Exception as e:
        logger.error(f"Failed to list SIP trunks: {e}")
        return {"trunks": [], "error": str(e)}


@router.post("/sip/provider")
async def configure_sip_provider(
    config: SIPProviderConfig,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Configure a SIP provider (creates trunk + dispatch rule in LiveKit AND saves to DB)."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()

        # First, configure in LiveKit
        result = await sip_service.configure_provider(
            provider_name=config.provider_name,
            server=config.server,
            username=config.username,
            password=config.password,
            phone_numbers=config.phone_numbers,
            allowed_ips=config.allowed_ips,
        )

        # If successful, save to database for persistence
        if result.get("success"):
            provider = await db.create_sip_provider(
                name=config.provider_name,
                server=config.server,
                username=config.username,
                password=config.password,
                phone_numbers=config.phone_numbers,
                allowed_ips=config.allowed_ips,
                created_by=current_user["email"],
            )
            
            if provider:
                # Update with LiveKit IDs
                await db.update_sip_provider_sync(
                    provider["id"],
                    livekit_trunk_id=result.get("trunk_id"),
                    livekit_rule_id=result.get("rule_id"),
                    sync_status="synced",
                )
                result["provider_id"] = provider["id"]

        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="sip_provider_configure",
            resource_type="sip_provider",
            resource_id=config.provider_name,
            new_value={"provider": config.provider_name, "server": config.server},
            ip_address=req.client.host if req.client else None
        )

        return result
    except Exception as e:
        logger.error(f"Failed to configure SIP provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sip/providers")
async def get_sip_providers(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all saved SIP providers from database."""
    providers = await db.get_sip_providers(active_only=True)
    return {"providers": providers}


@router.delete("/sip/providers/{provider_id}")
async def delete_sip_provider(
    provider_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Delete a SIP provider (from DB and LiveKit)."""
    try:
        # Get provider to find LiveKit IDs
        provider = await db.get_sip_provider(provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        # Delete from LiveKit if trunk exists
        if provider.get("livekit_trunk_id"):
            from src.services.livekit_sip import get_sip_service
            sip_service = get_sip_service()
            
            # Delete dispatch rule first
            if provider.get("livekit_rule_id"):
                await sip_service.delete_dispatch_rule(provider["livekit_rule_id"])
            
            # Delete trunk
            await sip_service.delete_inbound_trunk(provider["livekit_trunk_id"])
        
        # Delete from database
        await db.delete_sip_provider(provider_id)
        
        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="sip_provider_delete",
            resource_type="sip_provider",
            resource_id=provider_id,
            old_value={"provider": provider["name"]},
            ip_address=req.client.host if req.client else None
        )
        
        return {"success": True, "message": f"Provider {provider['name']} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SIP provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sip/sync")
async def sync_sip_providers(
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Manually sync all SIP providers from database to LiveKit."""
    try:
        from src.services.livekit_sip import sync_sip_providers_on_startup
        result = await sync_sip_providers_on_startup()
        
        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="sip_providers_sync",
            resource_type="sip_provider",
            new_value=result,
            ip_address=req.client.host if req.client else None
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to sync SIP providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sip/trunks/{trunk_id}")
async def delete_sip_trunk(
    trunk_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Delete a SIP trunk."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        
        success = await sip_service.delete_inbound_trunk(trunk_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete trunk")
        
        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="sip_trunk_delete",
            resource_type="sip_trunk",
            resource_id=trunk_id,
            ip_address=req.client.host if req.client else None
        )
        
        return {"success": True, "message": f"Trunk {trunk_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SIP trunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sip/rules")
async def list_sip_rules(
    current_user: dict = Depends(get_current_admin),
):
    """List all SIP dispatch rules."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        rules = await sip_service.list_dispatch_rules()
        return {"rules": rules}
    except Exception as e:
        logger.error(f"Failed to list SIP rules: {e}")
        return {"rules": [], "error": str(e)}


@router.delete("/sip/rules/{rule_id}")
async def delete_sip_rule(
    rule_id: str,
    req: Request,
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Delete a SIP dispatch rule."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        
        success = await sip_service.delete_dispatch_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete rule")
        
        # Audit log
        await db.create_audit_log(
            user_id=current_user.get("id"),
            user_email=current_user["email"],
            action="sip_rule_delete",
            resource_type="sip_rule",
            resource_id=rule_id,
            ip_address=req.client.host if req.client else None
        )
        
        return {"success": True, "message": f"Rule {rule_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SIP rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SIPValidateRequest(BaseModel):
    provider_name: str
    server: str
    username: str = ""  # Optional for inbound-only
    password: str = ""  # Optional for inbound-only
    phone_numbers: List[str] = []
    allowed_ips: List[str] = []


@router.post("/sip/validate")
async def validate_sip_config(
    config: SIPValidateRequest,
    current_user: dict = Depends(get_current_admin),
):
    """Validate SIP provider configuration without creating it."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        
        result = await sip_service.validate_provider_config(
            provider_name=config.provider_name,
            server=config.server,
            username=config.username,
            password=config.password,
            phone_numbers=config.phone_numbers,
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to validate SIP config: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
        }


@router.get("/sip/test-connection")
async def test_sip_connection(
    current_user: dict = Depends(get_current_admin),
):
    """Test connection to LiveKit SIP service."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        
        result = await sip_service.test_livekit_connection()
        return result
    except Exception as e:
        logger.error(f"Failed to test SIP connection: {e}")
        return {
            "connected": False,
            "message": str(e),
        }


@router.get("/sip/health")
async def sip_health_check(
    current_user: dict = Depends(get_current_admin),
):
    """Comprehensive SIP health check."""
    try:
        from src.services.livekit_sip import get_sip_service
        sip_service = get_sip_service()
        
        # Test LiveKit connection
        connection_test = await sip_service.test_livekit_connection()
        
        # Get current status
        status = await sip_service.get_sip_status()
        
        # Check for issues
        issues = []
        
        if not connection_test["connected"]:
            issues.append({
                "severity": "error",
                "message": f"Cannot connect to LiveKit: {connection_test['message']}",
            })
        
        # Check for orphaned rules (rules without matching trunks)
        if status.get("rules"):
            trunk_ids = {t["id"] for t in status.get("trunks", [])}
            for rule in status["rules"]:
                for rule_trunk_id in rule.get("trunk_ids", []):
                    if rule_trunk_id not in trunk_ids:
                        issues.append({
                            "severity": "warning",
                            "message": f"Dispatch rule '{rule['name']}' references non-existent trunk {rule_trunk_id}",
                        })
        
        # Check for trunks without dispatch rules
        if status.get("trunks"):
            rules_trunk_ids = set()
            for rule in status.get("rules", []):
                rules_trunk_ids.update(rule.get("trunk_ids", []))
            
            for trunk in status["trunks"]:
                if trunk["id"] not in rules_trunk_ids:
                    issues.append({
                        "severity": "warning",
                        "message": f"Trunk '{trunk['name']}' has no dispatch rules - calls will not be routed",
                    })
        
        return {
            "healthy": len([i for i in issues if i["severity"] == "error"]) == 0,
            "connection": connection_test,
            "status": status.get("status", "unknown"),
            "trunks_count": status.get("trunks_count", 0),
            "rules_count": status.get("rules_count", 0),
            "issues": issues,
        }
    except Exception as e:
        logger.error(f"SIP health check failed: {e}")
        return {
            "healthy": False,
            "connection": {"connected": False, "message": str(e)},
            "status": "error",
            "issues": [{"severity": "error", "message": str(e)}],
        }


@router.get("/sip/events")
async def get_sip_events(
    event_type: Optional[str] = Query(None),
    trunk_id: Optional[str] = Query(None),
    caller_number: Optional[str] = Query(None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get SIP events with filtering."""
    events = await db.get_sip_events(
        event_type=event_type,
        trunk_id=trunk_id,
        caller_number=caller_number,
        limit=limit,
        offset=offset,
    )
    return {"events": events, "count": len(events)}


@router.get("/sip/events/stats")
async def get_sip_event_stats(
    hours: int = Query(default=24, ge=1, le=720),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get SIP event statistics."""
    from datetime import datetime, timedelta
    from_date = datetime.utcnow() - timedelta(hours=hours)
    stats = await db.get_sip_event_stats(from_date=from_date)
    return stats


@router.get("/sip/trunk-statuses")
async def get_sip_trunk_statuses(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get status for all SIP trunks."""
    statuses = await db.get_trunk_statuses()
    return {"statuses": statuses}


@router.get("/sip/analytics")
async def get_sip_analytics(
    days: int = Query(default=7, ge=1, le=90),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get SIP analytics for the specified period."""
    analytics = await db.get_sip_analytics(days=days)
    return analytics


class SIPEventCreate(BaseModel):
    event_type: str
    trunk_id: Optional[str] = None
    trunk_name: Optional[str] = None
    call_id: Optional[str] = None
    room_name: Optional[str] = None
    from_uri: Optional[str] = None
    to_uri: Optional[str] = None
    caller_number: Optional[str] = None
    status_code: Optional[int] = None
    status_message: Optional[str] = None
    duration_seconds: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Optional[dict] = None
    source_ip: Optional[str] = None


@router.post("/sip/events")
async def create_sip_event(
    event: SIPEventCreate,
    req: Request,
    db: DatabaseService = Depends(get_database_service)
):
    """Log a SIP event (called by LiveKit webhook or agent)."""
    # Note: This endpoint doesn't require admin auth - it's called by internal services
    event_id = await db.create_sip_event(
        event_type=event.event_type,
        trunk_id=event.trunk_id,
        trunk_name=event.trunk_name,
        call_id=event.call_id,
        room_name=event.room_name,
        from_uri=event.from_uri,
        to_uri=event.to_uri,
        caller_number=event.caller_number,
        status_code=event.status_code,
        status_message=event.status_message,
        duration_seconds=event.duration_seconds,
        error_message=event.error_message,
        metadata=event.metadata,
        source_ip=event.source_ip or (req.client.host if req.client else None),
    )
    
    # Update trunk status if trunk_id is provided
    if event.trunk_id:
        is_success = event.event_type in ["call_connected", "call_completed"]
        is_failed = event.event_type in ["call_failed", "auth_failed", "timeout"]
        
        await db.update_trunk_status(
            trunk_id=event.trunk_id,
            trunk_name=event.trunk_name or event.trunk_id,
            status="connected" if is_success else ("error" if is_failed else "unknown"),
            last_call_at=datetime.utcnow() if event.event_type.startswith("call_") else None,
            increment_total=event.event_type == "call_incoming",
            increment_success=is_success,
            increment_failed=is_failed,
            duration_seconds=event.duration_seconds,
            error=event.error_message if is_failed else None,
        )
    
    return {"success": bool(event_id), "event_id": event_id}


# =============================================================================
# AUDIT LOG ENDPOINTS
# =============================================================================

@router.get("/audit-logs")
async def get_audit_logs(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get audit logs."""
    logs = await db.get_audit_logs(limit=limit, offset=offset)
    return {"logs": logs}


# =============================================================================
# ERROR LOGS ENDPOINTS
# =============================================================================

@router.get("/error-logs")
async def get_error_logs(
    service: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """Get error logs."""
    logs = await db.get_error_logs(service=service, level=level, limit=limit)
    return {"logs": logs}


# =============================================================================
# SYSTEM HEALTH ENDPOINTS
# =============================================================================

@router.get("/health")
async def admin_health_check(current_user: dict = Depends(get_current_admin)):
    """Admin-only health check with detailed status."""
    from src.config import settings
    
    services = {
        "livekit": {
            "configured": bool(settings.livekit_api_key),
            "url": settings.livekit_url
        },
        "openai": {
            "configured": bool(settings.openai_api_key),
            "model": settings.openai_model
        },
        "elevenlabs": {
            "configured": bool(settings.elevenlabs_api_key),
            "voice_id": settings.elevenlabs_voice_id
        },
        "database": {
            "configured": bool(settings.postgres_url),
            "url": settings.postgres_url[:30] + "..." if settings.postgres_url else None
        },
        "shopify": {
            "configured": bool(settings.shopify_store_url and settings.shopify_access_token)
        }
    }
    
    return {
        "status": "healthy",
        "services": services,
        "agent_language": settings.agent_language,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/verify-config")
async def verify_agent_config(
    current_user: dict = Depends(get_current_admin),
    db: DatabaseService = Depends(get_database_service)
):
    """
    Verify what the agent is actually using.
    Shows: DB prompts, DB settings, what's loaded vs fallback.
    Use this to debug configuration issues.
    """
    from src.agents.prompts import (
        get_system_prompt, get_greeting, get_closing, 
        get_agent_language, get_prompts_content, load_knowledge_base,
        refresh_cache, _cache
    )
    from src.config import settings
    
    # Force refresh cache to get latest from DB
    await refresh_cache()
    
    # Get current agent language
    db_language = get_agent_language()
    env_language = settings.agent_language
    
    # Get prompts content from DB
    prompts_en = get_prompts_content("en")
    prompts_el = get_prompts_content("el")
    
    # Get knowledge base content
    kb_en = load_knowledge_base("en")
    kb_el = load_knowledge_base("el")
    
    # Get the actual system prompt that would be used
    system_prompt = get_system_prompt(db_language)
    greeting = get_greeting(db_language)
    closing = get_closing(db_language)
    
    # Get all settings from DB
    all_settings = await db.get_all_settings()
    
    return {
        "language": {
            "from_database": db_language,
            "from_env": env_language,
            "active": db_language,  # DB takes priority
        },
        "prompts": {
            "en": {
                "loaded_from": "database" if prompts_en else "fallback",
                "length": len(prompts_en) if prompts_en else 0,
                "preview": prompts_en[:200] + "..." if prompts_en and len(prompts_en) > 200 else prompts_en,
            },
            "el": {
                "loaded_from": "database" if prompts_el else "fallback",
                "length": len(prompts_el) if prompts_el else 0,
                "preview": prompts_el[:200] + "..." if prompts_el and len(prompts_el) > 200 else prompts_el,
            }
        },
        "knowledge_base": {
            "en": {
                "loaded_from": "database" if kb_en else "empty",
                "length": len(kb_en) if kb_en else 0,
            },
            "el": {
                "loaded_from": "database" if kb_el else "empty",
                "length": len(kb_el) if kb_el else 0,
            }
        },
        "active_config": {
            "language": db_language,
            "system_prompt_length": len(system_prompt),
            "system_prompt_preview": system_prompt[:300] + "..." if len(system_prompt) > 300 else system_prompt,
            "greeting": greeting,
            "closing": closing,
        },
        "database_settings": all_settings,
        "cache_status": {
            "last_fetch": _cache.get("last_fetch", 0),
            "ttl": _cache.get("ttl", 60),
            "kb_languages": list(_cache.get("kb_content", {}).keys()),
            "prompts_languages": list(_cache.get("prompts_content", {}).keys()),
        },
        "timestamp": datetime.utcnow().isoformat()
    }
