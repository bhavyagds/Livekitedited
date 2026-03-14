"""
Meallion Voice AI - FastAPI Application
Main entry point for the web server and API.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

from src.config import settings
from src.api.health import router as health_router
from src.api.admin import router as admin_router
from src.services.shopify import get_shopify_service
from src.services.email import get_email_service
from src.services.database import init_db, get_database_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Manages startup and shutdown events.
    """
    # Startup
    logger.info("Starting Meallion Voice AI...")
    logger.info(f"LiveKit URL: {settings.livekit_url}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"ElevenLabs Voice: {settings.elevenlabs_voice_id}")
    
    # Initialize PostgreSQL database tables
    if settings.postgres_url:
        try:
            await init_db()
            logger.info("PostgreSQL database initialized")
            
            # Clean up orphaned calls on startup
            db = get_database_service()
            cleaned = await db.cleanup_orphaned_calls(max_age_minutes=5)
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} orphaned active calls")
            
            # Sync with LiveKit rooms
            try:
                from src.services.livekit_rooms import get_room_service
                room_service = get_room_service()
                rooms = await room_service.list_rooms()
                active_room_names = [r["name"] for r in rooms]
                synced = await db.sync_calls_with_livekit(active_room_names)
                if synced > 0:
                    logger.info(f"Synced {synced} calls with LiveKit rooms")
            except Exception as e:
                logger.warning(f"Could not sync with LiveKit: {e}")
            
            # Sync SIP providers from database to LiveKit
            try:
                from src.services.livekit_sip import sync_sip_providers_on_startup
                sip_result = await sync_sip_providers_on_startup()
                if sip_result.get("synced", 0) > 0:
                    logger.info(f"Synced {sip_result['synced']} SIP providers to LiveKit")
                if sip_result.get("failed", 0) > 0:
                    logger.warning(f"Failed to sync {sip_result['failed']} SIP providers: {sip_result.get('errors', [])}")
            except Exception as e:
                logger.warning(f"Could not sync SIP providers: {e}")
                
        except Exception as e:
            logger.warning(f"Could not initialize PostgreSQL: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meallion Voice AI...")
    
    # Close service connections
    shopify = get_shopify_service()
    await shopify.close()


# Create FastAPI app
app = FastAPI(
    title="Meallion Voice AI",
    description="Elena - AI Voice Receptionist for Meallion",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(admin_router)


# Token request model
class TokenRequest(BaseModel):
    """Request model for LiveKit token generation."""
    room: str
    identity: str
    name: Optional[str] = None


class TokenResponse(BaseModel):
    """Response model for LiveKit token."""
    token: str
    url: str
    room: str


def generate_livekit_token(
    room_name: str,
    participant_identity: str,
    participant_name: Optional[str] = None,
) -> str:
    """
    Generate a LiveKit access token.
    
    Args:
        room_name: The room to join
        participant_identity: Unique identifier for the participant
        participant_name: Display name for the participant
        
    Returns:
        JWT access token string
    """
    now = datetime.utcnow()
    exp = now + timedelta(hours=2)
    
    claims = {
        "iss": settings.livekit_api_key,
        "sub": participant_identity,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "nbf": int(now.timestamp()),
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
        "metadata": "",
        "name": participant_name or participant_identity,
    }
    
    token = jwt.encode(
        claims,
        settings.livekit_api_secret,
        algorithm="HS256",
    )
    
    return token


@app.post("/api/token", response_model=TokenResponse)
async def create_token(request: TokenRequest):
    """
    Generate a LiveKit access token for web clients.
    
    This endpoint is called by the web frontend to get a token
    for connecting to the LiveKit room.
    """
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured",
        )
    
    token = generate_livekit_token(
        room_name=request.room,
        participant_identity=request.identity,
        participant_name=request.name,
    )
    
    return TokenResponse(
        token=token,
        url=settings.livekit_public_url,
        room=request.room,
    )


@app.get("/api/config")
async def get_client_config():
    """
    Get client configuration.
    Returns non-sensitive configuration for the web client.
    """
    return {
        "livekit_url": settings.livekit_public_url,
        "brand": {
            "name": "Meallion",
            "tagline": "Premium Greek Food Delivery",
        },
    }


# Mount static files
static_dir = Path(__file__).parent / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the web frontend."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        content={"message": "Meallion Voice AI API", "docs": "/docs"},
        status_code=200,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"},
    )


# Agent runner script
def run_agent():
    """
    Run the Elena voice agent.
    This is called separately from the web server.
    """
    from src.agents.elena import run_agent as start_agent
    start_agent()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
