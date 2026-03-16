"""
Meallion Voice AI - Health Check Endpoints
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    services: dict


class ServiceStatus(BaseModel):
    """Individual service status."""
    status: str
    message: str


@router.get("/health")
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns:
        Health status with service checks
    """
    services = {}
    
    # Check configuration
    services["config"] = {
        "status": "healthy",
        "message": "Configuration loaded",
    }
    
    # Check LiveKit config
    if settings.livekit_url and settings.livekit_api_key:
        services["livekit"] = {
            "status": "configured",
            "message": f"URL: {settings.livekit_url}",
        }
    else:
        services["livekit"] = {
            "status": "not_configured",
            "message": "LiveKit credentials missing",
        }
    
    # Check ElevenLabs config
    if settings.elevenlabs_api_key:
        services["elevenlabs"] = {
            "status": "configured",
            "message": f"Voice ID: {settings.elevenlabs_voice_id}",
        }
    else:
        services["elevenlabs"] = {
            "status": "not_configured",
            "message": "ElevenLabs API key missing",
        }
    
    # Check OpenAI config
    if settings.openai_api_key:
        services["openai"] = {
            "status": "configured",
            "message": f"Model: {settings.openai_model}",
        }
    else:
        services["openai"] = {
            "status": "not_configured",
            "message": "OpenAI API key missing",
        }
    
    # Check Shopify config
    if settings.shopify_store_url and settings.shopify_access_token:
        services["shopify"] = {
            "status": "configured",
            "message": f"Store: {settings.shopify_store_url}",
        }
    else:
        services["shopify"] = {
            "status": "not_configured",
            "message": "Shopify credentials missing",
        }
    
    # Check Yuboto SIP config
    if settings.yuboto_sip_username and settings.yuboto_sip_password:
        services["yuboto_sip"] = {
            "status": "configured",
            "message": f"Phone: {settings.yuboto_phone_number}",
        }
    else:
        services["yuboto_sip"] = {
            "status": "not_configured",
            "message": "Yuboto SIP credentials missing",
        }
    
    # Determine overall status
    not_configured = [k for k, v in services.items() if v["status"] == "not_configured"]
    
    if not not_configured:
        overall_status = "healthy"
    elif len(not_configured) <= 2:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services=services,
    )


@router.get("/health/ready")
async def readiness_check() -> dict:
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to accept traffic.
    """
    # Check critical services
    critical_ok = (
        settings.openai_api_key and
        settings.elevenlabs_api_key
    )
    
    if critical_ok:
        return {"ready": True}
    else:
        return {"ready": False, "reason": "Missing critical configuration"}


@router.get("/health/live")
async def liveness_check() -> dict:
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive.
    """
    return {"alive": True}
