import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# If running outside docker, replace "postgres" hostname with "localhost"
from src.models.admin import SIPProvider
from src.services.database import get_database_service
from src.config import settings
from src.services.livekit_sip import sync_sip_providers_on_startup

def _parse_allowed_ips(raw: str) -> list[str]:
    if not raw:
        return []
    return [ip.strip() for ip in raw.split(",") if ip.strip()]

async def main():
    db = get_database_service()
    
    provider_name = "Yuboto"
    server = settings.yuboto_sip_server
    username = settings.yuboto_sip_username
    password = settings.yuboto_sip_password
    phone = settings.yuboto_phone_number
    allowed_ips = _parse_allowed_ips(os.getenv("YUBOTO_ALLOWED_IPS", ""))
    if not allowed_ips:
        allowed_ips = ["0.0.0.0/0"]
    
    print(f"Checking for existing {provider_name} provider...")
    
    # Check if exists
    from src.services.database import get_db
    async with get_db() as session:
        from sqlalchemy import select
        result = await session.execute(select(SIPProvider).where(SIPProvider.name == provider_name))
        existing = result.scalars().first()
        
        if existing:
            print(f"Provider {provider_name} already exists. Updating...")
            existing.server = server
            existing.username = username
            existing.password_encrypted = db._encrypt_password(password) if password else ""
            existing.phone_numbers = [phone]
            existing.allowed_ips = allowed_ips
            await session.commit()
        else:
            print(f"Creating new provider {provider_name}...")
            new_provider = SIPProvider(
                name=provider_name,
                server=server,
                username=username,
                password_encrypted=db._encrypt_password(password) if password else "",
                phone_numbers=[phone],
                allowed_ips=allowed_ips,
                is_active=True,
                sync_status="pending"
            )
            session.add(new_provider)
            await session.commit()
            
    print("Running SIP sync...")
    result = await sync_sip_providers_on_startup()
    print(f"Sync result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
