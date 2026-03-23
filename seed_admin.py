import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# If running outside docker, replace "postgres" hostname with "localhost"
db_url = os.getenv('POSTGRES_URL', '')
if "@postgres:5432" in db_url:
    os.environ['POSTGRES_URL'] = db_url.replace("@postgres:5432", "@localhost:5433")

from src.config import settings
from src.services.database import get_database_service, init_db
from src.api.admin import hash_password

async def seed_admin():
    print("Initializing database tables...")
    await init_db()
    
    admin_email = settings.admin_email
    admin_pass = settings.admin_password
    
    print(f"Checking for admin user: {admin_email}")
    db = get_database_service()
    existing = await db.get_admin_by_email(admin_email)
    
    if not existing:
        print("Creating admin user...")
        hashed = hash_password(admin_pass)
        await db.create_admin_user(admin_email, hashed, "Admin")
        print(f"Created admin user: {admin_email} / {admin_pass}")
    else:
        print("Admin user already exists!")
        
    print("Database seeding complete. You can now log into the admin dashboard at http://localhost:3001")

if __name__ == "__main__":
    asyncio.run(seed_admin())
