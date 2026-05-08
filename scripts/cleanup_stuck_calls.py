
import asyncio
import sys
import logging
from livekit import api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def cleanup(force=False):
    """
    Cleans up stuck LiveKit rooms.
    If force=True, it will suggest manual Redis commands if API deletion fails.
    """
    url = "http://localhost:7880"  # Adjust if running from outside container
    api_key = "devkey"
    api_secret = "secret"
    
    # Try to connect via API
    logger.info(f"Connecting to LiveKit at {url}...")
    lk_api = api.LiveKitAPI(url, api_key, api_secret)
    
    try:
        # 1. List active rooms
        rooms_res = await lk_api.room.list_rooms(api.ListRoomsRequest())
        active_rooms = rooms_res.rooms
        
        if not active_rooms:
            logger.info("No active rooms found via API.")
        else:
            logger.info(f"Found {len(active_rooms)} active rooms.")
            for room in active_rooms:
                logger.info(f" - {room.name} ({room.sid})")
                
                # Try to delete gracefully
                try:
                    logger.info(f"   Attempting to delete room {room.name}...")
                    await lk_api.room.delete_room(api.DeleteRoomRequest(room=room.name))
                    logger.info(f"   Successfully deleted {room.name}")
                except Exception as e:
                    logger.error(f"   Failed to delete {room.name} via API: {e}")
                    if "could not find object" in str(e).lower() or force:
                        logger.warning(f"   Room {room.name} appears stuck in Redis.")

        # 2. If force or rooms remain, provide the hard cleanup commands
        print("\n" + "="*50)
        print("MANUAL FORCE CLEANUP (If rooms are still stuck):")
        print("="*50)
        print("Run these commands in your terminal to clear Redis and restart LiveKit:")
        print("\n1. Clear LiveKit keys from Redis:")
        print('   docker exec meallion-redis redis-cli FLUSHALL')
        print("\n2. Restart LiveKit and Agent:")
        print('   docker-compose restart livekit agent')
        print("="*50)

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        await lk_api.aclose()

if __name__ == "__main__":
    force_mode = "--force" in sys.argv
    asyncio.run(cleanup(force=force_mode))
