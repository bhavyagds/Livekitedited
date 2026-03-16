"""
LiveKit Room Management Service
Handles room listing, participant management, and session control.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveKitRoomService:
    """Service to manage LiveKit rooms and participants."""
    
    def __init__(self):
        self._api = None
        self._initialized = False
    
    async def _get_api(self):
        """Get or create LiveKit API client."""
        if self._api is None:
            try:
                from livekit import api
                from src.config import settings
                
                # Convert ws:// to http:// for API calls
                api_url = settings.livekit_url.replace("ws://", "http://").replace("wss://", "https://")
                
                self._api = api.LiveKitAPI(
                    api_url,
                    settings.livekit_api_key,
                    settings.livekit_api_secret,
                )
                self._initialized = True
                logger.info(f"LiveKit Room API client initialized: {api_url}")
            except Exception as e:
                logger.error(f"Failed to initialize LiveKit API: {e}")
                raise
        return self._api
    
    async def list_rooms(self) -> List[Dict]:
        """List all active rooms."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.ListRoomsRequest()
            response = await lk_api.room.list_rooms(request)
            
            rooms = []
            for room in response.rooms:
                rooms.append({
                    "sid": room.sid,
                    "name": room.name,
                    "num_participants": room.num_participants,
                    "num_publishers": room.num_publishers,
                    "max_participants": room.max_participants,
                    "creation_time": room.creation_time,
                    "turn_password": room.turn_password,
                    "metadata": room.metadata,
                    "active_recording": room.active_recording,
                })
            
            return rooms
        except Exception as e:
            logger.error(f"Failed to list rooms: {e}")
            return []
    
    async def get_room(self, room_name: str) -> Optional[Dict]:
        """Get details of a specific room."""
        rooms = await self.list_rooms()
        for room in rooms:
            if room["name"] == room_name:
                return room
        return None
    
    async def list_participants(self, room_name: str) -> List[Dict]:
        """List all participants in a room."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.ListParticipantsRequest(room=room_name)
            response = await lk_api.room.list_participants(request)
            
            participants = []
            for p in response.participants:
                participants.append({
                    "sid": p.sid,
                    "identity": p.identity,
                    "name": p.name,
                    "state": str(p.state),
                    "joined_at": p.joined_at,
                    "metadata": p.metadata,
                    "is_publisher": p.is_publisher,
                    "permission": {
                        "can_subscribe": p.permission.can_subscribe if p.permission else True,
                        "can_publish": p.permission.can_publish if p.permission else True,
                    } if p.permission else None,
                })
            
            return participants
        except Exception as e:
            logger.error(f"Failed to list participants in {room_name}: {e}")
            return []
    
    async def remove_participant(self, room_name: str, identity: str) -> bool:
        """Remove a participant from a room."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.RoomParticipantIdentity(
                room=room_name,
                identity=identity,
            )
            await lk_api.room.remove_participant(request)
            logger.info(f"Removed participant {identity} from room {room_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove participant {identity} from {room_name}: {e}")
            return False
    
    async def delete_room(self, room_name: str) -> bool:
        """Delete/end a room (terminates all participants)."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.DeleteRoomRequest(room=room_name)
            await lk_api.room.delete_room(request)
            logger.info(f"Deleted room: {room_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete room {room_name}: {e}")
            return False
    
    async def mute_participant(
        self,
        room_name: str,
        identity: str,
        track_sid: str,
        muted: bool = True,
    ) -> bool:
        """Mute or unmute a participant's track."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.MuteRoomTrackRequest(
                room=room_name,
                identity=identity,
                track_sid=track_sid,
                muted=muted,
            )
            await lk_api.room.mute_published_track(request)
            logger.info(f"{'Muted' if muted else 'Unmuted'} track {track_sid} for {identity}")
            return True
        except Exception as e:
            logger.error(f"Failed to mute track: {e}")
            return False
    
    async def get_active_sessions(self) -> List[Dict]:
        """Get all active sessions with details."""
        rooms = await self.list_rooms()
        
        sessions = []
        for room in rooms:
            participants = await self.list_participants(room["name"])
            
            # Determine call type from room name
            is_sip = "sip" in room["name"].lower()
            call_type = "sip" if is_sip else "web"
            
            # Calculate duration
            creation_time = room.get("creation_time", 0)
            if creation_time:
                duration = int(datetime.utcnow().timestamp()) - creation_time
            else:
                duration = 0
            
            sessions.append({
                "room_sid": room["sid"],
                "room_name": room["name"],
                "call_type": call_type,
                "num_participants": room["num_participants"],
                "participants": participants,
                "creation_time": creation_time,
                "duration_seconds": duration,
                "metadata": room.get("metadata"),
                "active_recording": room.get("active_recording", False),
            })
        
        return sessions


# Global instance
_room_service: Optional[LiveKitRoomService] = None


def get_room_service() -> LiveKitRoomService:
    """Get the global room service instance."""
    global _room_service
    if _room_service is None:
        _room_service = LiveKitRoomService()
    return _room_service
