"""
Meallion Voice AI - Background Audio Service
Plays ambient audio during calls via LiveKit audio track.

Uses a dedicated thread for playback to ensure smooth audio
even when the main event loop is busy with agent processing.
"""

import logging
import asyncio
import httpx
import io
import threading
import time
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Try to import audio processing libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available - background audio disabled")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logger.warning("pydub not available - background audio disabled")


@dataclass
class BackgroundAudioConfig:
    """Configuration for background audio."""
    enabled: bool = False
    url: str = ""
    volume: float = 0.1  # 0.0 to 1.0


class BackgroundAudioPlayer:
    """
    Plays background audio during LiveKit calls.
    
    Uses a dedicated thread for playback to ensure smooth, uninterrupted
    audio even when the main event loop is busy with agent processing.
    
    Usage:
        player = BackgroundAudioPlayer(config)
        await player.start(room)
        # ... call in progress ...
        await player.stop()
    """
    
    SAMPLE_RATE = 48000  # LiveKit standard
    CHANNELS = 1  # Mono
    FRAME_DURATION_MS = 20  # 20ms frames
    SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    
    def __init__(self, config: BackgroundAudioConfig):
        self.config = config
        self._audio_data: Optional[np.ndarray] = None
        self._playing = False
        self._stop_event = threading.Event()  # Thread-safe stop signal
        self._thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bg_audio")
        self._audio_source = None
        self._track = None
        self._loop = None  # Reference to main event loop for frame capture
    
    async def load_audio(self) -> bool:
        """Load and prepare audio from URL with local file caching."""
        if not HAS_NUMPY or not HAS_PYDUB:
            logger.warning("Background audio requires numpy and pydub")
            return False
        
        if not self.config.url:
            logger.warning("No background audio URL configured")
            return False
        
        try:
            import hashlib
            import os
            
            # Create a cache directory
            cache_dir = "/tmp/bg_audio_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create cache filename from URL hash
            url_hash = hashlib.md5(self.config.url.encode()).hexdigest()[:16]
            cache_file = os.path.join(cache_dir, f"bg_audio_{url_hash}.raw")
            
            audio_bytes = None
            
            # Try to load from cache first
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        audio_bytes = f.read()
                    logger.info(f"Background audio loaded from cache ({len(audio_bytes)} bytes)")
                except Exception as e:
                    logger.debug(f"Cache read failed: {e}")
                    audio_bytes = None
            
            # Download if not cached
            if audio_bytes is None:
                logger.info(f"Loading background audio from: {self.config.url[:50]}...")
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.config.url, timeout=30.0)
                    response.raise_for_status()
                    audio_bytes = response.content
                
                logger.info(f"Downloaded {len(audio_bytes)} bytes")
                
                # Save to cache for next time
                try:
                    with open(cache_file, 'wb') as f:
                        f.write(audio_bytes)
                    logger.debug(f"Saved to cache: {cache_file}")
                except Exception as e:
                    logger.debug(f"Cache write failed: {e}")
            
            # Load with pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Convert to mono, 48kHz, 16-bit
            audio = audio.set_channels(self.CHANNELS)
            audio = audio.set_frame_rate(self.SAMPLE_RATE)
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Apply volume
            if self.config.volume < 1.0:
                # Convert volume (0-1) to dB reduction
                db_reduction = 20 * np.log10(max(self.config.volume, 0.001))
                audio = audio + db_reduction
            
            # Convert to numpy array (normalized float32)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # Normalize to -1.0 to 1.0
            
            self._audio_data = samples
            logger.info(f"Background audio loaded: {len(samples) / self.SAMPLE_RATE:.1f}s duration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load background audio: {e}")
            return False
    
    async def start(self, room) -> bool:
        """Start playing background audio in the room."""
        if not self.config.enabled:
            logger.debug("Background audio disabled")
            return False
        
        if self._playing:
            logger.warning("Background audio already playing")
            return False
        
        # Load audio if not loaded
        if self._audio_data is None:
            if not await self.load_audio():
                return False
        
        try:
            from livekit import rtc
            
            # Store reference to the event loop for thread-safe frame capture
            self._loop = asyncio.get_running_loop()
            
            # Create audio source - this generates audio independently
            self._audio_source = rtc.AudioSource(self.SAMPLE_RATE, self.CHANNELS)
            
            # Create local audio track with unique name
            self._track = rtc.LocalAudioTrack.create_audio_track(
                "ambient_music",
                self._audio_source,
            )
            
            # Publish as a separate track that won't interfere with voice
            # Using SOURCE_UNKNOWN to keep it separate from microphone/voice
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_UNKNOWN,
                # Don't let LiveKit do any processing on this track
                dtx=False,  # Disable discontinuous transmission
                red=False,  # Disable redundant encoding
            )
            
            publication = await room.local_participant.publish_track(self._track, options)
            logger.info(f"Background audio track published: {publication.sid}")
            
            # Start playback in a dedicated thread for uninterrupted playback
            self._playing = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._playback_loop_threaded,
                name="bg_audio_playback",
                daemon=True,
            )
            self._thread.start()
            logger.info("🎵 Background audio thread started")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start background audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _playback_loop_threaded(self):
        """Main playback loop running in a dedicated thread.
        
        Uses threading instead of asyncio for uninterrupted playback.
        The main event loop can be busy with agent processing, but this
        thread keeps running independently with precise timing.
        """
        if self._audio_data is None or self._audio_source is None:
            return
        
        from livekit import rtc
        
        frame_size = self.SAMPLES_PER_FRAME
        total_samples = len(self._audio_data)
        position = 0
        
        logger.info("🎵 Background audio playback started (threaded mode)")
        
        # Pre-compute frame timing for precise playback
        frame_duration = self.FRAME_DURATION_MS / 1000.0
        
        # Pre-allocate frame to avoid GC pauses
        frame = rtc.AudioFrame.create(
            self.SAMPLE_RATE,
            self.CHANNELS,
            frame_size,
        )
        frame_array = np.frombuffer(frame.data, dtype=np.int16)
        
        # Use monotonic time for consistent timing
        next_frame_time = time.monotonic()
        frames_sent = 0
        
        try:
            while not self._stop_event.is_set() and self._playing:
                # Get frame data (with looping)
                end_pos = position + frame_size
                
                if end_pos <= total_samples:
                    frame_data = self._audio_data[position:end_pos]
                else:
                    # Loop: wrap around to beginning
                    remaining = total_samples - position
                    frame_data = np.concatenate([
                        self._audio_data[position:],
                        self._audio_data[:frame_size - remaining]
                    ])
                    end_pos = frame_size - remaining
                
                position = end_pos % total_samples
                
                # Copy data to pre-allocated frame (faster than creating new)
                np.copyto(frame_array, (frame_data * 32767).astype(np.int16))
                
                # Send to LiveKit - thread-safe submission to event loop
                try:
                    if self._loop and self._loop.is_running():
                        # Submit coroutine to the main event loop from this thread
                        future = asyncio.run_coroutine_threadsafe(
                            self._audio_source.capture_frame(frame),
                            self._loop
                        )
                        # Wait briefly for completion (don't block too long)
                        try:
                            future.result(timeout=0.05)  # 50ms timeout
                            frames_sent += 1
                        except TimeoutError:
                            # Frame might still be queued, continue
                            pass
                        except Exception as e:
                            if frames_sent % 500 == 0:
                                logger.debug(f"Frame capture issue: {e}")
                except Exception as e:
                    if frames_sent % 500 == 0:
                        logger.debug(f"Frame submission issue: {e}")
                
                # Precise timing using monotonic clock and thread sleep
                next_frame_time += frame_duration
                current_time = time.monotonic()
                sleep_time = next_frame_time - current_time
                
                if sleep_time > 0:
                    # Sleep precisely (thread sleep, not affected by event loop)
                    time.sleep(sleep_time)
                elif sleep_time < -0.1:
                    # We're way behind (>100ms), reset timing
                    next_frame_time = time.monotonic()
                    if frames_sent % 100 == 0:
                        logger.debug(f"🎵 Background audio timing reset (was {-sleep_time:.3f}s behind)")
                # else: slightly behind, just continue without sleep
                
        except Exception as e:
            logger.error(f"🎵 Background audio playback error: {e}")
        finally:
            logger.info(f"🎵 Background audio stopped (sent {frames_sent} frames)")
    
    async def stop(self):
        """Stop playing background audio."""
        self._playing = False
        self._stop_event.set()  # Signal thread to stop
        
        # Wait for thread to finish (with timeout)
        if self._thread and self._thread.is_alive():
            # Run the join in a thread to avoid blocking the event loop
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._thread.join(timeout=1.0)
            )
            if self._thread.is_alive():
                logger.warning("Background audio thread did not stop cleanly")
            self._thread = None
        
        if self._track:
            # Track will be automatically unpublished when room disconnects
            self._track = None
        
        self._audio_source = None
        self._loop = None
        logger.info("Background audio stopped")
    
    def set_volume(self, volume: float):
        """Update volume (0.0 to 1.0). Requires reload to take effect."""
        self.config.volume = max(0.0, min(1.0, volume))


def get_background_audio_config() -> BackgroundAudioConfig:
    """Get background audio config from database settings."""
    try:
        from src.agents.prompts import get_agent_setting

        enabled = get_agent_setting("bg_audio_enabled")
        url = get_agent_setting("bg_audio_url")
        volume = get_agent_setting("bg_audio_volume")

        if enabled is None or url is None or volume is None:
            logger.warning("Background audio settings missing; background audio disabled")
            return BackgroundAudioConfig()

        return BackgroundAudioConfig(
            enabled=bool(enabled),
            url=str(url),
            volume=float(volume),
        )
    except Exception as e:
        logger.warning(f"Could not load background audio config: {e}")
        return BackgroundAudioConfig()


async def create_background_audio_player() -> Optional[BackgroundAudioPlayer]:
    """Create and initialize a background audio player."""
    config = get_background_audio_config()
    
    if not config.enabled:
        logger.debug("Background audio is disabled")
        return None
    
    if not config.url:
        logger.debug("No background audio URL configured")
        return None
    
    player = BackgroundAudioPlayer(config)
    return player
