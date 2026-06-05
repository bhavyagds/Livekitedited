import { useState, useCallback, useEffect } from 'react';
import { LiveKitRoom, RoomAudioRenderer } from '@livekit/components-react';
import '@livekit/components-styles';
import { VoiceAgent } from './components/VoiceAgent';
import { LandingPage } from './components/LandingPage';
import './App.css';

interface ConnectionState {
  token: string;
  url: string;
  room: string;
}

function App() {
  const [connection, setConnection] = useState<ConnectionState | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Warm up backend caches on page load to reduce first-call latency.
    fetch('/api/warmup').catch(() => {
      // Best-effort only; ignore warmup failures.
    });
  }, []);

  // Keep AudioContext active on iOS devices during interactions
  useEffect(() => {
    const resumeAudio = async () => {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        // Find existing contexts or test state
        const tempCtx = new AudioContextClass();
        if (tempCtx.state === 'suspended') {
          await tempCtx.resume().catch(() => {});
        }
        tempCtx.close().catch(() => {});
      }
    };

    window.addEventListener('touchend', resumeAudio);
    window.addEventListener('click', resumeAudio);
    return () => {
      window.removeEventListener('touchend', resumeAudio);
      window.removeEventListener('click', resumeAudio);
    };
  }, []);

  const handleConnect = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      // Pre-unlock audio context for iOS/Safari before async fetch
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        const audioCtx = new AudioContextClass();
        if (audioCtx.state === 'suspended') {
          audioCtx.resume().catch(() => {});
        }
      }
      // Request mic permissions synchronously within the click gesture chain
      await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.warn('Microphone pre-acquisition warning:', err);
    }

    try {
      const response = await fetch('/api/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          room: `web-${Date.now()}`,
          identity: `user-${Math.random().toString(36).substring(2, 9)}`,
          name: 'Web User',
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get connection token');
      }

      const data = await response.json();
      setConnection(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed');
    } finally {
      setIsConnecting(false);
    }
  };

  const handleDisconnect = useCallback(() => {
    console.log('Disconnecting from room...');
    setConnection(null);
  }, []);

  const handleError = useCallback((err: Error) => {
    console.error('Room error:', err);
    // On error, also disconnect and return to landing
    setConnection(null);
  }, []);

  if (!connection) {
    return (
      <LandingPage
        onConnect={handleConnect}
        isConnecting={isConnecting}
        error={error}
      />
    );
  }

  // Optimized room configuration for low-latency audio streaming and quick reconnection on mobile networks
  const roomOptions = {
    publishDefaults: {
      audioBitrate: 20000, // 20kbps is ideal for low-latency Opus voice streaming
      dtx: true,          // Discontinuous transmission reduces mobile bandwidth usage
    },
    adaptiveStream: false, // Disables visual stream adjustments since this is voice-only
    dynacast: false,
    reconnectPolicy: {
      nextRetryDelayInMs: (context: any) => {
        // Quick reconnects for mobile network switches
        return Math.min(1000 * Math.pow(1.5, context.retryCount), 6000);
      }
    }
  };

  return (
    <LiveKitRoom
      token={connection.token}
      serverUrl={connection.url}
      connect={true}
      audio={true}
      video={false}
      onDisconnected={handleDisconnect}
      onError={handleError}
      options={roomOptions}
      className="livekit-room"
    >
      <VoiceAgent onDisconnect={handleDisconnect} />
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}

export default App;
