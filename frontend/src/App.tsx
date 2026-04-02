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

  const handleConnect = async () => {
    setIsConnecting(true);
    setError(null);

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

  return (
    <LiveKitRoom
      token={connection.token}
      serverUrl={connection.url}
      connect={true}
      audio={true}
      video={false}
      onDisconnected={handleDisconnect}
      onError={handleError}
      className="livekit-room"
    >
      <VoiceAgent onDisconnect={handleDisconnect} />
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}

export default App;
