import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  useRoomContext,
  useTrackToggle,
} from '@livekit/components-react';
import { Track, RoomEvent } from 'livekit-client';
import {
  PhoneOff,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { Logo } from './Logo';
import { BackgroundPattern } from './BackgroundPattern';
import { AudioVisualizer } from './AudioVisualizer';
import { formatMessage } from '../utils/formatMessage';

interface VoiceAgentProps {
  onDisconnect: () => void;
}

type AgentState = 'idle' | 'listening' | 'thinking' | 'speaking';

interface InfoItem {
  type: string;
  icon: string;
  label: string;
  value: string;
}

interface TimelineEntry {
  id: string;
  kind: 'user' | 'agent' | 'info';
  text?: string;
  items?: InfoItem[];
  timestamp: Date;
}

const AGENT_AVATAR = '\u{1F469}\u200D\u{1F4BC}';
const USER_AVATAR = '\u{1F464}';

export function VoiceAgent({ onDisconnect }: VoiceAgentProps) {
  const room = useRoomContext();
  const [agentState, setAgentState] = useState<AgentState>('idle');
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeakerMuted, setIsSpeakerMuted] = useState(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Track toggle for microphone
  const { toggle: toggleMic, enabled: micEnabled } = useTrackToggle({
    source: Track.Source.Microphone,
  });

  // Monitor room disconnection and participant leaving
  useEffect(() => {
    const handleDisconnected = () => {
      console.log('Room disconnected');
      onDisconnect();
    };

    const handleParticipantDisconnected = (participant: any) => {
      // Check if the agent (Elena) left
      if (participant.identity?.startsWith('agent-')) {
        console.log('Agent disconnected, ending session');
        onDisconnect();
      }
    };

    const handleRoomStateChanged = (state: string) => {
      console.log('Room state changed:', state);
      if (state === 'disconnected') {
        onDisconnect();
      }
    };

    room.on(RoomEvent.Disconnected, handleDisconnected);
    room.on(RoomEvent.ParticipantDisconnected, handleParticipantDisconnected);
    room.on(RoomEvent.ConnectionStateChanged, handleRoomStateChanged);
    
    return () => {
      room.off(RoomEvent.Disconnected, handleDisconnected);
      room.off(RoomEvent.ParticipantDisconnected, handleParticipantDisconnected);
      room.off(RoomEvent.ConnectionStateChanged, handleRoomStateChanged);
    };
  }, [room, onDisconnect]);


  // Auto-scroll to latest message when timeline changes
  useEffect(() => {
    // Small delay to ensure DOM is updated
    const timer = setTimeout(() => {
      if (transcriptEndRef.current) {
        transcriptEndRef.current.scrollTo({
          top: transcriptEndRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [timeline, agentState]);

  // Handle data messages from agent
  useEffect(() => {
    const handleData = (payload: Uint8Array) => {
      try {
        const message = JSON.parse(new TextDecoder().decode(payload));
        console.log('ðŸ“¨ Received message:', message);
        
        // Handle chat transcripts (user + agent)
        if (message.type === 'transcript' && (message.speaker === 'user' || message.speaker === 'agent')) {
          const entry: TimelineEntry = {
            id: `${message.speaker}-${Date.now()}-${Math.random()}`,
            kind: message.speaker,
            text: message.text,
            timestamp: new Date(),
          };
          setTimeline(prev => [...prev.slice(-19), entry]);
        }
        
        // Handle agent info cards (key information only)
        if (message.type === 'info' && message.items && message.items.length > 0) {
          console.log('ðŸ“Š Info items received:', JSON.stringify(message.items, null, 2));
          const items = message.items.map((item: InfoItem) => ({
            type: item.type || 'unknown',
            icon: item.icon || '-',
            label: item.label || 'Info',
            value: item.value || ''
          }));
          console.log('ðŸ“Š Processed items:', items);
          const entry: TimelineEntry = {
            id: `info-${Date.now()}-${Math.random()}`,
            kind: 'info',
            items: items,
            timestamp: new Date(),
          };
          setTimeline(prev => [...prev.slice(-19), entry]);
        }
        
        if (message.type === 'state') {
          setAgentState(message.state as AgentState);
        }
      } catch (e) {
        console.error('Error parsing data message:', e);
      }
    };

    room.on(RoomEvent.DataReceived, handleData);
    return () => {
      room.off(RoomEvent.DataReceived, handleData);
    };
  }, [room]);

  // Initial state - start as idle, backend will send greeting
  useEffect(() => {
    setAgentState('idle');
  }, []);

  const handleMuteToggle = useCallback(() => {
    toggleMic();
    setIsMuted(!micEnabled);
  }, [toggleMic, micEnabled]);

  const handleSpeakerToggle = useCallback(() => {
    setIsSpeakerMuted(prev => !prev);
  }, []);

  // Keep all current and newly mounted audio elements in sync with speaker mute state.
  useEffect(() => {
    const syncAudioMuteState = () => {
      document.querySelectorAll('audio').forEach(audioEl => {
        audioEl.muted = isSpeakerMuted;
      });
    };

    syncAudioMuteState();

    const observer = new MutationObserver(() => {
      syncAudioMuteState();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    return () => observer.disconnect();
  }, [isSpeakerMuted]);

  const getStatusText = () => {
    switch (agentState) {
      case 'listening':
        return 'Elena is listening...';
      case 'thinking':
        return 'Elena is thinking...';
      case 'speaking':
        return 'Elena is speaking...';
      default:
        return 'Connected';
    }
  };

  return (
    <div className="voice-agent">
      <BackgroundPattern />

      {/* Header with logo */}
      <header className="call-header">
        <Logo size="small" />
      </header>

      {/* Main call layout - side by side on desktop */}
      <div className="call-layout">
        {/* Left Panel - Agent Avatar & Controls */}
        <motion.div
          className="agent-panel"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="agent-panel-content">
            <div className={`avatar-container ${agentState}`}>
              <div className="avatar-ring ring-1" />
              <div className="avatar-ring ring-2" />
              <div className="avatar-ring ring-3" />
              <div className="avatar">
                <span className="avatar-icon">{AGENT_AVATAR}</span>
              </div>
              <div className={`status-indicator ${agentState}`} />
            </div>

            <div className="agent-info">
              <h2 className="agent-name">Elena</h2>
              <motion.p
                className="status-text"
                key={agentState}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
              >
                {getStatusText()}
              </motion.p>
            </div>

            <AudioVisualizer isActive={agentState === 'speaking' || agentState === 'listening'} />

            {/* Controls */}
            <div className="controls-section">
              <button
                className={`control-button ${isMuted ? 'muted' : ''}`}
                onClick={handleMuteToggle}
                title={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
              </button>

              <button
                className="control-button disconnect"
                onClick={onDisconnect}
                title="End Call"
              >
                <PhoneOff size={28} />
              </button>

              <button
                className={`control-button ${isSpeakerMuted ? 'muted' : ''}`}
                onClick={handleSpeakerToggle}
                title={isSpeakerMuted ? 'Unmute Speaker' : 'Mute Speaker'}
              >
                {isSpeakerMuted ? <VolumeX size={24} /> : <Volume2 size={24} />}
              </button>
            </div>
          </div>
        </motion.div>

        {/* Right Panel - Chat / Transcript */}
        <motion.div
          className="chat-panel"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          <div className="chat-header">
            <h3>Conversation</h3>
            <span className="message-count">{timeline.length} messages</span>
          </div>
          
          <div className="chat-container" ref={transcriptEndRef}>
            <AnimatePresence mode="popLayout">
              {timeline.length === 0 && agentState !== 'speaking' ? (
                <motion.div 
                  className="chat-empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <span className="chat-empty-icon">...</span>
                  <p>Start speaking to begin the conversation</p>
                </motion.div>
              ) : timeline.length === 0 && agentState === 'speaking' ? (
                <motion.div 
                  className="chat-empty speaking"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <span className="chat-empty-icon">...</span>
                  <p>Elena is speaking...</p>
                </motion.div>
              ) : (
                [...timeline].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()).map((entry) => 
                  entry.kind === 'user' ? (
                    <motion.div
                      key={entry.id}
                      className="chat-message user"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="message-avatar">{USER_AVATAR}</div>
                      <div className="message-content">
                        <span className="message-speaker">You</span>
                        <p className="message-text">{entry.text ? formatMessage(entry.text) : ''}</p>
                      </div>
                    </motion.div>
                  ) : entry.kind === 'agent' ? (
                    <motion.div
                      key={entry.id}
                      className="chat-message elena"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="message-avatar">{AGENT_AVATAR}</div>
                      <div className="message-content">
                        <span className="message-speaker">Elena</span>
                        <p className="message-text">{entry.text ? formatMessage(entry.text) : ''}</p>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key={entry.id}
                      className="info-card"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="info-card-header">
                        <span className="info-card-icon">*</span>
                        <span className="info-card-title">Elena found</span>
                      </div>
                      <div className="info-card-items">
                        {entry.items && entry.items.length > 0 ? (
                          entry.items.map((item, idx) => (
                            <div
                              key={`${entry.id}-item-${idx}`}
                              className={`info-item info-${item.type}`}
                            >
                              <span className="info-item-icon">{item.icon}</span>
                              <span className="info-item-label">{item.label}</span>
                              <span className="info-item-value">{item.value}</span>
                            </div>
                          ))
                        ) : (
                          <div className="info-item">
                            <span className="info-item-value">Processing...</span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )
                )
              )}
              
              {/* Thinking indicator */}
              {agentState === 'thinking' && (
                <motion.div
                  className="chat-message elena typing"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="message-avatar">{AGENT_AVATAR}</div>
                  <div className="message-content">
                    <span className="message-speaker">Elena</span>
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </div>
  );
}


