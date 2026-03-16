import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  getActiveSessions,
  terminateSession,
  removeParticipant,
  getCalls,
  getCallTranscript,
  type LiveSession,
} from '@/lib/api'
import {
  Phone,
  Monitor,
  PhoneIncoming,
  Users,
  Clock,
  Trash2,
  RefreshCw,
  Loader2,
  AlertCircle,
  CheckCircle,
  MessageSquare,
  X,
  User,
  Activity,
} from 'lucide-react'

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (mins < 60) return `${mins}m ${secs}s`
  const hours = Math.floor(mins / 60)
  const remainingMins = mins % 60
  return `${hours}h ${remainingMins}m`
}

function formatTimestamp(timestamp: number): string {
  if (!timestamp) return 'Unknown'
  const date = new Date(timestamp * 1000)
  return date.toLocaleTimeString()
}

export default function Sessions() {
  const queryClient = useQueryClient()
  const [successMessage, setSuccessMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [selectedTranscript, setSelectedTranscript] = useState<{
    callId: string
    transcript: string | null
  } | null>(null)

  // Fetch active sessions
  const { data: sessionsData, isLoading: sessionsLoading, refetch: refetchSessions } = useQuery({
    queryKey: ['active-sessions'],
    queryFn: getActiveSessions,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  // Fetch recent calls with transcripts
  const { data: recentCallsData, isLoading: callsLoading } = useQuery({
    queryKey: ['recent-calls-with-transcripts'],
    queryFn: () => getCalls(1, 20),
    refetchInterval: 30000,
  })

  // Terminate session mutation
  const terminateMutation = useMutation({
    mutationFn: terminateSession,
    onSuccess: (result) => {
      if (result.success) {
        showSuccess('Session terminated successfully')
        queryClient.invalidateQueries({ queryKey: ['active-sessions'] })
      } else {
        showError('Failed to terminate session')
      }
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to terminate session')
    },
  })

  // Remove participant mutation
  const removeParticipantMutation = useMutation({
    mutationFn: ({ roomName, identity }: { roomName: string; identity: string }) =>
      removeParticipant(roomName, identity),
    onSuccess: (result) => {
      if (result.success) {
        showSuccess('Participant removed')
        queryClient.invalidateQueries({ queryKey: ['active-sessions'] })
      } else {
        showError('Failed to remove participant')
      }
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to remove participant')
    },
  })

  // Load transcript
  const loadTranscript = async (callId: string) => {
    try {
      const result = await getCallTranscript(callId)
      setSelectedTranscript({
        callId,
        transcript: result.transcript,
      })
    } catch (err) {
      showError('Failed to load transcript')
    }
  }

  const showSuccess = (msg: string) => {
    setSuccessMessage(msg)
    setErrorMessage('')
    setTimeout(() => setSuccessMessage(''), 5000)
  }

  const showError = (msg: string) => {
    setErrorMessage(msg)
    setSuccessMessage('')
    setTimeout(() => setErrorMessage(''), 5000)
  }

  const handleTerminate = (roomName: string) => {
    if (confirm(`Are you sure you want to terminate session "${roomName}"? This will disconnect all participants.`)) {
      terminateMutation.mutate(roomName)
    }
  }

  const handleRemoveParticipant = (roomName: string, identity: string) => {
    if (confirm(`Remove participant "${identity}" from the session?`)) {
      removeParticipantMutation.mutate({ roomName, identity })
    }
  }

  const sessions = sessionsData?.sessions || []
  const recentCalls = recentCallsData?.calls || []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Live Sessions</h1>
          <p className="text-slate-500 mt-1">
            Monitor active calls and view conversation transcripts.
          </p>
        </div>
        <Button variant="outline" onClick={() => refetchSessions()} disabled={sessionsLoading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${sessionsLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Success/Error Messages */}
      {successMessage && (
        <div className="flex items-center gap-2 p-4 text-green-600 bg-green-50 rounded-lg">
          <CheckCircle className="w-5 h-5" />
          {successMessage}
        </div>
      )}

      {errorMessage && (
        <div className="flex items-center gap-2 p-4 text-red-600 bg-red-50 rounded-lg">
          <AlertCircle className="w-5 h-5" />
          {errorMessage}
        </div>
      )}

      {/* Active Sessions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-500" />
            Active Sessions
            {sessions.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-sm rounded-full">
                {sessions.length} live
              </span>
            )}
          </CardTitle>
          <CardDescription>
            Currently active voice sessions. You can monitor and terminate them.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sessionsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <Phone className="w-12 h-12 mx-auto mb-3 text-slate-300" />
              <p className="font-medium">No active sessions</p>
              <p className="text-sm mt-1">Sessions will appear here when calls are in progress.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {sessions.map((session: LiveSession) => (
                <div
                  key={session.room_sid}
                  className="border rounded-lg p-4 bg-white hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <div
                        className={`p-2 rounded-lg ${
                          session.call_type === 'sip' ? 'bg-orange-100' : 'bg-purple-100'
                        }`}
                      >
                        {session.call_type === 'sip' ? (
                          <PhoneIncoming className="w-5 h-5 text-orange-600" />
                        ) : (
                          <Monitor className="w-5 h-5 text-purple-600" />
                        )}
                      </div>
                      <div>
                        <h4 className="font-semibold text-slate-900">{session.room_name}</h4>
                        <div className="flex items-center gap-4 mt-1 text-sm text-slate-500">
                          <span className="flex items-center gap-1">
                            <Users className="w-4 h-4" />
                            {session.num_participants} participants
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            {formatDuration(session.duration_seconds)}
                          </span>
                          <span className="text-xs text-slate-400">
                            Started: {formatTimestamp(session.creation_time)}
                          </span>
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleTerminate(session.room_name)}
                      disabled={terminateMutation.isPending}
                      className="text-red-600 hover:bg-red-50 hover:border-red-300"
                    >
                      <Trash2 className="w-4 h-4 mr-1" />
                      End Call
                    </Button>
                  </div>

                  {/* Participants */}
                  {session.participants.length > 0 && (
                    <div className="mt-4 pt-4 border-t">
                      <p className="text-xs font-medium text-slate-500 mb-2">PARTICIPANTS</p>
                      <div className="flex flex-wrap gap-2">
                        {session.participants.map((p) => (
                          <div
                            key={p.sid}
                            className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 rounded-full text-sm"
                          >
                            <User className="w-3 h-3 text-slate-400" />
                            <span>{p.identity || p.name || 'Unknown'}</span>
                            <button
                              onClick={() => handleRemoveParticipant(session.room_name, p.identity)}
                              className="p-0.5 hover:bg-slate-200 rounded"
                              title="Remove participant"
                            >
                              <X className="w-3 h-3 text-slate-400 hover:text-red-500" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Calls with Transcripts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Recent Calls & Transcripts
          </CardTitle>
          <CardDescription>
            View conversation transcripts from completed calls.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {callsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
            </div>
          ) : recentCalls.length === 0 ? (
            <div className="text-center py-8 text-slate-500">
              No recent calls
            </div>
          ) : (
            <div className="space-y-2">
              {recentCalls.map((call: any) => (
                <div
                  key={call.id}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-slate-50"
                >
                  <div className="flex items-center gap-3">
                    {call.call_type === 'web' ? (
                      <Monitor className="w-4 h-4 text-purple-500" />
                    ) : (
                      <PhoneIncoming className="w-4 h-4 text-orange-500" />
                    )}
                    <div>
                      <p className="font-medium text-sm">
                        {call.caller_number || call.caller_name || call.room_name || 'Unknown'}
                      </p>
                      <p className="text-xs text-slate-500">
                        {new Date(call.started_at).toLocaleString()} • {call.duration_seconds}s •{' '}
                        <span
                          className={
                            call.status === 'completed' ? 'text-green-600' : 'text-red-600'
                          }
                        >
                          {call.status}
                        </span>
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => loadTranscript(call.id)}
                  >
                    <MessageSquare className="w-4 h-4 mr-1" />
                    Transcript
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Transcript Modal */}
      {selectedTranscript && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="font-semibold">Call Transcript</h3>
              <button
                onClick={() => setSelectedTranscript(null)}
                className="p-1 hover:bg-slate-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1">
              {selectedTranscript.transcript ? (
                <div className="space-y-2 font-mono text-sm">
                  {selectedTranscript.transcript.split('\n').map((line, i) => {
                    const isUser = line.startsWith('User:')
                    const isAgent = line.startsWith('Agent:')
                    return (
                      <div
                        key={i}
                        className={`p-2 rounded ${
                          isUser
                            ? 'bg-blue-50 text-blue-900'
                            : isAgent
                            ? 'bg-green-50 text-green-900'
                            : 'bg-slate-50'
                        }`}
                      >
                        {line}
                      </div>
                    )
                  })}
                </div>
              ) : (
                <p className="text-center text-slate-500 py-8">
                  No transcript available for this call.
                </p>
              )}
            </div>
            <div className="p-4 border-t">
              <Button variant="outline" onClick={() => setSelectedTranscript(null)} className="w-full">
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
