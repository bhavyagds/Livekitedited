import axios from 'axios'
import { useAuthStore } from '@/store/auth'

const API_BASE_URL = '/api/admin'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Track request sources to identify background refetches
let requestId = 0

// Add request interceptor to include auth token
api.interceptors.request.use((config) => {
  // Use the same key as the auth store - check both localStorage and store state
  const storeState = useAuthStore.getState()
  const token = storeState.token || localStorage.getItem('admin_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  } else {
    // Remove Authorization header if no token to avoid sending invalid tokens
    delete config.headers.Authorization
  }
  
  // Mark if this is a background request (when tab is not visible)
  if (document.hidden || document.visibilityState === 'hidden') {
    config.headers['X-Is-Background'] = 'true'
  }
  
  // Add request ID for tracking
  config.headers['X-Request-ID'] = String(++requestId)
  
  return config
})

// Track if we just logged in to prevent immediate logout
let lastLoginTime = 0
const LOGIN_GRACE_PERIOD = 2000 // 2 seconds grace period after login

// Track user activity to detect if logout should happen
let lastUserActivity = Date.now()

// Track user activity (mouse movement, clicks, keyboard, scroll)
// This helps distinguish between user-initiated actions and automatic refetches
if (typeof window !== 'undefined') {
  const updateActivity = () => {
    lastUserActivity = Date.now()
  }
  
  window.addEventListener('mousemove', updateActivity, { passive: true })
  window.addEventListener('click', updateActivity, { passive: true })
  window.addEventListener('keydown', updateActivity, { passive: true })
  window.addEventListener('scroll', updateActivity, { passive: true })
  window.addEventListener('touchstart', updateActivity, { passive: true })
}

// Check if user was recently active (within last 5 seconds)
const isUserActive = () => {
  return Date.now() - lastUserActivity < 5000
}

// Track consecutive 401 errors to avoid logout on transient errors
let consecutive401Errors = 0
let last401Time = 0
const MAX_CONSECUTIVE_401 = 3 // Only logout after 3 consecutive 401s
const RESET_401_WINDOW = 10000 // Reset counter after 10 seconds

// Add response interceptor to handle errors
api.interceptors.response.use(
  (response) => {
    // Reset 401 counter on successful response
    consecutive401Errors = 0
    return response
  },
  (error) => {
    // Only handle 401 errors for authenticated endpoints
    // Don't logout if the error is from the login endpoint itself
    if (error.response?.status === 401 && error.config?.url !== '/login') {
      const currentPath = window.location.pathname
      const timeSinceLogin = Date.now() - lastLoginTime
      const now = Date.now()
      
      // Don't logout if we just logged in (grace period)
      if (timeSinceLogin < LOGIN_GRACE_PERIOD) {
        return Promise.reject(error)
      }
      
      // Reset counter if enough time has passed
      if (now - last401Time > RESET_401_WINDOW) {
        consecutive401Errors = 0
      }
      consecutive401Errors++
      last401Time = now
      
      // Only logout if we're not already on the login page
      if (currentPath !== '/login') {
        const { logout, token } = useAuthStore.getState()
        const storedToken = localStorage.getItem('admin_token')
        
        // Check if this is a background/automatic refetch request
        const isBackgroundRequest = document.hidden || 
                                   document.visibilityState === 'hidden' ||
                                   error.config?.headers?.['X-Is-Background'] === 'true'
        
        // Check if this is a React Query refetch (has refetch metadata)
        const isRefetch = error.config?.headers?.['X-React-Query-Refetch'] === 'true' ||
                         error.config?._retryCount > 0
        
        // Only logout if:
        // 1. We actually had a token
        // 2. This is NOT a background request
        // 3. This is NOT a React Query refetch
        // 4. User was recently active OR we have multiple consecutive 401s
        // 5. We have enough consecutive 401s to indicate real auth failure
        const shouldLogout = (token || storedToken) && 
                             !isBackgroundRequest && 
                             !isRefetch &&
                             (isUserActive() || consecutive401Errors >= MAX_CONSECUTIVE_401) &&
                             consecutive401Errors >= MAX_CONSECUTIVE_401
        
        if (shouldLogout) {
          console.warn('Logging out due to authentication failure', {
            consecutive401Errors,
            isBackgroundRequest,
            isRefetch,
            isUserActive: isUserActive()
          })
          logout()
          // Use replace to avoid adding to history
          window.location.replace('/login')
        }
      }
    }
    return Promise.reject(error)
  }
)

// =============================================================================
// AUTH API
// =============================================================================

export interface LoginRequest {
  email: string
  password: string
}

export interface LoginResponse {
  token: string
  user: {
    id: string
    email: string
    name: string
  }
}

export async function login(email: string, password: string): Promise<LoginResponse> {
  const response = await api.post<LoginResponse>('/login', { email, password })
  // Track login time to prevent immediate logout
  lastLoginTime = Date.now()
  // Token will be stored by the auth store's login function
  return response.data
}

export async function logout(): Promise<void> {
  try {
    await api.post('/logout')
  } finally {
    // Logout is handled by the auth store
  }
}

export async function getCurrentUser() {
  const response = await api.get('/me')
  return response.data
}

// =============================================================================
// KNOWLEDGE BASE API
// =============================================================================

export interface KnowledgeBase {
  source: 'database' | 'file'
  version_id?: string
  version_number?: number
  content: any
  file_name?: string
  updated_at: string
  updated_by?: string
}

export interface KBVersion {
  id: string
  version_number: number
  content: any
  file_name?: string
  file_size?: number
  changed_by?: string
  change_summary?: string
  is_active: boolean
  created_at: string
}

export async function getKnowledgeBase(): Promise<KnowledgeBase> {
  const response = await api.get<KnowledgeBase>('/kb')
  return response.data
}

export async function getKBVersions(limit = 20): Promise<{ versions: KBVersion[] }> {
  const response = await api.get<{ versions: KBVersion[] }>('/kb/versions', {
    params: { limit },
  })
  return response.data
}

export async function uploadKnowledgeBase(
  file: File,
  changeSummary: string
): Promise<{ success: boolean; version_id: string; message: string }> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('change_summary', changeSummary)

  const response = await api.post('/kb/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function rollbackKBVersion(
  versionId: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.post(`/kb/rollback/${versionId}`)
  return response.data
}

export async function downloadKnowledgeBase(): Promise<Blob> {
  const response = await api.get('/kb/download', {
    responseType: 'blob',
  })
  return response.data
}

// =============================================================================
// KB ITEMS API (Individual FAQ entries)
// =============================================================================

export interface KBItem {
  id: string
  category: string
  question: string
  answer: string
  keywords: string[]
  language: string
  is_active: boolean
  display_order: number
  created_at?: string
  updated_at?: string
}

export interface KBItemsResponse {
  items: KBItem[]
  categories: string[]
  total: number
}

export async function getKBItems(
  category?: string,
  language?: string,
  includeInactive = false
): Promise<KBItemsResponse> {
  const response = await api.get<KBItemsResponse>('/kb/items', {
    params: { category, language, include_inactive: includeInactive },
  })
  return response.data
}

export async function getKBItem(itemId: string): Promise<KBItem> {
  const response = await api.get<KBItem>(`/kb/items/${itemId}`)
  return response.data
}

export async function createKBItem(item: {
  category: string
  question: string
  answer: string
  keywords?: string[]
  language?: string
  display_order?: number
}): Promise<{ success: boolean; item: KBItem }> {
  const response = await api.post('/kb/items', item)
  return response.data
}

export async function updateKBItem(
  itemId: string,
  item: {
    category?: string
    question?: string
    answer?: string
    keywords?: string[]
    language?: string
    is_active?: boolean
    display_order?: number
  }
): Promise<{ success: boolean; message: string }> {
  const response = await api.put(`/kb/items/${itemId}`, item)
  return response.data
}

export async function deleteKBItem(
  itemId: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/kb/items/${itemId}`)
  return response.data
}

export async function importKBFromFile(): Promise<{
  success: boolean
  message: string
  count: number
}> {
  const response = await api.post('/kb/import')
  return response.data
}

// =============================================================================
// LANGUAGES API
// =============================================================================

export interface Language {
  id: string
  code: string
  name: string
  native_name: string
  flag_emoji?: string
  is_default: boolean
  is_active: boolean
}

export async function getLanguages(includeInactive = false): Promise<{ languages: Language[] }> {
  const response = await api.get('/languages', {
    params: { include_inactive: includeInactive },
  })
  return response.data
}

export async function createLanguage(lang: {
  code: string
  name: string
  native_name: string
  flag_emoji?: string
  is_default?: boolean
}): Promise<{ success: boolean; language: Language }> {
  const response = await api.post('/languages', lang)
  return response.data
}

export async function updateLanguage(
  languageId: string,
  lang: {
    code?: string
    name?: string
    native_name?: string
    flag_emoji?: string
    is_default?: boolean
    is_active?: boolean
  }
): Promise<{ success: boolean; message: string }> {
  const response = await api.put(`/languages/${languageId}`, lang)
  return response.data
}

export async function deleteLanguage(
  languageId: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/languages/${languageId}`)
  return response.data
}

// =============================================================================
// PROMPTS API
// =============================================================================

export interface PromptVersion {
  id: string
  version_number: number
  language: string
  prompt_type: string
  content: string
  changed_by?: string
  change_summary?: string
  is_active: boolean
  created_at: string
}

export async function getPrompts(): Promise<{
  file_content: string
  db_prompts: PromptVersion[]
  file_path: string
}> {
  const response = await api.get('/prompts')
  return response.data
}

export async function getPromptVersions(
  language?: string,
  promptType?: string,
  limit = 20
): Promise<{ versions: PromptVersion[] }> {
  const response = await api.get<{ versions: PromptVersion[] }>('/prompts/versions', {
    params: { language, prompt_type: promptType, limit },
  })
  return response.data
}

// =============================================================================
// CALLS API
// =============================================================================

export interface Call {
  id: string
  call_sid?: string
  room_name?: string
  caller_number?: string
  caller_name?: string
  call_type: 'inbound' | 'outbound' | 'web'
  status: 'active' | 'completed' | 'failed' | 'missed' | 'busy'
  started_at: string
  ended_at?: string
  duration_seconds?: number
  disconnect_reason?: string
  metadata_json?: any
  transcript?: string
  sentiment_score?: number
  created_at: string
}

export interface CallsResponse {
  calls: Call[]
  total: number
  page: number
  page_size: number
}

export async function getCalls(
  page = 1,
  pageSize = 20,
  status?: string,
  callType?: string
): Promise<CallsResponse> {
  const response = await api.get<CallsResponse>('/calls', {
    params: { page, page_size: pageSize, status, call_type: callType },
  })
  return response.data
}

export async function getCall(callId: string): Promise<Call> {
  const response = await api.get<Call>(`/calls/${callId}`)
  return response.data
}

// =============================================================================
// ANALYTICS API
// =============================================================================

export interface AnalyticsSummary {
  total_calls: number
  successful_calls: number
  failed_calls: number
  missed_calls: number
  web_calls: number
  sip_calls: number
  avg_duration_seconds: number
  total_duration_seconds: number
  success_rate?: number
}

export interface TodayStats {
  calls: number
  successful: number
  failed: number
}

export interface AnalyticsResponse {
  summary: AnalyticsSummary
  today: TodayStats
}

export async function getAnalytics(days = 30): Promise<AnalyticsResponse> {
  const response = await api.get<AnalyticsResponse>('/analytics', {
    params: { days },
  })
  return response.data
}

// =============================================================================
// SIP CONFIG API
// =============================================================================

export interface SIPConfig {
  content: string
  file_path: string
  versions: SIPConfigVersion[]
}

export interface SIPConfigVersion {
  id: string
  version_number: number
  content: string
  changed_by?: string
  change_summary?: string
  is_active: boolean
  created_at: string
}

export async function getSIPConfig(): Promise<SIPConfig> {
  const response = await api.get<SIPConfig>('/sip-config')
  return response.data
}

export async function updateSIPConfig(
  content: string,
  changeSummary: string
): Promise<{ success: boolean; message: string; version_id?: string }> {
  const response = await api.put('/sip-config', {
    content,
    change_summary: changeSummary,
  })
  return response.data
}

export async function getSIPConfigVersions(
  limit = 20
): Promise<{ versions: SIPConfigVersion[] }> {
  const response = await api.get<{ versions: SIPConfigVersion[] }>('/sip-config/versions', {
    params: { limit },
  })
  return response.data
}

// =============================================================================
// LIVEKIT SIP API (Hot-reload without restart)
// =============================================================================

export interface SIPTrunk {
  id: string
  name: string
  numbers: string[]
  allowed_addresses: string[]
  metadata?: string
}

export interface SIPRule {
  id: string
  name: string
  trunk_ids: string[]
  metadata?: string
}

export interface SIPStatus {
  status: string
  trunks_count: number
  trunks: SIPTrunk[]
  rules_count: number
  rules: SIPRule[]
  error?: string
}

export async function getSIPStatus(): Promise<SIPStatus> {
  const response = await api.get<SIPStatus>('/sip/status')
  return response.data
}

export async function getSIPTrunks(): Promise<{ trunks: SIPTrunk[] }> {
  const response = await api.get<{ trunks: SIPTrunk[] }>('/sip/trunks')
  return response.data
}

export async function getSIPRules(): Promise<{ rules: SIPRule[] }> {
  const response = await api.get<{ rules: SIPRule[] }>('/sip/rules')
  return response.data
}

export async function configureSIPProvider(config: {
  provider_name: string
  server: string
  username: string
  password: string
  phone_numbers: string[]
  allowed_ips?: string[]
}): Promise<{
  success: boolean
  trunk_id?: string
  rule_id?: string
  provider_id?: string
  message?: string
  error?: string
}> {
  const response = await api.post('/sip/provider', config)
  return response.data
}

// Saved SIP Providers (persistent in database)
export interface SavedSIPProvider {
  id: string
  name: string
  server: string
  username: string
  phone_numbers: string[]
  allowed_ips: string[]
  is_active: boolean
  livekit_trunk_id?: string
  livekit_rule_id?: string
  sync_status: 'pending' | 'synced' | 'failed' | 'deleted'
  sync_error?: string
  last_sync_at?: string
  created_at: string
}

export async function getSavedSIPProviders(): Promise<{ providers: SavedSIPProvider[] }> {
  const response = await api.get<{ providers: SavedSIPProvider[] }>('/sip/providers')
  return response.data
}

export async function deleteSavedSIPProvider(
  providerId: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/sip/providers/${providerId}`)
  return response.data
}

export async function syncSIPProviders(): Promise<{
  synced: number
  failed: number
  errors: string[]
}> {
  const response = await api.post('/sip/sync')
  return response.data
}

export async function deleteSIPTrunk(
  trunkId: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/sip/trunks/${trunkId}`)
  return response.data
}

export async function deleteSIPRule(
  ruleId: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/sip/rules/${ruleId}`)
  return response.data
}

export interface SIPValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
  validated_numbers?: string[]
  validated_server?: string
}

export async function validateSIPConfig(config: {
  provider_name: string
  server: string
  username?: string
  password?: string
  phone_numbers?: string[]
  allowed_ips?: string[]
}): Promise<SIPValidationResult> {
  const response = await api.post<SIPValidationResult>('/sip/validate', config)
  return response.data
}

export interface SIPConnectionTest {
  connected: boolean
  message: string
}

export async function testSIPConnection(): Promise<SIPConnectionTest> {
  const response = await api.get<SIPConnectionTest>('/sip/test-connection')
  return response.data
}

export interface SIPHealthCheck {
  healthy: boolean
  connection: SIPConnectionTest
  status: string
  trunks_count: number
  rules_count: number
  issues: Array<{
    severity: 'error' | 'warning'
    message: string
  }>
}

export async function getSIPHealth(): Promise<SIPHealthCheck> {
  const response = await api.get<SIPHealthCheck>('/sip/health')
  return response.data
}

// SIP Events and Analytics
export interface SIPEvent {
  id: string
  event_type: string
  trunk_id?: string
  trunk_name?: string
  call_id?: string
  room_name?: string
  from_uri?: string
  to_uri?: string
  caller_number?: string
  status_code?: number
  status_message?: string
  duration_seconds?: number
  error_message?: string
  metadata?: Record<string, any>
  source_ip?: string
  created_at: string
}

export interface SIPTrunkStatusData {
  id: string
  trunk_id: string
  trunk_name: string
  provider_name?: string
  status: string
  last_call_at?: string
  total_calls: number
  successful_calls: number
  failed_calls: number
  success_rate: number
  avg_duration_seconds: number
  last_error?: string
  last_error_at?: string
  updated_at: string
}

export interface SIPEventStats {
  total_events: number
  by_type: Record<string, number>
  by_trunk: Array<{ trunk_id: string; trunk_name: string; count: number }>
  avg_call_duration: number
  period: { from: string; to: string }
}

export interface SIPAnalytics {
  daily: Array<{ date: string; total: number; connected: number; failed: number }>
  hourly_today: Record<number, number>
  top_callers: Array<{ number: string; count: number }>
  period_days: number
}

export async function getSIPEvents(params?: {
  event_type?: string
  trunk_id?: string
  caller_number?: string
  limit?: number
  offset?: number
}): Promise<{ events: SIPEvent[]; count: number }> {
  const response = await api.get('/sip/events', { params })
  return response.data
}

export async function getSIPEventStats(hours: number = 24): Promise<SIPEventStats> {
  const response = await api.get<SIPEventStats>('/sip/events/stats', {
    params: { hours },
  })
  return response.data
}

export async function getSIPTrunkStatuses(): Promise<{ statuses: SIPTrunkStatusData[] }> {
  const response = await api.get<{ statuses: SIPTrunkStatusData[] }>('/sip/trunk-statuses')
  return response.data
}

export async function getSIPAnalytics(days: number = 7): Promise<SIPAnalytics> {
  const response = await api.get<SIPAnalytics>('/sip/analytics', {
    params: { days },
  })
  return response.data
}

// =============================================================================
// LIVE SESSIONS API
// =============================================================================

export interface SessionParticipant {
  sid: string
  identity: string
  name: string
  state: string
  joined_at: number
  metadata?: string
  is_publisher: boolean
}

export interface LiveSession {
  room_sid: string
  room_name: string
  call_type: 'web' | 'sip'
  num_participants: number
  participants: SessionParticipant[]
  creation_time: number
  duration_seconds: number
  metadata?: string
  active_recording: boolean
}

export async function getActiveSessions(): Promise<{ sessions: LiveSession[]; count: number }> {
  const response = await api.get<{ sessions: LiveSession[]; count: number }>('/sessions')
  return response.data
}

export async function getSessionDetails(roomName: string): Promise<{
  room: Record<string, any>
  participants: SessionParticipant[]
}> {
  const response = await api.get(`/sessions/${encodeURIComponent(roomName)}`)
  return response.data
}

export async function terminateSession(roomName: string): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/sessions/${encodeURIComponent(roomName)}`)
  return response.data
}

export async function removeParticipant(
  roomName: string,
  identity: string
): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(
    `/sessions/${encodeURIComponent(roomName)}/participants/${encodeURIComponent(identity)}`
  )
  return response.data
}

export async function getCallTranscript(callId: string): Promise<{
  call_id: string
  transcript: string | null
  message?: string
}> {
  const response = await api.get(`/calls/${callId}/transcript`)
  return response.data
}

// =============================================================================
// SYSTEM SETTINGS API
// =============================================================================

export async function getSettings(): Promise<{ settings: Record<string, any> }> {
  const response = await api.get<{ settings: Record<string, any> }>('/settings')
  return response.data
}

export async function getSetting(key: string): Promise<{ key: string; value: any }> {
  const response = await api.get<{ key: string; value: any }>(`/settings/${key}`)
  return response.data
}

export async function updateSetting(
  key: string,
  value: any,
  description?: string
): Promise<{ success: boolean; key: string; value: any }> {
  const response = await api.put(`/settings/${key}`, { value, description })
  return response.data
}

export async function initDefaultSettings(): Promise<{
  success: boolean
  settings: Record<string, any>
}> {
  const response = await api.post('/settings/init')
  return response.data
}

// =============================================================================
// LOGS API
// =============================================================================

export interface AuditLog {
  id: string
  user_id?: string
  user_email?: string
  action: string
  resource_type?: string
  resource_id?: string
  old_value?: any
  new_value?: any
  ip_address?: string
  user_agent?: string
  created_at: string
}

export interface ErrorLog {
  id: string
  service?: string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
  message: string
  context?: any
  stack_trace?: string
  created_at: string
}

export async function getAuditLogs(
  limit = 100,
  offset = 0
): Promise<{ logs: AuditLog[] }> {
  const response = await api.get<{ logs: AuditLog[] }>('/audit-logs', {
    params: { limit, offset },
  })
  return response.data
}

export async function getErrorLogs(
  service?: string,
  level?: string,
  limit = 100
): Promise<{ logs: ErrorLog[] }> {
  const response = await api.get<{ logs: ErrorLog[] }>('/error-logs', {
    params: { service, level, limit },
  })
  return response.data
}

// =============================================================================
// HEALTH API
// =============================================================================

export interface HealthStatus {
  status: string
  services: {
    livekit: { configured: boolean; url?: string }
    openai: { configured: boolean; model?: string }
    elevenlabs: { configured: boolean; voice_id?: string }
    supabase?: { configured: boolean; url?: string }
    shopify?: { configured: boolean }
  }
  agent_language: string
  timestamp: string
}

export async function getHealth(): Promise<HealthStatus> {
  const response = await api.get<HealthStatus>('/health')
  return response.data
}
