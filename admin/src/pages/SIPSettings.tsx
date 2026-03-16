import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Phone,
  Plus,
  Trash2,
  Save,
  CheckCircle,
  AlertCircle,
  Shield,
  Server,
  RefreshCw,
  Loader2,
  ExternalLink,
  Activity,
  Clock,
  PhoneIncoming,
  PhoneOff,
  BarChart3,
  List,
  Database,
  Upload,
} from 'lucide-react'
import {
  getSIPStatus,
  configureSIPProvider,
  deleteSIPTrunk,
  deleteSIPRule,
  validateSIPConfig,
  testSIPConnection,
  getSIPHealth,
  getSIPEvents,
  getSIPEventStats,
  getSIPTrunkStatuses,
  getSIPAnalytics,
  getSavedSIPProviders,
  deleteSavedSIPProvider,
  syncSIPProviders,
  type SIPEvent,
  type SIPTrunkStatusData,
  type SavedSIPProvider,
} from '@/lib/api'

interface SIPProviderTemplate {
  id: string
  name: string
  server: string
  icon: string
  color: string
  description: string
  docsUrl: string
}

const PROVIDER_TEMPLATES: SIPProviderTemplate[] = [
  {
    id: 'twilio',
    name: 'Twilio',
    server: 'sip.twilio.com',
    icon: '📞',
    color: 'bg-red-500',
    description: 'Twilio Elastic SIP Trunking',
    docsUrl: 'https://www.twilio.com/docs/sip-trunking',
  },
  {
    id: 'vonage',
    name: 'Vonage',
    server: 'sip.nexmo.com',
    icon: '🔊',
    color: 'bg-purple-500',
    description: 'Vonage SIP Connect',
    docsUrl: 'https://developer.vonage.com/en/sip',
  },
  {
    id: 'plivo',
    name: 'Plivo',
    server: 'sip.plivo.com',
    icon: '📱',
    color: 'bg-green-500',
    description: 'Plivo SIP Trunking',
    docsUrl: 'https://www.plivo.com/docs/voice/concepts/sip-trunking/',
  },
  {
    id: 'yuboto',
    name: 'Yuboto',
    server: 'sip.yuboto-telephony.gr',
    icon: '🇬🇷',
    color: 'bg-blue-500',
    description: 'Greek VoIP Provider',
    docsUrl: 'https://yuboto.com',
  },
  {
    id: 'custom',
    name: 'Custom',
    server: '',
    icon: '⚙️',
    color: 'bg-gray-500',
    description: 'Configure any SIP provider',
    docsUrl: '',
  },
]

export default function SIPSettings() {
  const queryClient = useQueryClient()
  const [successMessage, setSuccessMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [validationErrors, setValidationErrors] = useState<string[]>([])
  const [validationWarnings, setValidationWarnings] = useState<string[]>([])
  const [isValidating, setIsValidating] = useState(false)
  const [isValidated, setIsValidated] = useState(false)
  
  // New provider form
  const [showAddForm, setShowAddForm] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<SIPProviderTemplate | null>(null)
  const [newProvider, setNewProvider] = useState({
    name: '',
    server: '',
    username: '',
    password: '',
    phoneNumbers: '',
    allowedIps: '',
  })

  // Fetch current SIP status from LiveKit
  const { data: sipStatus, isLoading, refetch } = useQuery({
    queryKey: ['sip-status'],
    queryFn: getSIPStatus,
    retry: false,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch SIP health
  const { data: sipHealth, refetch: refetchHealth } = useQuery({
    queryKey: ['sip-health'],
    queryFn: getSIPHealth,
    retry: false,
    refetchInterval: 60000, // Refresh every minute
  })

  // Fetch SIP events (recent logs)
  const { data: sipEventsData, refetch: refetchEvents } = useQuery({
    queryKey: ['sip-events'],
    queryFn: () => getSIPEvents({ limit: 50 }),
    retry: false,
    refetchInterval: 30000,
  })

  // Fetch SIP event stats
  const { data: sipStats } = useQuery({
    queryKey: ['sip-event-stats'],
    queryFn: () => getSIPEventStats(24),
    retry: false,
    refetchInterval: 60000,
  })

  // Fetch trunk statuses
  const { data: trunkStatusData } = useQuery({
    queryKey: ['sip-trunk-statuses'],
    queryFn: getSIPTrunkStatuses,
    retry: false,
    refetchInterval: 30000,
  })

  // Fetch SIP analytics
  const { data: sipAnalytics } = useQuery({
    queryKey: ['sip-analytics'],
    queryFn: () => getSIPAnalytics(7),
    retry: false,
    refetchInterval: 300000, // 5 minutes
  })

  // Fetch saved SIP providers from database
  const { data: savedProvidersData, refetch: refetchSavedProviders } = useQuery({
    queryKey: ['sip-saved-providers'],
    queryFn: getSavedSIPProviders,
    retry: false,
    refetchInterval: 30000,
  })

  // Sync providers mutation
  const syncProvidersMutation = useMutation({
    mutationFn: syncSIPProviders,
    onSuccess: (result) => {
      if (result.synced > 0) {
        showSuccess(`Synced ${result.synced} provider(s) to LiveKit`)
      }
      if (result.failed > 0) {
        showError(`Failed to sync ${result.failed} provider(s): ${result.errors.join(', ')}`)
      }
      queryClient.invalidateQueries({ queryKey: ['sip-status'] })
      queryClient.invalidateQueries({ queryKey: ['sip-saved-providers'] })
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to sync providers')
    },
  })

  // Delete saved provider mutation
  const deleteSavedProviderMutation = useMutation({
    mutationFn: deleteSavedSIPProvider,
    onSuccess: () => {
      showSuccess('Provider deleted successfully!')
      queryClient.invalidateQueries({ queryKey: ['sip-status'] })
      queryClient.invalidateQueries({ queryKey: ['sip-saved-providers'] })
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to delete provider')
    },
  })

  // Configure provider mutation
  const configureProviderMutation = useMutation({
    mutationFn: configureSIPProvider,
    onSuccess: (result) => {
      if (result.success) {
        showSuccess(`Provider configured successfully! Trunk ID: ${result.trunk_id}`)
        setShowAddForm(false)
        setSelectedTemplate(null)
        setNewProvider({ name: '', server: '', username: '', password: '', phoneNumbers: '', allowedIps: '' })
        queryClient.invalidateQueries({ queryKey: ['sip-status'] })
      } else {
        showError(result.error || 'Failed to configure provider')
      }
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to configure provider')
    },
  })

  // Delete trunk mutation
  const deleteTrunkMutation = useMutation({
    mutationFn: deleteSIPTrunk,
    onSuccess: () => {
      showSuccess('Trunk deleted successfully!')
      queryClient.invalidateQueries({ queryKey: ['sip-status'] })
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to delete trunk')
    },
  })

  // Delete rule mutation
  const deleteRuleMutation = useMutation({
    mutationFn: deleteSIPRule,
    onSuccess: () => {
      showSuccess('Dispatch rule deleted successfully!')
      queryClient.invalidateQueries({ queryKey: ['sip-status'] })
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to delete rule')
    },
  })

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

  const handleSelectTemplate = (template: SIPProviderTemplate) => {
    setSelectedTemplate(template)
    setShowAddForm(true)
    
    // Default allowed IPs for known providers
    let defaultAllowedIps = ''
    if (template.id === 'twilio') {
      defaultAllowedIps = [
        '54.172.60.0/23',    // Twilio US East
        '54.244.51.0/24',    // Twilio US West
        '54.171.127.0/24',   // Twilio Ireland
        '54.65.63.0/24',     // Twilio Tokyo
        '54.169.127.0/24',   // Twilio Singapore
        '54.252.254.0/24',   // Twilio Sydney
        '177.71.206.0/24',   // Twilio São Paulo
        '35.156.191.0/24',   // Twilio Frankfurt
      ].join(', ')
    } else if (template.id === 'vonage') {
      defaultAllowedIps = '0.0.0.0/0'  // Allow all for Vonage
    } else if (template.id === 'plivo') {
      defaultAllowedIps = '0.0.0.0/0'  // Allow all for Plivo
    }
    
    setNewProvider({
      name: template.name,
      server: template.server,
      username: '',
      password: '',
      phoneNumbers: '',
      allowedIps: defaultAllowedIps,
    })
  }

  const handleValidateProvider = async () => {
    setIsValidating(true)
    setValidationErrors([])
    setValidationWarnings([])
    setIsValidated(false)

    const phoneNumbers = newProvider.phoneNumbers
      .split(',')
      .map((p) => p.trim())
      .filter(Boolean)

    const allowedIps = newProvider.allowedIps
      .split(',')
      .map((ip) => ip.trim())
      .filter(Boolean)

    try {
      const result = await validateSIPConfig({
        provider_name: newProvider.name,
        server: newProvider.server,
        username: newProvider.username || '',
        password: newProvider.password || '',
        phone_numbers: phoneNumbers,
        allowed_ips: allowedIps,
      })

      setValidationErrors(result.errors || [])
      setValidationWarnings(result.warnings || [])
      setIsValidated(result.valid)

      if (result.valid) {
        showSuccess('Configuration validated successfully!')
      }
    } catch (err: any) {
      setValidationErrors([err.response?.data?.detail || 'Validation failed'])
    } finally {
      setIsValidating(false)
    }
  }

  const handleConfigureProvider = () => {
    if (!newProvider.name || !newProvider.server) {
      showError('Please fill in provider name and server')
      return
    }

    const phoneNumbers = newProvider.phoneNumbers
      .split(',')
      .map((p) => p.trim())
      .filter(Boolean)

    const allowedIps = newProvider.allowedIps
      .split(',')
      .map((ip) => ip.trim())
      .filter(Boolean)

    configureProviderMutation.mutate({
      provider_name: newProvider.name,
      server: newProvider.server,
      username: newProvider.username,
      password: newProvider.password,
      phone_numbers: phoneNumbers,
      allowed_ips: allowedIps.length > 0 ? allowedIps : ['0.0.0.0/0'],
    })
  }

  // Reset validation when form changes
  const handleFormChange = (field: string, value: string) => {
    setNewProvider({ ...newProvider, [field]: value })
    setIsValidated(false)
    setValidationErrors([])
    setValidationWarnings([])
  }

  const handleDeleteTrunk = (trunkId: string) => {
    if (confirm('Are you sure you want to delete this SIP trunk? This will disconnect all calls using this trunk.')) {
      deleteTrunkMutation.mutate(trunkId)
    }
  }

  const handleDeleteRule = (ruleId: string) => {
    if (confirm('Are you sure you want to delete this dispatch rule?')) {
      deleteRuleMutation.mutate(ruleId)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">SIP Settings</h1>
          <p className="text-slate-500 mt-1">
            Configure SIP providers for inbound call handling. Changes apply immediately without restart.
          </p>
        </div>
        <Button variant="outline" onClick={() => refetch()} disabled={isLoading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
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

      {/* Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="w-5 h-5" />
            SIP Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center gap-2 text-slate-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading status...
            </div>
          ) : sipStatus?.error ? (
            <div className="text-red-600">
              <AlertCircle className="w-4 h-4 inline mr-2" />
              {sipStatus.error}
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className={`text-3xl font-bold ${sipHealth?.healthy ? 'text-green-600' : 'text-red-600'}`}>
                  {sipHealth?.healthy ? '✓' : '✗'}
                </p>
                <p className="text-sm text-slate-500">Health</p>
              </div>
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className="text-3xl font-bold text-blue-600">{sipStatus?.trunks_count || 0}</p>
                <p className="text-sm text-slate-500">Active Trunks</p>
              </div>
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className="text-3xl font-bold text-green-600">{sipStatus?.rules_count || 0}</p>
                <p className="text-sm text-slate-500">Dispatch Rules</p>
              </div>
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className={`text-3xl font-bold ${sipHealth?.connection?.connected ? 'text-green-600' : 'text-red-600'}`}>
                  {sipHealth?.connection?.connected ? 'Connected' : 'Disconnected'}
                </p>
                <p className="text-sm text-slate-500">LiveKit</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Health Issues */}
      {sipHealth?.issues && sipHealth.issues.length > 0 && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-yellow-800">
              <AlertCircle className="w-5 h-5" />
              Configuration Issues
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {sipHealth.issues.map((issue, i) => (
                <li
                  key={i}
                  className={`flex items-start gap-2 ${
                    issue.severity === 'error' ? 'text-red-700' : 'text-yellow-700'
                  }`}
                >
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{issue.message}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Add Provider Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5" />
            Add SIP Provider
          </CardTitle>
          <CardDescription>
            Select a provider template or configure a custom SIP trunk.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!showAddForm ? (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {PROVIDER_TEMPLATES.map((template) => (
                <button
                  key={template.id}
                  onClick={() => handleSelectTemplate(template)}
                  className="p-4 border rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors text-center"
                >
                  <div className="text-3xl mb-2">{template.icon}</div>
                  <h4 className="font-semibold">{template.name}</h4>
                  <p className="text-xs text-slate-500 mt-1">{template.description}</p>
                </button>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold flex items-center gap-2">
                  {selectedTemplate?.icon} Configure {selectedTemplate?.name}
                </h4>
                <Button variant="outline" size="sm" onClick={() => setShowAddForm(false)}>
                  Cancel
                </Button>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Provider Name *</Label>
                  <Input
                    value={newProvider.name}
                    onChange={(e) => handleFormChange('name', e.target.value)}
                    placeholder="My Provider"
                  />
                </div>
                <div>
                  <Label>SIP Server *</Label>
                  <Input
                    value={newProvider.server}
                    onChange={(e) => handleFormChange('server', e.target.value)}
                    placeholder="sip.provider.com"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Domain or IP address of your SIP provider
                  </p>
                </div>
                <div>
                  <Label>Username / SIP ID (optional for inbound)</Label>
                  <Input
                    value={newProvider.username}
                    onChange={(e) => handleFormChange('username', e.target.value)}
                    placeholder="Leave empty for IP-based auth"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    For Twilio inbound: leave empty (uses IP allowlist)
                  </p>
                </div>
                <div>
                  <Label>Password (optional for inbound)</Label>
                  <Input
                    type="password"
                    value={newProvider.password}
                    onChange={(e) => handleFormChange('password', e.target.value)}
                    placeholder="Leave empty for IP-based auth"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Only needed for outbound or if provider requires auth
                  </p>
                </div>
                <div className="col-span-2">
                  <Label>Phone Numbers (comma separated)</Label>
                  <Input
                    value={newProvider.phoneNumbers}
                    onChange={(e) => handleFormChange('phoneNumbers', e.target.value)}
                    placeholder="+30211234567, +30698765432"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Enter the phone numbers in E.164 format (e.g., +30211234567)
                  </p>
                </div>
                <div className="col-span-2">
                  <Label>Allowed IP Ranges (comma separated, CIDR notation)</Label>
                  <Input
                    value={newProvider.allowedIps}
                    onChange={(e) => handleFormChange('allowedIps', e.target.value)}
                    placeholder="54.172.60.0/23, 54.244.51.0/24, 0.0.0.0/0"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    IP ranges that can send SIP traffic. Use 0.0.0.0/0 to allow all. Pre-filled for known providers.
                  </p>
                </div>
              </div>

              {/* Validation Results */}
              {validationErrors.length > 0 && (
                <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                  <h4 className="font-semibold text-red-700 mb-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Validation Errors
                  </h4>
                  <ul className="text-sm text-red-600 space-y-1 list-disc list-inside">
                    {validationErrors.map((error, i) => (
                      <li key={i}>{error}</li>
                    ))}
                  </ul>
                </div>
              )}

              {validationWarnings.length > 0 && (
                <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <h4 className="font-semibold text-yellow-700 mb-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Warnings
                  </h4>
                  <ul className="text-sm text-yellow-600 space-y-1 list-disc list-inside">
                    {validationWarnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {isValidated && (
                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <p className="text-green-700 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" />
                    Configuration validated! Ready to configure.
                  </p>
                </div>
              )}

              {selectedTemplate?.docsUrl && (
                <a
                  href={selectedTemplate.docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-blue-600 hover:underline flex items-center gap-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  View {selectedTemplate.name} Documentation
                </a>
              )}

              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={handleValidateProvider}
                  disabled={isValidating || !newProvider.name || !newProvider.server}
                  className="flex-1"
                >
                  {isValidating ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Validating...
                    </>
                  ) : (
                    <>
                      <Shield className="w-4 h-4 mr-2" />
                      Validate Config
                    </>
                  )}
                </Button>

                <Button
                  onClick={handleConfigureProvider}
                  disabled={configureProviderMutation.isPending || !isValidated}
                  className="flex-1"
                >
                  {configureProviderMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Configuring...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Configure Provider
                    </>
                  )}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Saved Providers (Database) */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                Saved Providers
              </CardTitle>
              <CardDescription>
                SIP providers saved in database. These are automatically synced to LiveKit on startup.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => syncProvidersMutation.mutate()}
              disabled={syncProvidersMutation.isPending}
            >
              {syncProvidersMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Syncing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Sync to LiveKit
                </>
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {savedProvidersData?.providers && savedProvidersData.providers.length > 0 ? (
            <div className="space-y-3">
              {savedProvidersData.providers.map((provider) => (
                <div
                  key={provider.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <h4 className="font-semibold">{provider.name}</h4>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          provider.sync_status === 'synced'
                            ? 'bg-green-100 text-green-700'
                            : provider.sync_status === 'failed'
                            ? 'bg-red-100 text-red-700'
                            : provider.sync_status === 'pending'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-slate-100 text-slate-700'
                        }`}
                      >
                        {provider.sync_status}
                      </span>
                    </div>
                    <p className="text-sm text-slate-500">
                      Server: {provider.server} | User: {provider.username}
                    </p>
                    {provider.phone_numbers?.length > 0 && (
                      <p className="text-sm text-slate-500">
                        Numbers: {provider.phone_numbers.join(', ')}
                      </p>
                    )}
                    {provider.allowed_ips?.length > 0 && (
                      <p className="text-sm text-slate-400">
                        Allowed IPs: {provider.allowed_ips.slice(0, 3).join(', ')}
                        {provider.allowed_ips.length > 3 && ` +${provider.allowed_ips.length - 3} more`}
                      </p>
                    )}
                    {provider.sync_error && (
                      <p className="text-sm text-red-600 mt-1">
                        Error: {provider.sync_error}
                      </p>
                    )}
                    {provider.last_sync_at && (
                      <p className="text-xs text-slate-400 mt-1">
                        Last synced: {new Date(provider.last_sync_at).toLocaleString()}
                      </p>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (confirm(`Delete provider "${provider.name}"? This will also remove it from LiveKit.`)) {
                        deleteSavedProviderMutation.mutate(provider.id)
                      }
                    }}
                    disabled={deleteSavedProviderMutation.isPending}
                  >
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500">
              <Database className="w-8 h-8 mx-auto mb-2 text-slate-300" />
              <p>No saved providers.</p>
              <p className="text-sm mt-1">Add a provider above - it will be saved and auto-synced on restarts.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active Trunks */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Phone className="w-5 h-5" />
            Active SIP Trunks
          </CardTitle>
          <CardDescription>
            Currently configured SIP inbound trunks. These receive calls from your providers.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sipStatus?.trunks && sipStatus.trunks.length > 0 ? (
            <div className="space-y-3">
              {sipStatus.trunks.map((trunk) => (
                <div
                  key={trunk.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div>
                    <h4 className="font-semibold">{trunk.name}</h4>
                    <p className="text-sm text-slate-500">
                      ID: {trunk.id.slice(0, 8)}...
                    </p>
                    {trunk.numbers?.length > 0 && (
                      <p className="text-sm text-slate-500">
                        Numbers: {trunk.numbers.join(', ')}
                      </p>
                    )}
                    {trunk.allowed_addresses?.length > 0 && (
                      <p className="text-sm text-slate-500">
                        Addresses: {trunk.allowed_addresses.join(', ')}
                      </p>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDeleteTrunk(trunk.id)}
                    disabled={deleteTrunkMutation.isPending}
                  >
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500">
              No SIP trunks configured. Add a provider above to get started.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active Dispatch Rules */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Dispatch Rules
          </CardTitle>
          <CardDescription>
            Rules that route incoming SIP calls to voice agent rooms.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sipStatus?.rules && sipStatus.rules.length > 0 ? (
            <div className="space-y-3">
              {sipStatus.rules.map((rule) => (
                <div
                  key={rule.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div>
                    <h4 className="font-semibold">{rule.name}</h4>
                    <p className="text-sm text-slate-500">
                      ID: {rule.id.slice(0, 8)}...
                    </p>
                    {rule.trunk_ids?.length > 0 && (
                      <p className="text-sm text-slate-500">
                        Linked Trunks: {rule.trunk_ids.length}
                      </p>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDeleteRule(rule.id)}
                    disabled={deleteRuleMutation.isPending}
                  >
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500">
              No dispatch rules configured. They are created automatically when you add a provider.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trunk Connection Status */}
      {trunkStatusData?.statuses && trunkStatusData.statuses.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Trunk Connection Status
            </CardTitle>
            <CardDescription>
              Real-time status and statistics for each configured SIP trunk.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {trunkStatusData.statuses.map((status) => (
                <div
                  key={status.trunk_id}
                  className="p-4 border rounded-lg"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          status.status === 'connected'
                            ? 'bg-green-500'
                            : status.status === 'error'
                            ? 'bg-red-500'
                            : 'bg-yellow-500'
                        }`}
                      />
                      <h4 className="font-semibold">{status.trunk_name}</h4>
                      {status.provider_name && (
                        <span className="text-sm text-slate-500">({status.provider_name})</span>
                      )}
                    </div>
                    <span
                      className={`text-sm font-medium ${
                        status.status === 'connected'
                          ? 'text-green-600'
                          : status.status === 'error'
                          ? 'text-red-600'
                          : 'text-yellow-600'
                      }`}
                    >
                      {status.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-4 text-sm">
                    <div>
                      <p className="text-slate-500">Total Calls</p>
                      <p className="font-semibold">{status.total_calls}</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Successful</p>
                      <p className="font-semibold text-green-600">{status.successful_calls}</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Failed</p>
                      <p className="font-semibold text-red-600">{status.failed_calls}</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Success Rate</p>
                      <p className="font-semibold">{status.success_rate}%</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Avg Duration</p>
                      <p className="font-semibold">{status.avg_duration_seconds}s</p>
                    </div>
                  </div>
                  {status.last_error && (
                    <div className="mt-3 p-2 bg-red-50 rounded text-sm text-red-700">
                      <strong>Last Error:</strong> {status.last_error}
                      {status.last_error_at && (
                        <span className="text-red-500 ml-2">
                          ({new Date(status.last_error_at).toLocaleString()})
                        </span>
                      )}
                    </div>
                  )}
                  {status.last_call_at && (
                    <p className="mt-2 text-xs text-slate-400">
                      Last call: {new Date(status.last_call_at).toLocaleString()}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* SIP Event Statistics */}
      {sipStats && sipStats.total_events > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              SIP Statistics (Last 24 Hours)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4 mb-4">
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className="text-3xl font-bold text-blue-600">{sipStats.total_events}</p>
                <p className="text-sm text-slate-500">Total Events</p>
              </div>
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className="text-3xl font-bold text-green-600">
                  {sipStats.by_type['call_connected'] || 0}
                </p>
                <p className="text-sm text-slate-500">Connected</p>
              </div>
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className="text-3xl font-bold text-red-600">
                  {sipStats.by_type['call_failed'] || 0}
                </p>
                <p className="text-sm text-slate-500">Failed</p>
              </div>
              <div className="text-center p-4 bg-slate-50 rounded-lg">
                <p className="text-3xl font-bold text-purple-600">
                  {sipStats.avg_call_duration}s
                </p>
                <p className="text-sm text-slate-500">Avg Duration</p>
              </div>
            </div>
            
            {/* Event type breakdown */}
            {Object.keys(sipStats.by_type).length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Events by Type</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(sipStats.by_type).map(([type, count]) => (
                    <span
                      key={type}
                      className="px-3 py-1 bg-slate-100 rounded-full text-sm"
                    >
                      {type.replace(/_/g, ' ')}: <strong>{count}</strong>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Recent SIP Events Log */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <List className="w-5 h-5" />
                SIP Event Log
              </CardTitle>
              <CardDescription>
                Recent SIP connection events and call logs.
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={() => refetchEvents()}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {sipEventsData?.events && sipEventsData.events.length > 0 ? (
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b">
                    <th className="text-left py-2 px-2">Time</th>
                    <th className="text-left py-2 px-2">Event</th>
                    <th className="text-left py-2 px-2">Trunk</th>
                    <th className="text-left py-2 px-2">Caller</th>
                    <th className="text-left py-2 px-2">Status</th>
                    <th className="text-left py-2 px-2">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {sipEventsData.events.map((event) => (
                    <tr key={event.id} className="border-b hover:bg-slate-50">
                      <td className="py-2 px-2 text-slate-500">
                        {new Date(event.created_at).toLocaleTimeString()}
                      </td>
                      <td className="py-2 px-2">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                            event.event_type === 'call_connected'
                              ? 'bg-green-100 text-green-700'
                              : event.event_type === 'call_incoming'
                              ? 'bg-blue-100 text-blue-700'
                              : event.event_type === 'call_completed'
                              ? 'bg-slate-100 text-slate-700'
                              : event.event_type.includes('failed') || event.event_type.includes('error')
                              ? 'bg-red-100 text-red-700'
                              : 'bg-yellow-100 text-yellow-700'
                          }`}
                        >
                          {event.event_type === 'call_incoming' && <PhoneIncoming className="w-3 h-3" />}
                          {event.event_type === 'call_connected' && <Phone className="w-3 h-3" />}
                          {event.event_type.includes('failed') && <PhoneOff className="w-3 h-3" />}
                          {event.event_type.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="py-2 px-2 text-slate-600">
                        {event.trunk_name || event.trunk_id?.slice(0, 8) || '-'}
                      </td>
                      <td className="py-2 px-2">
                        {event.caller_number || event.from_uri?.split('@')[0]?.replace('sip:', '') || '-'}
                      </td>
                      <td className="py-2 px-2">
                        {event.status_code ? (
                          <span
                            className={`text-xs ${
                              event.status_code >= 200 && event.status_code < 300
                                ? 'text-green-600'
                                : event.status_code >= 400
                                ? 'text-red-600'
                                : 'text-yellow-600'
                            }`}
                          >
                            {event.status_code} {event.status_message}
                          </span>
                        ) : (
                          '-'
                        )}
                      </td>
                      <td className="py-2 px-2 text-slate-500">
                        {event.duration_seconds ? `${event.duration_seconds}s` : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500">
              <Clock className="w-8 h-8 mx-auto mb-2 text-slate-300" />
              <p>No SIP events recorded yet.</p>
              <p className="text-sm mt-1">Events will appear here when calls come in.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Info Card */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="pt-6">
          <h3 className="font-semibold text-blue-900 mb-2">How SIP Integration Works</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>1. Add your SIP provider credentials above</li>
            <li>2. A trunk and dispatch rule are created in LiveKit automatically</li>
            <li>3. Configure your provider to send calls to your LiveKit SIP endpoint</li>
            <li>4. Incoming calls are routed to the voice agent immediately</li>
            <li>5. Changes apply instantly - no restart required!</li>
            <li>6. View call logs and connection status in real-time above</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
