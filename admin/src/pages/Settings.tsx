import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Settings as SettingsIcon,
  Save,
  CheckCircle,
  AlertCircle,
  Globe,
  Volume2,
  Brain,
  RefreshCw,
  Plus,
  Trash2,
  Star,
  Loader2,
  Music,
  Play,
  Pause,
  Zap,
} from 'lucide-react'
import {
  getHealth,
  getLanguages,
  createLanguage,
  updateLanguage,
  deleteLanguage,
  getSettings,
  updateSetting,
  initDefaultSettings,
} from '@/lib/api'

export default function Settings() {
  const queryClient = useQueryClient()
  const [saveSuccess, setSaveSuccess] = useState('')
  const [saveError, setSaveError] = useState('')
  const [showAddLanguage, setShowAddLanguage] = useState(false)
  const [newLang, setNewLang] = useState({ code: '', name: '', native_name: '', flag_emoji: '' })
  const [hasChanges, setHasChanges] = useState(false)

  // Local state for settings
  const [agentLanguage, setAgentLanguage] = useState('el')
  const [voiceSpeed, setVoiceSpeed] = useState(0.6)
  const [voiceStability, setVoiceStability] = useState(0.45)
  const [greetingEnabled, setGreetingEnabled] = useState(true)
  const [abuseDetectionEnabled, setAbuseDetectionEnabled] = useState(true)
  const [autoLanguageSwitch, setAutoLanguageSwitch] = useState(true)
  
  // Background audio settings
  const [bgAudioEnabled, setBgAudioEnabled] = useState(false)
  const [bgAudioUrl, setBgAudioUrl] = useState('')
  const [bgAudioVolume, setBgAudioVolume] = useState(0.1)
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false)
  const audioPreviewRef = useRef<HTMLAudioElement | null>(null)
  
  // LLM settings
  const [llmProvider, setLlmProvider] = useState('openai')
  const [openaiModel, setOpenaiModel] = useState('gpt-4o-mini')
  const [groqModel, setGroqModel] = useState('llama-3.3-70b-versatile')

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    retry: false,
  })

  const { data: languagesData, isLoading: langsLoading } = useQuery({
    queryKey: ['languages'],
    queryFn: () => getLanguages(true),
    retry: false,
  })

  const { data: settingsData, isLoading: settingsLoading } = useQuery({
    queryKey: ['system-settings'],
    queryFn: getSettings,
    retry: false,
  })

  // Load settings from database
  useEffect(() => {
    if (settingsData?.settings) {
      const s = settingsData.settings
      if (s.agent_language !== undefined) setAgentLanguage(s.agent_language)
      if (s.agent_voice_speed !== undefined) setVoiceSpeed(s.agent_voice_speed)
      if (s.agent_voice_stability !== undefined) setVoiceStability(s.agent_voice_stability)
      if (s.agent_greeting_enabled !== undefined) setGreetingEnabled(s.agent_greeting_enabled)
      if (s.abuse_detection_enabled !== undefined) setAbuseDetectionEnabled(s.abuse_detection_enabled)
      if (s.auto_language_switch !== undefined) setAutoLanguageSwitch(s.auto_language_switch)
      // Background audio settings
      if (s.bg_audio_enabled !== undefined) setBgAudioEnabled(s.bg_audio_enabled)
      if (s.bg_audio_url !== undefined) setBgAudioUrl(s.bg_audio_url)
      if (s.bg_audio_volume !== undefined) setBgAudioVolume(s.bg_audio_volume)
      // LLM settings
      if (s.llm_provider !== undefined) setLlmProvider(s.llm_provider)
      if (s.openai_model !== undefined) setOpenaiModel(s.openai_model)
      if (s.groq_model !== undefined) setGroqModel(s.groq_model)
    }
  }, [settingsData])
  
  // Cleanup audio preview on unmount
  useEffect(() => {
    return () => {
      if (audioPreviewRef.current) {
        audioPreviewRef.current.pause()
        audioPreviewRef.current = null
      }
    }
  }, [])

  const languages = languagesData?.languages || []
  const services = health?.services || {}

  // Mutations
  const updateSettingMutation = useMutation({
    mutationFn: ({ key, value }: { key: string; value: any }) => updateSetting(key, value),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-settings'] })
    },
  })

  const initSettingsMutation = useMutation({
    mutationFn: initDefaultSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-settings'] })
      showSuccess('Default settings initialized!')
    },
  })

  const createLangMutation = useMutation({
    mutationFn: createLanguage,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['languages'] })
      setShowAddLanguage(false)
      setNewLang({ code: '', name: '', native_name: '', flag_emoji: '' })
      showSuccess('Language added!')
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to create language')
    },
  })

  const updateLangMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) => updateLanguage(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['languages'] })
      showSuccess('Language updated!')
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to update language')
    },
  })

  const deleteLangMutation = useMutation({
    mutationFn: deleteLanguage,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['languages'] })
      showSuccess('Language removed')
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to delete language')
    },
  })

  const showSuccess = (msg: string) => {
    setSaveSuccess(msg)
    setSaveError('')
    setTimeout(() => setSaveSuccess(''), 3000)
  }

  const showError = (msg: string) => {
    setSaveError(msg)
    setSaveSuccess('')
    setTimeout(() => setSaveError(''), 5000)
  }

  const handleSaveAll = async () => {
    try {
      // Save all settings
      await Promise.all([
        updateSettingMutation.mutateAsync({ key: 'agent_language', value: agentLanguage }),
        updateSettingMutation.mutateAsync({ key: 'agent_voice_speed', value: voiceSpeed }),
        updateSettingMutation.mutateAsync({ key: 'agent_voice_stability', value: voiceStability }),
        updateSettingMutation.mutateAsync({ key: 'agent_greeting_enabled', value: greetingEnabled }),
        updateSettingMutation.mutateAsync({ key: 'abuse_detection_enabled', value: abuseDetectionEnabled }),
        updateSettingMutation.mutateAsync({ key: 'auto_language_switch', value: autoLanguageSwitch }),
        // Background audio settings
        updateSettingMutation.mutateAsync({ key: 'bg_audio_enabled', value: bgAudioEnabled }),
        updateSettingMutation.mutateAsync({ key: 'bg_audio_url', value: bgAudioUrl }),
        updateSettingMutation.mutateAsync({ key: 'bg_audio_volume', value: bgAudioVolume }),
        // LLM settings
        updateSettingMutation.mutateAsync({ key: 'llm_provider', value: llmProvider }),
        updateSettingMutation.mutateAsync({ key: 'openai_model', value: openaiModel }),
        updateSettingMutation.mutateAsync({ key: 'groq_model', value: groqModel }),
      ])
      setHasChanges(false)
      showSuccess('Settings saved! Changes apply immediately without restart.')
    } catch (err: any) {
      showError(err.response?.data?.detail || 'Failed to save settings')
    }
  }
  
  // Audio preview functions
  const toggleAudioPreview = () => {
    if (!bgAudioUrl) {
      showError('Please enter an audio URL first')
      return
    }
    
    if (isPreviewPlaying) {
      audioPreviewRef.current?.pause()
      setIsPreviewPlaying(false)
    } else {
      if (!audioPreviewRef.current) {
        audioPreviewRef.current = new Audio(bgAudioUrl)
        audioPreviewRef.current.loop = true
        audioPreviewRef.current.onended = () => setIsPreviewPlaying(false)
        audioPreviewRef.current.onerror = () => {
          showError('Failed to load audio file')
          setIsPreviewPlaying(false)
        }
      } else {
        audioPreviewRef.current.src = bgAudioUrl
      }
      audioPreviewRef.current.volume = bgAudioVolume
      audioPreviewRef.current.play()
      setIsPreviewPlaying(true)
    }
  }
  
  // Update preview volume when slider changes
  useEffect(() => {
    if (audioPreviewRef.current) {
      audioPreviewRef.current.volume = bgAudioVolume
    }
  }, [bgAudioVolume])

  const handleAddLanguage = () => {
    if (!newLang.code || !newLang.name || !newLang.native_name) {
      showError('Code, name, and native name are required')
      return
    }
    createLangMutation.mutate({
      code: newLang.code.toLowerCase(),
      name: newLang.name,
      native_name: newLang.native_name,
      flag_emoji: newLang.flag_emoji,
    })
  }

  const setDefaultLanguage = (langId: string) => {
    updateLangMutation.mutate({ id: langId, data: { is_default: true } })
  }

  const handleSettingChange = (setter: (v: any) => void, value: any) => {
    setter(value)
    setHasChanges(true)
  }

  // Auto-save language immediately when changed
  const handleLanguageChange = async (newLang: string) => {
    setAgentLanguage(newLang)
    try {
      await updateSettingMutation.mutateAsync({ key: 'agent_language', value: newLang })
      showSuccess(`Language changed to ${newLang === 'el' ? 'Greek' : 'English'}! Next call will use this language.`)
    } catch (err: any) {
      showError(err.response?.data?.detail || 'Failed to save language')
    }
  }

  const isSaving = updateSettingMutation.isPending

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Settings</h1>
          <p className="text-slate-500 mt-1">
            Configure the AI agent behavior. Changes apply immediately without restart.
          </p>
        </div>
        <Button onClick={handleSaveAll} disabled={!hasChanges || isSaving}>
          {isSaving ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save className="w-4 h-4 mr-2" />
              Save All Settings
            </>
          )}
        </Button>
      </div>

      {/* Success/Error Messages */}
      {saveSuccess && (
        <div className="flex items-center gap-2 p-4 text-green-600 bg-green-50 rounded-lg">
          <CheckCircle className="w-5 h-5" />
          {saveSuccess}
        </div>
      )}

      {saveError && (
        <div className="flex items-center gap-2 p-4 text-red-600 bg-red-50 rounded-lg">
          <AlertCircle className="w-5 h-5" />
          {saveError}
        </div>
      )}

      {/* Service Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5" />
            Service Status
          </CardTitle>
          <CardDescription>Current status of connected services</CardDescription>
        </CardHeader>
        <CardContent>
          {healthLoading ? (
            <p className="text-slate-500">Checking services...</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(services).map(([name, status]: [string, any]) => (
                <div
                  key={name}
                  className={`p-4 rounded-lg border ${
                    status.configured || status.status === 'configured' || status.status === 'healthy'
                      ? 'border-green-200 bg-green-50'
                      : 'border-red-200 bg-red-50'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        status.configured || status.status === 'configured' || status.status === 'healthy'
                          ? 'bg-green-500'
                          : 'bg-red-500'
                      }`}
                    />
                    <span className="font-medium capitalize">{name.replace('_', ' ')}</span>
                  </div>
                  <p className="text-xs text-slate-500">
                    {status.message || (status.configured ? 'Connected' : 'Not configured')}
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Languages Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Globe className="w-5 h-5" />
                Languages
              </CardTitle>
              <CardDescription>
                Manage supported languages for prompts and knowledge base
              </CardDescription>
            </div>
            <Button onClick={() => setShowAddLanguage(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add Language
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Add Language Form */}
          {showAddLanguage && (
            <div className="p-4 border-2 border-blue-200 bg-blue-50 rounded-lg space-y-4">
              <h4 className="font-medium">Add New Language</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <Label>Code</Label>
                  <Input
                    placeholder="e.g., de"
                    value={newLang.code}
                    onChange={(e) => setNewLang({ ...newLang, code: e.target.value })}
                    maxLength={5}
                  />
                </div>
                <div className="space-y-1">
                  <Label>Name</Label>
                  <Input
                    placeholder="e.g., German"
                    value={newLang.name}
                    onChange={(e) => setNewLang({ ...newLang, name: e.target.value })}
                  />
                </div>
                <div className="space-y-1">
                  <Label>Native Name</Label>
                  <Input
                    placeholder="e.g., Deutsch"
                    value={newLang.native_name}
                    onChange={(e) => setNewLang({ ...newLang, native_name: e.target.value })}
                  />
                </div>
                <div className="space-y-1">
                  <Label>Flag Emoji</Label>
                  <Input
                    placeholder="e.g., 🇩🇪"
                    value={newLang.flag_emoji}
                    onChange={(e) => setNewLang({ ...newLang, flag_emoji: e.target.value })}
                    maxLength={4}
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleAddLanguage} disabled={createLangMutation.isPending}>
                  {createLangMutation.isPending ? 'Adding...' : 'Add Language'}
                </Button>
                <Button variant="outline" onClick={() => setShowAddLanguage(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* Languages List */}
          {langsLoading ? (
            <p className="text-slate-500">Loading languages...</p>
          ) : languages.length === 0 ? (
            <p className="text-slate-500 text-center py-4">
              No languages configured. Click "Add Language" to get started.
            </p>
          ) : (
            <div className="space-y-2">
              {languages.map((lang) => (
                <div
                  key={lang.id}
                  className={`flex items-center justify-between p-4 rounded-lg border ${
                    lang.is_default ? 'border-blue-500 bg-blue-50' : 'border-slate-200'
                  } ${!lang.is_active ? 'opacity-50' : ''}`}
                >
                  <div className="flex items-center gap-4">
                    <span className="text-2xl">{lang.flag_emoji || '🌐'}</span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{lang.name}</span>
                        <span className="text-sm text-slate-500">({lang.native_name})</span>
                        <code className="text-xs px-1.5 py-0.5 bg-slate-100 rounded">
                          {lang.code}
                        </code>
                        {lang.is_default && (
                          <span className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded-full">
                            Default
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {!lang.is_default && lang.is_active && (
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Set as default"
                        onClick={() => setDefaultLanguage(lang.id)}
                      >
                        <Star className="w-4 h-4" />
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      onClick={() => {
                        if (confirm(`Remove ${lang.name}?`)) {
                          deleteLangMutation.mutate(lang.id)
                        }
                      }}
                      disabled={lang.is_default}
                      title={lang.is_default ? 'Cannot delete default language' : 'Delete'}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Default Agent Language */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Globe className="w-5 h-5" />
              Default Agent Language
            </CardTitle>
            <CardDescription>
              Select the primary language for the AI agent (applies immediately)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {settingsLoading ? (
              <p className="text-slate-500">Loading...</p>
            ) : (
              <div className="space-y-2">
                <Label>Agent Language</Label>
                <div className="flex flex-wrap gap-2">
                  {languages
                    .filter((l) => l.is_active)
                    .map((lang) => (
                      <Button
                        key={lang.code}
                        variant={agentLanguage === lang.code ? 'default' : 'outline'}
                        onClick={() => handleLanguageChange(lang.code)}
                        disabled={updateSettingMutation.isPending}
                      >
                        {lang.flag_emoji} {lang.name}
                      </Button>
                    ))}
                  {languages.filter((l) => l.is_active).length === 0 && (
                    <>
                      <Button
                        variant={agentLanguage === 'el' ? 'default' : 'outline'}
                        onClick={() => handleLanguageChange('el')}
                        disabled={updateSettingMutation.isPending}
                      >
                        🇬🇷 Greek
                      </Button>
                      <Button
                        variant={agentLanguage === 'en' ? 'default' : 'outline'}
                        onClick={() => handleLanguageChange('en')}
                        disabled={updateSettingMutation.isPending}
                      >
                        🇬🇧 English
                      </Button>
                    </>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Voice Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Volume2 className="w-5 h-5" />
              Voice
            </CardTitle>
            <CardDescription>Configure text-to-speech settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Voice Speed: {voiceSpeed.toFixed(2)}</Label>
              <input
                type="range"
                min="0.5"
                max="1.5"
                step="0.05"
                value={voiceSpeed}
                onChange={(e) => handleSettingChange(setVoiceSpeed, parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>Slow</span>
                <span>Normal</span>
                <span>Fast</span>
              </div>
            </div>
            <div className="space-y-2">
              <Label>Voice Stability: {voiceStability.toFixed(2)}</Label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={voiceStability}
                onChange={(e) => handleSettingChange(setVoiceStability, parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>Variable</span>
                <span>Stable</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* LLM Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              LLM Configuration
            </CardTitle>
            <CardDescription>
              Choose AI model provider for faster responses
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Provider Selection */}
            <div className="space-y-2">
              <Label>Provider</Label>
              <div className="flex gap-2">
                <Button
                  variant={llmProvider === 'openai' ? 'default' : 'outline'}
                  onClick={() => handleSettingChange(setLlmProvider, 'openai')}
                  className="flex-1"
                >
                  <Brain className="w-4 h-4 mr-2" />
                  OpenAI
                </Button>
                <Button
                  variant={llmProvider === 'groq' ? 'default' : 'outline'}
                  onClick={() => handleSettingChange(setLlmProvider, 'groq')}
                  className="flex-1"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Groq
                </Button>
              </div>
              {llmProvider === 'groq' && (
                <p className="text-xs text-green-600 flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  Groq is 10x faster than OpenAI
                </p>
              )}
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label>Model {llmProvider === 'openai' ? '(OpenAI)' : '(Groq)'}</Label>
              {llmProvider === 'openai' ? (
                <select
                  value={openaiModel}
                  onChange={(e) => handleSettingChange(setOpenaiModel, e.target.value)}
                  className="w-full p-2 border rounded-lg bg-white"
                >
                  <option value="gpt-4o-mini">gpt-4o-mini (Fast, Good quality)</option>
                  <option value="gpt-4o">gpt-4o (Slower, Best quality)</option>
                  <option value="gpt-3.5-turbo">gpt-3.5-turbo (Fastest, OK quality)</option>
                </select>
              ) : (
                <select
                  value={groqModel}
                  onChange={(e) => handleSettingChange(setGroqModel, e.target.value)}
                  className="w-full p-2 border rounded-lg bg-white"
                >
                  <option value="llama-3.3-70b-versatile">llama-3.3-70b-versatile (Best quality)</option>
                  <option value="llama-3.1-8b-instant">llama-3.1-8b-instant (Fastest)</option>
                  <option value="mixtral-8x7b-32768">mixtral-8x7b-32768 (Good balance)</option>
                </select>
              )}
            </div>

            {/* Info */}
            <div className="p-3 bg-slate-50 rounded-lg text-sm text-slate-600">
              <p className="font-medium mb-1">API Keys</p>
              <p className="text-xs">
                API keys are configured in environment variables (OPENAI_API_KEY, GROQ_API_KEY).
                Contact your administrator to update API keys.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Background Audio Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Music className="w-5 h-5" />
              Background Audio
            </CardTitle>
            <CardDescription>
              Ambient sound during calls (works for web and phone)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Enable/Disable Toggle */}
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <Label>Enable Background Audio</Label>
                <p className="text-xs text-slate-500">Play ambient sound during calls</p>
              </div>
              <button
                onClick={() => handleSettingChange(setBgAudioEnabled, !bgAudioEnabled)}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  bgAudioEnabled ? 'bg-blue-600' : 'bg-slate-300'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                    bgAudioEnabled ? 'translate-x-6' : ''
                  }`}
                />
              </button>
            </div>
            
            {/* Audio URL */}
            <div className="space-y-2">
              <Label>Audio URL</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="https://example.com/ambient.mp3"
                  value={bgAudioUrl}
                  onChange={(e) => handleSettingChange(setBgAudioUrl, e.target.value)}
                  disabled={!bgAudioEnabled}
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={toggleAudioPreview}
                  disabled={!bgAudioEnabled || !bgAudioUrl}
                  title={isPreviewPlaying ? 'Stop Preview' : 'Preview Audio'}
                >
                  {isPreviewPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
              </div>
              <p className="text-xs text-slate-500">
                Use a direct link to MP3 file (looped during call)
              </p>
            </div>
            
            {/* Volume Slider */}
            <div className="space-y-2">
              <Label>Volume: {Math.round(bgAudioVolume * 100)}%</Label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={bgAudioVolume}
                onChange={(e) => handleSettingChange(setBgAudioVolume, parseFloat(e.target.value))}
                className="w-full"
                disabled={!bgAudioEnabled}
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>Off</span>
                <span>Low</span>
                <span>Medium</span>
              </div>
            </div>
            
            {/* Preset Audio Options */}
            <div className="space-y-2">
              <Label>Quick Presets</Label>
              <div className="flex flex-wrap gap-2">
                {[
                  { name: 'Office', url: 'https://cdn.pixabay.com/download/audio/2022/03/10/audio_6c5e5c5c5c.mp3' },
                  { name: 'Cafe', url: 'https://cdn.pixabay.com/download/audio/2021/09/06/audio_f1c8e8e8e8.mp3' },
                  { name: 'Nature', url: 'https://cdn.pixabay.com/download/audio/2022/01/18/audio_d1d1d1d1d1.mp3' },
                ].map((preset) => (
                  <Button
                    key={preset.name}
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      handleSettingChange(setBgAudioUrl, preset.url)
                      if (isPreviewPlaying) {
                        audioPreviewRef.current?.pause()
                        setIsPreviewPlaying(false)
                      }
                    }}
                    disabled={!bgAudioEnabled}
                  >
                    {preset.name}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Behavior Settings */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SettingsIcon className="w-5 h-5" />
              Behavior
            </CardTitle>
            <CardDescription>Configure agent behavior</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <Label>Greeting Message</Label>
                  <p className="text-sm text-slate-500">Enable automatic greeting on call start</p>
                </div>
                <button
                  onClick={() => handleSettingChange(setGreetingEnabled, !greetingEnabled)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    greetingEnabled ? 'bg-blue-600' : 'bg-slate-300'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      greetingEnabled ? 'translate-x-6' : ''
                    }`}
                  />
                </button>
              </div>
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <Label>Auto Language Switch</Label>
                  <p className="text-sm text-slate-500">Match the caller's Greek or English mid-call</p>
                </div>
                <button
                  onClick={() => handleSettingChange(setAutoLanguageSwitch, !autoLanguageSwitch)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    autoLanguageSwitch ? 'bg-blue-600' : 'bg-slate-300'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      autoLanguageSwitch ? 'translate-x-6' : ''
                    }`}
                  />
                </button>
              </div>
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <Label>Abuse Detection</Label>
                  <p className="text-sm text-slate-500">Detect and handle abusive callers</p>
                </div>
                <button
                  onClick={() => handleSettingChange(setAbuseDetectionEnabled, !abuseDetectionEnabled)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    abuseDetectionEnabled ? 'bg-blue-600' : 'bg-slate-300'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      abuseDetectionEnabled ? 'translate-x-6' : ''
                    }`}
                  />
                </button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Info */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="pt-6">
          <h3 className="font-semibold text-blue-900 mb-2">No Restart Required</h3>
          <p className="text-sm text-blue-800">
            All settings are applied in real-time. The AI agent will use the new configuration 
            within 60 seconds of saving. No Docker restart is needed.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
