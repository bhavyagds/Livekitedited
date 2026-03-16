import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Save,
  CheckCircle,
  AlertCircle,
  Globe,
  MessageSquare,
  RotateCcw,
} from 'lucide-react'
import { getLanguages } from '@/lib/api'
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/admin',
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('admin_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export default function Prompts() {
  const queryClient = useQueryClient()
  const [selectedLanguage, setSelectedLanguage] = useState<string>('el')
  const [content, setContent] = useState<string>('')
  const [originalContent, setOriginalContent] = useState<string>('')
  const [successMessage, setSuccessMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [hasChanges, setHasChanges] = useState(false)

  // Fetch languages
  const { data: languagesData } = useQuery({
    queryKey: ['languages'],
    queryFn: () => getLanguages(),
    retry: false,
  })
  const languages = languagesData?.languages || []

  // Fetch prompts content for selected language
  const { data: promptsData, isLoading } = useQuery({
    queryKey: ['prompts-content', selectedLanguage],
    queryFn: async () => {
      const res = await api.get(`/prompts/content/${selectedLanguage}`)
      return res.data
    },
    retry: false,
  })

  // Update content when data loads
  useEffect(() => {
    if (promptsData?.content) {
      setContent(promptsData.content)
      setOriginalContent(promptsData.content)
      setHasChanges(false)
    } else if (promptsData === null || (promptsData && !promptsData.content)) {
      setContent('')
      setOriginalContent('')
      setHasChanges(false)
    }
  }, [promptsData])

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      const res = await api.put(`/prompts/content/${selectedLanguage}`, { content })
      return res.data
    },
    onSuccess: () => {
      setOriginalContent(content)
      setHasChanges(false)
      queryClient.invalidateQueries({ queryKey: ['prompts-content'] })
      showSuccess('Prompts saved successfully!')
    },
    onError: (err: any) => {
      showError(err.response?.data?.detail || 'Failed to save')
    },
  })

  const showSuccess = (msg: string) => {
    setSuccessMessage(msg)
    setErrorMessage('')
    setTimeout(() => setSuccessMessage(''), 3000)
  }

  const showError = (msg: string) => {
    setErrorMessage(msg)
    setSuccessMessage('')
    setTimeout(() => setErrorMessage(''), 5000)
  }

  const handleContentChange = (value: string) => {
    setContent(value)
    setHasChanges(value !== originalContent)
  }

  const handleReset = () => {
    setContent(originalContent)
    setHasChanges(false)
  }

  const handleSave = () => {
    saveMutation.mutate()
  }

  const getLanguageLabel = (code: string) => {
    const lang = languages.find((l) => l.code === code)
    return lang ? `${lang.flag_emoji || ''} ${lang.name}` : code.toUpperCase()
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Prompts</h1>
          <p className="text-slate-500 mt-1">
            Customize the AI agent's system prompt, greeting, and responses for each language.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleReset} disabled={!hasChanges}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button onClick={handleSave} disabled={!hasChanges || saveMutation.isPending}>
            <Save className="w-4 h-4 mr-2" />
            {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
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

      {/* Language Tabs */}
      <div className="flex items-center gap-2">
        <Globe className="w-5 h-5 text-slate-500" />
        <div className="flex gap-1">
          {languages.map((lang) => (
            <Button
              key={lang.code}
              size="sm"
              variant={selectedLanguage === lang.code ? 'default' : 'outline'}
              onClick={() => setSelectedLanguage(lang.code)}
            >
              {lang.flag_emoji} {lang.name}
            </Button>
          ))}
          {languages.length === 0 && (
            <>
              <Button
                size="sm"
                variant={selectedLanguage === 'el' ? 'default' : 'outline'}
                onClick={() => setSelectedLanguage('el')}
              >
                🇬🇷 Greek
              </Button>
              <Button
                size="sm"
                variant={selectedLanguage === 'en' ? 'default' : 'outline'}
                onClick={() => setSelectedLanguage('en')}
              >
                🇬🇧 English
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Content Editor */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Agent Prompts ({getLanguageLabel(selectedLanguage)})
          </CardTitle>
          <CardDescription>
            Define the AI agent's personality, greeting messages, responses, and behavior rules.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="h-96 flex items-center justify-center text-slate-500">
              Loading...
            </div>
          ) : (
            <textarea
              value={content}
              onChange={(e) => handleContentChange(e.target.value)}
              placeholder={`Enter prompts for ${getLanguageLabel(selectedLanguage)}...

Example format:

## Agent Identity
Name: Elena
Role: Customer service assistant for Meallion

## Greeting
Γεια σας! Είμαι η Έλενα από τη Meallion. Πώς μπορώ να σας βοηθήσω;

## Personality
- Friendly and professional
- Speak naturally, like a real person
- Be helpful and patient
- Keep responses concise

## Handling Orders
When customer asks about an order:
1. Ask for their order number or phone
2. Look up order status
3. Provide clear update

## Closing
Thank the customer and wish them a nice day.

## Error Response
If you don't understand:
"Συγγνώμη, δεν κατάλαβα. Μπορείτε να επαναλάβετε;"

## Do NOT
- Never discuss competitors
- Never make promises about delivery times
- Never share customer data
`}
              className="w-full h-[600px] p-4 border rounded-lg resize-none font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              spellCheck={false}
            />
          )}
        </CardContent>
      </Card>

      {/* Tips */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="pt-6">
          <h3 className="font-semibold text-blue-900 mb-2">Tips for Writing Prompts</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use clear headings (## Section Name) to organize</li>
            <li>• Write in the language the agent should speak</li>
            <li>• Include greeting and closing messages</li>
            <li>• Define personality traits and behavior rules</li>
            <li>• Add "Do NOT" section for things to avoid</li>
            <li>• Be specific about how to handle common scenarios</li>
          </ul>
        </CardContent>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-600">{content.length.toLocaleString()}</p>
              <p className="text-sm text-slate-500">Characters</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-3xl font-bold text-green-600">
                {content.split(/\s+/).filter(Boolean).length.toLocaleString()}
              </p>
              <p className="text-sm text-slate-500">Words</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-600">
                {content.split('\n').filter(Boolean).length}
              </p>
              <p className="text-sm text-slate-500">Lines</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
