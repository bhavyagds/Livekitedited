import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Save,
  CheckCircle,
  AlertCircle,
  Globe,
  FileText,
  RotateCcw,
} from 'lucide-react'
import { getLanguages } from '@/lib/api'
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/admin',
  headers: { 'Content-Type': 'application/json' },
})

// Add auth header
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('admin_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export default function KnowledgeBase() {
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

  // Fetch KB content for selected language
  const { data: kbData, isLoading } = useQuery({
    queryKey: ['kb-content', selectedLanguage],
    queryFn: async () => {
      const res = await api.get(`/kb/content/${selectedLanguage}`)
      return res.data
    },
    retry: false,
  })

  // Update content when data loads
  useEffect(() => {
    if (kbData?.content) {
      setContent(kbData.content)
      setOriginalContent(kbData.content)
      setHasChanges(false)
    } else if (kbData === null || (kbData && !kbData.content)) {
      // No content yet for this language
      setContent('')
      setOriginalContent('')
      setHasChanges(false)
    }
  }, [kbData])

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      const res = await api.put(`/kb/content/${selectedLanguage}`, { content })
      return res.data
    },
    onSuccess: () => {
      setOriginalContent(content)
      setHasChanges(false)
      queryClient.invalidateQueries({ queryKey: ['kb-content'] })
      showSuccess('Knowledge base saved successfully!')
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
          <h1 className="text-3xl font-bold text-slate-900">Knowledge Base</h1>
          <p className="text-slate-500 mt-1">
            Edit the AI agent's knowledge for each language. Changes apply immediately.
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
            <FileText className="w-5 h-5" />
            Knowledge Base Content ({getLanguageLabel(selectedLanguage)})
          </CardTitle>
          <CardDescription>
            Enter all the information the AI agent should know. Include FAQs, product info, 
            company details, policies, and any other relevant content.
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
              placeholder={`Enter knowledge base content for ${getLanguageLabel(selectedLanguage)}...

Example format:

## About Company
Our company is called Meallion. We provide premium ready meals...

## Products
- Protein Boost: High protein meals (34-66g per serving)
- Signature: Gourmet dishes by Chef Lambros
- Plant-Based: Vegetarian options

## Ordering
- Minimum order: 2 meals
- Free delivery for 5+ meals
- Delivery area: Athens

## Contact
Phone: +30 211 9555 451
Email: hello@meallion.gr

## FAQ
Q: How long do meals last?
A: Up to 7 days in fridge at 0-4°C

Q: How to heat?
A: Microwave 3 min at 600W or oven 120°C for 20 min
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
          <h3 className="font-semibold text-blue-900 mb-2">Tips for Writing Knowledge Base</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use clear headings (## Section Name) to organize content</li>
            <li>• Write in the language selected above</li>
            <li>• Include common questions and answers</li>
            <li>• Add product details, pricing, and policies</li>
            <li>• Keep information concise and accurate</li>
            <li>• The AI will use this content to answer customer questions</li>
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
