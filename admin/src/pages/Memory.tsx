import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getMemoryItems,
  createMemoryItem,
  updateMemoryItem,
  deleteMemoryItem,
  type MemoryItem,
} from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Brain,
  Plus,
  Pencil,
  Trash2,
  Check,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  ToggleLeft,
  ToggleRight,
  MessageSquare,
} from 'lucide-react'

// ─── Inline-editable cell ────────────────────────────────────────────────────
function EditableCell({
  value,
  onSave,
  multiline = false,
  placeholder = '',
}: {
  value: string
  onSave: (v: string) => void
  multiline?: boolean
  placeholder?: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)

  const commit = () => {
    if (draft.trim() !== value) onSave(draft.trim())
    setEditing(false)
  }

  const cancel = () => {
    setDraft(value)
    setEditing(false)
  }

  if (!editing) {
    return (
      <div
        className="group flex items-start gap-1 cursor-pointer min-h-[2rem]"
        onClick={() => { setDraft(value); setEditing(true) }}
      >
        <span className={`flex-1 text-sm leading-snug ${value ? '' : 'text-slate-400 italic'}`}>
          {value || placeholder}
        </span>
        <Pencil className="w-3.5 h-3.5 text-slate-300 group-hover:text-slate-500 mt-0.5 shrink-0" />
      </div>
    )
  }

  return (
    <div className="flex items-start gap-1">
      {multiline ? (
        <textarea
          className="flex-1 text-sm border rounded px-2 py-1 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[80px]"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Escape') cancel() }}
        />
      ) : (
        <input
          className="flex-1 text-sm border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') cancel() }}
        />
      )}
      <button onClick={commit} className="p-1 text-green-600 hover:bg-green-50 rounded shrink-0">
        <Check className="w-4 h-4" />
      </button>
      <button onClick={cancel} className="p-1 text-slate-400 hover:bg-slate-100 rounded shrink-0">
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

// ─── Add-item modal ──────────────────────────────────────────────────────────
function AddMemoryModal({
  initialQuestion = '',
  initialAnswer = '',
  onClose,
  onSave,
  isSaving,
}: {
  initialQuestion?: string
  initialAnswer?: string
  onClose: () => void
  onSave: (q: string, a: string, c: string) => void
  isSaving: boolean
}) {
  const [question, setQuestion] = useState(initialQuestion)
  const [answer, setAnswer] = useState(initialAnswer)
  const [comment, setComment] = useState('')

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full">
        <div className="flex items-center justify-between p-5 border-b">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-600" />
            <h3 className="font-semibold text-slate-900">Add to Memory</h3>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-5 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wide">
              Question
            </label>
            <textarea
              className="w-full border rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 min-h-[70px]"
              placeholder="What should Elena be asked?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wide">
              Answer
            </label>
            <textarea
              className="w-full border rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 min-h-[70px]"
              placeholder="How should Elena respond?"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wide">
              Comment <span className="text-slate-400 font-normal">(optional)</span>
            </label>
            <input
              className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
              placeholder="Admin note or context…"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
            />
          </div>
        </div>
        <div className="flex justify-end gap-2 px-5 pb-5">
          <Button variant="outline" onClick={onClose} disabled={isSaving}>
            Cancel
          </Button>
          <Button
            onClick={() => onSave(question, answer, comment)}
            disabled={!question.trim() || !answer.trim() || isSaving}
            className="bg-purple-600 hover:bg-purple-700 text-white"
          >
            {isSaving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Plus className="w-4 h-4 mr-2" />}
            Save to Memory
          </Button>
        </div>
      </div>
    </div>
  )
}

// ─── Main page ───────────────────────────────────────────────────────────────
export default function Memory() {
  const queryClient = useQueryClient()
  const [successMsg, setSuccessMsg] = useState('')
  const [errorMsg, setErrorMsg] = useState('')
  const [showAddModal, setShowAddModal] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const showSuccess = (msg: string) => { setSuccessMsg(msg); setErrorMsg(''); setTimeout(() => setSuccessMsg(''), 4000) }
  const showError = (msg: string) => { setErrorMsg(msg); setSuccessMsg(''); setTimeout(() => setErrorMsg(''), 5000) }

  // ── Queries ──────────────────────────────────────────────────────────────
  const { data, isLoading } = useQuery({
    queryKey: ['memory-items'],
    queryFn: () => getMemoryItems(false),
  })

  // ── Mutations ────────────────────────────────────────────────────────────
  const createMutation = useMutation({
    mutationFn: (item: { question: string; answer: string; comment?: string }) =>
      createMemoryItem(item),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory-items'] })
      showSuccess('Memory item saved!')
      setShowAddModal(false)
    },
    onError: () => showError('Failed to save memory item'),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, ...rest }: { id: string; question?: string; answer?: string; comment?: string; is_active?: boolean }) =>
      updateMemoryItem(id, rest),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory-items'] })
      showSuccess('Memory item updated!')
    },
    onError: () => showError('Failed to update memory item'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteMemoryItem(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory-items'] })
      showSuccess('Memory item deleted')
      setConfirmDelete(null)
    },
    onError: () => showError('Failed to delete memory item'),
  })

  const items: MemoryItem[] = data?.items || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-purple-600" />
            </div>
            Memory
          </h1>
          <p className="text-slate-500 mt-1 ml-1">
            Train Elena with long-term Q&amp;A context. Active items are injected into every session.
          </p>
        </div>
        <Button
          onClick={() => setShowAddModal(true)}
          className="bg-purple-600 hover:bg-purple-700 text-white gap-2"
        >
          <Plus className="w-4 h-4" />
          Add Memory
        </Button>
      </div>

      {/* Alerts */}
      {successMsg && (
        <div className="flex items-center gap-2 p-4 text-green-700 bg-green-50 border border-green-200 rounded-lg">
          <CheckCircle className="w-5 h-5 shrink-0" />
          {successMsg}
        </div>
      )}
      {errorMsg && (
        <div className="flex items-center gap-2 p-4 text-red-700 bg-red-50 border border-red-200 rounded-lg">
          <AlertCircle className="w-5 h-5 shrink-0" />
          {errorMsg}
        </div>
      )}

      {/* Stats bar */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="border-0 bg-purple-50">
          <CardContent className="pt-4 pb-4">
            <p className="text-2xl font-bold text-purple-700">{items.length}</p>
            <p className="text-xs text-purple-500 font-medium mt-0.5">Total Entries</p>
          </CardContent>
        </Card>
        <Card className="border-0 bg-green-50">
          <CardContent className="pt-4 pb-4">
            <p className="text-2xl font-bold text-green-700">{items.filter((i) => i.is_active).length}</p>
            <p className="text-xs text-green-500 font-medium mt-0.5">Active (injected)</p>
          </CardContent>
        </Card>
        <Card className="border-0 bg-slate-50">
          <CardContent className="pt-4 pb-4">
            <p className="text-2xl font-bold text-slate-500">{items.filter((i) => !i.is_active).length}</p>
            <p className="text-xs text-slate-400 font-medium mt-0.5">Inactive</p>
          </CardContent>
        </Card>
      </div>

      {/* Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <MessageSquare className="w-4 h-4 text-purple-500" />
            Memory Entries
          </CardTitle>
          <CardDescription>
            Click any cell in Question, Answer, or Comment to edit inline. Toggle the switch to activate/deactivate.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
            </div>
          ) : items.length === 0 ? (
            <div className="text-center py-16 text-slate-500">
              <Brain className="w-12 h-12 mx-auto mb-3 text-slate-200" />
              <p className="font-medium">No memory entries yet</p>
              <p className="text-sm mt-1 text-slate-400">Click "Add Memory" to start training Elena.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-slate-50 text-left">
                    <th className="px-4 py-3 font-semibold text-slate-600 w-[28%]">Question</th>
                    <th className="px-4 py-3 font-semibold text-slate-600 w-[32%]">Answer</th>
                    <th className="px-4 py-3 font-semibold text-slate-600 w-[22%]">Comment</th>
                    <th className="px-4 py-3 font-semibold text-slate-600 text-center w-[10%]">Active</th>
                    <th className="px-4 py-3 font-semibold text-slate-600 text-center w-[8%]">Delete</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {items.map((item) => (
                    <tr
                      key={item.id}
                      className={`hover:bg-slate-50 transition-colors ${!item.is_active ? 'opacity-50' : ''}`}
                    >
                      {/* Question */}
                      <td className="px-4 py-3 align-top">
                        <EditableCell
                          value={item.question}
                          multiline
                          placeholder="Click to edit question…"
                          onSave={(v) => updateMutation.mutate({ id: item.id, question: v })}
                        />
                      </td>

                      {/* Answer */}
                      <td className="px-4 py-3 align-top">
                        <EditableCell
                          value={item.answer}
                          multiline
                          placeholder="Click to edit answer…"
                          onSave={(v) => updateMutation.mutate({ id: item.id, answer: v })}
                        />
                      </td>

                      {/* Comment */}
                      <td className="px-4 py-3 align-top">
                        <EditableCell
                          value={item.comment || ''}
                          placeholder="Add a note…"
                          onSave={(v) => updateMutation.mutate({ id: item.id, comment: v })}
                        />
                      </td>

                      {/* Active toggle */}
                      <td className="px-4 py-3 align-top text-center">
                        <button
                          onClick={() => updateMutation.mutate({ id: item.id, is_active: !item.is_active })}
                          className="inline-flex items-center justify-center"
                          title={item.is_active ? 'Deactivate' : 'Activate'}
                        >
                          {item.is_active ? (
                            <ToggleRight className="w-7 h-7 text-green-500 hover:text-green-600" />
                          ) : (
                            <ToggleLeft className="w-7 h-7 text-slate-300 hover:text-slate-500" />
                          )}
                        </button>
                      </td>

                      {/* Delete */}
                      <td className="px-4 py-3 align-top text-center">
                        {confirmDelete === item.id ? (
                          <div className="flex items-center gap-1 justify-center">
                            <button
                              onClick={() => deleteMutation.mutate(item.id)}
                              className="p-1 text-red-600 hover:bg-red-50 rounded"
                              disabled={deleteMutation.isPending}
                            >
                              <Check className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => setConfirmDelete(null)}
                              className="p-1 text-slate-400 hover:bg-slate-100 rounded"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setConfirmDelete(item.id)}
                            className="p-1 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add Modal */}
      {showAddModal && (
        <AddMemoryModal
          onClose={() => setShowAddModal(false)}
          onSave={(q, a, c) => createMutation.mutate({ question: q, answer: a, comment: c || undefined })}
          isSaving={createMutation.isPending}
        />
      )}

    </div>
  )
}
