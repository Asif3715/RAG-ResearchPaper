import React, { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowUp, MessageSquarePlus, PanelLeft, Paperclip, Upload, X } from 'lucide-react'
import { ApiClient, DocumentItem, IngestionStatusItem } from './api'
import { AdvancedSettings } from './components/AdvancedSettings'
import { ChatMessage, Message, StreamingMessage } from './components/ChatMessage'
import { DocumentList } from './components/DocumentList'

const DEFAULT_BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'

const PLACEHOLDERS = [
  'Ask about methodology…',
  'What is the main contribution?',
  'Summarize the experimental setup…',
  'Compare results across sections…',
]

const STARTER_PROMPTS = [
  'What is the main contribution of this paper?',
  'What dataset was used in the experiments?',
  'Summarize the method in two sentences.',
]

type Toast = { id: number; message: string; type: 'error' | 'success' }

function formatApiError(text: string): string {
  try {
    const body = JSON.parse(text) as { detail?: string | Array<{ msg?: string }> }
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg || JSON.stringify(item)).join('; ')
    }
  } catch {
    // not JSON
  }
  return text.trim() || 'Failed to get answer'
}

export default function App() {
  const client = useMemo(() => new ApiClient(DEFAULT_BACKEND_URL), [])
  const [docs, setDocs] = useState<DocumentItem[]>([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [selectMode, setSelectMode] = useState(false)
  const [activeChatDocIds, setActiveChatDocIds] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [streamingAnswer, setStreamingAnswer] = useState('')
  const [uploadStatus, setUploadStatus] = useState<IngestionStatusItem[]>([])
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(true)
  const [searchMode, setSearchMode] = useState<'hybrid' | 'simple'>('hybrid')
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [expandedCitations, setExpandedCitations] = useState<Record<number, boolean>>({})
  const [placeholderIdx, setPlaceholderIdx] = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const uploadInputRef = useRef<HTMLInputElement>(null)
  const previousActiveRef = useRef(false)

  const activeChatDocObjects = docs.filter((doc) => activeChatDocIds.includes(doc.doc_id))

  const addToast = (message: string, type: 'error' | 'success' = 'error') => {
    const id = Date.now()
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 5000)
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingAnswer])

  useEffect(() => {
    setDocsLoading(true)
    client
      .documents()
      .then(setDocs)
      .catch(() => setDocs([]))
      .finally(() => setDocsLoading(false))
  }, [client])

  useEffect(() => {
    refreshIngestionStatus()
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setPlaceholderIdx((idx) => (idx + 1) % PLACEHOLDERS.length)
    }, 6000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!sidebarOpen) {
      setOpenMenuId(null)
    }
  }, [sidebarOpen])

  const toggleSidebar = () => {
    setSidebarOpen((prev) => {
      if (prev) {
        setOpenMenuId(null)
      }
      return !prev
    })
  }

  useEffect(() => {
    const active = uploadStatus.some((item) => item.state === 'queued' || item.state === 'running')
    if (previousActiveRef.current && !active) {
      addToast('Document processing completed', 'success')
    }
    previousActiveRef.current = active
    if (!active) return
    const timer = window.setInterval(() => {
      refreshIngestionStatus()
      refreshDocs()
    }, 1500)
    return () => window.clearInterval(timer)
  }, [uploadStatus])

  const refreshDocs = async () => {
    try {
      setDocs(await client.documents())
    } catch {
      addToast('Failed to refresh documents')
    } finally {
      setDocsLoading(false)
    }
  }

  const refreshIngestionStatus = async () => {
    try {
      const data = await client.ingestionStatuses()
      setUploadStatus(data.items ?? [])
    } catch {
      // background poll
    }
  }

  const uploadFiles = async (files: FileList | File[] | null) => {
    if (!files?.length) return
    setIsUploading(true)
    try {
      await client.upload(Array.from(files))
      await refreshIngestionStatus()
      await refreshDocs()
      addToast('Upload started', 'success')
    } catch (err: unknown) {
      addToast(err instanceof Error ? err.message : 'Failed to upload files')
    } finally {
      setIsUploading(false)
    }
  }

  const newChat = () => {
    setMessages([])
    setStreamingAnswer('')
    setQuery('')
    setExpandedCitations({})
  }

  const runQuery = async (question: string) => {
    if (!question.trim() || loading) return
    setLoading(true)
    setStreamingAnswer('')
    setQuery('')
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    let assistantText = ''
    try {
      const response = await fetch(
        client.streamUrl({ query: question, topK, docIds: activeChatDocIds, rerank, searchMode }),
      )
      if (!response.ok || !response.body) throw new Error(await response.text())
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let latestSources: Message['sources'] = []

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''
        for (const event of events) {
          const line = event.split('\n').find((entry) => entry.startsWith('data: '))
          if (!line) continue
          const lineText = line.slice(6).trim()
          if (lineText === '[DONE]') continue
          try {
            const payload = JSON.parse(lineText) as {
              type: string
              content?: string
              final?: Array<Record<string, unknown>>
            }
            if (payload.type === 'meta') {
              latestSources = (payload.final ?? []).map((item) => ({
                title: String(item.doc_title || item.doc_id || ''),
                type: String(item.type || 'text'),
                content: String(item.content || ''),
                relevance_score: Number(item.rerank_score || item.rrf_score || 0),
              }))
            } else if (payload.type === 'token' && payload.content) {
              assistantText += payload.content
              setStreamingAnswer(assistantText)
            }
          } catch {
            // skip malformed SSE
          }
        }
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: assistantText || 'No response received.',
        sources: latestSources,
      }
      setMessages((prev) => [...prev, assistantMessage])
      setStreamingAnswer('')
    } catch (err: unknown) {
      const message = formatApiError(err instanceof Error ? err.message : String(err))
      addToast(message)
      if (!assistantText) {
        setMessages((prev) => [...prev, { role: 'assistant', content: message }])
      }
    } finally {
      setLoading(false)
    }
  }

  const scopeToDoc = (docId: string) => {
    setActiveChatDocIds((prev) => (prev.includes(docId) && prev.length === 1 ? [] : [docId]))
  }

  const toggleSelect = (docId: string) => {
    setSelectedDocs((prev) => (prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]))
  }

  const deleteDoc = async (docId: string) => {
    if (!window.confirm('Delete this document and all its indexed sections?')) return
    try {
      await client.deleteDocument(docId)
      setActiveChatDocIds((prev) => prev.filter((id) => id !== docId))
      setSelectedDocs((prev) => prev.filter((id) => id !== docId))
      await refreshDocs()
      addToast('Document deleted', 'success')
    } catch {
      addToast('Failed to delete document')
    }
  }

  const deleteSelected = async () => {
    if (!selectedDocs.length) return
    const toDelete = [...selectedDocs]
    try {
      await Promise.all(toDelete.map((docId) => client.deleteDocument(docId)))
      setSelectedDocs([])
      setSelectMode(false)
      setActiveChatDocIds((prev) => prev.filter((id) => !toDelete.includes(id)))
      await refreshDocs()
      addToast('Documents deleted', 'success')
    } catch {
      addToast('Failed to delete documents')
    }
  }

  const renameDoc = async (docId: string, title: string) => {
    try {
      await client.renameDocument(docId, title)
      await refreshDocs()
      addToast('Document renamed', 'success')
    } catch {
      addToast('Failed to rename document')
    }
  }

  const showEmptyChat = !docsLoading && messages.length === 0 && !streamingAnswer
  const hasDocs = docs.length > 0

  return (
    <div
      className="app-shell"
      onDrop={(e) => {
        e.preventDefault()
        setIsDragging(false)
        if (e.dataTransfer.files?.length) uploadFiles(e.dataTransfer.files)
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        setIsDragging(false)
      }}
    >
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} aria-hidden="true" />}

      <DocumentList
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        docs={docs}
        activeChatDocIds={activeChatDocIds}
        uploadStatus={uploadStatus}
        selectMode={selectMode}
        selectedDocs={selectedDocs}
        isUploading={isUploading}
        openMenuId={openMenuId}
        onOpenMenu={setOpenMenuId}
        onScope={scopeToDoc}
        onToggleSelect={toggleSelect}
        onDelete={deleteDoc}
        onRename={renameDoc}
        onRefresh={refreshDocs}
        onUploadClick={() => uploadInputRef.current?.click()}
        onToggleSelectMode={() => {
          setSelectMode((prev) => !prev)
          setSelectedDocs([])
        }}
      />

      <main className={`main ${isDragging ? 'drag-over' : ''}`}>
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className={`sources-toggle ${sidebarOpen ? 'active' : ''}`}
              onClick={toggleSidebar}
              aria-expanded={sidebarOpen}
              aria-label={sidebarOpen ? 'Hide sources' : 'Show sources'}
            >
              <PanelLeft size={18} />
              <span>Sources</span>
              {docs.length > 0 && <span className="sources-count">{docs.length}</span>}
            </button>
            <div className="scope-bar">
            {activeChatDocObjects.length ? (
              <>
                <span className="scope-label">Scoped to</span>
                <div className="scope-chip">
                  <span>{activeChatDocObjects[0].title || activeChatDocObjects[0].doc_id}</span>
                  <button type="button" className="icon-button" onClick={() => setActiveChatDocIds([])} aria-label="Clear scope">
                    <X size={14} />
                  </button>
                </div>
              </>
            ) : (
              <span className="scope-label muted">All sources</span>
            )}
            </div>
          </div>
          <div className="topbar-actions">
            {selectMode && selectedDocs.length > 0 && (
              <button type="button" className="danger-btn subtle" onClick={() => setShowDeleteModal(true)}>
                Delete {selectedDocs.length}
              </button>
            )}
            <button type="button" className="ghost-button" onClick={newChat}>
              <MessageSquarePlus size={16} />
              New chat
            </button>
          </div>
        </header>

        <section className="canvas">
          {showEmptyChat && (
            <div className="empty-state">
              {!hasDocs ? (
                <>
                  <h2>Upload a paper to begin</h2>
                  <p>Drop a PDF here or open Sources to upload. Then ask focused questions about methods, results, and claims.</p>
                  <button
                    type="button"
                    className="upload-btn upload-btn-centered"
                    onClick={() => {
                      setSidebarOpen(true)
                      uploadInputRef.current?.click()
                    }}
                  >
                    <span className="upload-btn-icon">
                      <Upload size={18} />
                    </span>
                    <span className="upload-btn-text">
                      <strong>Upload PDF</strong>
                      <span>Or drag and drop into the window</span>
                    </span>
                  </button>
                </>
              ) : (
                <>
                  <h2>Ask about your papers</h2>
                  <p>
                    {activeChatDocObjects.length
                      ? `Questions are scoped to “${activeChatDocObjects[0].title || activeChatDocObjects[0].doc_id}”.`
                      : 'Open Sources to scope a single paper, or ask across your full library.'}
                  </p>
                  <div className="starter-prompts">
                    {STARTER_PROMPTS.map((prompt) => (
                      <button key={prompt} type="button" className="starter-chip" onClick={() => runQuery(prompt)}>
                        {prompt}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          <div className="chat-scroll">
            <div className="messages">
              {messages.map((message, idx) => (
                <ChatMessage
                  key={idx}
                  message={message}
                  citationsExpanded={expandedCitations[idx] ?? false}
                  onToggleCitations={() =>
                    setExpandedCitations((prev) => ({ ...prev, [idx]: !prev[idx] }))
                  }
                  onCopy={(text) => {
                    navigator.clipboard.writeText(text)
                    addToast('Copied to clipboard', 'success')
                  }}
                />
              ))}
              {(streamingAnswer || (loading && !streamingAnswer)) && (
                <StreamingMessage content={streamingAnswer} loading={loading} />
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </section>

        <section className="composer-shell">
          <div className="composer">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  runQuery(query)
                }
              }}
              placeholder={docs.length ? PLACEHOLDERS[placeholderIdx] : 'Upload a PDF first…'}
              rows={1}
              disabled={docs.length === 0}
            />
            <div className="composer-row">
              <div className="composer-tools">
                <button type="button" className="icon-button" onClick={() => uploadInputRef.current?.click()} title="Upload PDF">
                  <Paperclip size={16} />
                </button>
                <AdvancedSettings
                  open={showAdvanced}
                  onToggle={() => setShowAdvanced((prev) => !prev)}
                  searchMode={searchMode}
                  onSearchMode={setSearchMode}
                  topK={topK}
                  onTopK={setTopK}
                  rerank={rerank}
                  onRerank={setRerank}
                />
              </div>
              <button
                type="button"
                className={`send-btn ${query.trim() ? 'active' : ''}`}
                onClick={() => runQuery(query)}
                disabled={loading || !query.trim() || docs.length === 0}
                aria-label="Send"
              >
                <ArrowUp size={18} />
              </button>
            </div>
          </div>
          <p className="disclaimer">Verify important claims against the original paper.</p>
        </section>
      </main>

      <input
        ref={uploadInputRef}
        className="hidden-input"
        type="file"
        multiple
        accept="application/pdf"
        onChange={(e) => uploadFiles(e.target.files)}
      />

      {showDeleteModal && (
        <div className="modal-backdrop" onClick={() => setShowDeleteModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>Delete selected documents?</h3>
            <p>This removes indexed sections and cannot be undone.</p>
            <div className="button-row">
              <button type="button" className="ghost-button" onClick={() => setShowDeleteModal(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="danger-btn"
                onClick={async () => {
                  await deleteSelected()
                  setShowDeleteModal(false)
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <p>{toast.message}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
