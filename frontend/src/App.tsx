import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { ApiClient, DocumentItem, IngestionStatusItem, QuerySource } from './api'

const DEFAULT_BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'
const EXAMPLES = [
  'What is the main contribution of this paper?',
  'What dataset was used in the experiments?',
  'Summarize the method in two sentences.',
]

type Message = { role: 'user' | 'assistant'; content: string; sources?: QuerySource[] }
type Toast = { id: number; message: string; type: 'error' | 'success' }

export default function App() {
  const client = useMemo(() => new ApiClient(DEFAULT_BACKEND_URL), [])
  const [docs, setDocs] = useState<DocumentItem[]>([])
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [activeChatDocIds, setActiveChatDocIds] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [sources, setSources] = useState<QuerySource[]>([])
  const [loading, setLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [streamingAnswer, setStreamingAnswer] = useState('')
  const [uploadStatus, setUploadStatus] = useState<IngestionStatusItem[]>([])
  const [uploadNotice, setUploadNotice] = useState('')
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(true)
  const [searchMode, setSearchMode] = useState<'hybrid' | 'simple'>('hybrid')
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [activeSources, setActiveSources] = useState<QuerySource[] | null>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const activeChatDocObjects = docs.filter((doc) => activeChatDocIds.includes(doc.doc_id))

  const addToast = (message: string, type: 'error' | 'success' = 'error') => {
    const id = Date.now()
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 5000)
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingAnswer])

  useEffect(() => {
    client.documents().then(setDocs).catch(() => setDocs([]))
  }, [client])

  const refreshDocs = async () => {
    try {
      const next = await client.documents()
      setDocs(next)
    } catch {
      addToast('Failed to refresh documents')
    }
  }

  const refreshIngestionStatus = async () => {
    try {
      const data = await client.ingestionStatuses()
      setUploadStatus(data.items ?? [])
    } catch {
      // Background poll, no toast
    }
  }

  useEffect(() => {
    refreshIngestionStatus()
  }, [])

  const previousActiveRef = useRef(false)

  useEffect(() => {
    const active = uploadStatus.some((item) => item.state === 'queued' || item.state === 'running')
    if (previousActiveRef.current && !active) {
      addToast('Document processing completed!', 'success')
      setUploadNotice('')
    }
    previousActiveRef.current = active
    if (!active) return
    const timer = window.setInterval(() => {
      refreshIngestionStatus()
      refreshDocs()
    }, 1500)
    return () => window.clearInterval(timer)
  }, [uploadStatus])

  const uploadFiles = async (files: FileList | File[] | null) => {
    if (!files?.length) return
    setIsUploading(true)
    setUploadNotice('Uploading documents...')
    try {
      await client.upload(Array.from(files))
      setUploadNotice('Processing upload pipeline...')
      await refreshIngestionStatus()
      await refreshDocs()
    } catch (err: any) {
      addToast(err.message || 'Failed to upload files')
      setUploadNotice('')
    } finally {
      setIsUploading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.length) {
      uploadFiles(e.dataTransfer.files)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runQuery(query)
    }
  }

  const runQuery = async (question: string) => {
    if (!question.trim()) return
    setLoading(true)
    setSources([])
    setStreamingAnswer('')
    setQuery('') // Auto-clear
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    let assistantText = ''
    try {
      const response = await fetch(client.streamUrl({ query: question, topK, docIds: activeChatDocIds, rerank, searchMode }))
      if (!response.ok || !response.body) throw new Error(await response.text())
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let latestSources: QuerySource[] = []

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
            const payload = JSON.parse(lineText) as { type: string; content?: string; final?: Array<Record<string, unknown>> }
            if (payload.type === 'meta') {
              latestSources = (payload.final ?? []).map((item) => ({
                title: String(item.doc_title || item.doc_id || ''),
                type: String(item.type || 'text'),
                content: String(item.content || ''),
                relevance_score: Number(item.rerank_score || item.rrf_score || 0),
              }))
              setSources(latestSources)
            } else if (payload.type === 'token' && payload.content) {
              assistantText += payload.content
              setStreamingAnswer(assistantText)
            }
          } catch (_parseErr) {
            // Skip malformed SSE payloads
          }
        }
      }
      setMessages((prev) => [...prev, { role: 'assistant', content: assistantText || 'No response received.', sources: latestSources }])
      setStreamingAnswer('')
      setActiveSources(latestSources.length ? latestSources : null)
    } catch (err: any) {
      console.error(err)
      addToast(err.message || 'Failed to get answer')
      if (!assistantText) {
        setMessages((prev) => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error answering your question.' }])
      }
    } finally {
      setLoading(false)
    }
  }

  const toggleDoc = (docId: string) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    )
  }

  const scopeToDoc = (docId: string) => {
    setActiveChatDocIds([docId])
  }

  const clearScope = () => {
    setActiveChatDocIds([])
  }

  const deleteSelected = async () => {
    if (!selectedDocs.length) return
    try {
      await client.deleteDocuments(selectedDocs)
      setSelectedDocs([])
      setActiveChatDocIds((prev) => prev.filter((id) => !selectedDocs.includes(id)))
      await refreshDocs()
      addToast('Documents deleted successfully', 'success')
    } catch {
      addToast('Failed to delete documents')
    }
  }

  const activeUploads = uploadStatus.filter((item) => item.state === 'queued' || item.state === 'running')
  const doneUploads = uploadStatus.filter((item) => item.state === 'done')

  return (
    <div className="app-shell" onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}>
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">RA</div>
          <div>
            <h2>RAG Analyst</h2>
            <p>Pro Workspace</p>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button type="button" className="sidebar-link active"><span className="material-symbols-outlined">description</span><span>Documents</span></button>
          <button type="button" className="sidebar-link" onClick={() => { setMessages([]); setSources([]) }}><span className="material-symbols-outlined">history</span><span>Clear Chat</span></button>
        </nav>

        <button className="primary-cta" onClick={() => document.getElementById('upload-input')?.click()} disabled={isUploading}>
          <span className="material-symbols-outlined">upload_file</span>
          {isUploading ? 'Uploading...' : 'Upload Document'}
        </button>

        {(isUploading || activeUploads.length > 0) && (
          <section id="library-anchor">
            <h3>Current Pipeline</h3>
            {uploadNotice && <div className="status-banner">{uploadNotice}</div>}
            <div className="pipeline-card">
              <div className="pipeline-step done">
                <span className="material-symbols-outlined">check_circle</span>
                <div><strong>Upload complete</strong><span>100% - upload finished</span></div>
              </div>
              <div className={`pipeline-step ${activeUploads.length ? 'active' : 'done'}`}>
                <span className="material-symbols-outlined">sync</span>
                <div><strong>{activeUploads.length ? 'Embedding...' : 'Chunking complete'}</strong><span>{activeUploads.length ? activeUploads[0]?.detail || 'Working through ingestion' : `${doneUploads.reduce((acc, item) => acc + Number(item.timeline?.length || 0), 0)} chunks created`}</span></div>
              </div>
            </div>
          </section>
        )}

        <section>
          <h3>Library</h3>
          <div className="library-toolbar">
            {selectedDocs.length > 0 && (
              <button className="icon-button" onClick={() => setShowDeleteModal(true)} title="Delete selected">
                <span className="material-symbols-outlined">delete</span>
              </button>
            )}
            <button className="icon-button" onClick={() => refreshDocs()} title="Refresh">
              <span className="material-symbols-outlined">refresh</span>
            </button>
          </div>
          
          {docs.length === 0 ? (
            <div className="empty-library">
              No documents yet.
            </div>
          ) : (
            <div className="doc-list">
              {docs.map((doc) => (
                <div key={doc.doc_id} className={`doc-card ${selectedDocs.includes(doc.doc_id) ? 'selected' : ''}`}>
                  <label className="doc-card-main">
                    <input type="checkbox" checked={selectedDocs.includes(doc.doc_id)} onChange={() => toggleDoc(doc.doc_id)} />
                    <span className="material-symbols-outlined doc-icon">picture_as_pdf</span>
                    <div className="doc-copy">
                      <div className="doc-title">{doc.title || doc.doc_id}</div>
                      <div className="doc-meta">{doc.metadata?.chunks ?? 0} chunks</div>
                    </div>
                    {activeChatDocIds.includes(doc.doc_id) && <span className="check-badge material-symbols-outlined">check</span>}
                  </label>
                  <div className="doc-actions">
                    <button type="button" className="ghost-button" onClick={() => scopeToDoc(doc.doc_id)}>Use for chat</button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="retrieval-config">
            <div className="config-row">
              <span>Mode</span>
              <div className="segmented">
                <button className={searchMode === 'hybrid' ? 'seg-active' : ''} onClick={() => setSearchMode('hybrid')}>Hybrid</button>
                <button className={searchMode === 'simple' ? 'seg-active' : ''} onClick={() => setSearchMode('simple')}>Simple</button>
              </div>
            </div>
            <div className="config-row">
              <span>Top-K</span>
              <span className="mono-pill">{topK}</span>
            </div>
            <input type="range" min={1} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
            <label className="toggle-row">
              <span>Reranker</span>
              <label className="toggle-switch">
                <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
                <span className="slider"></span>
              </label>
            </label>
          </div>
        </section>

        <input id="upload-input" className="hidden-input" type="file" multiple accept="application/pdf" onChange={(e) => uploadFiles(e.target.files)} />
      </aside>

      <main className={`main ${isDragging ? 'drag-over' : ''}`}>
        <header className="topbar">
          <div className="topbar-brand">
            <span className="material-symbols-outlined" style={{color: 'var(--accent-primary)'}}>hub</span>
            RAG Analyst
          </div>
          <nav className="topbar-nav">
            <span>Models</span>
            <span className="active">Knowledge Base</span>
            <span>Analytics</span>
          </nav>
          <div className="topbar-actions">
            {activeSources && <button className="ghost-button" onClick={() => setActiveSources(null)}>Close References</button>}
            <button className="new-session-btn" onClick={() => { setMessages([]); setActiveSources(null); setStreamingAnswer(''); setQuery('') }}>New Session</button>
          </div>
        </header>

        <div className="context-bar">
          <span className="context-label">Context:</span>
          {activeChatDocObjects.length ? (
            <div className="context-chip">
              <span className="material-symbols-outlined">picture_as_pdf</span>
              <span>{activeChatDocObjects[0].title || activeChatDocObjects[0].doc_id}</span>
              <button className="chip-close" onClick={clearScope}><span className="material-symbols-outlined">close</span></button>
            </div>
          ) : (
            <div className="context-empty">Global scope (all documents)</div>
          )}
        </div>

        <section className="canvas">
          {activeChatDocIds.length === 0 && messages.length === 0 && !streamingAnswer && (
            <div className="empty-state">
              <div className="empty-card">
                <div className="empty-icon"><span className="material-symbols-outlined">find_in_page</span></div>
                <h2>Explore your documents</h2>
                <p>Choose a document from the sidebar or upload a new one to start chatting and generating insights.</p>
                <div className="example-prompts">
                  {EXAMPLES.map((ex, idx) => (
                    <button key={idx} className="example-chip ghost-button" onClick={() => runQuery(ex)}>{ex}</button>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="chat-scroll">
            <div className="messages">
              {messages.map((message, idx) => (
                <div key={idx} className={`message-row ${message.role}`}>
                  <div className={`avatar ${message.role}`}>{message.role === 'user' ? 'You' : 'AI'}</div>
                  <div className={`message-bubble ${message.role}`}>
                    {message.role === 'assistant' ? <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</ReactMarkdown> : <div>{message.content}</div>}
                    {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                      <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-light)', paddingTop: '8px' }}>
                        <button className="ghost-button" onClick={() => setActiveSources(message.sources!)} style={{ fontSize: '12px', padding: '4px 8px' }}>
                          <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>library_books</span> References ({message.sources.length})
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {streamingAnswer && (
                <div className="message-row assistant">
                  <div className="avatar assistant">AI</div>
                  <div className="message-bubble assistant"><ReactMarkdown>{streamingAnswer}</ReactMarkdown></div>
                </div>
              )}
              {loading && !streamingAnswer && (
                <div className="message-row assistant">
                  <div className="avatar assistant">AI</div>
                  <div className="message-bubble assistant">
                    <div className="typing-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
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
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..." 
              rows={1} 
            />
            <div className="composer-row">
              <div className="composer-tools">
                <button className="icon-button" onClick={() => document.getElementById('upload-input')?.click()} title="Upload PDF">
                  <span className="material-symbols-outlined">attach_file</span>
                </button>
              </div>
              <div className="composer-send">
                <button className="send-btn" onClick={() => runQuery(query)} disabled={loading || !query.trim()}>
                  <span className="material-symbols-outlined">arrow_upward</span>
                </button>
              </div>
            </div>
          </div>
          <div className="disclaimer">RAG Analyst AI can make mistakes. Consider verifying important technical details.</div>
        </section>
      </main>

      {activeSources && (
        <aside className="right-panel">
          <div className="right-panel-header">
            <h3>References</h3>
            <button className="icon-button" onClick={() => setActiveSources(null)}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
          <div className="right-panel-content">
            {activeSources.map((src, idx) => (
              <div key={idx} className="reference-card">
                <h4>{src.title}</h4>
                <div className="ref-score">Relevance: {(src.relevance_score * 100).toFixed(1)}%</div>
                <p>{src.content}</p>
                <button className="ghost-button copy-btn" onClick={() => { navigator.clipboard.writeText(src.content); addToast('Copied to clipboard', 'success'); }}>
                  Copy text
                </button>
              </div>
            ))}
          </div>
        </aside>
      )}

      {showDeleteModal && (
        <div className="modal-backdrop" onClick={() => setShowDeleteModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-icon"><span className="material-symbols-outlined">warning</span></div>
            <h3>Delete Document?</h3>
            <p>This action cannot be undone. All embeddings and chat history for the selected documents will be permanently removed.</p>
            <div className="button-row">
              <button className="ghost-button" onClick={() => setShowDeleteModal(false)}>Cancel</button>
              <button className="danger-btn" onClick={async () => { await deleteSelected(); setShowDeleteModal(false) }}>Delete</button>
            </div>
          </div>
        </div>
      )}

      <div className="toast-container">
        {toasts.map(toast => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span className="material-symbols-outlined">{toast.type === 'error' ? 'error' : 'check_circle'}</span>
            <p>{toast.message}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
