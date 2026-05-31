import {
  Check,
  FileText,
  Loader2,
  MoreHorizontal,
  RefreshCw,
  Trash2,
  Upload,
  X,
  ExternalLink,
} from 'lucide-react'
import { DocumentItem, IngestionStatusItem } from '../api'
import { DismissLayer } from './DismissLayer'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'

type Props = {
  open: boolean
  onClose: () => void
  docs: DocumentItem[]
  activeChatDocIds: string[]
  uploadStatus: IngestionStatusItem[]
  selectMode: boolean
  selectedDocs: string[]
  isUploading: boolean
  openMenuId: string | null
  onOpenMenu: (docId: string | null) => void
  onScope: (docId: string) => void
  onToggleSelect: (docId: string) => void
  onDelete: (docId: string) => void
  onRename: (docId: string, title: string) => void
  onRefresh: () => void
  onUploadClick: () => void
  onToggleSelectMode: () => void
}

function docStatusLabel(doc: DocumentItem, ingestion?: IngestionStatusItem): string {
  if (ingestion && (ingestion.state === 'queued' || ingestion.state === 'running')) {
    return `${capitalize(ingestion.stage || 'Processing')}…`
  }
  if (doc.status === 'error') return 'Error'
  const chunks = Number(doc.metadata?.chunks ?? 0)
  if (chunks > 0) return `Indexed · ${chunks} sections`
  if (doc.status === 'done') return 'Indexed'
  return doc.status || 'Ready'
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

export function DocumentList({
  open,
  onClose,
  docs,
  activeChatDocIds,
  uploadStatus,
  selectMode,
  selectedDocs,
  isUploading,
  openMenuId,
  onOpenMenu,
  onScope,
  onToggleSelect,
  onDelete,
  onRename,
  onRefresh,
  onUploadClick,
  onToggleSelectMode,
}: Props) {
  const activeUploads = uploadStatus.filter((item) => item.state === 'queued' || item.state === 'running')
  const ingestionByDoc = new Map(uploadStatus.map((item) => [item.doc_id, item]))

  return (
    <aside className={`sidebar ${open ? 'open' : ''}`} aria-hidden={!open}>
      <div className="sidebar-header">
        <div className="brand-block">
          <div className="brand-mark">
            <FileText size={18} strokeWidth={1.75} />
          </div>
          <div>
            <h1>Paper Assistant</h1>
            <p>Sources</p>
          </div>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close sources panel">
          <X size={18} />
        </button>
      </div>

      <button type="button" className="upload-btn" onClick={onUploadClick} disabled={isUploading}>
        <span className="upload-btn-icon">
          <Upload size={18} strokeWidth={1.75} />
        </span>
        <span className="upload-btn-text">
          <strong>{isUploading ? 'Uploading…' : 'Upload PDF'}</strong>
          <span>Drop a file or click to browse</span>
        </span>
      </button>

      {activeUploads.length > 0 && (
        <section className="ingestion-panel">
          <h3>Processing</h3>
          <div className="ingestion-list">
            {activeUploads.map((item) => (
              <div key={item.doc_id} className="ingestion-row">
                <Loader2 size={14} className="spin" />
                <div className="ingestion-copy">
                  <strong>{item.title || 'Document'}</strong>
                  <span>{item.detail || `${item.stage || 'Working'}…`}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="sources-section">
        <div className="section-header">
          <h3>Sources</h3>
          <div className="section-actions">
            <button type="button" className="text-button" onClick={onToggleSelectMode}>
              {selectMode ? 'Done' : 'Select'}
            </button>
            <button type="button" className="icon-button" onClick={onRefresh} title="Refresh">
              <RefreshCw size={15} />
            </button>
          </div>
        </div>

        {docs.length === 0 ? (
          <div className="empty-library">Upload a PDF to ask questions about it.</div>
        ) : (
          <div className="doc-list">
            {docs.map((doc) => {
              const scoped = activeChatDocIds.includes(doc.doc_id)
              const selected = selectedDocs.includes(doc.doc_id)
              const ingestion = ingestionByDoc.get(doc.doc_id)
              const status = docStatusLabel(doc, ingestion)
              const isError = doc.status === 'error' || ingestion?.state === 'error'

              return (
                <div
                  key={doc.doc_id}
                  className={`doc-row ${scoped ? 'scoped' : ''} ${selected ? 'selected' : ''}`}
                >
                  {selectMode && (
                    <input
                      type="checkbox"
                      className="doc-select"
                      checked={selected}
                      onChange={() => onToggleSelect(doc.doc_id)}
                      aria-label={`Select ${doc.title || doc.doc_id}`}
                    />
                  )}
                  <button
                    type="button"
                    className="doc-row-main"
                    onClick={() => (selectMode ? onToggleSelect(doc.doc_id) : onScope(doc.doc_id))}
                  >
                    <FileText size={16} className="doc-icon" />
                    <div className="doc-copy">
                      <div className="doc-title">{doc.title || doc.doc_id}</div>
                      <div className={`doc-meta ${isError ? 'error' : ''}`}>{status}</div>
                    </div>
                    {scoped && !selectMode && <Check size={16} className="scoped-badge" />}
                  </button>
                  <div className="doc-menu-wrap">
                    <button
                      type="button"
                      className="icon-button"
                      aria-label="Document actions"
                      onClick={(e) => {
                        e.stopPropagation()
                        onOpenMenu(openMenuId === doc.doc_id ? null : doc.doc_id)
                      }}
                    >
                      <MoreHorizontal size={16} />
                    </button>
                    {openMenuId === doc.doc_id && (
                      <>
                        <DismissLayer onDismiss={() => onOpenMenu(null)} />
                        <div className="doc-menu">
                          <a
                            href={`${BACKEND_URL}/documents/${doc.doc_id}/pdf`}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '8px' }}
                            onClick={() => onOpenMenu(null)}
                          >
                            <ExternalLink size={14} />
                            View PDF
                          </a>
                          <button
                            type="button"
                            onClick={() => {
                              onOpenMenu(null)
                              const title = window.prompt('Rename document', doc.title || doc.doc_id)
                              if (title?.trim()) onRename(doc.doc_id, title.trim())
                            }}
                          >
                            Rename
                          </button>
                          <button
                            type="button"
                            className="danger-text"
                            onClick={() => {
                              onOpenMenu(null)
                              onDelete(doc.doc_id)
                            }}
                          >
                            <Trash2 size={14} />
                            Delete
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>
    </aside>
  )
}
