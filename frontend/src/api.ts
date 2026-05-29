export type DocumentItem = {
  doc_id: string
  title?: string | null
  status?: string | null
  metadata?: Record<string, unknown>
}

export type QuerySource = {
  title: string
  type: string
  content: string
  relevance_score: number
}

export type QueryResponse = {
  answer: string
  sources: QuerySource[]
  latency_ms: number
}

export type StreamMeta = {
  type: 'meta'
  query: string
  candidates: Array<Record<string, unknown>>
  final: Array<Record<string, unknown>>
}

export type StreamToken = {
  type: 'token'
  content: string
}

export type StreamDone = {
  type: 'done'
}

export type StreamEvent = StreamMeta | StreamToken | StreamDone

export type IngestionStatusItem = {
  doc_id: string
  title?: string | null
  state?: string
  stage?: string
  detail?: string
  updated_at?: string | null
  timeline?: Array<Record<string, unknown>>
}

export type DeleteDocumentsResponse = {
  deleted: string[]
  status: string
}

export type RenameDocumentResponse = DocumentItem

const headers = { 'Content-Type': 'application/json' }

export class ApiClient {
  constructor(private baseUrl: string) {}

  async upload(files: File[]) {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    const response = await fetch(`${this.baseUrl}/upload`, { method: 'POST', body: form })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async ingestionStatus(docId: string): Promise<IngestionStatusItem> {
    const response = await fetch(`${this.baseUrl}/ingestion/${docId}`)
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async query(payload: { query: string; top_k: number; doc_ids: string[]; rerank: boolean; search_mode: string }): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async documents(): Promise<DocumentItem[]> {
    const response = await fetch(`${this.baseUrl}/documents`)
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async document(docId: string): Promise<DocumentItem> {
    const response = await fetch(`${this.baseUrl}/documents/${docId}`)
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async deleteDocument(docId: string) {
    const response = await fetch(`${this.baseUrl}/documents/${docId}`, { method: 'DELETE' })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async deleteDocuments(docIds?: string[]) {
    const url = new URL(`${this.baseUrl}/documents`)
    if (docIds?.length) docIds.forEach((id) => url.searchParams.append('doc_ids', id))
    const response = await fetch(url.toString(), { method: 'DELETE' })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async clearAllDocuments(): Promise<DeleteDocumentsResponse> {
    const url = new URL(`${this.baseUrl}/documents`)
    url.searchParams.set('clear_all', 'true')
    const response = await fetch(url.toString(), { method: 'DELETE' })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async renameDocument(docId: string, title: string): Promise<RenameDocumentResponse> {
    const response = await fetch(`${this.baseUrl}/documents/${docId}`, {
      method: 'PATCH',
      headers,
      body: JSON.stringify({ title }),
    })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async ingestionStatuses(): Promise<{ items: IngestionStatusItem[] }> {
    const response = await fetch(`${this.baseUrl}/ingestion`)
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  streamUrl(payload: { query: string; topK: number; docIds: string[]; rerank: boolean; searchMode: string }) {
    const url = new URL(`${this.baseUrl}/answer/stream`)
    url.searchParams.set('q', payload.query)
    url.searchParams.set('top_k_initial', String(Math.max(payload.topK, 10)))
    url.searchParams.set('top_k_final', String(payload.topK))
    url.searchParams.set('rerank', String(payload.rerank))
    url.searchParams.set('search_mode', payload.searchMode)
    payload.docIds.forEach((id) => url.searchParams.append('doc_ids', id))
    return url.toString()
  }
}
