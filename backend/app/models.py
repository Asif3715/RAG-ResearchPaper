from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    doc_id: str
    status: str
    title: str | None = None


class IngestionStatusItem(BaseModel):
    doc_id: str
    title: str | None = None
    state: str = "queued"
    stage: str = "queued"
    detail: str = ""
    updated_at: str | None = None
    timeline: list[dict[str, Any]] = Field(default_factory=list)


class DeleteDocumentsResponse(BaseModel):
    deleted: list[str]
    status: str


class RenameDocumentRequest(BaseModel):
    title: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: str = Field(default="hybrid")
    rerank: bool = Field(default=True)
    doc_ids: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    title: str = ""
    type: str = "text"
    content: str = ""
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: int


class DocumentItem(BaseModel):
    doc_id: str
    title: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    details: dict[str, Any] = Field(default_factory=dict)
