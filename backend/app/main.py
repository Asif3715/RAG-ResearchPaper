from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.dependencies import get_embedding_client, get_groq_generator, get_retrieval_pipeline, get_vector_db
from backend.app.models import (
    DeleteDocumentsResponse,
    DocumentItem,
    HealthResponse,
    IngestionStatusItem,
    QueryRequest,
    QueryResponse,
    RenameDocumentRequest,
    SourceItem,
    UploadResponse,
)
from backend.app.services.embeddings.client import normalize_text
from backend.app.services.indexing.ingest import ingest_parsed_document
from backend.app.services.indexing.cleanup import delete_all_document_artifacts, delete_document_artifacts
from backend.app.services.indexing.registry import list_documents, rename_document_record, upsert_document
from backend.app.services.indexing.status import fail_status, get_status, init_status, list_statuses
from backend.app.services.pdf.parser import parse_pdf_bytes


logger = logging.getLogger("paperassistant.api")
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_vector_db().create_collections()
    except Exception:
        pass
    yield

app = FastAPI(title="RAG Analyst API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LAST_QUERY_AT = 0.0


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        latency = int((time.time() - start) * 1000)
        logger.info("%s %s -> %s in %sms", request.method, request.url.path, response.status_code, latency)
        return response
    except Exception:
        latency = int((time.time() - start) * 1000)
        logger.exception("%s %s failed in %sms", request.method, request.url.path, latency)
        raise


def _rate_limit_query() -> None:
    global _LAST_QUERY_AT
    now = time.time()
    if now - _LAST_QUERY_AT < 2:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before querying again.")
    _LAST_QUERY_AT = now


def _ingest_background(parsed: dict) -> None:
    doc_id = parsed.get("doc_id", "unknown")
    try:
        result = ingest_parsed_document(parsed)
        upsert_document({
            "doc_id": doc_id,
            "title": parsed.get("title"),
            "status": "done",
            "metadata": {"chunks": result.get("ingested", len(parsed.get("chunks", [])))},
        })
    except Exception as exc:
        fail_status(doc_id, f"{type(exc).__name__}: {exc}")
        upsert_document({
            "doc_id": doc_id,
            "title": parsed.get("title"),
            "status": "error",
            "metadata": {"error": str(exc)},
        })


def _safe_list_qdrant_documents() -> list[dict]:
    try:
        return get_vector_db().list_documents()
    except Exception as exc:
        logger.warning("Qdrant unavailable for document listing: %s", exc)
        return []


def _merged_document_sources() -> list[dict]:
    return _safe_list_qdrant_documents() + list_documents() + list_statuses()


def _merge_metadata(current: dict | None, incoming: dict | None) -> dict:
    base = dict(current or {})
    new = dict(incoming or {})
    base_chunks = int(base.get("chunks") or 0)
    new_chunks = int(new.get("chunks") or 0)
    if new_chunks or base_chunks:
        base["chunks"] = max(base_chunks, new_chunks)
    for key, value in new.items():
        if key == "chunks":
            continue
        if key == "error" and base.get("chunks"):
            continue
        if value is not None and value != "":
            base[key] = value
    return base


def _merge_document_items(items: list[dict]) -> list[DocumentItem]:
    merged: dict[str, dict] = {}
    for item in items:
        doc_id = item.get("doc_id")
        if not doc_id:
            continue
        current = merged.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": item.get("title"),
                "status": item.get("status") or item.get("state"),
                "metadata": {},
            },
        )
        if item.get("title") and (not current.get("title") or current.get("title") == doc_id):
            current["title"] = item.get("title")
        incoming_status = item.get("status") or item.get("state")
        if incoming_status:
            if incoming_status != "error" or current.get("status") in (None, "error"):
                current["status"] = incoming_status
        if isinstance(item.get("metadata"), dict):
            current["metadata"] = _merge_metadata(current.get("metadata"), item.get("metadata"))
    return [DocumentItem(**item) for item in merged.values()]


def _retrieval_error_detail(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "Connection refused" in text or "ConnectError" in text:
        return (
            "Cannot connect to Qdrant. Check QDRANT_URL and QDRANT_API_KEY in .env "
            "(for Qdrant Cloud use your cluster HTTPS URL and API key from the cloud console)."
        )
    if "Not found" in text and "collection" in text.lower():
        return "Qdrant collections are missing. Re-upload a PDF or restart the API after Qdrant is running."
    return f"Retrieval failed: {text}"


def _collect_doc_ids() -> list[str]:
    items = _merged_document_sources()
    return list({str(item["doc_id"]) for item in items if item.get("doc_id")})


@app.get("/health", response_model=HealthResponse)
def health():
    details: dict[str, object] = {}
    status = "ok"
    try:
        get_vector_db().client.get_collections()
        details["qdrant"] = "ok"
    except Exception as exc:
        status = "error"
        details["qdrant"] = str(exc)

    try:
        emb = get_embedding_client()
        details["embeddings_model"] = emb.model
    except Exception as exc:
        status = "error"
        details["embeddings"] = str(exc)

    try:
        groq = get_groq_generator()
        details["groq_model"] = groq.model
        details["groq_api_key_present"] = bool(groq.api_key)
    except Exception as exc:
        status = "error"
        details["groq"] = str(exc)

    return HealthResponse(status=status, details=details)


@app.post("/upload", response_model=list[UploadResponse])
async def upload_pdfs(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files provided")

    responses = []
    for file in files:
        data = await file.read()
        parsed = parse_pdf_bytes(data, original_name=file.filename or "uploaded.pdf")
        parsed["ingestion"] = {"status": "queued"}
        init_status(parsed["doc_id"], parsed.get("title"))
        background_tasks.add_task(_ingest_background, parsed)
        responses.append(UploadResponse(doc_id=parsed["doc_id"], status="queued", title=parsed.get("title")))

    return responses


@app.get("/ingestion/{doc_id}", response_model=IngestionStatusItem)
def ingestion_status(doc_id: str):
    status = get_status(doc_id)
    if not status:
        return IngestionStatusItem(doc_id=doc_id, state="unknown", stage="unknown", detail="No status found")
    return IngestionStatusItem(**status)


@app.get("/ingestion", response_model=dict[str, list[IngestionStatusItem]])
def ingestion_statuses():
    return {"items": [IngestionStatusItem(**item) for item in list_statuses()]}


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    _rate_limit_query()
    pipeline = get_retrieval_pipeline()
    start = time.time()
    result = pipeline.retrieve_with_candidates(
        payload.query,
        top_k_initial=max(payload.top_k, 10),
        top_k_final=payload.top_k,
        doc_ids=payload.doc_ids or None,
        rerank=payload.rerank,
        search_mode=payload.search_mode,
    )
    generator = get_groq_generator()
    answer = generator.generate_with_citations(payload.query, result.get("final", []))
    latency_ms = int((time.time() - start) * 1000)

    sources = [
        SourceItem(
            title=item.get("doc_title") or item.get("doc_id", ""),
            type=item.get("type", "text"),
            content=item.get("content", ""),
            relevance_score=float(item.get("rerank_score") or item.get("rrf_score") or 0.0),
        )
        for item in result.get("final", [])
    ]
    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)


@app.get("/documents", response_model=list[DocumentItem])
def documents():
    return _merge_document_items(_merged_document_sources())


@app.get("/documents/{doc_id}", response_model=DocumentItem)
def document(doc_id: str):
    for item in _merge_document_items(_merged_document_sources()):
        if item.doc_id == doc_id:
            return item
    raise HTTPException(status_code=404, detail="Document not found")


@app.patch("/documents/{doc_id}", response_model=DocumentItem)
def rename_document(doc_id: str, payload: RenameDocumentRequest):
    rename_document_record(doc_id, payload.title)
    for item in _merge_document_items(_merged_document_sources()):
        if item.doc_id == doc_id:
            return DocumentItem(
                doc_id=doc_id,
                title=payload.title,
                status=item.status,
                metadata=item.metadata,
            )
    return DocumentItem(doc_id=doc_id, title=payload.title, status="indexed", metadata={})


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    get_vector_db().delete_document(doc_id)
    delete_document_artifacts(doc_id)
    return {"doc_id": doc_id, "status": "deleted"}


@app.delete("/documents", response_model=DeleteDocumentsResponse)
def delete_documents(
    doc_ids: list[str] | None = Query(default=None),
    clear_all: bool = Query(default=False),
):
    vector_db = get_vector_db()
    if clear_all:
        ids = _collect_doc_ids()
        vector_db.delete_all_documents()
        delete_all_document_artifacts(ids)
        return DeleteDocumentsResponse(deleted=ids, status="deleted")

    if not doc_ids:
        raise HTTPException(status_code=400, detail="Provide doc_ids query parameters or set clear_all=true")

    ids = list(dict.fromkeys(doc_ids))
    for doc_id in ids:
        vector_db.delete_document(doc_id)
        delete_document_artifacts(doc_id)
    return DeleteDocumentsResponse(deleted=ids, status="deleted")


@app.get("/answer/stream")
def answer_documents_stream(q: str, top_k_initial: int = 30, top_k_final: int = 5, rerank: bool = True, search_mode: str = "hybrid", doc_ids: list[str] | None = Query(default=None)):
    generator = get_groq_generator()
    if not generator.api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    allowed = set(doc_ids or [])
    try:
        pipeline_results = get_retrieval_pipeline().retrieve_with_candidates(
            query=q,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            rerank=rerank,
            doc_ids=list(allowed) if allowed else None,
            search_mode=search_mode,
        )
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        raise HTTPException(status_code=503, detail=_retrieval_error_detail(e))

    def event_stream():
        yield "data: " + json.dumps({
            "type": "meta",
            "query": q,
            "candidates": pipeline_results.get("candidates", []),
            "final": pipeline_results["final"],
        }) + "\n\n"

        candidates = pipeline_results["final"]
        if not candidates:
            yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant chunks were found for your query. Please try rephrasing or ensure documents are uploaded.'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        try:
            for token in generator.stream(q, candidates):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"data: {json.dumps({'type': 'token', 'content': f'\\n\\n[Error generating response: {str(e)}]' })}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
