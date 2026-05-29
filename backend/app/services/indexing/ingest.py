from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services.embeddings.client import EmbeddingClient
from backend.app.services.indexing.chunk_normalizer import normalize_chunk
from backend.app.services.indexing.rag_chunker import chunk_for_retrieval
from backend.app.services.indexing.status import complete_status, fail_status, init_status, update_status
from backend.app.services.retrieval.sparse import build_sparse_vector
from backend.app.services.storage.qdrant_client import QdrantVectorDB


def ingest_parsed_document(parsed: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    doc_id = parsed["doc_id"]
    title = parsed.get("title", "")
    init_status(doc_id, title)
    update_status(doc_id, "chunking", "running", "Chunking parsed content for retrieval")
    chunks = chunk_for_retrieval(parsed)
    if not chunks:
        update_status(doc_id, "chunking", "done", "No chunks to ingest")
        return {"doc_id": doc_id, "ingested": 0}

    embedding_client = EmbeddingClient(config_path=config_path)
    vector_db = QdrantVectorDB(config_path=config_path)
    vector_db.create_collections()

    normalized_chunks = [normalize_chunk(chunk, doc_id=doc_id, title=title) for chunk in chunks]
    payload_chunks = []
    ingested_total = 0
    batch_size = 24
    total = len(normalized_chunks)
    for start in range(0, total, batch_size):
        batch = normalized_chunks[start : start + batch_size]
        update_status(doc_id, "embedding", "running", f"Embedding chunks {start + 1}-{min(start + batch_size, total)} of {total}")
        texts = [chunk["content"] or "" for chunk in batch]
        dense_vectors = embedding_client.embed_batch(texts)
        for chunk, dense_vector in zip(batch, dense_vectors):
            sparse_vector = build_sparse_vector(chunk["content"])
            payload_chunks.append(
                {
                    **chunk,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                }
            )
        update_status(doc_id, "storage", "running", f"Storing {len(payload_chunks)} chunks in Qdrant")
        vector_db.upsert_batch(payload_chunks)
        ingested_total += len(payload_chunks)
        payload_chunks.clear()

    complete_status(doc_id, f"Ingested {ingested_total} chunks")
    return {"doc_id": doc_id, "ingested": ingested_total}
