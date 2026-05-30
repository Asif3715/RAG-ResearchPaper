from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.services.embeddings.client import EmbeddingClient, RerankClient
from backend.app.services.retrieval.sparse import build_sparse_vector
from backend.app.services.retrieval.rrf_score import rrf_fuse
from backend.app.services.storage.qdrant_client import QdrantVectorDB


@dataclass
class RetrievedChunk:
    id: str
    doc_id: str
    doc_title: str
    type: str
    content: str
    metadata: dict[str, Any]
    caption: str | None = None
    context: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalPipeline:
    def __init__(
        self,
        config_path: str | None = None,
        embedding_client: EmbeddingClient | None = None,
        rerank_client: RerankClient | None = None,
        vector_db: QdrantVectorDB | None = None,
    ):
        self.embedding_client = embedding_client or EmbeddingClient(config_path=config_path)
        self.rerank_client = rerank_client or RerankClient(config_path=config_path)
        self.vector_db = vector_db or QdrantVectorDB(config_path=config_path)

    def retrieve(self, query: str, top_k_initial: int = 30, top_k_final: int = 5, doc_ids: list[str] | None = None, rerank: bool = True) -> list[RetrievedChunk]:
        query_vector = self.embedding_client.embed_text(query)
        sparse_vector = build_sparse_vector(query)
        candidates = self.vector_db.hybrid_search(query_vector=query_vector, sparse_vector=sparse_vector, top_k=top_k_initial, doc_ids=doc_ids)
        structured = self._filter_chunks([self._to_chunk(item) for item in candidates], doc_ids)
        if not structured:
            return []

        if rerank:
            rerank_inputs = [chunk.content or chunk.caption or "" for chunk in structured]
            reranked = self.rerank_client.rerank(query=query, documents=rerank_inputs, top_n=min(top_k_final, len(structured)))

            for item in reranked:
                idx = item["index"]
                if 0 <= idx < len(structured):
                    structured[idx].rerank_score = float(item["score"])

        structured.sort(key=lambda x: (x.rerank_score or 0.0, x.rrf_score or 0.0), reverse=True)
        return structured[:top_k_final]

    def retrieve_with_candidates(self, query: str, top_k_initial: int = 30, top_k_final: int = 5, doc_ids: list[str] | None = None, rerank: bool = True, search_mode: str = "hybrid") -> dict[str, Any]:
        query_vector = self.embedding_client.embed_text(query)
        sparse_vector = build_sparse_vector(query)
        if search_mode == "simple":
            candidates = self.vector_db.search_dense_filtered(query_vector=query_vector, top_k=top_k_initial, doc_ids=doc_ids)
        else:
            candidates = self.vector_db.hybrid_search(query_vector=query_vector, sparse_vector=sparse_vector, top_k=top_k_initial, doc_ids=doc_ids)
        structured = self._filter_chunks([self._to_chunk(item) for item in candidates], doc_ids)
        if not structured:
            return {"query": query, "candidates": [], "final": []}

        if rerank:
            rerank_inputs = [chunk.content or chunk.caption or "" for chunk in structured]
            reranked = self.rerank_client.rerank(query=query, documents=rerank_inputs, top_n=min(top_k_final, len(structured)))

            for item in reranked:
                idx = item["index"]
                if 0 <= idx < len(structured):
                    structured[idx].rerank_score = float(item["score"])

        structured.sort(key=lambda x: (x.rerank_score or 0.0, x.rrf_score or 0.0), reverse=True)
        return {
            "query": query,
            "candidates": [self._serialize_chunk(chunk) for chunk in structured[:top_k_initial]],
            "final": [self._serialize_chunk(chunk) for chunk in structured[:top_k_final]],
        }

    @staticmethod
    def _filter_chunks(chunks: list[RetrievedChunk], doc_ids: list[str] | None) -> list[RetrievedChunk]:
        if not doc_ids:
            return chunks
        allowed = {doc_id for doc_id in doc_ids if doc_id}
        return [chunk for chunk in chunks if chunk.doc_id in allowed]

    @staticmethod
    def _to_chunk(item: dict[str, Any]) -> RetrievedChunk:
        payload = item.get("payload", {})
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        return RetrievedChunk(
            id=str(payload.get("chunk_id") or item.get("id")),
            doc_id=str(payload.get("doc_id", "")),
            doc_title=str(payload.get("doc_title", payload.get("doc_id", ""))),
            type=str(payload.get("type", "text")),
            content=str(payload.get("content", "")),
            metadata=metadata,
            caption=payload.get("caption"),
            context=payload.get("context"),
            rrf_score=float(item.get("rrf_score", 0.0)),
            dense_score=float(item.get("score", 0.0)),
        )

    @staticmethod
    def _serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "type": chunk.type,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "caption": chunk.caption,
            "context": chunk.context,
            "dense_score": chunk.dense_score,
            "sparse_score": chunk.sparse_score,
            "rrf_score": chunk.rrf_score,
            "rerank_score": chunk.rerank_score,
        }
