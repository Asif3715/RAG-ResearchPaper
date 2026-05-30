from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from backend.app.services.retrieval.rrf_score import rrf_fuse
from backend.app.services.pdf.parser import load_parsed_output
from backend.app.services.storage.config import load_config


@dataclass
class QdrantSettings:
    url: str
    api_key: str
    dense_collection: str = "paper_chunks_dense"
    sparse_collection: str = "paper_chunks_sparse"
    dense_size: int = 1024


class QdrantVectorDB:
    def __init__(self, config_path: str | None = None):
        cfg = load_config(config_path)
        q = cfg.get("vector_store", {}).get("qdrant", {})
        self.settings = QdrantSettings(
            url=q.get("url", ""),
            api_key=q.get("api_key", ""),
            dense_collection=q.get("dense_collection", "paper_chunks_dense"),
            sparse_collection=q.get("sparse_collection", "paper_chunks_sparse"),
            dense_size=int(q.get("dense_size", 1024)),
        )
        self.client = QdrantClient(
            url=self.settings.url or None,
            api_key=self.settings.api_key or None,
            check_compatibility=False,
        )
        self._sparse_vector_names: tuple[str, ...] | None = None

    def create_collections(self) -> None:
        dense_name = self.settings.dense_collection
        sparse_name = self.settings.sparse_collection

        existing = {c.name for c in self.client.get_collections().collections}

        if dense_name not in existing:
            self.client.create_collection(
                collection_name=dense_name,
                vectors_config=rest.VectorParams(size=self.settings.dense_size, distance=rest.Distance.COSINE),
            )

        if sparse_name not in existing:
            self.client.create_collection(
                collection_name=sparse_name,
                vectors_config={},
                sparse_vectors_config={"sparse": rest.SparseVectorParams()},
            )

        self.client.create_payload_index(
            collection_name=dense_name,
            field_name="doc_id",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=sparse_name,
            field_name="doc_id",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )

    def upsert_chunk(self, chunk: dict[str, Any], dense_vector: list[float], sparse_vector: dict[str, Any]) -> None:
        point_id = self._point_id(chunk["id"])
        payload = {
            "doc_id": chunk["doc_id"],
            "doc_title": chunk.get("doc_title") or chunk.get("title") or chunk["doc_id"],
            "chunk_id": chunk["id"],
            "type": chunk["type"],
            "content": chunk.get("content") or chunk.get("caption") or "",
            "metadata": chunk.get("metadata", {}),
            "caption": chunk.get("caption"),
            "context": chunk.get("context"),
        }
        self.client.upsert(
            collection_name=self.settings.dense_collection,
            points=[rest.PointStruct(id=point_id, vector=dense_vector, payload=payload)],
        )
        sparse_key = self._primary_sparse_vector_name()
        self.client.upsert(
            collection_name=self.settings.sparse_collection,
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector={
                        sparse_key: rest.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        )
                    },
                    payload=payload,
                )
            ],
        )

    def upsert_batch(self, chunks: list[dict[str, Any]]) -> None:
        dense_points = []
        sparse_points = []
        sparse_key = self._primary_sparse_vector_name()
        for chunk in chunks:
            point_id = self._point_id(chunk["id"])
            dense_vector = chunk["dense_vector"]
            sparse_vector = chunk["sparse_vector"]
            payload = {
                "doc_id": chunk["doc_id"],
                "doc_title": chunk.get("doc_title") or chunk.get("title") or chunk["doc_id"],
                "chunk_id": chunk["id"],
                "type": chunk["type"],
                "content": chunk.get("content") or chunk.get("caption") or "",
                "metadata": chunk.get("metadata", {}),
                "caption": chunk.get("caption"),
                "context": chunk.get("context"),
            }
            dense_points.append(rest.PointStruct(id=point_id, vector=dense_vector, payload=payload))
            sparse_points.append(
                rest.PointStruct(
                    id=point_id,
                    vector={
                        sparse_key: rest.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        )
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.settings.dense_collection, points=dense_points)
        self.client.upsert(collection_name=self.settings.sparse_collection, points=sparse_points)

    def search_dense(self, query_vector: list[float], top_k: int = 30) -> list[dict[str, Any]]:
        return self.search_dense_filtered(query_vector=query_vector, top_k=top_k)

    def search_dense_filtered(self, query_vector: list[float], top_k: int = 30, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        query_filter = None
        if doc_ids:
            query_filter = rest.Filter(must=[rest.FieldCondition(key="doc_id", match=rest.MatchAny(any=doc_ids))])
        dense_hits = self.client.query_points(
            collection_name=self.settings.dense_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        ).points
        return [self._hit_to_dict(hit) for hit in dense_hits]

    def _primary_sparse_vector_name(self) -> str:
        names = self._sparse_vector_name_candidates()
        return names[0] if names else "sparse"

    def _sparse_vector_name_candidates(self) -> tuple[str, ...]:
        if self._sparse_vector_names is not None:
            return self._sparse_vector_names
        names: list[str] = []
        try:
            info = self.client.get_collection(self.settings.sparse_collection)
            config = getattr(info, "config", None)
            params = getattr(config, "params", None) if config else None
            sparse_cfg = getattr(params, "sparse_vectors", None) if params else None
            if isinstance(sparse_cfg, dict):
                names = list(sparse_cfg.keys())
        except Exception:
            pass
        if not names:
            names = ["sparse", "bm25"]
        elif "sparse" not in names and "bm25" in names:
            names = ["bm25", "sparse"]
        else:
            names = list(dict.fromkeys([*names, "sparse", "bm25"]))
        self._sparse_vector_names = tuple(names)
        return self._sparse_vector_names

    def search_sparse(self, sparse_vector: dict[str, Any], top_k: int = 30, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        query_filter = None
        if doc_ids:
            query_filter = rest.Filter(must=[rest.FieldCondition(key="doc_id", match=rest.MatchAny(any=doc_ids))])
        query = rest.SparseVector(indices=sparse_vector.get("indices", []), values=sparse_vector.get("values", []))
        last_exc: Exception | None = None
        for using in self._sparse_vector_name_candidates():
            try:
                sparse_hits = self.client.query_points(
                    collection_name=self.settings.sparse_collection,
                    query=query,
                    using=using,
                    limit=top_k,
                    with_payload=True,
                    query_filter=query_filter,
                ).points
                return [self._hit_to_dict(hit) for hit in sparse_hits]
            except Exception as exc:
                last_exc = exc
                logger.debug("Sparse search with vector name %r failed: %s", using, exc)
        if last_exc:
            raise last_exc
        return []

    def hybrid_search(self, query_vector: list[float], sparse_vector: dict[str, Any], top_k: int = 30, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        dense_hits = self.search_dense_filtered(query_vector=query_vector, top_k=top_k, doc_ids=doc_ids)
        try:
            sparse_hits = self.search_sparse(sparse_vector=sparse_vector, top_k=top_k, doc_ids=doc_ids)
            return rrf_fuse([dense_hits, sparse_hits])[:top_k]
        except Exception as exc:
            logger.warning("Sparse leg unavailable, using dense search only: %s", exc)
            return dense_hits[:top_k]

    def delete_document(self, doc_id: str) -> None:
        flt = rest.Filter(must=[rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id))])
        self.client.delete(collection_name=self.settings.dense_collection, points_selector=rest.FilterSelector(filter=flt))
        self.client.delete(collection_name=self.settings.sparse_collection, points_selector=rest.FilterSelector(filter=flt))

    def delete_documents(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            self.delete_document(doc_id)

    def delete_all_documents(self) -> None:
        dense_name = self.settings.dense_collection
        sparse_name = self.settings.sparse_collection
        for name in (dense_name, sparse_name):
            try:
                self.client.delete_collection(collection_name=name)
            except Exception:
                pass
        self.create_collections()

    def list_documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for collection_name in (self.settings.dense_collection,):
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = getattr(point, "payload", {}) or {}
                    doc_id = str(payload.get("doc_id", ""))
                    if not doc_id:
                        continue
                    entry = docs.setdefault(
                        doc_id,
                        {
                            "doc_id": doc_id,
                            "title": payload.get("doc_title") or payload.get("title") or doc_id,
                            "status": "indexed",
                            "metadata": {"chunks": 0},
                        },
                    )
                    if entry.get("title") == doc_id or not entry.get("title"):
                        cached = load_parsed_output(doc_id)
                        if cached and cached.get("title"):
                            entry["title"] = cached.get("title")
                    entry["metadata"]["chunks"] = int(entry["metadata"].get("chunks", 0)) + 1
                    if payload.get("doc_title") and (entry.get("title") == doc_id or not entry.get("title")):
                        entry["title"] = payload.get("doc_title")
                if offset is None:
                    break
        return list(docs.values())

    @staticmethod
    def _hit_to_dict(hit: Any) -> dict[str, Any]:
        return {
            "id": str(hit.id),
            "score": float(hit.score),
            "payload": getattr(hit, "payload", {}) or {},
        }

    @staticmethod
    def _point_id(source_id: str) -> str:
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
