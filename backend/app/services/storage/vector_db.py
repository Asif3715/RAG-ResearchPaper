from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class VectorRecord:
    id: str
    doc_id: str
    chunk_id: str
    text: str
    payload: dict[str, Any]


class VectorDBClient(Protocol):
    def create_collections(self) -> None: ...

    def upsert_chunk(self, chunk: dict[str, Any], dense_vector: list[float], sparse_vector: dict[str, Any]) -> None: ...

    def upsert_batch(self, chunks: list[dict[str, Any]]) -> None: ...

    def hybrid_search(self, query_vector: list[float], top_k: int = 30) -> list[dict[str, Any]]: ...

    def delete_document(self, doc_id: str) -> None: ...
