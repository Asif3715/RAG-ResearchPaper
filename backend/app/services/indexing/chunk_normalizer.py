from __future__ import annotations

from typing import Any


def normalize_chunk(chunk: dict[str, Any], doc_id: str, title: str) -> dict[str, Any]:
    content = chunk.get("content") or ""
    metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), dict) else {}
    return {
        "doc_id": doc_id,
        "doc_title": title,
        "title": title,
        "id": chunk.get("id"),
        "type": chunk.get("type", "text"),
        "content": content,
        "metadata": metadata,
        "page_number": metadata.get("page_number"),
        "page_range": metadata.get("page_range"),
        "source": metadata.get("source"),
        "chunk_index": metadata.get("chunk_index"),
    }
