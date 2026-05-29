from __future__ import annotations

from typing import Any


def chunk_for_retrieval(parsed: dict[str, Any], max_chars: int = 1200, overlap_chars: int = 150) -> list[dict[str, Any]]:
    chunks = parsed.get("chunks", [])
    retrieval_chunks: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        text = (chunk.get("content") or "").strip()
        if not text:
            continue
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), dict) else {}
        retrieval_chunks.append(
            {
                "id": chunk.get("id") or f"{parsed.get('doc_id', 'doc')}_retrieval_{idx}",
                "type": "text",
                "content": text[:max_chars],
                "metadata": metadata,
                "doc_id": parsed.get("doc_id", "doc"),
                "doc_title": parsed.get("title", ""),
            }
        )
    return retrieval_chunks
