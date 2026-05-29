from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.config import EXTRACTED_DIR


REGISTRY_PATH = EXTRACTED_DIR.parent / "documents.json"


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"items": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(data: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_document(doc: dict[str, Any]) -> None:
    registry = _load_registry()
    items = [item for item in registry.get("items", []) if item.get("doc_id") != doc.get("doc_id")]
    items.append(doc)
    registry["items"] = items
    _save_registry(registry)


def list_documents() -> list[dict[str, Any]]:
    return list(_load_registry().get("items", []))


def delete_document_record(doc_id: str) -> None:
    registry = _load_registry()
    registry["items"] = [item for item in registry.get("items", []) if item.get("doc_id") != doc_id]
    _save_registry(registry)


def rename_document_record(doc_id: str, title: str) -> None:
    registry = _load_registry()
    items = registry.get("items", [])
    updated = False
    for item in items:
        if item.get("doc_id") == doc_id:
            item["title"] = title
            updated = True
            break
    if not updated:
        items.append({"doc_id": doc_id, "title": title, "status": "indexed", "metadata": {}})
    registry["items"] = items
    _save_registry(registry)
