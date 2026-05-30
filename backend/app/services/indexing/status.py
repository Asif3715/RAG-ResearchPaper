from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import DATA_DIR


STATUS_PATH = DATA_DIR / "ingestion_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_store() -> dict[str, dict[str, Any]]:
    if not STATUS_PATH.exists():
        return {}
    try:
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    items = data.get("items", [])
    if isinstance(items, list):
        return {item["doc_id"]: item for item in items if isinstance(item, dict) and item.get("doc_id")}
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _save_store(store: dict[str, dict[str, Any]]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps({"items": list(store.values())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def init_status(doc_id: str, title: str | None = None) -> dict[str, Any]:
    status = {
        "doc_id": doc_id,
        "title": title,
        "state": "queued",
        "stage": "queued",
        "detail": "Awaiting ingestion",
        "updated_at": _now(),
        "timeline": [
            {"stage": "queued", "state": "queued", "detail": "Awaiting ingestion", "at": _now()}
        ],
    }
    store = _load_store()
    store[doc_id] = status
    _save_store(store)
    return status


def update_status(doc_id: str, stage: str, state: str, detail: str) -> dict[str, Any]:
    store = _load_store()
    if doc_id not in store:
        init_status(doc_id)
        store = _load_store()
    status = store[doc_id]
    status.update({"stage": stage, "state": state, "detail": detail, "updated_at": _now()})
    status.setdefault("timeline", []).append({"stage": stage, "state": state, "detail": detail, "at": _now()})
    store[doc_id] = status
    _save_store(store)
    return status


def get_status(doc_id: str) -> dict[str, Any] | None:
    return _load_store().get(doc_id)


def complete_status(doc_id: str, detail: str = "Ingestion complete") -> dict[str, Any]:
    return update_status(doc_id, "done", "done", detail)


def fail_status(doc_id: str, detail: str) -> dict[str, Any]:
    return update_status(doc_id, "error", "error", detail)


def clear_status(doc_id: str) -> None:
    store = _load_store()
    if doc_id in store:
        del store[doc_id]
        _save_store(store)


def clear_all_statuses() -> None:
    if STATUS_PATH.exists():
        STATUS_PATH.unlink()


def list_statuses() -> list[dict[str, Any]]:
    return list(_load_store().values())
