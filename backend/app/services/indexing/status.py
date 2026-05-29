from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


_STATUS_STORE: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    _STATUS_STORE[doc_id] = status
    return status


def update_status(doc_id: str, stage: str, state: str, detail: str) -> dict[str, Any]:
    status = _STATUS_STORE.get(doc_id) or init_status(doc_id)
    status.update({"stage": stage, "state": state, "detail": detail, "updated_at": _now()})
    status.setdefault("timeline", []).append({"stage": stage, "state": state, "detail": detail, "at": _now()})
    _STATUS_STORE[doc_id] = status
    return status


def get_status(doc_id: str) -> dict[str, Any] | None:
    return _STATUS_STORE.get(doc_id)


def complete_status(doc_id: str, detail: str = "Ingestion complete") -> dict[str, Any]:
    return update_status(doc_id, "done", "done", detail)


def fail_status(doc_id: str, detail: str) -> dict[str, Any]:
    return update_status(doc_id, "error", "error", detail)


def list_statuses() -> list[dict[str, Any]]:
    return list(_STATUS_STORE.values())
