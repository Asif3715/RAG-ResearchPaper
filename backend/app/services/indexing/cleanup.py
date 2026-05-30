from __future__ import annotations

from backend.app.core.config import EXTRACTED_DIR, UPLOADS_DIR
from backend.app.services.indexing.registry import delete_document_record
from backend.app.services.indexing.status import clear_all_statuses, clear_status
from backend.app.services.pdf.parser import delete_parsed_output


def delete_document_artifacts(doc_id: str) -> None:
    delete_parsed_output(doc_id)
    delete_document_record(doc_id)
    clear_status(doc_id)
    upload_path = UPLOADS_DIR / f"{doc_id}.pdf"
    if upload_path.exists():
        upload_path.unlink()


def delete_all_document_artifacts(doc_ids: list[str]) -> None:
    for doc_id in doc_ids:
        delete_document_artifacts(doc_id)
    clear_all_statuses()
    for path in EXTRACTED_DIR.glob("*.json"):
        path.unlink(missing_ok=True)
    for path in UPLOADS_DIR.glob("*.pdf"):
        path.unlink(missing_ok=True)
