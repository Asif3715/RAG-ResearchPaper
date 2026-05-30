from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload


def test_query_schema_rejects_empty_query():
    response = client.post("/query", json={"query": "", "top_k": 5})
    assert response.status_code == 422


def test_documents_endpoint_returns_list(monkeypatch):
    class FakeVectorDB:
        def list_documents(self):
            return []

    monkeypatch.setattr("backend.app.main.get_vector_db", lambda: FakeVectorDB())
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_delete_documents_requires_doc_ids(monkeypatch):
    class FakeVectorDB:
        def delete_document(self, doc_id: str):
            raise AssertionError("should not delete without doc_ids")

    monkeypatch.setattr("backend.app.main.get_vector_db", lambda: FakeVectorDB())
    response = client.delete("/documents")
    assert response.status_code == 400


def test_delete_documents_deletes_only_requested_ids(monkeypatch):
    deleted: list[str] = []

    class FakeVectorDB:
        def delete_document(self, doc_id: str) -> None:
            deleted.append(doc_id)

    monkeypatch.setattr("backend.app.main.get_vector_db", lambda: FakeVectorDB())
    monkeypatch.setattr("backend.app.main.delete_document_artifacts", lambda doc_id: deleted.append(f"artifact:{doc_id}"))

    response = client.delete("/documents", params=[("doc_ids", "doc-a"), ("doc_ids", "doc-b")])
    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted"] == ["doc-a", "doc-b"]
    assert "doc-a" in deleted
    assert "doc-b" in deleted


def test_delete_single_document_endpoint(monkeypatch):
    deleted: list[str] = []

    class FakeVectorDB:
        def delete_document(self, doc_id: str) -> None:
            deleted.append(doc_id)

    monkeypatch.setattr("backend.app.main.get_vector_db", lambda: FakeVectorDB())
    monkeypatch.setattr("backend.app.main.delete_document_artifacts", lambda doc_id: None)

    response = client.delete("/documents/doc-only")
    assert response.status_code == 200
    assert deleted == ["doc-only"]
