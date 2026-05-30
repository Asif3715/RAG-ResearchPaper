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
