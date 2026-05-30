from backend.app.services.indexing.status import STATUS_PATH, complete_status, get_status, init_status, update_status


def test_status_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.services.indexing.status.STATUS_PATH", tmp_path / "ingestion_status.json")
    init_status("doc-1", "Title")
    update_status("doc-1", "chunking", "running", "Chunking")
    complete_status("doc-1")
    status = get_status("doc-1")
    assert status is not None
    assert status["state"] == "done"
    assert len(status["timeline"]) >= 3
