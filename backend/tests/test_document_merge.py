from backend.app.main import _merge_document_items


def test_merge_preserves_chunk_count_over_error_metadata():
    items = [
        {
            "doc_id": "doc-1",
            "title": "DenseNet",
            "status": "indexed",
            "metadata": {"chunks": 142},
        },
        {
            "doc_id": "doc-1",
            "title": "DenseNet",
            "status": "error",
            "metadata": {"error": "sparse vector missing"},
        },
    ]
    merged = _merge_document_items(items)
    assert len(merged) == 1
    assert merged[0].metadata["chunks"] == 142
    assert merged[0].status == "indexed"
