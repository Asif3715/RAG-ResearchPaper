from backend.app.services.retrieval.rrf_score import rrf_fuse


def test_rrf_fusion_orders_results():
    dense = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
    sparse = [{"id": "b", "score": 0.95}, {"id": "c", "score": 0.7}]

    fused = rrf_fuse([dense, sparse])

    assert fused[0]["id"] in {"a", "b"}
    assert len(fused) == 3
    assert all("rrf_score" in item for item in fused)
