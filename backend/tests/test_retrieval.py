from backend.app.services.retrieval.rrf_score import rrf_fuse
from backend.app.services.retrieval.sparse import build_sparse_vector


def test_rrf_prefers_consensus_items():
    dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    sparse = [{"id": "b"}, {"id": "c"}, {"id": "d"}]

    fused = rrf_fuse([dense, sparse])

    assert fused[0]["id"] in {"b", "c"}
    assert len(fused) == 4
    assert all("rrf_score" in item for item in fused)


def test_sparse_vector_is_stable_for_query_terms():
    sparse = build_sparse_vector("hybrid retrieval pipeline for research papers")
    assert len(sparse["indices"]) == len(sparse["values"]) == len(sparse["terms"])
    assert all(isinstance(i, int) for i in sparse["indices"])
