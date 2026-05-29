from backend.app.services.retrieval.sparse import build_sparse_vector


def test_sparse_vector_contains_terms_and_values():
    sparse = build_sparse_vector("This is a small test for the sparse vector builder")
    assert "terms" in sparse
    assert "indices" in sparse
    assert "values" in sparse
    assert len(sparse["terms"]) == len(sparse["indices"]) == len(sparse["values"])
