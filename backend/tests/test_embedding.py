from backend.app.services.embeddings.client import EmbeddingClient, RerankClient, cache_key, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello   world \n") == "hello world"


def test_cache_key_changes_with_model():
    text = "hello world"
    assert cache_key("bge-m3", text) != cache_key("other-model", text)


def test_cache_key_changes_with_text():
    assert cache_key("bge-m3", "a") != cache_key("bge-m3", "b")


def test_default_models_use_hf_names(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    emb = EmbeddingClient(cache_dir=None)
    rerank = RerankClient()
    assert emb.model == "BAAI/bge-m3"
    assert rerank.model == "BAAI/bge-reranker-v2-m3"
