from backend.app.services.storage.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_path_points_at_backend_yaml():
    assert DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert DEFAULT_CONFIG_PATH.parent.name == "backend"
    assert DEFAULT_CONFIG_PATH.exists()


def test_load_config_reads_embeddings_and_rerank():
    cfg = load_config()
    assert "embeddings" in cfg
    assert "rerank" in cfg
    assert cfg["embeddings"].get("model")
    assert cfg["vector_store"]["qdrant"]["dense_collection"]


def test_env_overrides_empty_yaml_qdrant_url(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://example.cloud.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    cfg = load_config()
    assert cfg["vector_store"]["qdrant"]["url"] == "https://example.cloud.qdrant.io:6333"
    assert cfg["vector_store"]["qdrant"]["api_key"] == "test-key"
