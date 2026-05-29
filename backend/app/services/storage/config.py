from __future__ import annotations

import os
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config.yaml"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


def load_env(path: str | Path | None = None) -> None:
    env_path = Path(path) if path else DEFAULT_ENV_PATH
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_config(path: str | Path | None = None) -> dict:
    load_env()
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {
            "vector_store": {
                "provider": "qdrant",
                "qdrant": {
                    "url": os.getenv("QDRANT_URL", ""),
                    "api_key": os.getenv("QDRANT_API_KEY", ""),
                    "dense_collection": "paper_chunks_dense",
                    "sparse_collection": "paper_chunks_sparse",
                    "dense_size": 1024,
                },
                "embeddings": {"hf_token": ""},
                "rerank": {"hf_token": ""},
            }
        }
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    vector_store = cfg.setdefault("vector_store", {})
    qdrant = vector_store.setdefault("qdrant", {})
    qdrant.setdefault("url", os.getenv("QDRANT_URL", ""))
    qdrant.setdefault("api_key", os.getenv("QDRANT_API_KEY", ""))
    qdrant.setdefault("dense_collection", "paper_chunks_dense")
    qdrant.setdefault("sparse_collection", "paper_chunks_sparse")
    qdrant.setdefault("dense_size", 1024)
    cfg.setdefault("embeddings", {})
    cfg.setdefault("rerank", {})
    cfg["embeddings"].setdefault("hf_token", os.getenv("HF_TOKEN", ""))
    cfg["rerank"].setdefault("hf_token", os.getenv("HF_TOKEN", ""))
    return cfg
