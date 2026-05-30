from __future__ import annotations

import os
from pathlib import Path

import yaml


# backend/app/services/storage -> parents[3] == backend/
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


def _prefer_env(env_key: str, yaml_value: str | None = None) -> str:
    env_value = os.getenv(env_key, "").strip()
    if env_value:
        return env_value
    return (yaml_value or "").strip() if yaml_value is not None else ""


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
            },
            "embeddings": {
                "hf_token": os.getenv("HF_TOKEN", ""),
                "model": "BAAI/bge-m3",
                "timeout": 60,
            },
            "rerank": {
                "hf_token": os.getenv("HF_TOKEN", ""),
                "model": "BAAI/bge-reranker-v2-m3",
                "timeout": 60,
            },
        }
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    vector_store = cfg.setdefault("vector_store", {})
    qdrant = vector_store.setdefault("qdrant", {})
    qdrant["url"] = _prefer_env("QDRANT_URL", qdrant.get("url"))
    qdrant["api_key"] = _prefer_env("QDRANT_API_KEY", qdrant.get("api_key"))
    qdrant.setdefault("dense_collection", "paper_chunks_dense")
    qdrant.setdefault("sparse_collection", "paper_chunks_sparse")
    qdrant.setdefault("dense_size", 1024)
    cfg.setdefault("embeddings", {})
    cfg.setdefault("rerank", {})
    cfg["embeddings"]["hf_token"] = _prefer_env("HF_TOKEN", cfg["embeddings"].get("hf_token"))
    cfg["rerank"]["hf_token"] = _prefer_env("HF_TOKEN", cfg["rerank"].get("hf_token"))
    return cfg
