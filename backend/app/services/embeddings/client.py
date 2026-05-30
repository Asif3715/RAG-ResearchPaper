from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient

from backend.app.services.storage.config import load_config


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "cache"


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def cache_key(model: str, text: str) -> str:
    normalized = normalize_text(text)
    raw = f"{model}:{normalized}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class DiskCache:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")


class RetrySession:
    def __init__(self, retries: int = 4, backoff: float = 1.5):
        self.retries = retries
        self.backoff = backoff

    def run(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(self.backoff ** attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed")


class EmbeddingClient:
    def __init__(self, config_path: str | None = None, cache_dir: Path | None = None):
        cfg = load_config(config_path)
        emb_cfg = cfg.get("embeddings", {})
        self.model = emb_cfg.get("model", "BAAI/bge-m3")
        self.hf_token = emb_cfg.get("hf_token") or os.getenv("HF_TOKEN", "")
        self.timeout = int(emb_cfg.get("timeout", 60))
        self.cache = DiskCache(cache_dir=cache_dir)
        self.session = RetrySession()
        self.client = InferenceClient(provider="hf-inference", api_key=self.hf_token)

    def embed_text(self, text: str) -> list[float]:
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = [normalize_text(text) for text in texts]
        results: list[list[float] | None] = [None] * len(normalized_texts)
        uncached_inputs: list[str] = []
        uncached_positions: list[int] = []

        for idx, text in enumerate(normalized_texts):
            key = cache_key(self.model, text)
            cached = self.cache.get(key)
            if cached is not None:
                results[idx] = cached
                continue
            uncached_inputs.append(text)
            uncached_positions.append(idx)

        if uncached_inputs:
            raw = self.session.run(self.client.feature_extraction, uncached_inputs, model=self.model)
            vectors = self._normalize_embeddings_response(raw)
            if len(vectors) != len(uncached_inputs):
                raise ValueError("embedding response length mismatch")
            for pos, text, vector in zip(uncached_positions, uncached_inputs, vectors):
                results[pos] = vector
                self.cache.set(cache_key(self.model, text), vector)

        return [vector for vector in results if vector is not None]

    @staticmethod
    def _normalize_embeddings_response(raw: Any) -> list[list[float]]:
        if isinstance(raw, dict):
            for key in ("embeddings", "embedding", "data", "vector"):
                if key in raw:
                    return EmbeddingClient._normalize_embeddings_response(raw[key])
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if not isinstance(raw, list) or not raw:
            raise ValueError("invalid embedding response")
        if isinstance(raw[0], (int, float)):
            return [[float(v) for v in raw]]
        if isinstance(raw[0], list):
            if raw and isinstance(raw[0][0], (int, float)):
                return [[float(v) for v in row] for row in raw]
            if raw and isinstance(raw[0][0], list):
                # Some providers return token embeddings per item; pool each item.
                return [EmbeddingClient._pool_embedding(item) for item in raw]
        raise ValueError("invalid embedding response shape")

    @staticmethod
    def _pool_embedding(raw: Any) -> list[float]:
        if isinstance(raw, dict):
            for key in ("embeddings", "embedding", "data", "vector"):
                if key in raw:
                    return EmbeddingClient._pool_embedding(raw[key])
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if not isinstance(raw, list) or not raw:
            raise ValueError("invalid embedding response")
        if isinstance(raw[0], (int, float)):
            return [float(v) for v in raw]
        if isinstance(raw[0], list):
            if raw and isinstance(raw[0][0], list):
                raw = raw[0]
            token_embeddings = [[float(v) for v in token] for token in raw]
            dims = len(token_embeddings[0])
            pooled = [0.0] * dims
            for token in token_embeddings:
                for i, value in enumerate(token):
                    pooled[i] += value
            count = float(len(token_embeddings))
            return [value / count for value in pooled]
        raise ValueError("invalid embedding response shape")


class RerankClient:
    def __init__(self, config_path: str | None = None):
        cfg = load_config(config_path)
        rerank_cfg = cfg.get("rerank", {})
        self.model = rerank_cfg.get("model", "BAAI/bge-reranker-v2-m3")
        self.hf_token = rerank_cfg.get("hf_token") or os.getenv("HF_TOKEN", "")
        self.timeout = int(rerank_cfg.get("timeout", 5))
        self.session = RetrySession(retries=1, backoff=0)
        self.client = InferenceClient(provider="hf-inference", api_key=self.hf_token, timeout=self.timeout)

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[dict[str, Any]]:
        import logging

        if not documents:
            return []

        query_text = normalize_text(query)
        normalized_docs = [normalize_text(doc) for doc in documents]

        try:
            ranked = self._rerank_via_inference_api(query_text, normalized_docs)
            if ranked:
                ranked.sort(key=lambda x: x["score"], reverse=True)
                return ranked[:top_n]
        except Exception as exc:
            logging.warning("Rerank API failed (%s); using positional fallback.", exc)

        return self._positional_fallback(documents, top_n)

    def _rerank_via_inference_api(self, query_text: str, documents: list[str]) -> list[dict[str, Any]]:
        import concurrent.futures

        def _score_doc(idx: int, document: str) -> dict[str, Any]:
            if hasattr(self.client, "sentence_similarity"):
                response = self.session.run(
                    self.client.sentence_similarity,
                    query_text,
                    document,
                    model=self.model,
                )
                score = float(response) if isinstance(response, (int, float)) else self._extract_label_score(response)
            else:
                pair_text = f"{query_text} [SEP] {document}"
                response = self.session.run(self.client.text_classification, pair_text, model=self.model)
                score = self._extract_label_score(response)
            return {"index": idx, "score": score, "document": documents[idx]}

        ranked: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_score_doc, idx, doc) for idx, doc in enumerate(documents)]
            for future in concurrent.futures.as_completed(futures):
                ranked.append(future.result())
        return ranked

    @staticmethod
    def _positional_fallback(documents: list[str], top_n: int) -> list[dict[str, Any]]:
        return [
            {"index": idx, "score": 1.0 / (idx + 1), "document": doc}
            for idx, doc in enumerate(documents)
        ][:top_n]

    @staticmethod
    def _extract_label_score(response: Any) -> float:
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict):
                return float(first.get("score", first.get("value", 0.0)))
            if isinstance(first, (int, float)):
                return float(first)
        if isinstance(response, dict):
            for key in ("score", "value", "label_score"):
                if key in response:
                    return float(response[key])
        raise ValueError("invalid rerank response")
