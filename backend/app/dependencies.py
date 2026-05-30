from __future__ import annotations

from functools import lru_cache

from backend.app.services.embeddings.client import EmbeddingClient, RerankClient
from backend.app.services.llm.groq_generator import GroqGenerator
from backend.app.services.retrieval.retrieval_pipeline import RetrievalPipeline
from backend.app.services.storage.qdrant_client import QdrantVectorDB


@lru_cache(maxsize=1)
def get_vector_db() -> QdrantVectorDB:
    return QdrantVectorDB()


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache(maxsize=1)
def get_rerank_client() -> RerankClient:
    return RerankClient()


@lru_cache(maxsize=1)
def get_groq_generator() -> GroqGenerator:
    return GroqGenerator()


@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(
        embedding_client=get_embedding_client(),
        rerank_client=get_rerank_client(),
        vector_db=get_vector_db(),
    )
