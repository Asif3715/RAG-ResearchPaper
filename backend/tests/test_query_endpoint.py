from backend.app.services.retrieval.retrieval_pipeline import RetrievalPipeline


def test_retrieval_pipeline_imports():
    pipeline = RetrievalPipeline
    assert pipeline is not None


def test_retrieval_pipeline_serialization_shape():
    assert hasattr(RetrievalPipeline, "retrieve_with_candidates")
