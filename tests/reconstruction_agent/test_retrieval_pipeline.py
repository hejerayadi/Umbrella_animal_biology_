from pathlib import Path

from backend.agents.reconstruction_agent.retrieval_pipeline import RetrievalPipeline


class FakeQdrantClient:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, collection_name, query_vector, limit=5, filter=None):
        self.calls.append((collection_name, query_vector, limit, filter))
        return self.results


def test_get_flanking_context_aggregates_results_and_fallbacks():
    seq = "A" * 2000
    client = FakeQdrantClient([
        {"id": "1", "score": 0.9, "payload": {"species_id": "sp1", "window_sequence": "C" * 2000, "is_extinct": False}},
        {"id": "2", "score": 0.8, "payload": {"species_id": "sp1", "window_sequence": "G" * 2000, "is_extinct": False}},
    ])
    pipeline = RetrievalPipeline(qdrant_url="http://localhost", collection_name="windows", top_k=2)
    pipeline.client = client

    left, right = pipeline.get_flanking_context("sp1", 1000, 1001, seq, False)

    assert left.startswith("C")
    assert right.startswith("G")
    assert len(left) >= 1000
    assert len(right) >= 1000


def test_get_flanking_context_falls_back_to_sequence_slices():
    seq = "A" * 5000
    client = FakeQdrantClient([])
    pipeline = RetrievalPipeline(qdrant_url="http://localhost", collection_name="windows")
    pipeline.client = client

    left, right = pipeline.get_flanking_context("sp1", 2000, 2005, seq, False)

    assert left == "A" * 1000
    assert right == "A" * 1000


def test_get_flanking_context_filters_out_mammoth_records():
    seq = "A" * 2000
    client = FakeQdrantClient([
        {"id": "mammoth", "score": 0.7, "payload": {"species_id": "sp1", "window_sequence": "C" * 2000, "is_extinct": True}},
        {"id": "living", "score": 0.6, "payload": {"species_id": "sp1", "window_sequence": "G" * 2000, "is_extinct": False}},
    ])
    pipeline = RetrievalPipeline(qdrant_url="http://localhost", collection_name="windows")
    pipeline.client = client

    left, right = pipeline.get_flanking_context("sp1", 1000, 1001, seq, False)

    assert left.startswith("G")
    assert right.startswith("G")
