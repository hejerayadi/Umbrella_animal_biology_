"""The retrieval boundary: the local fixture retriever, and the real adapter's
refusal to start without Chahd's contract."""
from __future__ import annotations

import pytest

from ..adapters.bioclip import deterministic_vector
from ..adapters.qdrant_mock import MockQdrantRetriever
from ..adapters.qdrant_real import RealQdrantRetriever
from ..config import QdrantConfig
from ..domain.errors import ErrorCode, RecognitionError

DIMENSION = 32


# --- the local fixture retriever -------------------------------------------

def test_mock_retriever_declares_local_development_mode():
    """It must be impossible to mistake a fixture result for the real thing."""
    retriever = MockQdrantRetriever(DIMENSION)
    assert retriever.mode == "mock_local_development"
    assert retriever.mode != "real_minimal"


def test_querying_a_species_centre_retrieves_that_species():
    retriever = MockQdrantRetriever(DIMENSION)
    query = deterministic_vector("panthera_leo", DIMENSION)

    outcome = retriever.search(query, top_k=10)

    assert outcome.references[0].species_id == "panthera_leo"
    assert outcome.references[0].similarity_score > 0.9
    assert outcome.mode == "mock_local_development"


def test_same_species_references_cluster_together():
    """Both of a species' references must score highly, or its mean-of-best-N
    is dragged down by its own siblings - which real embeddings never do."""
    retriever = MockQdrantRetriever(DIMENSION)
    query = deterministic_vector("panthera_leo", DIMENSION)

    outcome = retriever.search(query, top_k=10)
    leo = [ref for ref in outcome.references if ref.species_id == "panthera_leo"]

    assert len(leo) == 2
    assert all(ref.similarity_score > 0.9 for ref in leo)


def test_different_species_do_not_cluster():
    retriever = MockQdrantRetriever(DIMENSION)
    query = deterministic_vector("panthera_leo", DIMENSION)

    outcome = retriever.search(query, top_k=10)
    others = [ref for ref in outcome.references if ref.species_id != "panthera_leo"]

    assert all(ref.similarity_score < 0.5 for ref in others)


def test_unknown_query_never_reaches_the_identification_threshold():
    """An image nothing was ingested for must never look like a confident match.

    Asserted on the AGGREGATED species score, which is what the confidence gate
    actually reads. An individual reference can drift high by chance - at
    dimension 32 two unrelated unit vectors have a cosine standard deviation of
    about 1/sqrt(32) = 0.18 - but averaging a species' best references pulls
    that back well below the threshold.
    """
    retriever = MockQdrantRetriever(DIMENSION)

    worst = 0.0
    for i in range(200):
        outcome = retriever.search(
            deterministic_vector(f"sha256:unknown-{i}", DIMENSION), top_k=30
        )
        by_species: dict[str, list[float]] = {}
        for ref in outcome.references:
            by_species.setdefault(ref.species_id, []).append(ref.similarity_score)
        aggregated = [
            sum(sorted(scores, reverse=True)[:3]) / len(sorted(scores, reverse=True)[:3])
            for scores in by_species.values()
        ]
        worst = max(worst, max(aggregated))

    # IDENTIFIED_MIN_SCORE is 0.75; the observed worst case sits far below it.
    assert worst < 0.60, f"an unknown image aggregated to {worst:.3f}"


def test_top_k_is_respected():
    retriever = MockQdrantRetriever(DIMENSION)
    query = deterministic_vector("anything", DIMENSION)
    assert len(retriever.search(query, top_k=3).references) == 3


def test_wrong_length_query_is_refused():
    retriever = MockQdrantRetriever(DIMENSION)
    with pytest.raises(RecognitionError) as caught:
        retriever.search([0.1] * 8, top_k=5)
    assert caught.value.code is ErrorCode.EMBEDDING_DIMENSION_MISMATCH


def test_points_with_incomplete_payloads_are_rejected_not_guessed():
    points = [
        {"vector_seed": "a", "payload": {
            "point_id": "good", "species_id": "panthera_leo", "scientific_name": "Panthera leo",
            "dataset_version": "v0", "embedding_mode": "mock_bioclip2"}},
        {"vector_seed": "b", "payload": {
            "point_id": "bad", "scientific_name": "Missing species_id",
            "dataset_version": "v0", "embedding_mode": "mock_bioclip2"}},
    ]
    retriever = MockQdrantRetriever(DIMENSION, points=points)
    outcome = retriever.search(deterministic_vector("a", DIMENSION), top_k=10)

    assert len(outcome.references) == 1
    assert outcome.rejected_payloads == 1


def test_empty_collection_returns_no_hits():
    retriever = MockQdrantRetriever(DIMENSION, points=[])
    outcome = retriever.search(deterministic_vector("a", DIMENSION), top_k=5)
    assert outcome.references == []


@pytest.mark.parametrize(
    "code", [ErrorCode.RETRIEVAL_UNAVAILABLE, ErrorCode.RETRIEVAL_TIMEOUT]
)
def test_failure_branches_are_reachable_offline(code):
    retriever = MockQdrantRetriever(DIMENSION, fail_with=code)
    with pytest.raises(RecognitionError) as caught:
        retriever.search(deterministic_vector("a", DIMENSION), top_k=5)
    assert caught.value.code is code


# --- the real adapter ------------------------------------------------------

def _config(**overrides) -> QdrantConfig:
    base = dict(
        url=None, collection=None, vector_name=None, expected_dimension=None,
        expected_distance=None, dataset_version=None, top_k_references=None,
        timeout_seconds=10.0,
    )
    base.update(overrides)
    return QdrantConfig(**base)


def test_real_adapter_refuses_to_start_without_the_contract():
    with pytest.raises(RecognitionError) as caught:
        RealQdrantRetriever(_config())
    assert caught.value.code is ErrorCode.QDRANT_CONTRACT_NOT_FROZEN


@pytest.mark.parametrize(
    "missing",
    ["url", "collection", "expected_dimension", "expected_distance",
     "dataset_version", "top_k_references"],
)
def test_every_contract_field_is_required(missing):
    complete = dict(
        url="http://example.invalid:6333", collection="c", expected_dimension=32,
        expected_distance="Cosine", dataset_version="v1", top_k_references=30,
    )
    complete[missing] = None

    with pytest.raises(RecognitionError) as caught:
        RealQdrantRetriever(_config(**complete))
    assert caught.value.code is ErrorCode.QDRANT_CONTRACT_NOT_FROZEN


def test_missing_contract_fields_are_reported():
    config = _config(url="http://example.invalid:6333")
    assert "collection" in config.missing_contract_fields
    assert "url" not in config.missing_contract_fields
    assert config.is_frozen is False


def test_a_complete_contract_is_frozen():
    config = _config(
        url="http://example.invalid:6333", collection="c", expected_dimension=32,
        expected_distance="Cosine", dataset_version="v1", top_k_references=30,
    )
    assert config.is_frozen is True
    assert config.missing_contract_fields == ()


def test_real_adapter_is_read_only_by_construction():
    """Ingestion belongs to the Qdrant owner. This class has no way to do it."""
    forbidden = ("upsert", "create_collection", "delete", "delete_collection",
                 "recreate_collection", "upload_points", "upload_collection", "set_payload")
    for name in forbidden:
        assert not hasattr(RealQdrantRetriever, name), name


def test_real_adapter_declares_real_mode():
    assert RealQdrantRetriever.mode == "real_minimal"
    assert RealQdrantRetriever.provider_name == "Qdrant"
