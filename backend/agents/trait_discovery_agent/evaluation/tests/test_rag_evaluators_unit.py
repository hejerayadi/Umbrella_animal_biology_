"""
Offline unit tests for rag_evaluators.py — no Qdrant/Neo4j/LLM needed, just
synthetic RetrievedDocument objects. These exist to (a) prove the metric
functions themselves are correct before trusting a live scorecard, and (b)
cover Step 6.2 of the guide directly: a malformed/incomplete payload must
fail loudly via validate_document(), not be silently treated as complete.

answer_relevancy is cross-encoder-based (see rag_evaluators.py), so these
tests monkeypatch rag_evaluators._score_relevance_pairs with a tiny
deterministic bag-of-words scorer instead of loading the real ~90MB MS
MARCO model -- keeps this suite fast and dependency-free while still
exercising the real per-claim scoring/off-topic-flagging code path (not
just re-testing token overlap under a new name).
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rag_evaluators  # noqa: E402
from kb.retrieval import RetrievedDocument, validate_document  # noqa: E402
from rag_evaluators import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    payload_contract,
)

# Small fixed vocabulary shared by all relevancy test cases below -- enough
# to make "shares a concept with the query" score high and "shares nothing"
# score exactly 0, without pulling in the real cross-encoder model.
_FAKE_RELEVANCE_VOCAB = {
    "response", "cold", "hallmark", "adaptation", "caffeine",
    "metabolism", "pathway", "unrelated",
}


def _fake_score_relevance_pairs(query: str, claims: list[str]) -> list[float]:
    query_tokens = rag_evaluators._tokenize(query) & _FAKE_RELEVANCE_VOCAB
    scores = []
    for claim in claims:
        claim_tokens = rag_evaluators._tokenize(claim) & _FAKE_RELEVANCE_VOCAB
        scores.append(1.0 if (query_tokens & claim_tokens) else 0.0)
    return scores


@pytest.fixture(autouse=True)
def _patch_relevance_scorer(monkeypatch):
    monkeypatch.setattr(rag_evaluators, "_score_relevance_pairs", _fake_score_relevance_pairs)


def _doc(id_: str, gene_symbol: str, **extra) -> RetrievedDocument:
    payload = {"gene_symbol": gene_symbol, **extra}
    return RetrievedDocument(id=id_, score=0.9, payload=payload)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def test_faithfulness_grounded_claim_scores_high():
    chunks = [_doc("1", "UCP1", go_name="response to cold")]
    result = faithfulness(["response to cold"], chunks)
    assert result.score == 1.0


def test_faithfulness_hallucinated_claim_scores_low():
    chunks = [_doc("1", "UCP1", go_name="response to cold")]
    result = faithfulness(["completely unrelated invented claim about spaceflight"], chunks)
    assert result.score == 0.0
    assert "unrelated invented claim about spaceflight" in result.detail


def test_faithfulness_no_claims_is_vacuously_faithful():
    result = faithfulness([], [])
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Answer relevancy
# ---------------------------------------------------------------------------

def test_answer_relevancy_on_topic():
    # claim shares the underlying concept with the gene/trait query -- with
    # the fake embedder that means shared vocabulary tokens (cold/adaptation).
    result = asyncio.run(answer_relevancy(
        ["response to cold is a hallmark of cold adaptation"], gene="UCP1", trait_name="cold adaptation"
    ))
    assert result.score > 0.0


def test_answer_relevancy_off_topic_pathway_flagged():
    # the guide's own example: a pathways answer about an unrelated pathway
    result = asyncio.run(answer_relevancy(
        ["unrelated caffeine metabolism pathway"], gene="UCP1", trait_name="cold adaptation"
    ))
    assert result.score == 0.0
    assert "unrelated caffeine metabolism pathway" in result.detail


def test_answer_relevancy_no_answer_scores_zero():
    result = asyncio.run(answer_relevancy([], gene="UCP1", trait_name="cold adaptation"))
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Context precision
# ---------------------------------------------------------------------------

def test_context_precision_all_relevant():
    chunks = [_doc("1", "UCP1"), _doc("2", "UCP1")]
    result = context_precision(chunks, gene="UCP1")
    assert result.score == 1.0


def test_context_precision_ambiguous_symbol_collision():
    # exactly the HR vs HRAS failure mode the gold dataset targets
    chunks = [_doc("1", "HRAS"), _doc("2", "HR"), _doc("3", "HRAS")]
    result = context_precision(chunks, gene="HRAS")
    assert round(result.score, 2) == round(2 / 3, 2)
    assert "2" in result.detail


def test_context_precision_empty_retrieval():
    result = context_precision([], gene="TRPV3")
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Context recall
# ---------------------------------------------------------------------------

def test_context_recall_full_coverage():
    chunks = [_doc("1", "TP53", go_name="DNA damage response apoptotic process")]
    result = context_recall(["DNA damage response", "apoptotic process"], chunks)
    assert result.score == 1.0


def test_context_recall_partial_gap_reported():
    chunks = [_doc("1", "HR", go_name="hair cycle")]
    result = context_recall(["hair cycle", "negative regulation of Wnt signaling pathway"], chunks)
    assert result.score == 0.5
    assert "negative regulation of Wnt signaling pathway" in result.detail


def test_context_recall_no_expected_terms_is_vacuous_pass():
    result = context_recall([], [])
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Step 6 — payload schema contract
# ---------------------------------------------------------------------------

def test_payload_contract_normal_retrieval_all_valid():
    chunks = [
        _doc(
            "1", "UCP1", go_id="GO:0009408", go_name="response to cold",
            source="GO REST API (QuickGO)", ingested_at="2026-01-01T00:00:00+00:00",
            schema_version=1,
        )
    ]
    check = payload_contract("go_annotations", chunks)
    assert check.valid_chunks == 1
    assert check.failures == []


def test_payload_contract_malformed_payload_fails_loudly():
    # Step 6.2: simulate a malformed/incomplete payload -- missing go_name
    # and schema_version, which validate_document() must catch, not silently
    # pass through as if complete.
    chunks = [_doc("1", "UCP1", go_id="GO:0009408", source="GO REST API (QuickGO)")]
    check = payload_contract("go_annotations", chunks)
    assert check.valid_chunks == 0
    assert len(check.failures) == 1
    assert "go_name" in check.failures[0]
    assert "schema_version" in check.failures[0]


def test_validate_document_directly_flags_missing_fields():
    # same check, bypassing payload_contract(), straight against the
    # architecture's own REQUIRED_FIELDS contract in kb/retrieval.py
    result = validate_document("uniprot_proteins", {"gene_symbol": "UCP1"})
    assert not result.is_valid
    assert "function_summary" in result.missing_fields
    assert "protein_name" in result.missing_fields
