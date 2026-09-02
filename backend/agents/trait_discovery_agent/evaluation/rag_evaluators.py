"""
Step 5 — the four RAGAS-style checks, implemented as plain comparison/overlap
functions (guide: "enough for a first pass" -- swap in the real `ragas`
library metric functions later if needed, keeping one function per metric so
scores stay reportable separately per node rather than one blended number).

Step 6 — payload schema contract check, wired in here via
`payload_contract()` so it shows up in the same scorecard rather than as a
separate ad-hoc test.

All four metrics operate on plain strings/lists so they work the same way
regardless of which node produced them -- capture.py is what's responsible
for turning each node's structured output into that flat shape.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from kb.retrieval import RetrievedDocument, ValidationResult, validate_document

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "via", "with", "gene", "protein",
}

# Below this relevance score a claim is flagged as off-topic in the detail
# string (the score itself is still the raw model output, not this threshold
# -- this only controls what gets called out for a human to look at).
_RELEVANCY_OFF_TOPIC_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# answer_relevancy scoring backend
#
# A first pass here used cosine similarity between a full question and each
# bare claim, embedded with the same all-MiniLM-L6-v2 model kb/retrieval.py
# uses for search. That flattened every collection into a 0.16-0.40 band
# even though faithfulness and context_precision both sat at ~1.00 -- i.e.
# the answers *were* correct, the metric just couldn't say so. Reformulating
# the query into several shorter, same-register anchors and max-pooling
# barely moved the number (confirmed against a live run), which rules out
# "wrong phrasing" as the cause: a bi-encoder trained for general sentence
# *similarity* just doesn't produce high absolute cosine scores for two
# genuinely-related but differently-worded short phrases -- there's no
# amount of query reformulation that fixes a ceiling built into the model.
#
# A cross-encoder trained specifically for query/passage *relevance*
# ranking (as opposed to paraphrase similarity) is the right tool for this
# job: it takes the (query, claim) pair together and outputs a single
# relevance judgment, which is what "is this claim relevant to the
# question" actually is. This only affects how the eval harness scores
# relevancy -- it doesn't touch kb/embeddings.py or require re-indexing
# Qdrant, since retrieval itself never used this model.
# ---------------------------------------------------------------------------

_cross_encoder = None

# Set RAG_EVAL_DEBUG_RELEVANCY=1 to log the *raw* (pre-sigmoid) cross-encoder
# logit for every single (query, claim) pair scored, plus a min/max/mean
# summary per call. answer_relevancy has been reported near-zero (0.00-0.13)
# across every collection -- including literature_support, whose claims come
# from a hardcoded mock dict, not an LLM -- which rules out "bad/garbled
# generated text" as the cause and points at this scoring step itself.
#
# Leading hypothesis: cross-encoder/ms-marco-MiniLM-L-6-v2 is trained on
# MS MARCO web query/passage pairs (full-sentence passages, search-engine
# click relevance). Our claims are short biomedical fragments -- bare GO
# term names, one-clause function summaries -- a very different text style
# from what the model calibrated its score scale on. It's plausible its raw
# logits for a *genuinely correct* match here still land around -3 to -8
# (sigmoid(-5) = 0.007, sigmoid(-3) = 0.047), which would look exactly like
# what the scorecard shows regardless of whether the claim is actually
# on-topic. This logging exists to check that against real numbers instead
# of guessing further -- if raw_scores for known-correct claims cluster
# solidly negative, the fix is recalibrating the score range (e.g.
# percentile/min-max rescaling against a per-query negative-control claim,
# or lowering the sigmoid temperature) rather than swapping models again.
import logging
import os

logger = logging.getLogger(__name__)
_DEBUG_RELEVANCY = os.environ.get("RAG_EVAL_DEBUG_RELEVANCY") == "1"


def _get_relevance_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _score_relevance_pairs(query: str, claims: list[str]) -> list[float]:
    """Raw (query, claim) relevance scores, sigmoid-mapped to [0, 1].

    Isolated into its own function (rather than inlined in answer_relevancy)
    so tests can monkeypatch this one call instead of needing the real
    ~90MB MS MARCO cross-encoder model.
    """
    if not claims:
        return []

    encoder = _get_relevance_cross_encoder()
    raw_scores = [float(s) for s in encoder.predict([(query, claim) for claim in claims])]
    sigmoid_scores = [max(0.0, min(1.0 / (1.0 + math.exp(-s)), 1.0)) for s in raw_scores]

    if _DEBUG_RELEVANCY:
        for claim, raw, sig in zip(claims, raw_scores, sigmoid_scores):
            logger.info(
                "[answer_relevancy] query=%r claim=%r raw_logit=%.3f sigmoid=%.4f",
                query, claim, raw, sig,
            )
        logger.info(
            "[answer_relevancy] batch summary: n=%d raw_min=%.3f raw_max=%.3f raw_mean=%.3f",
            len(raw_scores), min(raw_scores), max(raw_scores), sum(raw_scores) / len(raw_scores),
        )

    return sigmoid_scores


def _claim_as_passage(gene: str, claim: str) -> str:
    """Turn a bare claim into something that reads as a passage.

    cross-encoder/ms-marco-MiniLM-L-6-v2 was trained on MS MARCO web
    query/passage pairs: full sentences competing for search-engine click
    relevance. A short controlled-vocabulary fragment like a GO term name
    ("keratinization") or a KEGG pathway title ("Thermogenesis - Mus
    musculus (house mouse)") has no verb, no mention of the gene, and no
    mention of the trait -- it doesn't resemble a passage at all, so the
    model scores it as noise (raw logit around -10 to -11 in practice)
    regardless of whether it's actually the correct answer. Confirmed via
    RAG_EVAL_DEBUG_RELEVANCY logging: the *longer*, sentence-shaped claims
    in the very same batches (full UniProt function paragraphs, literature
    summaries that spell out "links FGF5 to hair length") score meaningfully
    higher, and none of that gap tracks with correctness -- every claim
    checked by hand was correct.

    The fix restates *which gene* the claim came from -- something the
    pipeline already knows for certain, since every claim scored here is
    already attached to one specific queried gene -- as an explicit
    sentence subject, e.g. "keratinization" becomes "KRT71: keratinization
    is reported.". It deliberately does NOT also splice trait_name into
    every claim: whether a claim actually relates to the trait is exactly
    the thing this metric is supposed to judge, so forcing the trait's
    words into every passage would inflate every score uniformly rather
    than fix the length/style bias, and would make a genuinely off-topic
    claim (e.g. a pathway with nothing to do with the queried trait) look
    on-topic just because the wrapper said so. Only the gene, and the
    claim's own text verbatim, go in.
    """
    claim = (claim or "").strip()
    if not claim:
        return claim
    return f"{gene}: {claim} is reported."


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _chunk_text(chunk: RetrievedDocument) -> str:
    """Flatten a retrieved document's payload into one searchable string."""
    payload = chunk.payload or {}
    parts = [
        str(payload.get(k, ""))
        for k in (
            "go_name", "pathway_name", "protein_name", "function_summary",
            "title", "short_summary", "reasoning",
        )
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Metric 1 — Faithfulness
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    score: float
    detail: str = ""


def faithfulness(answer_claims: list[str], context_chunks: list[RetrievedDocument]) -> MetricResult:
    """Does every claim in the generated answer trace back to something
    actually present in the retrieved context?

    First-pass implementation: for each claim, require meaningful token
    overlap with at least one retrieved chunk. A claim with no supporting
    chunk at all is the unfaithful (hallucinated) case this is meant to
    catch -- e.g. protein_data stating a function that isn't in any
    retrieved UniProt chunk.
    """
    if not answer_claims:
        return MetricResult(score=1.0, detail="no claims made, nothing to be unfaithful about")

    context_token_sets = [_tokenize(_chunk_text(c)) for c in context_chunks]
    unsupported: list[str] = []

    for claim in answer_claims:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            continue
        supported = any(
            len(claim_tokens & ctx_tokens) / len(claim_tokens) >= 0.5
            for ctx_tokens in context_token_sets
        )
        if not supported:
            unsupported.append(claim)

    score = 1.0 - (len(unsupported) / len(answer_claims))
    detail = f"unsupported claims: {unsupported}" if unsupported else "all claims grounded in retrieved context"
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Metric 2 — Answer relevancy
# ---------------------------------------------------------------------------

async def answer_relevancy(answer_claims: list[str], gene: str, trait_name: str) -> MetricResult:
    """Does the generated answer actually address the gene/trait asked about,
    rather than being generic or off-topic?

    Cross-encoder implementation: scores each claim against a
    "What is the role of {gene} in {trait_name}?" query as a direct
    relevance judgment (see _score_relevance_pairs), not paraphrase
    similarity. Replaces two earlier attempts that both undersold correct
    answers -- pure token overlap scored a claim as *completely* off-topic
    (0.0) whenever it shared zero literal words with the query (e.g. "p53
    signaling pathway" vs trait_name "tumor suppression": a correct answer,
    near-zero literal overlap); a bi-encoder cosine-similarity version fixed
    that false-negative but flattened every collection into a 0.16-0.40
    band regardless of answer quality, because a model trained for sentence
    *similarity* doesn't produce high absolute scores for two genuinely
    related but differently-worded short phrases -- confirmed in practice
    across several query reformulations, so it's the scoring approach, not
    the phrasing, that was capping every collection.
    """
    if not answer_claims:
        return MetricResult(score=0.0, detail="no answer produced")

    query = f"What is the role of {gene} in {trait_name}?"
    # Bare fragments (GO term names, KEGG pathway titles) don't read as
    # passages to a cross-encoder trained on MS MARCO passages -- see
    # _claim_as_passage for why -- so restate each one as a short sentence
    # before scoring. off_topic/detail below still report the *original*
    # claim text so the scorecard reads naturally.
    passages = [_claim_as_passage(gene, claim) for claim in answer_claims]
    # CrossEncoder.predict is a synchronous, CPU-bound call -- offload it so
    # this coroutine doesn't block the event loop the other capture_* calls
    # in run_eval.py share.
    scores = await asyncio.to_thread(_score_relevance_pairs, query, passages)

    off_topic = [claim for claim, sim in zip(answer_claims, scores) if sim < _RELEVANCY_OFF_TOPIC_THRESHOLD]
    score = sum(scores) / len(scores)
    detail = f"off-topic claims: {off_topic}" if off_topic else "answer stays on-topic for gene/trait"
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Metric 3 — Context precision
# ---------------------------------------------------------------------------

def context_precision(context_chunks: list[RetrievedDocument], gene: str) -> MetricResult:
    """Of the chunks retrieved, how many were actually about the right gene?

    First-pass implementation: a chunk is "relevant" if its payload's
    gene_symbol matches the queried gene (case-insensitive exact match --
    this is the check that catches the ambiguous/synonym-symbol cases like
    HR vs HRAS or MARCH1 vs the calendar month).
    """
    if not context_chunks:
        return MetricResult(score=0.0, detail="nothing retrieved")

    relevant = [
        c for c in context_chunks
        if str(c.payload.get("gene_symbol", "")).upper() == gene.upper()
    ]
    score = len(relevant) / len(context_chunks)
    irrelevant_ids = [c.id for c in context_chunks if c not in relevant]
    detail = (
        f"{len(relevant)}/{len(context_chunks)} chunks match gene_symbol={gene!r}"
        + (f"; off-gene hits: {irrelevant_ids}" if irrelevant_ids else "")
    )
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Metric 4 — Context recall
# ---------------------------------------------------------------------------

def _term_surfaced(term: str, context_blob: str, context_tokens: set[str]) -> bool:
    """A term counts as recalled if it's a literal substring, OR if every
    content word in it shows up somewhere in the retrieved context.

    Live QuickGO/UniProt text legitimately varies in ways an exact substring
    check punishes for no good reason -- hyphenation ("G-protein coupled"
    vs "G protein-coupled"), word order, or one field spelling out what
    another abbreviates. The token fallback still requires every meaningful
    word in the expected term to be present (not just any one of them), so
    it doesn't turn into a trivially lenient check -- it only forgives
    surface formatting, not missing content.
    """
    normalized_term = term.lower().replace("-", " ")
    normalized_blob = context_blob.replace("-", " ")
    if normalized_term in normalized_blob:
        return True
    term_tokens = _tokenize(term)
    return bool(term_tokens) and term_tokens.issubset(context_tokens)


def context_recall(expected_terms: list[str], context_chunks: list[RetrievedDocument]) -> MetricResult:
    """Of everything relevant that exists in the KB for this gene, how much
    did retrieval actually surface? Compares Step 3's expected_* list
    against the retrieved chunks' text (case-insensitive substring, with a
    token-subset fallback for surface-level wording differences -- see
    _term_surfaced).
    """
    if not expected_terms:
        return MetricResult(score=1.0, detail="no expected terms recorded for this gene")

    context_blob = " ".join(_chunk_text(c) for c in context_chunks).lower()
    context_tokens = _tokenize(context_blob)
    found, missing = [], []
    for term in expected_terms:
        if _term_surfaced(term, context_blob, context_tokens):
            found.append(term)
        else:
            missing.append(term)

    score = len(found) / len(expected_terms)
    detail = f"missing: {missing}" if missing else "all expected terms surfaced"
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Step 6 — payload schema contract check (architecture-specific, not a
# generic RAGAS metric, but reported in the same scorecard).
# ---------------------------------------------------------------------------

@dataclass
class ContractCheck:
    collection: str
    total_chunks: int
    valid_chunks: int
    failures: list[str] = field(default_factory=list)


def payload_contract(collection: str, context_chunks: list[RetrievedDocument]) -> ContractCheck:
    """Confirms every retrieved chunk satisfies kb/retrieval.py's
    REQUIRED_FIELDS contract for its collection -- Step 6.1. Any missing
    field here is exactly the "fail loudly on drift" signal the contract is
    designed to surface; Step 6.2 (deliberately malformed payload -> loud
    failure) belongs in a dedicated unit test against validate_document()
    directly (see tests/test_rag_evaluators_unit.py), not this scorecard
    pass, since it needs a synthetic bad payload rather than live data.
    """
    check = ContractCheck(collection=collection, total_chunks=len(context_chunks), valid_chunks=0)
    for chunk in context_chunks:
        result: ValidationResult = validate_document(collection, chunk.payload)
        if result.is_valid:
            check.valid_chunks += 1
        else:
            check.failures.append(f"id={chunk.id} missing={result.missing_fields}")
    return check