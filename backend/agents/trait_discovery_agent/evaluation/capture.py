"""
Step 4 — run retrieval and capture what actually came back.

One shared helper, `capture_node`, used for all four retrieval-facing nodes
(gene_mapper, pathways, protein_data, literature_support) instead of
duplicating the same "run node -> pull context -> pull answer" shape four
times, per the Sprint 4 guide.

Each node has a different "context" surface (see Step 1 of the guide):
  - gene_mapper       -> Qdrant `go_annotations`
  - pathways          -> Qdrant `kegg_pathways`
  - protein_data      -> Qdrant `uniprot_proteins`
  - literature_support -> cross-agent call to the Literature Agent (no
                           Qdrant collection -- kb/sources/literature_agent_client.py
                           explicitly never caches evidence content)

so `capture_node` takes a small per-node adapter describing how to get the
query text, which Qdrant collection (if any) to search, and how to turn the
subagent's structured output into a flat list of "claims" for the RAGAS-style
checks in rag_evaluators.py.

--- live-KB identifier resolution (added) ---------------------------------
gene_mapper_agent/pathways_agent/protein_data_agent are ID-based, not
symbol-based: they expect a pre-resolved UniProt accession / KEGG gene id in
`input.context`, never look one up themselves (see each subagent's
`_select_*_for_gene`). protein_data_agent is the one exception -- it takes a
bare NCBI taxonomy id and resolves the gene symbol itself via UniProt's
`gene:{symbol} AND organism_id:{tax_id}` search.

So for a *live* run (this file, not the mock-only path) we resolve those ids
here, once per gene, before calling the real subagent:
  - protein_data: species_name -> tax_id (SPECIES_TAX_ID), nothing else needed.
  - gene_mapper:  needs a UniProt accession. We get it for free from the same
    UniProt lookup used for protein_data (list_uniprot_candidates), so no
    separate resolution step.
  - pathways:     needs a KEGG gene id (e.g. "hsa:7157"). KEGG's `conv`
    endpoint maps a UniProt accession straight to it, so we chain off the
    same accession. See `_resolve_kegg_gene_id` below.

None of this changes the subagents' own contracts (workflows/nodes/*.py
still populate `context` the same way they always did) -- it's purely how
this eval script fills in the same context dict for an offline gene list
that doesn't come with pre-resolved ids attached, per the module docstring's
"swap in the real ones" note.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.retrieval import RetrievedDocument, semantic_search  # noqa: E402
from kb.sources.kegg_client import _assert_kegg_academic_use_only  # noqa: E402
from kb.sources.uniprot_client import _list_uniprot_candidates_raw  # noqa: E402
from schemas.inputs import (  # noqa: E402
    GeneMapperInput,
    LiteratureSupportInput,
    PathwaysInput,
    ProteinDataInput,
)
from subagents.gene_mapper import gene_mapper_agent  # noqa: E402
from subagents.literature_support.mock import mock_literature_support  # noqa: E402
from subagents.pathways import pathways_agent  # noqa: E402
from subagents.protein_data import protein_data_agent  # noqa: E402

# NCBI taxonomy id / KEGG organism code per species_name value used in
# rag_test_cases.yaml. Add an entry here before adding a gene from a new
# species to the eval set.
SPECIES_TAX_ID: dict[str, int] = {
    "Homo sapiens": 9606,
    "Mus musculus": 10090,
}
SPECIES_KEGG_ORG: dict[str, str] = {
    "Homo sapiens": "hsa",
    "Mus musculus": "mmu",
}

KEGG_CONV_URL = "https://rest.kegg.jp/conv/{org}/uniprot:{accession}"


async def _resolve_uniprot_accession(gene: str, tax_id: int) -> Optional[str]:
    """First reviewed UniProt accession for gene+species, or None on zero hits.

    Reuses the exact call protein_data_agent makes internally
    (list_uniprot_candidates) so "does this gene resolve at all" is answered
    the same way for both the protein_data context and as the upstream input
    gene_mapper/pathways need.
    """
    candidates = await _list_uniprot_candidates_raw(gene, tax_id)
    return candidates[0]["source_accession"] if candidates else None


async def _resolve_kegg_gene_id(accession: str, org_code: str) -> Optional[str]:
    """UniProt accession -> KEGG gene id (e.g. "hsa:7157") via KEGG's `conv`
    endpoint. Returns None if KEGG has no entry for this accession."""
    _assert_kegg_academic_use_only()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(KEGG_CONV_URL.format(org=org_code, accession=accession))
        resp.raise_for_status()
        line = resp.text.strip()
        if not line or "\t" not in line:
            return None
        _source, kegg_gene_id = line.split("\t", 1)
        return kegg_gene_id.strip() or None


@dataclass
class NodeCapture:
    """Everything Step 4 asks us to capture, for one gene at one node."""

    node: str
    gene: str
    collection: Optional[str]          # None for literature_support (no Qdrant surface)
    query_text: str
    context_chunks: list[RetrievedDocument] = field(default_factory=list)
    answer_claims: list[str] = field(default_factory=list)
    # Separate from answer_claims on purpose: faithfulness/context_recall
    # need the bare, literally-retrieved fact (does this term/summary trace
    # back to a retrieved chunk?), while answer_relevancy needs something
    # that reads as a passage (see rag_evaluators._claim_as_passage). Folding
    # the LLM's free-text reasoning into answer_claims made faithfulness
    # collapse (go_annotations 1.00 -> 0.00, kegg_pathways 1.00 -> 0.33) --
    # the reasoning legitimately introduces words absent from the retrieved
    # chunk, which token-overlap faithfulness reads as hallucination even
    # though the *fact itself* (the picked term) is still fully grounded.
    # Defaults to answer_claims for nodes with nothing extra to add
    # (protein_data, literature_support already return sentence-shaped
    # text); gene_mapper/pathways override it below with name+reasoning.
    relevancy_claims: list[str] = field(default_factory=list)
    raw_output: Any = None
    error: Optional[str] = None
    # True when zero claims is a genuine "nothing to report" outcome (no
    # UniProt/KEGG id to resolve, zero reviewed hits, etc.) rather than the
    # answer being off-topic. Keeps answer_relevancy from being penalized for
    # a coverage gap it wasn't designed to measure -- see run_eval.py.
    unresolved: bool = False


async def _capture_context(
    collection: Optional[str], query_text: str, top_k: int, gene: Optional[str] = None
) -> list[RetrievedDocument]:
    """Semantic search, scoped to the gene under evaluation when one is given.

    Without this filter, `query_text` (just "{trait_name} {gene}") is the
    *only* thing keeping results on-topic, and it isn't a reliable enough
    embedding signal to beat cross-gene collisions -- exactly the cases
    rag_test_cases.yaml deliberately includes (HR vs HRAS, MARCH1 vs the
    calendar month, ASIP vs its "Agouti" synonym). Every writer
    (gene_mapper/pathways/protein_data __init__.py) stores the queried gene
    verbatim as payload["gene_symbol"], so an exact-match filter on the same
    string is safe and doesn't need fuzzy/case handling.

    Because that filter already pins every result to the correct gene,
    raising top_k here costs nothing on context_precision (it's a hard
    filter, not a re-rank) but directly buys context_recall: a
    well-annotated gene can have dozens of real GO biological_process
    terms, and the old top_k=5 was throwing away everything past the 5
    most semantically-similar-to-the-query-text ones before recall was
    even computed. Bumped from 5 -> 25 in the capture_* defaults below.
    """
    if collection is None:
        return []
    filters = {"gene_symbol": gene} if gene else None
    return await semantic_search(collection, query_text, top_k=top_k, filters=filters)


async def capture_gene_mapper(gene: str, trait_name: str, species_name: str, top_k: int = 25) -> NodeCapture:
    query_text = f"{trait_name} {gene}"
    cap = NodeCapture(node="gene_mapper", gene=gene, collection="go_annotations", query_text=query_text)
    try:
        tax_id = SPECIES_TAX_ID.get(species_name)
        accession = await _resolve_uniprot_accession(gene, tax_id) if tax_id else None
        if accession is None:
            # No reviewed UniProt entry to key off of -> QuickGO can't be
            # queried either. Report it as gene_mapper's own unmatched case
            # rather than silently returning an empty answer with status
            # COMPLETED-looking metrics.
            cap.raw_output = None
            cap.answer_claims = []
            cap.unresolved = True
            # Nothing was ever going to be written to Qdrant on this branch
            # (the agent is never called), so capturing context here vs.
            # earlier makes no difference -- kept after the branch purely so
            # both branches share one capture call at the end of this function.
            cap.context_chunks = await _capture_context("go_annotations", query_text, top_k, gene=gene)
            return cap

        out = await gene_mapper_agent(GeneMapperInput(
            trait_name=trait_name, gene_list=[gene], species_name=species_name,
            instruction=f"Map GO annotation for {gene}",
            context={"uniprot_accessions": {gene: accession}},
        ))
        cap.raw_output = out
        picks = [a for a in out.go_annotations if a.gene_symbol == gene]
        # answer_claims stays the bare go_name: it's the literal fact that
        # must trace back to a retrieved chunk (faithfulness/context_recall).
        cap.answer_claims = [a.go_name for a in picks]
        # relevancy_claims adds the LLM's own reasoning where one exists
        # (multi-candidate pick) -- see NodeCapture.relevancy_claims and
        # rag_evaluators._claim_as_passage for why a bare term needs this to
        # read as a passage. Single-candidate/fallback picks have no
        # reasoning to add, so those fall back to the bare go_name too.
        cap.relevancy_claims = [
            f"{a.go_name}. {a.reasoning}".strip() if a.reasoning else a.go_name
            for a in picks
        ]
        cap.unresolved = not cap.answer_claims
        # Capture context AFTER the agent call, not before: gene_mapper_agent
        # writes this gene's own pick *and* every other real QuickGO
        # candidate into Qdrant as it runs (the "KB enrichment" fix). Search-
        # ing beforehand scores context_recall/precision against whatever
        # existed prior to this call -- for any gene queried for the first
        # time that's an empty collection, which floors recall at 0.00
        # regardless of retrieval quality (this is what was driving
        # sparse_go=0.00 and the low happy_path average -- a cold-KB
        # artifact, not a retrieval defect). Searching after this call means
        # the score reflects the KB state a real subsequent query would see.
        cap.context_chunks = await _capture_context("go_annotations", query_text, top_k, gene=gene)
    except Exception as exc:  # pragma: no cover - live-dependency failures surface as eval findings
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


async def capture_pathways(gene: str, trait_name: str, species_name: str, top_k: int = 25) -> NodeCapture:
    query_text = f"{trait_name} {gene} pathway"
    cap = NodeCapture(node="pathways", gene=gene, collection="kegg_pathways", query_text=query_text)
    try:
        tax_id = SPECIES_TAX_ID.get(species_name)
        org_code = SPECIES_KEGG_ORG.get(species_name)
        kegg_gene_id = None
        if tax_id and org_code:
            accession = await _resolve_uniprot_accession(gene, tax_id)
            if accession:
                kegg_gene_id = await _resolve_kegg_gene_id(accession, org_code)

        if kegg_gene_id is None:
            cap.raw_output = None
            cap.answer_claims = []
            cap.unresolved = True
            cap.context_chunks = await _capture_context("kegg_pathways", query_text, top_k, gene=gene)
            return cap

        out = await pathways_agent(PathwaysInput(
            gene_list=[gene], trait_name=trait_name,
            instruction=f"Find pathway for {gene}",
            context={"kegg_gene_ids": {gene: kegg_gene_id}},
        ))
        cap.raw_output = out
        # Same split as capture_gene_mapper: bare pathway_name for
        # faithfulness/context_recall grounding, name+reasoning for
        # answer_relevancy's passage-shaped scoring.
        cap.answer_claims = [p.pathway_name for p in out.pathways]
        cap.relevancy_claims = [
            f"{p.pathway_name}. {p.reasoning}".strip() if p.reasoning else p.pathway_name
            for p in out.pathways
        ]
        cap.unresolved = not cap.answer_claims
        # See capture_gene_mapper: capture context after the write, not before.
        cap.context_chunks = await _capture_context("kegg_pathways", query_text, top_k, gene=gene)
    except Exception as exc:  # pragma: no cover
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


async def capture_protein_data(gene: str, trait_name: str, species_name: str, top_k: int = 25) -> NodeCapture:
    query_text = f"{trait_name} {gene} protein function"
    cap = NodeCapture(node="protein_data", gene=gene, collection="uniprot_proteins", query_text=query_text)
    try:
        tax_id = SPECIES_TAX_ID.get(species_name)
        if tax_id is None:
            cap.error = f"no tax_id mapping for species_name={species_name!r}"
            return cap

        out = await protein_data_agent(ProteinDataInput(
            gene_list=[gene], trait_name=trait_name,
            instruction=f"Find protein function for {gene}",
            context={"tax_id": tax_id},
        ))
        cap.raw_output = out
        cap.answer_claims = [p.function_summary for p in out.proteins]
        # protein_data_agent degrades zero-hit genes into missing_genes
        # rather than raising (see subagents/protein_data/__init__.py), so a
        # gene landing there is a genuine "no reviewed entry", same
        # coverage-gap shape as the gene_mapper/pathways unresolved case
        # above -- not an off-topic answer.
        cap.unresolved = not cap.answer_claims and gene in (out.missing_genes or [])
        # See capture_gene_mapper: capture context after the write, not
        # before, so recall/precision reflect this call's own enrichment.
        cap.context_chunks = await _capture_context("uniprot_proteins", query_text, top_k, gene=gene)
    except Exception as exc:  # pragma: no cover
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


async def capture_literature_support(gene: str, trait_name: str) -> NodeCapture:
    # No Qdrant collection for this node (see module docstring) -- "context" is
    # whatever the Literature Agent cross-agent call returned this run.
    cap = NodeCapture(node="literature_support", gene=gene, collection=None, query_text=trait_name)
    try:
        out = await mock_literature_support(LiteratureSupportInput(
            trait_name=trait_name, gene_list=[gene],
            instruction=f"Find literature support for {gene}", context={},
        ))
        cap.raw_output = out
        cap.context_chunks = [
            RetrievedDocument(id=r.pmid, score=1.0, payload={
                "pmid": r.pmid, "title": r.title, "year": r.year, "short_summary": r.short_summary,
            })
            for r in out.evidence
        ]
        cap.answer_claims = [r.short_summary for r in out.evidence]
        # Zero evidence here means the mock/live literature source has
        # nothing indexed for this trait+gene (e.g. mock_literature_support's
        # _MOCK_LITERATURE_DB only covers a couple of trait_name keys) -- a
        # coverage gap, same shape as gene_mapper/pathways/protein_data's
        # unresolved case, not an off-topic answer. Without this,
        # answer_relevancy() short-circuits empty claims to a hard 0.0 and
        # run_eval.py has no unresolved flag to exclude it on, silently
        # folding every zero-evidence gene into the average as if it were a
        # bad (rather than absent) answer.
        cap.unresolved = not cap.answer_claims
    except Exception as exc:  # pragma: no cover
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


CAPTURE_FNS: dict[str, Callable[..., Awaitable[NodeCapture]]] = {
    "gene_mapper": capture_gene_mapper,
    "pathways": capture_pathways,
    "protein_data": capture_protein_data,
    "literature_support": capture_literature_support,
}
