"""
Gene Mapper Agent — GO biological-process annotation with LLM-guided
disambiguation.

For each gene:
  1. List all QuickGO biological_process candidates for the gene's UniProt
     accession (with one retry on transient QuickGO errors, §9).
  2. If 0 candidates → unmatched (no LLM call).
  3. If 1 candidate  → straight through (no real decision needed).
  4. If >1 candidates → LLM holds list_go_candidates/resolve_go_term_name as
                         bound tools via a bind_tools loop (llm_pick.py) and
                         picks the most trait-relevant GO term itself, with
                         deterministic fallback on LLM failure (§9).

Everything that gets monkeypatched by name in tests (list_go_candidates,
resolve_go_term_name, fetch_go_annotation, _llm_pick_candidate) is imported
directly into this module's namespace, and gene_mapper_agent() — which looks
those names up as bare globals — lives here too, so patching this module's
attributes actually changes what gene_mapper_agent() calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from schemas.inputs import GeneMapperInput
from schemas.outputs import GeneMapperOutput, GOAnnotation
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.go_client import (
    fetch_go_annotation,
    _list_go_candidates_raw as list_go_candidates,
    _resolve_go_term_name_raw as resolve_go_term_name,
)

from .llm_pick import _llm_pick_candidate
from .mock import mock_gene_mapper

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

__all__ = ["gene_mapper_agent", "mock_gene_mapper"]


async def gene_mapper_agent(input: GeneMapperInput) -> GeneMapperOutput:
    annotations: list[GOAnnotation] = []
    unmatched: list[str] = []

    for gene in input.gene_list:
        uniprot_accession = input.context.get("uniprot_accessions", {}).get(gene)
        if not uniprot_accession:
            # §8: nothing to disambiguate — never reaches the LLM.
            unmatched.append(gene)
            continue

        # ---- fetch candidates from QuickGO (with one retry, §9) ----
        candidates: list[dict] = []
        try:
            candidates = await list_go_candidates(gene, uniprot_accession)
        except Exception as exc:
            logger.warning("QuickGO error for %s, retrying once: %s", gene, exc)
            try:
                candidates = await list_go_candidates(gene, uniprot_accession)
            except Exception as exc2:
                logger.error("QuickGO failed for %s after retry: %s", gene, exc2)
                unmatched.append(gene)
                continue

        if not candidates:
            unmatched.append(gene)
            continue

        # ---- single candidate: straight through, no LLM (§8) ----
        if len(candidates) == 1:
            go_id = candidates[0]["go_id"]
            go_name = candidates[0].get("go_name") or await resolve_go_term_name(go_id)
            if not go_name:
                unmatched.append(gene)
                continue
            entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)

        # ---- multi-candidate: LLM pick via bind_tools (§8) ----
        else:
            try:
                go_id, go_name, reasoning = await _llm_pick_candidate(
                    input.trait_name, gene, candidates, uniprot_accession
                )
                # ---- grounding rule (§0.1): validate against actual tool output ----
                valid_ids = {c["go_id"] for c in candidates}
                if go_id not in valid_ids:
                    raise RuntimeError(
                        f"LLM picked invalid go_id {go_id} not in {valid_ids}"
                    )
                logger.info("LLM picked %s (%s) for %s: %s", go_id, go_name, gene, reasoning)
                entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)
            except Exception as exc:
                # ---- LLM/NIM unavailable or invalid → deterministic fallback (§9) ----
                logger.warning(
                    "LLM pick failed for %s, falling back to first candidate: %s", gene, exc
                )
                fallback = await fetch_go_annotation(gene, uniprot_accession)
                if fallback is None:
                    unmatched.append(gene)
                    continue
                entry = fallback

        # ---- cache layer (§6) ----
        dedup_key = f"go:{entry.go_id}:{entry.gene_symbol}"
        cached = await get_cached("go_annotations", dedup_key)
        if cached:
            annotations.append(entry)
            continue

        await upsert_point(
            "go_annotations",
            dedup_key,
            text_to_embed=entry.go_name,
            payload={
                "gene_symbol": entry.gene_symbol,
                "go_id": entry.go_id,
                "go_name": entry.go_name,
                "source": "GO REST API (QuickGO)",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        annotations.append(entry)

    # §9: FAILED if no annotations resolved OR any gene is unmatched
    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(
        status=status,
        go_annotations=annotations,
        unmatched_genes=unmatched,
    )
