"""
Protein Data Agent — UniProt reviewed-entry selection with LLM-guided
disambiguation.

For the same gene list Pathways receives (they run in parallel — see the
Functional Evidence subgraph):
  1. List every reviewed UniProt hit for the gene+species pair.
  2. If 0 hits    → missing_genes (no LLM call).
  3. If 1 hit     → straight through (no real decision needed).
  4. If >1 hits   → LLM holds list_uniprot_candidates as a bound tool via a
                    bind_tools loop (llm_pick.py) and picks the trait-relevant
                    entry itself, with deterministic fallback on LLM failure
                    (§9).

Everything that gets called/monkeypatched by name in tests
(list_uniprot_candidates, fetch_uniprot, _llm_pick_protein) is imported
directly into this module's namespace, and protein_data_agent()/
_select_protein_for_gene() — which look those names up as bare globals —
live here too, so patching this module's attributes actually changes what
they call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from schemas.inputs import ProteinDataInput
from schemas.outputs import ProteinDataOutput, ProteinEntry
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.uniprot_client import (
    fetch_uniprot,
    _list_uniprot_candidates_raw as list_uniprot_candidates,
)

from .llm_pick import _llm_pick_protein
from .mock import mock_protein_data_agent

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

__all__ = ["protein_data_agent", "mock_protein_data_agent"]


class _NoReviewedHits(Exception):
    """Raised when list_uniprot_candidates succeeded but came back empty —
    distinguishes a genuine zero-hit gene (§8: skip straight to missing_genes,
    no fallback call) from an LLM-pick failure or a candidate-lookup error
    (§9: worth retrying via the deterministic fallback)."""


async def _select_protein_for_gene(
    gene: str, tax_id: int, trait_name: str
) -> ProteinEntry | None:
    """
    Raises _NoReviewedHits on zero hits — caller records it in missing_genes
    without ever trying an LLM call or a fallback fetch for that gene (a
    fresh identical query would just re-confirm the same empty result).
    Returns None only when a real decision was attempted and failed (LLM
    pick), which the caller falls back to fetch_uniprot for.
    """
    # --- Discover candidates (§9: surfaced, one retry handled by the caller) ---
    candidates = await list_uniprot_candidates(gene, tax_id)
    if not candidates:
        raise _NoReviewedHits(gene)  # zero reviewed hits, §8: never reaches the LLM

    # --- one hit: straight through, no real decision (§8) ---
    if len(candidates) == 1:
        c = candidates[0]
        return ProteinEntry(
            gene_symbol=gene,
            protein_name=c["protein_name"],
            function_summary=c["function_summary"],
            source_accession=c["source_accession"],
        )

    # --- several hits: LLM pick via bind_tools (§8) ---
    try:
        accession, protein_name, function_summary, reasoning = await _llm_pick_protein(
            trait_name, gene, candidates, tax_id
        )
        # --- grounding rule (§0.1): validate against actual tool output ---
        valid_accessions = {c["source_accession"] for c in candidates}
        if accession not in valid_accessions:
            raise RuntimeError(
                f"LLM picked invalid source_accession {accession} not in {valid_accessions}"
            )
        logger.info(
            "LLM picked %s (%s) for %s: %s", accession, protein_name, gene, reasoning
        )
        return ProteinEntry(
            gene_symbol=gene,
            protein_name=protein_name,
            function_summary=function_summary,
            source_accession=accession,
        )
    except Exception as exc:
        logger.warning("LLM pick failed for %s, deferring to fallback: %s", gene, exc)
        return None  # caller falls back to fetch_uniprot (§9)


async def protein_data_agent(input: ProteinDataInput) -> ProteinDataOutput:
    tax_id = input.context.get("tax_id")
    proteins: list[ProteinEntry] = []
    missing: list[str] = []

    for gene in input.gene_list:
        try:
            entry = await _select_protein_for_gene(gene, tax_id, input.trait_name)
        except _NoReviewedHits:
            # §8: genuine zero-hit gene — skip the fallback call entirely,
            # a fresh identical query would just re-confirm the same result.
            missing.append(gene)
            continue
        except Exception as exc:
            logger.warning("list_uniprot_candidates failed for %s: %s", gene, exc)
            entry = None

        # --- Deterministic fallback (§9) --------------------------------
        if entry is None:
            logger.info("Falling back to deterministic fetch_uniprot for %s", gene)
            try:
                entry = await fetch_uniprot(gene, tax_id)
            except Exception as exc:
                logger.warning("Deterministic fallback also failed for %s: %s", gene, exc)
                entry = None

        if entry is None:
            missing.append(gene)
            continue

        # --- Cache / dedup (§6) ------------------------
        dedup_key = f"uniprot:{entry.source_accession}:{tax_id}"
        cached = await get_cached("uniprot_proteins", dedup_key)
        if cached:
            proteins.append(entry)
            continue

        await upsert_point(
            "uniprot_proteins",
            dedup_key,
            text_to_embed=entry.function_summary,
            payload={
                "gene_symbol": entry.gene_symbol,
                "protein_name": entry.protein_name,
                "function_summary": entry.function_summary,
                "species_tax_id": tax_id,
                "source": "UniProt REST API",
                "source_accession": entry.source_accession,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        proteins.append(entry)

    # §9: No proteins resolved at all → status=FAILED
    status = AgentStatus.COMPLETED if proteins else AgentStatus.FAILED
    return ProteinDataOutput(status=status, proteins=proteins, missing_genes=missing)
