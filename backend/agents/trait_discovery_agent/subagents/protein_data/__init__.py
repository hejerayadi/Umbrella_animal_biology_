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


async def _enrich_other_protein_candidates(
    gene: str, tax_id: int, candidates: list[dict], picked_accession: str
) -> None:
    """Ingest every other reviewed UniProt hit for this gene+species besides
    the one being returned as this call's answer. Unlike gene_mapper/
    pathways, no extra network round-trip is needed here -- list_uniprot_
    candidates already returns full protein_name/function_summary for every
    candidate, not just IDs -- so this is purely widening what gets indexed
    from data we already fetched."""
    others = [c for c in candidates if c["source_accession"] != picked_accession]
    for c in others:
        if not c.get("function_summary"):
            continue
        cand_key = f"uniprot:{c['source_accession']}:{tax_id}"
        if await get_cached("uniprot_proteins", cand_key):
            continue
        await upsert_point(
            "uniprot_proteins",
            cand_key,
            text_to_embed=c["function_summary"],
            payload={
                "gene_symbol": gene,
                "protein_name": c["protein_name"],
                "function_summary": c["function_summary"],
                "species_tax_id": tax_id,
                "source": "UniProt REST API",
                "source_accession": c["source_accession"],
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )


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
        accession, _llm_protein_name, _llm_function_summary, reasoning = await _llm_pick_protein(
            trait_name, gene, candidates, tax_id
        )
        # --- grounding rule (§0.1): validate against actual tool output ---
        by_accession = {c["source_accession"]: c for c in candidates}
        if accession not in by_accession:
            raise RuntimeError(
                f"LLM picked invalid source_accession {accession} not in {set(by_accession)}"
            )
        # Use the ORIGINAL candidate's protein_name/function_summary, not the
        # model's own retyped copy of them. The grounding check above only
        # ever validated the accession -- nothing stopped the model from
        # paraphrasing or shortening function_summary when it echoed it back
        # in its JSON reply, and that paraphrase is what got embedded into
        # Qdrant (text_to_embed=entry.function_summary) and scored for
        # context_recall. Since this collection is dominated by single-
        # candidate genes (no "other candidates" to enrich against), the
        # winner's fidelity to the real UniProt text *is* most of the recall
        # score, so a dropped phrase here reads as a structural retrieval
        # ceiling rather than the one-off paraphrasing loss it actually is.
        # Pulling straight from `candidates` mirrors how the "other
        # candidates" enrichment below already avoids this exact risk.
        picked = by_accession[accession]
        protein_name = picked["protein_name"]
        function_summary = picked["function_summary"]
        logger.info(
            "LLM picked %s (%s) for %s: %s", accession, protein_name, gene, reasoning
        )
        # ---- KB enrichment: index every other reviewed hit too (see
        # gene_mapper_agent/pathways_agent for the same fix + rationale) ----
        try:
            await _enrich_other_protein_candidates(gene, tax_id, candidates, accession)
        except Exception as enrich_exc:
            logger.warning("protein KB enrichment failed for %s: %s", gene, enrich_exc)
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
        # No separate "already cached -> skip" gate here: upsert_point()
        # already does the right thing internally (hash-compares
        # text_to_embed against what's stored and only re-embeds on a real
        # change, see kb/qdrant_store.py). A gate here that short-circuits
        # on mere *existence* rather than content would silently freeze
        # whatever text got embedded first -- exactly what happened to the
        # LLM-paraphrased function_summary before the fidelity fix in
        # _select_protein_for_gene above: the corrected text could never
        # reach Qdrant because this line always saw an existing point for
        # that accession and returned before upsert_point was even called.
        dedup_key = f"uniprot:{entry.source_accession}:{tax_id}"
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
