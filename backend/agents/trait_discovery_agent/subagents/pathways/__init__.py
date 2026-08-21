"""
Pathways Agent — KEGG pathway selection with LLM-guided relevance ranking.

For each gene:
  1. List all KEGG pathways linked to the gene (not just the first).
  2. If 0 links   → malformed_ids (no LLM call).
  3. If 1 link    → straight through (no real decision needed).
  4. If >1 links  → LLM holds list_pathway_candidates/fetch_pathway_name as
                    bound tools via a bind_tools loop (llm_pick.py) and picks
                    the most trait-relevant pathway itself, with
                    deterministic fallback on LLM failure (§9).

Everything that gets called/monkeypatched by name in tests (list_pathway_
candidates, fetch_pathway_name, fetch_pathway, _llm_pick_pathway) is imported
directly into this module's namespace, and pathways_agent()/
_select_pathway_for_gene() — which look those names up as bare globals —
live here too, so patching this module's attributes actually changes what
they call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from schemas.inputs import PathwaysInput
from schemas.outputs import PathwayEntry, PathwaysOutput
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.kegg_client import (
    fetch_pathway,
    _list_pathway_candidates_raw as list_pathway_candidates,
    _fetch_pathway_name_raw as fetch_pathway_name,
)

from .llm_pick import _llm_pick_pathway
from .mock import mock_pathways_agent

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

__all__ = ["pathways_agent", "mock_pathways_agent"]


async def _select_pathway_for_gene(
    gene: str, kegg_gene_id: str, trait_name: str
) -> PathwayEntry | None:
    """
    Returns None on malformed (zero links) — caller records it in malformed_ids
    without ever trying an LLM call or a fallback fetch for that gene.
    """
    # --- Discover candidates (§9: surfaced, one retry handled inside the client) ---
    candidates = await list_pathway_candidates(kegg_gene_id)
    if not candidates:
        return None  # malformed — zero links, §8: never reaches the LLM

    # --- one link: straight through, no real decision (§8) ---
    if len(candidates) == 1:
        pid = candidates[0]["pathway_id"]
        name = await fetch_pathway_name(pid)
        return PathwayEntry(
            pathway_id=pid,
            pathway_name=name or pid,
            reasoning="Only one KEGG link available.",
        )

    # --- several links: LLM pick via bind_tools (§8) ---
    try:
        pathway_id, pathway_name, reasoning = await _llm_pick_pathway(
            trait_name, gene, candidates, kegg_gene_id
        )
        # --- grounding rule (§0.1): validate against actual tool output ---
        valid_ids = {c["pathway_id"] for c in candidates}
        if pathway_id not in valid_ids:
            raise RuntimeError(
                f"LLM picked invalid pathway_id {pathway_id} not in {valid_ids}"
            )
        return PathwayEntry(
            pathway_id=pathway_id, pathway_name=pathway_name, reasoning=reasoning
        )
    except Exception as exc:
        logger.warning("LLM pick failed for %s, deferring to fallback: %s", gene, exc)
        return None  # caller falls back to fetch_pathway (§9)


async def pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways: list[PathwayEntry] = []
    malformed: list[str] = []

    for gene in input.gene_list:
        kegg_gene_id = input.context.get("kegg_gene_ids", {}).get(gene)
        if not kegg_gene_id:
            malformed.append(gene)
            continue

        try:
            entry = await _select_pathway_for_gene(gene, kegg_gene_id, input.trait_name)
        except Exception as exc:
            logger.warning("list_pathway_candidates failed for %s: %s", gene, exc)
            entry = None

        # --- Deterministic fallback (§9) --------------------------------
        if entry is None:
            logger.info("Falling back to deterministic fetch_pathway for %s", gene)
            try:
                entry = await fetch_pathway(kegg_gene_id)
            except Exception as exc:
                logger.warning("Deterministic fallback also failed for %s: %s", gene, exc)
                entry = None

        if entry is None:
            malformed.append(gene)
            continue

        # --- Cache / dedup (§6) ------------------------
        dedup_key = f"kegg:{entry.pathway_id}:{gene}"
        cached = await get_cached("kegg_pathways", dedup_key)
        if cached:
            pathways.append(entry)
            continue

        await upsert_point(
            "kegg_pathways",
            dedup_key,
            text_to_embed=entry.pathway_name,
            payload={
                "gene_symbol": gene,
                "pathway_id": entry.pathway_id,
                "pathway_name": entry.pathway_name,
                "source": "KEGG REST API",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        pathways.append(entry)

    # §9: No pathways resolved for any gene → status=FAILED
    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)
