from __future__ import annotations

import asyncio
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
from workflows.llm import invoke_json_with_fallback

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

# Cap concurrent KEGG name lookups so a gene with many candidate pathways
# doesn't fire off a dozen requests at once and trip KEGG's own rate limits.
_NAME_RESOLVE_CONCURRENCY = 3


async def _resolve_missing_pathway_names(candidates: list[dict]) -> list[dict]:
    """Fill in pathway_name for every candidate up front.

    list_pathway_candidates never returns names (§ kegg_client), so without
    this the LLM has to spend a tool-call turn per candidate just to find out
    what each one is called before it can even start deciding. Resolving
    them here means the model sees a fully-named list on turn 1 and — in the
    common case — doesn't need to call a tool at all.
    """
    sem = asyncio.Semaphore(_NAME_RESOLVE_CONCURRENCY)

    async def _fill(c: dict) -> dict:
        if c.get("pathway_name"):
            return c
        async with sem:
            name = await fetch_pathway_name(c["pathway_id"])
        return {**c, "pathway_name": name}

    return await asyncio.gather(*(_fill(c) for c in candidates))

_SYSTEM_PROMPT = (
    "You are the Pathways agent. Given a gene, a list of KEGG pathways (already "
    "resolved to their human-readable names), and the trait under investigation, "
    "select the single most trait-relevant pathway.\n\n"
    "Rules:\n"
    "- Prefer the pathway whose name plausibly relates to the trait.\n"
    "- If none look trait-relevant, keep the FIRST one in the list and say so in your "
    "reasoning. Never fabricate a better-sounding match.\n"
    "- Only report a pathway_id and pathway_name that appear in the candidate list below "
    "— never one you recall from elsewhere.\n\n"
    "Reply with ONLY a JSON object with exactly these keys (no markdown fences, no extra "
    'text): "pathway_id", "pathway_name", "reasoning".'
)


async def _llm_pick_pathway(
    trait_name: str,
    gene_symbol: str,
    candidates: list[dict],
) -> tuple[str, str, str]:
    """
    Ask the LLM to pick the best pathway. Candidates are pre-resolved (every
    pathway_name already filled in by _resolve_missing_pathway_names), so this
    is a single tool-free JSON completion — no bind_tools loop, no turn limit,
    no risk of the model re-fetching unnamed data it doesn't need.

    Returns (pathway_id, pathway_name, reasoning). Raises on any failure — the
    caller is responsible for the deterministic fallback (§9).
    """
    candidates_text = "\n".join(
        f"- {c['pathway_id']}: {c.get('pathway_name') or '(name unavailable)'}"
        for c in candidates
    )

    human_prompt = (
        f"Gene symbol: {gene_symbol}\n"
        f"Trait under investigation: {trait_name}\n\n"
        f"Candidate pathways (this is the complete list — do not assume others exist):\n"
        f"{candidates_text}\n\n"
        "Reply with ONLY the final JSON object."
    )

    parsed = await invoke_json_with_fallback(
        _SYSTEM_PROMPT,
        human_prompt,
        temperature=0.1,
    )

    picked_id = parsed.get("pathway_id")
    picked_name = parsed.get("pathway_name") or ""
    reasoning = parsed.get("reasoning", "")

    if not picked_id:
        raise RuntimeError(f"Model did not return a pathway_id: {parsed!r}")

    return picked_id, picked_name, reasoning


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
    candidates = await _resolve_missing_pathway_names(candidates)
    try:
        pathway_id, pathway_name, reasoning = await _llm_pick_pathway(
            trait_name, gene, candidates
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


# =========================================================================== #
#  Mock agent (offline / CI)                                                  #
# =========================================================================== #

_MOCK_KEGG_DB = {
    "UCP1": PathwayEntry(
        pathway_id="ko00071",
        pathway_name="Fatty acid degradation",
        reasoning="Mock: relevant to metabolic heat production.",
    ),
    "PRDM16": PathwayEntry(
        pathway_id="ko04928",
        pathway_name="Thermogenesis",
        reasoning="Mock: directly related to brown adipose tissue.",
    ),
    "FGF5": PathwayEntry(
        pathway_id="ko04010",
        pathway_name="MAPK signaling pathway",
        reasoning="Mock: growth factor signaling.",
    ),
}


async def mock_pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways, malformed = [], []
    for gene in input.gene_list:
        if gene in _MOCK_KEGG_DB:
            pathways.append(_MOCK_KEGG_DB[gene])
        else:
            malformed.append(gene)

    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)