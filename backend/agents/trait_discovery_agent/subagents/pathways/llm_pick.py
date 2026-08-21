"""
Multi-candidate KEGG pathway selection for the Pathways agent (§8, multi-link
branch only — single-link and malformed cases never reach this module).

Implements the bind_tools loop required by §2/§5: the model holds
list_pathway_candidates/fetch_pathway_name as bound tools and calls them
itself to resolve any names it needs before answering.
"""
from __future__ import annotations

import logging

from kb.sources.kegg_client import (
    list_pathway_candidates as _list_pathway_candidates_tool,
    fetch_pathway_name as _fetch_pathway_name_tool,
    fetch_pathway_names as _fetch_pathway_names_tool,
)
from workflows.llm import invoke_tool_loop_with_fallback, MAX_TOOL_TURNS

logger = logging.getLogger(__name__)

# §2/§5: the tools the model holds via bind_tools and calls itself for a
# multi-candidate decision — the same KEGG calls the node uses for branching
# (§8) and the deterministic fallback (§9), wrapped with @tool.
# fetch_pathway_names (plural) lets the model resolve every unresolved
# candidate in one turn instead of one per turn — some genes link to a
# dozen+ real KEGG pathways, and paying a full LLM round-trip per name
# doesn't scale.
_PATHWAYS_TOOLS = [
    _list_pathway_candidates_tool,
    _fetch_pathway_name_tool,
    _fetch_pathway_names_tool,
]

_SYSTEM_PROMPT = (
    "You are the Pathways agent. Given a gene, its KEGG gene id, a list of candidate "
    "KEGG pathway ids linked to that gene, and the trait under investigation, select "
    "the single most trait-relevant pathway.\n\n"
    "You have tools available:\n"
    "- fetch_pathway_names(pathway_ids): look up human-readable names for candidate "
    "pathway ids you don't already recognize. Pass ALL of the ones you need in ONE "
    "call — do not call this (or fetch_pathway_name) once per id; that wastes a full "
    "turn per candidate for no benefit.\n"
    "- fetch_pathway_name(pathway_id): same lookup for a single id, if you only need one.\n"
    "- list_pathway_candidates(kegg_gene_id): re-fetch the candidate list from KEGG if "
    "you need to double-check it. This is rarely necessary since the complete list is "
    "already given to you below.\n\n"
    "Rules:\n"
    "- Prefer the pathway whose name plausibly relates to the trait.\n"
    "- If none look trait-relevant, keep the FIRST one in the list and say so in your "
    "reasoning. Never fabricate a better-sounding match.\n"
    "- Only report a pathway_id that appears in the candidate list below — never one "
    "you recall from elsewhere, and never one you have not resolved a name for.\n\n"
    "Once you have enough information, reply with ONLY a JSON object with exactly these "
    'keys (no markdown fences, no extra text, no further tool calls): "pathway_id", '
    '"pathway_name", "reasoning".'
)


async def _llm_pick_pathway(
    trait_name: str,
    gene_symbol: str,
    candidates: list[dict],
    kegg_gene_id: str = "",
) -> tuple[str, str, str]:
    """
    Ask the LLM to pick the best pathway via a bind_tools loop (§2/§5): the
    model holds list_pathway_candidates/fetch_pathway_name as bound tools and
    calls them itself to resolve any names it needs before answering.

    Returns (pathway_id, pathway_name, reasoning). Raises on any failure — the
    caller is responsible for the deterministic fallback (§9).
    """
    candidates_text = "\n".join(
        f"- {c['pathway_id']}" + (f": {c['pathway_name']}" if c.get("pathway_name") else "")
        for c in candidates
    )

    human_prompt = (
        f"Gene symbol: {gene_symbol}\n"
        f"KEGG gene id (for list_pathway_candidates, if you need it): {kegg_gene_id}\n"
        f"Trait under investigation: {trait_name}\n\n"
        f"Candidate pathway ids (this is the complete list — do not assume others exist):\n"
        f"{candidates_text}\n\n"
        "Resolve names for any candidates you don't recognize, then reply with ONLY the "
        "final JSON object."
    )

    # Same reasoning as the gene_mapper equivalent: with fetch_pathway_names
    # (batch), ~2-3 turns is enough regardless of candidate count, so no need
    # to scale max_turns with the candidate count — that scaling is what let
    # a gene with a dozen+ real KEGG links balloon into a multi-minute
    # decision when the model resolved names one at a time.
    parsed, tool_call_log = await invoke_tool_loop_with_fallback(
        _SYSTEM_PROMPT,
        human_prompt,
        _PATHWAYS_TOOLS,
        temperature=0.1,
        max_turns=MAX_TOOL_TURNS,
    )
    logger.debug("pathways tool_call_log for %s: %s", gene_symbol, tool_call_log)

    picked_id = parsed.get("pathway_id")
    picked_name = parsed.get("pathway_name") or ""
    reasoning = parsed.get("reasoning", "")

    if not picked_id:
        raise RuntimeError(f"Model did not return a pathway_id: {parsed!r}")

    return picked_id, picked_name, reasoning
