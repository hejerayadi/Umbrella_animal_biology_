"""
Multi-candidate GO-term selection for the Gene Mapper agent (§8, multi-link
branch only — single-link and malformed cases never reach this module).

Implements the bind_tools loop required by §2/§5: the model holds
list_go_candidates/resolve_go_term_name as bound tools and calls them itself
to resolve any names it needs before answering.
"""
from __future__ import annotations

import logging

from kb.sources.go_client import (
    list_go_candidates as _list_go_candidates_tool,
    resolve_go_term_name as _resolve_go_term_name_tool,
    resolve_go_term_names as _resolve_go_term_names_tool,
)
from workflows.llm import invoke_tool_loop_with_fallback, MAX_TOOL_TURNS

logger = logging.getLogger(__name__)

# §2/§5: the tools the model holds via bind_tools and calls itself for a
# multi-candidate decision. These are the *same* QuickGO calls the node uses
# for branching (§8) and the deterministic fallback (§9) — just wrapped with
# @tool so the LLM can invoke them directly instead of being handed a
# pre-resolved answer. resolve_go_term_names (plural) lets the model resolve
# every unresolved candidate in one turn instead of one per turn — a gene can
# have a dozen+ real GO annotations, and paying a full LLM round-trip per
# name doesn't scale.
_GENE_MAPPER_TOOLS = [
    _list_go_candidates_tool,
    _resolve_go_term_name_tool,
    _resolve_go_term_names_tool,
]

_SYSTEM_PROMPT = (
    "You are the Gene Mapper agent. You're given a trait, a species, and a gene already "
    "confirmed relevant to that trait — your job is to attach the correct GO "
    "biological-process annotation to it, choosing from the candidate GO ids listed "
    "below.\n\n"
    "You have tools available:\n"
    "- resolve_go_term_names(go_ids): look up human-readable names for candidate GO ids "
    "you don't already recognize. Pass ALL of the ones you need in ONE call — do not "
    "call this (or resolve_go_term_name) once per id; that wastes a full turn per "
    "candidate for no benefit.\n"
    "- resolve_go_term_name(go_id): same lookup for a single id, if you only need one.\n"
    "- list_go_candidates(gene_symbol, uniprot_accession): re-fetch the candidate list "
    "from QuickGO if you need to double-check it. This is rarely necessary since the "
    "complete list is already given to you below.\n\n"
    "Rules:\n"
    "- Pick the candidate whose name is most plausibly related to the trait as described.\n"
    "- If none of the candidates look trait-relevant, still pick the closest one but say "
    "so plainly in your reasoning — do not invent a better-sounding GO term.\n"
    "- Only report a go_id that appears in the candidate list below — never one you "
    "recall from elsewhere, and never one you have not resolved a name for.\n\n"
    "Once you have enough information, reply with ONLY a JSON object with exactly these "
    'keys (no markdown fences, no extra text, no further tool calls): "go_id", '
    '"go_name", "reasoning".'
)


async def _llm_pick_candidate(
    trait_name: str,
    gene_symbol: str,
    candidates: list[dict],
    uniprot_accession: str = "",
) -> tuple[str, str, str]:
    """
    Ask the LLM to pick the best candidate via a bind_tools loop (§2/§5): the
    model holds list_go_candidates/resolve_go_term_name as bound tools and
    calls them itself to resolve any names it needs before answering.

    Returns (go_id, go_name, reasoning). Raises on any failure — the caller is
    responsible for the deterministic fallback (§9).
    """
    candidates_text = "\n".join(
        f"- {c['go_id']}" + (f": {c['go_name']}" if c.get("go_name") else "")
        for c in candidates
    )

    human_prompt = (
        f"Trait: {trait_name}\n"
        f"Gene: {gene_symbol}\n"
        f"UniProt accession (for list_go_candidates, if you need it): {uniprot_accession}\n\n"
        f"Candidate GO ids (this is the complete list — do not assume others exist):\n"
        f"{candidates_text}\n\n"
        "Resolve names for any candidates you don't recognize, then reply with ONLY the "
        "final JSON object."
    )

    # With resolve_go_term_names (batch), the model needs ~2-3 turns
    # regardless of candidate count: one batch resolve + one final answer,
    # maybe one list_go_candidates double-check. No need to scale max_turns
    # with the candidate count anymore — that scaling is what let a gene with
    # a dozen+ real candidates (each needing its own LLM round-trip if the
    # model insists on one-by-one calls) balloon to a multi-minute decision.
    # MAX_TOOL_TURNS is enough headroom even for a model that doesn't batch.
    parsed, tool_call_log = await invoke_tool_loop_with_fallback(
        _SYSTEM_PROMPT,
        human_prompt,
        _GENE_MAPPER_TOOLS,
        temperature=0.1,
        max_turns=MAX_TOOL_TURNS,
    )
    logger.debug("gene_mapper tool_call_log for %s: %s", gene_symbol, tool_call_log)

    picked_id = parsed.get("go_id")
    picked_name = parsed.get("go_name") or "unknown"
    reasoning = parsed.get("reasoning", "")

    if not picked_id:
        raise RuntimeError(f"Model did not return a go_id: {parsed!r}")

    return picked_id, picked_name, reasoning
