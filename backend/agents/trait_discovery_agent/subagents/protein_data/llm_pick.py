"""
Multi-candidate UniProt entry selection for the Protein Data agent (§8,
multi-hit branch only — single-hit and zero-hit cases never reach this
module).

Narrower than the Pathways/Gene Mapper equivalents (§2): list_uniprot_
candidates already returns protein_name and function_summary on every
candidate, so there's no separate name-resolution tool to hand the model —
one bound tool, one structured pick, no multi-step plan. bind_tools is kept
(rather than dropping to a no-tools call) only so the model can re-fetch the
candidate list to double-check it, matching §2/§5; this is expected to be
rare since the complete list is already given to it below.
"""
from __future__ import annotations

import logging

from kb.sources.uniprot_client import list_uniprot_candidates as _list_uniprot_candidates_tool
from workflows.llm import invoke_tool_loop_with_fallback, MAX_TOOL_TURNS

logger = logging.getLogger(__name__)

# §2/§5: the only tool the model holds via bind_tools for this decision — a
# re-fetch/double-check of the same UniProt call the node already used for
# branching (§8) and that backs the deterministic fallback (§9).
_PROTEIN_DATA_TOOLS = [_list_uniprot_candidates_tool]

_SYSTEM_PROMPT = (
    "You are the Protein Data agent. You're given a gene, a species tax_id, the trait "
    "these genes were selected for, and every reviewed UniProt entry available for "
    "that gene+species pair. Pick the entry that is actually about the trait-relevant "
    "function.\n\n"
    "You have a tool available:\n"
    "- list_uniprot_candidates(gene_symbol, tax_id): re-fetch the candidate list from "
    "UniProt if you need to double-check it. This is rarely necessary since the "
    "complete list is already given to you below.\n\n"
    "Rules:\n"
    "- Prefer the entry whose function_summary is plausibly related to the trait.\n"
    "- If they look interchangeable, take the first and say so plainly in your "
    "reasoning — do not invent a better-sounding function.\n"
    "- Only report a source_accession that appears in the candidate list below — "
    "never one you recall from elsewhere.\n\n"
    "Once you have enough information, reply with ONLY a JSON object with exactly these "
    'keys (no markdown fences, no extra text, no further tool calls): "source_accession", '
    '"protein_name", "function_summary", "reasoning".'
)


async def _llm_pick_protein(
    trait_name: str,
    gene_symbol: str,
    candidates: list[dict],
    tax_id: int | None = None,
) -> tuple[str, str, str, str]:
    """
    Ask the LLM to pick the best reviewed UniProt entry via a bind_tools loop
    (§2/§5): the model holds list_uniprot_candidates as a bound tool and may
    call it to double-check the candidate list before answering.

    Returns (source_accession, protein_name, function_summary, reasoning).
    Raises on any failure — the caller is responsible for the deterministic
    fallback (§9).
    """
    candidates_text = "\n".join(
        f"- {c['source_accession']}: {c['protein_name']} — {c['function_summary']}"
        for c in candidates
    )

    human_prompt = (
        f"Gene symbol: {gene_symbol}\n"
        f"Species tax_id (for list_uniprot_candidates, if you need it): {tax_id}\n"
        f"Trait under investigation: {trait_name}\n\n"
        f"Candidate reviewed UniProt entries (this is the complete list — do not "
        f"assume others exist):\n"
        f"{candidates_text}\n\n"
        "Reply with ONLY the final JSON object."
    )

    # A single bound tool and pre-resolved candidate text: one real decision
    # should resolve in 1-2 turns, so MAX_TOOL_TURNS is ample headroom even
    # for a model that insists on double-checking via the tool first.
    parsed, tool_call_log = await invoke_tool_loop_with_fallback(
        _SYSTEM_PROMPT,
        human_prompt,
        _PROTEIN_DATA_TOOLS,
        temperature=0.1,
        max_turns=MAX_TOOL_TURNS,
    )
    logger.debug("protein_data tool_call_log for %s: %s", gene_symbol, tool_call_log)

    picked_accession = parsed.get("source_accession")
    picked_protein_name = parsed.get("protein_name") or ""
    picked_function_summary = parsed.get("function_summary") or ""
    reasoning = parsed.get("reasoning", "")

    if not picked_accession:
        raise RuntimeError(f"Model did not return a source_accession: {parsed!r}")

    return picked_accession, picked_protein_name, picked_function_summary, reasoning
