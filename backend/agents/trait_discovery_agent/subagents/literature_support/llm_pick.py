"""
Sufficiency judgment + judgment-based de-duplication for the Literature
Support agent (design guide §3/§4/§5).

Evidence has already been fetched and exact-deduped by pmid (deterministic,
§3) by the time this runs. The LLM holds the same request_literature_evidence
tool (in case it wants to double-check or fetch more before answering),
dedupe_by_pmid (rarely needed — already applied), and
write_trait_gene_relationship (to persist the trait->gene edge once it
decides the evidence is sufficient) as bound tools via bind_tools, mirroring
gene_mapper's llm_pick.py: candidates/evidence are handed to the model as
context, and the tools are there for it to use itself rather than being
pre-resolved into the prompt.
"""
from __future__ import annotations

import logging

from kb.sources.literature_agent_client import (
    request_literature_evidence as _request_literature_evidence_tool,
    dedupe_by_pmid as _dedupe_by_pmid_tool,
    write_trait_gene_relationship as _write_trait_gene_relationship_tool,
)
from workflows.llm import invoke_tool_loop_with_fallback, MAX_TOOL_TURNS

logger = logging.getLogger(__name__)

_LITERATURE_SUPPORT_TOOLS = [
    _request_literature_evidence_tool,
    _dedupe_by_pmid_tool,
    _write_trait_gene_relationship_tool,
]

_SYSTEM_PROMPT = (
    "You are the Literature Support agent. You are given literature evidence "
    "already retrieved for a trait + gene set (you do not search PubMed yourself). "
    "Decide whether this evidence is sufficient to support a conclusion, or whether "
    "it's thin enough to warrant escalating to the Literature Agent for a deeper pass.\n\n"
    "Evidence is sufficient if at least one record's summary plausibly and directly "
    "supports the trait-gene relationship, and there is no unresolved contradiction "
    "between records. If it's thin or inconclusive, do NOT guess a conclusion — "
    "recommend escalation and describe precisely what's missing in your "
    "prompt_to_target_agent.\n\n"
    "You have tools available:\n"
    "- request_literature_evidence(trait_name, gene_list): re-fetch evidence from the "
    "Literature Agent if you genuinely need more before deciding. The evidence you were "
    "already given below is usually enough — don't call this reflexively.\n"
    "- dedupe_by_pmid(records): exact de-dup by pmid. The evidence below has already "
    "had this applied; use it only if you fetch additional records yourself.\n"
    "- write_trait_gene_relationship(trait_name, gene_symbol, pmid): call this once per "
    "gene you conclude the evidence supports, using the single strongest supporting pmid "
    "for that gene. Only call it when you are marking the evidence sufficient.\n\n"
    "Rules:\n"
    "- Never report a pmid, title, or summary that wasn't in the evidence you were given "
    "(or that a tool call actually returned).\n"
    "- If two records describe the same finding under different pmids (e.g. a preprint "
    "and its published version), note that as a duplicate in your reasoning rather than "
    "double-counting them as independent support.\n\n"
    "Once you have enough information, reply with ONLY a JSON object with exactly these "
    'keys (no markdown fences, no extra text, no further tool calls): "sufficient" '
    '(true/false), "supporting_pmids" (list of pmid strings drawn only from the evidence '
    'given/returned), "reasoning" (string), "prompt_to_target_agent" (string, empty if '
    'sufficient — otherwise a precise description of what evidence is missing).'
)


async def _llm_judge_sufficiency(
    trait_name: str,
    gene_list: list[str],
    evidence: list[dict],
    thin_flag: bool,
) -> tuple[bool, list[str], str, str]:
    """Ask the LLM to judge evidence sufficiency via a bind_tools loop (§3/§5).

    Returns (sufficient, supporting_pmids, reasoning, prompt_to_target_agent).
    Raises on any failure — the caller is responsible for the deterministic
    count-only fallback (§9).
    """
    evidence_text = "\n".join(
        f"- pmid={r['pmid']} ({r.get('year', '?')}) {r.get('title', '')}: "
        f"{r.get('short_summary', '')}"
        for r in evidence
    )

    human_prompt = (
        f"Trait: {trait_name}\n"
        f"Gene(s): {gene_list}\n\n"
        f"Evidence already retrieved and exact-deduped by pmid "
        f"({len(evidence)} record(s), thin-by-count-threshold={thin_flag}):\n"
        f"{evidence_text or '(none)'}\n\n"
        "Decide sufficiency, call write_trait_gene_relationship for any gene the "
        "evidence supports, then reply with ONLY the final JSON object."
    )

    parsed, tool_call_log = await invoke_tool_loop_with_fallback(
        _SYSTEM_PROMPT,
        human_prompt,
        _LITERATURE_SUPPORT_TOOLS,
        temperature=0.1,
        max_turns=MAX_TOOL_TURNS,
    )
    logger.debug("literature_support tool_call_log for %s: %s", trait_name, tool_call_log)

    if "sufficient" not in parsed:
        raise RuntimeError(f"Model did not return a sufficiency verdict: {parsed!r}")

    sufficient = bool(parsed["sufficient"])
    supporting_pmids = [str(p) for p in parsed.get("supporting_pmids", [])]
    reasoning = parsed.get("reasoning", "")
    prompt_to_target_agent = parsed.get("prompt_to_target_agent", "")

    return sufficient, supporting_pmids, reasoning, prompt_to_target_agent
