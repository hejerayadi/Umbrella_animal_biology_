"""
Literature Support Agent — evidence sufficiency judgment with LLM-guided
escalation (design guide §5).

Flow:
  1. Cross-agent call to the Literature Agent for evidence records (with one
     retry on transient errors, §9) — this agent never queries PubMed itself.
  2. Exact de-dup by pmid (deterministic, §3).
  3. No evidence at all → FAILED, no LLM call (§8/§10).
  4. Otherwise: LLM holds request_literature_evidence/dedupe_by_pmid/
     write_trait_gene_relationship as bound tools via a bind_tools loop
     (llm_pick.py) and judges sufficiency itself, with deterministic
     count-only fallback on LLM failure (§9).

Everything that gets monkeypatched by name in tests (request_literature_evidence,
dedupe_by_pmid, _llm_judge_sufficiency) is imported directly into this module's
namespace, and literature_support_agent() — which looks those names up as bare
globals — lives here too, so patching this module's attributes actually changes
what literature_support_agent() calls.
"""
from __future__ import annotations

import logging

from schemas.inputs import LiteratureSupportInput
from schemas.outputs import LiteratureSupportOutput, LiteratureRecord
from schemas.common import AgentStatus
from kb.sources.literature_agent_client import (
    _request_literature_evidence_raw as request_literature_evidence,
    _dedupe_by_pmid_raw as dedupe_by_pmid,
)

from .llm_pick import _llm_judge_sufficiency
from .mock import mock_literature_support, _THIN_EVIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

__all__ = ["literature_support_agent", "mock_literature_support"]


def _to_literature_records(records: list[dict]) -> list[LiteratureRecord]:
    return [
        LiteratureRecord(
            pmid=r["pmid"],
            title=r.get("title", ""),
            year=int(r.get("year") or 0),
            short_summary=r.get("short_summary", ""),
        )
        for r in records
    ]


async def literature_support_agent(input: LiteratureSupportInput) -> LiteratureSupportOutput:
    # ---- fetch evidence from the Literature Agent (with one retry, §9) ----
    try:
        raw_records = await request_literature_evidence(input.trait_name, input.gene_list)
    except Exception as exc:
        logger.warning("Literature Agent error, retrying once: %s", exc)
        try:
            raw_records = await request_literature_evidence(input.trait_name, input.gene_list)
        except Exception as exc2:
            logger.error("Literature Agent unreachable after retry: %s", exc2)
            return LiteratureSupportOutput(status=AgentStatus.FAILED, evidence=[])

    deduped = dedupe_by_pmid(raw_records)
    if not deduped:
        # §8/§10: no evidence at all → FAILED, never reaches the LLM.
        return LiteratureSupportOutput(status=AgentStatus.FAILED, evidence=[])

    thin_flag = len(deduped) <= _THIN_EVIDENCE_THRESHOLD
    evidence = _to_literature_records(deduped)
    known_pmids = {r["pmid"] for r in deduped}

    # ---- LLM sufficiency judgment via bind_tools (§3/§5) ----
    try:
        sufficient, supporting_pmids, reasoning, prompt_to_target_agent = (
            await _llm_judge_sufficiency(
                input.trait_name, input.gene_list, deduped, thin_flag
            )
        )

        # ---- grounding rule (§0.1): reject citations not in the evidence ----
        invented = [p for p in supporting_pmids if p not in known_pmids]
        if invented:
            raise RuntimeError(
                f"LLM cited pmid(s) not present in evidence: {invented}"
            )

        if sufficient:
            logger.info(
                "LLM judged evidence sufficient for %s: %s", input.trait_name, reasoning
            )
            return LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=evidence)

        logger.info(
            "LLM judged evidence thin for %s: %s", input.trait_name, reasoning
        )
        return LiteratureSupportOutput(
            status=AgentStatus.NEEDS_AGENT,
            evidence=evidence,
            target_agent="Literature Agent",
            prompt_to_target_agent=prompt_to_target_agent or (
                f"Find additional peer-reviewed evidence for trait '{input.trait_name}' "
                f"and genes {input.gene_list}. Existing evidence is thin "
                f"({len(deduped)} record(s))."
            ),
        )
    except Exception as exc:
        # ---- LLM/NIM unavailable or invalid → deterministic fallback (§9) ----
        logger.warning(
            "LLM sufficiency judgment failed for %s, falling back to count-only "
            "threshold: %s", input.trait_name, exc,
        )
        if thin_flag:
            return LiteratureSupportOutput(
                status=AgentStatus.NEEDS_AGENT,
                evidence=evidence,
                target_agent="Literature Agent",
                prompt_to_target_agent=(
                    f"Find additional peer-reviewed evidence for trait '{input.trait_name}' "
                    f"and genes {input.gene_list}. Existing evidence is thin "
                    f"({len(deduped)} record(s))."
                ),
            )
        return LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=evidence)
