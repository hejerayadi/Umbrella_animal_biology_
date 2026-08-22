"""
Cross-agent client for the Literature Support subagent (design guide §5/§6).

Literature Support does not own a source client the way Gene Mapper/Pathways/
Protein Data do (§0.5) — it never talks to PubMed or Semantic Scholar itself.
Instead it makes an A2A call to the external Literature Agent
(agent_cards/literature_support.json's `may_need`), using the same
AgentRequest/AgentResult envelope every worker agent in this system speaks
(instruction + context in, status/output out — see schema.py, api.py).

Two independent pieces live here:

  - request_literature_evidence: the actual cross-agent HTTP call.
  - dedupe_by_pmid: a deterministic, exact pre-pass (no LLM) that both the
    node and the bind_tools loop can call.
  - write_trait_gene_relationship: the Neo4j knowledge-graph write
    (kb/neo4j_store.py). Scalar-only signature (trait_name, gene_symbol,
    pmid) — never evidence content. Fails soft per §9: a write failure must
    not fail the whole subagent; the evidence itself is still valid.
"""
from __future__ import annotations

import logging
import os

import httpx
from langchain_core.tools import tool

from schemas.outputs import LiteratureRecord
from kb.sources._http_retry import request_with_retry
from kb.neo4j_store import upsert_trait_gene_relationship

logger = logging.getLogger(__name__)

# The Literature Agent speaks the same POST /execute contract as every other
# worker agent in this system (see trait_discovery_agent/api.py, gene_agent/
# api.py). No docker-compose service exists for it yet in this repo — it's
# owned/deployed by another team — so this is overridable via env rather than
# hardcoded to a port that may not match their deployment.
LITERATURE_AGENT_URL = os.environ.get(
    "LITERATURE_AGENT_URL", "http://literature-agent:8000/execute"
)


# --------------------------------------------------------------------------- #
#  Raw implementations.
#
#  These are what the agent calls directly for the deterministic pre-fetch
#  and count-only fallback (§3/§9), and what tests monkeypatch. The @tool-
#  wrapped versions below are thin adapters over these so the *same* calls
#  are also exposed to the LLM via bind_tools (§5).
# --------------------------------------------------------------------------- #

async def _request_literature_evidence_raw(
    trait_name: str, gene_list: list[str]
) -> list[dict]:
    """A2A call to the Literature Agent for pmid/title/year/summary records.

    Sends the same {instruction, context} envelope every worker agent in
    this system accepts, asking specifically for paper retrieval + summari-
    zation (two of the Literature Agent's stated capabilities) rather than
    scientific writing. Expects an AgentResult-shaped body back and reads
    evidence out of output["discovery"]["records"] — per the Literature
    Agent's card, `discovery` is populated for retrieval/summarization asks
    and `writing` is left null.
    """
    payload = {
        "instruction": (
            f"Find peer-reviewed literature evidence for the trait "
            f"'{trait_name}' in relation to the gene(s) {gene_list}. "
            f"Return paper records (pmid, title, year, short_summary) — "
            f"no drafting or writing needed."
        ),
        "context": {
            "task": "literature_evidence",
            "trait_name": trait_name,
            "gene_list": gene_list,
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await request_with_retry(
            client, "POST", LITERATURE_AGENT_URL, json=payload
        )
        body = resp.json()

    if body.get("status") not in (None, "completed"):
        # A non-completed AgentResult (e.g. the Literature Agent itself
        # failed or needs escalation) has no usable evidence.
        logger.warning(
            "Literature Agent returned non-completed status=%s", body.get("status")
        )
        return []

    discovery = ((body.get("output") or {}).get("discovery")) or {}
    raw_records = discovery.get("records", [])

    records: list[dict] = []
    for r in raw_records:
        pmid = r.get("pmid")
        if not pmid:
            continue
        records.append({
            "pmid": str(pmid),
            "title": r.get("title", ""),
            "year": r.get("year", 0),
            "short_summary": r.get("short_summary") or r.get("summary", ""),
        })
    return records


def _dedupe_by_pmid_raw(records: list[dict]) -> list[dict]:
    """Exact de-dup by pmid, first-seen-wins. Deterministic — no LLM judgment
    (that's a separate, judgment-based dedup the LLM itself does over the
    *content*, e.g. a preprint and its published version sharing different
    pmids; this pass only catches literal duplicate pmids)."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in records:
        pmid = r.get("pmid")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        deduped.append(r)
    return deduped


async def _write_trait_gene_relationship_raw(
    trait_name: str, gene_symbol: str, pmid: str
) -> bool:
    """Writes a trait->gene edge backed by a pmid to the Neo4j knowledge
    graph (kb/neo4j_store.py). This function's own signature is part of the
    "never cache evidence content" guarantee (design guide §6): it accepts
    only trait_name/gene_symbol/pmid, so there is no parameter through which
    a title or short_summary could reach the graph, even after future edits.
    Per §9, this fails soft — the caller treats a False return as a warning,
    not a reason to downgrade the subagent's status, since the evidence
    itself is still valid and already returned to the caller."""
    return await upsert_trait_gene_relationship(trait_name, gene_symbol, pmid)


# --------------------------------------------------------------------------- #
#  bind_tools-facing wrappers (guide §5: model gets direct, typed access)
# --------------------------------------------------------------------------- #

@tool
async def request_literature_evidence(trait_name: str, gene_list: list[str]) -> list[dict]:
    """Cross-agent call to the Literature Agent for pmid/title/year/summary records."""
    return await _request_literature_evidence_raw(trait_name, gene_list)


@tool
async def dedupe_by_pmid(records: list[dict]) -> list[dict]:
    """Exact de-dup by pmid — a deterministic pre-pass before judging relevance."""
    return _dedupe_by_pmid_raw(records)


@tool
async def write_trait_gene_relationship(trait_name: str, gene_symbol: str, pmid: str) -> bool:
    """Writes a trait->gene edge backed by a pmid to the Neo4j knowledge graph."""
    return await _write_trait_gene_relationship_raw(trait_name, gene_symbol, pmid)


def _records_to_literature_records(records: list[dict]) -> list[LiteratureRecord]:
    return [
        LiteratureRecord(
            pmid=r["pmid"],
            title=r.get("title", ""),
            year=int(r.get("year") or 0),
            short_summary=r.get("short_summary", ""),
        )
        for r in records
    ]
