"""Where the Literature Agent's papers come from.

The single seam between the discovery pipeline and the outside world. The
LangGraph in `knowledge_discovery.py` never reaches a paper source directly,
so wiring up PubMed / Semantic Scholar later is a change to this file alone.

The real source is the Retrieval & Knowledge Processing subagent
(`subagents/retrieval_knowledge/`): LLM query expansion, Firecrawl web search,
OpenAlex academic search and semantic ranking. `search_papers` delegates to it
and returns what it found.

The placeholder below is the fallback, not the default. It is used only when
retrieval is unconfigured or returns nothing, and every result it produces
carries `is_placeholder=True` so that no caller - this agent, the Global
Orchestrator, or the Responder that writes the user's final answer - can
mistake these for retrieved literature. That flag is the point of this module:
on a platform whose premise is never inventing scientific results, unlabelled
fake papers flowing into the shared context is the failure worth preventing.
Real results never carry the flag, so the distinction stays visible either way.
"""
from __future__ import annotations

import logging

from ..retrieval_knowledge import search_literature

_logger = logging.getLogger(__name__)

# Replaced wholesale once a real client exists. Kept deliberately obvious
# rather than realistic - plausible-looking fake titles would be worse.
_PLACEHOLDER_PAPERS = ["Paper 1", "Paper 2", "Paper 3"]

PLACEHOLDER_NOTICE = (
    "Placeholder results: this agent is not yet connected to PubMed or "
    "Semantic Scholar and these are not real publications."
)


def search_papers(query: str) -> dict:
    """Search the literature for `query`.

    Returns the discovery payload. The shape is what `knowledge_discovery.py`
    formats and what ends up under the `discovery` key of the agent's output,
    so a real implementation must keep these keys:

        papers      list[str]   human-readable citation lines. Fed to the
                                writing subagent as the only references it is
                                allowed to cite.
        records     list[dict]  structured evidence, one dict per paper, with
                                keys pmid / title / year / short_summary. The
                                Trait Discovery Agent reads this directly
                                (kb/sources/literature_agent_client.py) to back
                                trait->gene edges in Neo4j.
        total_found int
        source      str

    `records` is deliberately EMPTY while this is a placeholder, even though
    `papers` is not. A record is only useful to the Trait Discovery Agent if it
    carries a pmid, and that agent writes the pmid it is given straight into
    the knowledge graph as evidence for a trait->gene link. Inventing pmids
    here would put fabricated citations into shared, persisted scientific
    state - a much worse failure than returning nothing. `papers` can stay
    populated because it never leaves this agent unflagged: it goes to the
    writing subagent, which carries `references_are_placeholder` alongside it.
    """
    result = search_literature(query)

    # An empty `papers` means retrieval could not run or found nothing. Fall
    # back rather than returning an empty success: the Responder treats a
    # completed-but-empty finding as a gap to write around from its own
    # memory, which is the outcome the placeholder notice exists to prevent.
    if result.get("papers"):
        return result

    _logger.info(
        "retrieval returned nothing (source=%s); using labelled placeholder",
        result.get("source"),
    )
    papers = list(_PLACEHOLDER_PAPERS)
    return {
        "papers": papers,
        "records": [],
        "total_found": len(papers),
        "source": "placeholder",
        "is_placeholder": True,
        "notice": PLACEHOLDER_NOTICE,
    }
