"""Where the Literature Agent's papers come from.

The single seam between the discovery pipeline and the outside world. The
LangGraph in `knowledge_discovery.py` never reaches a paper source directly,
so wiring up PubMed / Semantic Scholar later is a change to this file alone.

Nothing is wired to a real source yet. `search_papers` returns a fixed
placeholder list, and every result it produces carries `is_placeholder=True`
so that no caller - this agent, the Global Orchestrator, or the Responder that
writes the user's final answer - can mistake these for retrieved literature.
That flag is the point of this module: on a platform whose premise is never
inventing scientific results, unlabelled fake papers flowing into the shared
context is the failure worth preventing. The Genome Agent deleted its own mock
for exactly this reason; this one stays only until the real client lands, and
announces itself while it does.
"""
from __future__ import annotations

# Replaced wholesale once a real client exists. Kept deliberately obvious
# rather than realistic - plausible-looking fake titles would be worse.
_PLACEHOLDER_PAPERS = ["Paper 1", "Paper 2", "Paper 3"]

PLACEHOLDER_NOTICE = (
    "Placeholder results: this agent is not yet connected to PubMed or "
    "Semantic Scholar and these are not real publications."
)


def search_papers(query: str) -> dict:
    """Search the literature for `query`.

    Returns the discovery payload: the papers found, how many, where they came
    from, and whether they are real. The shape is what `knowledge_discovery.py`
    formats and what ends up under the `discovery` key of the agent's output,
    so a real implementation must keep these keys.
    """
    papers = list(_PLACEHOLDER_PAPERS)
    return {
        "papers": papers,
        "total_found": len(papers),
        "source": "placeholder",
        "is_placeholder": True,
        "notice": PLACEHOLDER_NOTICE,
    }
