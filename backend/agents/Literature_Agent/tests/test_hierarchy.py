"""The Literature Agent's three-level hierarchy is actually wired.

    LiteratureOrchestrator
      |- KnowledgeDiscoveryOrchestrator
      |    `- retrieval_knowledge          (subagents/retrieval_knowledge/)
      `- ScientificWritingOrchestrator
           |- WritingSupportAgent          (article text generation)
           `- PublicationSupportAgent      (subagents/publication_support/)

Both leaves were unreachable before: the retrieval subagent was never
referenced by anything, and the publication pipeline could not be imported
outside its own folder. These tests pin the wiring, so a regression shows up
here rather than as a silently degraded answer.

No network: every leaf is stubbed. What is under test is that the graph calls
the leaf at all, and that a failing leaf degrades instead of raising.
"""
from __future__ import annotations

import pytest

from ..schema import AgentRequest, AgentStatus
from ..subagents.discovery import sources
from ..subagents.writing.scientific_writing import PublicationSupportAgent


# ---------------------------------------------------------------------------
# discovery -> retrieval_knowledge
# ---------------------------------------------------------------------------


def test_discovery_delegates_to_the_retrieval_subagent(monkeypatch) -> None:
    """`search_papers` must call the retrieval leaf, not return the placeholder."""
    called: list[str] = []

    def fake_search(query: str, limit: int = 5) -> dict:
        called.append(query)
        return {
            "papers": ["Author, A. Wolf genomics (2024) https://doi.org/10.x/y"],
            "records": [{"pmid": None, "doi": "10.x/y", "title": "Wolf genomics"}],
            "total_found": 1,
            "source": "retrieval_knowledge",
        }

    monkeypatch.setattr(sources, "search_literature", fake_search)
    result = sources.search_papers("wolf genomics")

    assert called == ["wolf genomics"]
    assert result["source"] == "retrieval_knowledge"
    # Real results must never carry the placeholder flag - that is what tells
    # the Responder it may present them as retrieved literature.
    assert "is_placeholder" not in result


def test_discovery_falls_back_to_labelled_placeholder_when_retrieval_is_empty(
    monkeypatch,
) -> None:
    """An empty retrieval must not surface as an empty success."""
    monkeypatch.setattr(
        sources,
        "search_literature",
        lambda query, limit=5: {
            "papers": [],
            "records": [],
            "total_found": 0,
            "source": "retrieval_unavailable",
        },
    )
    result = sources.search_papers("wolf genomics")

    assert result["is_placeholder"] is True
    assert result["notice"] == sources.PLACEHOLDER_NOTICE


def test_retrieval_adapter_never_invents_a_pmid() -> None:
    """The engine searches Firecrawl and OpenAlex, never PubMed.

    The Trait Discovery Agent writes the pmid it is handed straight into Neo4j
    as evidence for a trait->gene edge, so a fabricated one would persist an
    invented citation into shared scientific state. It skips records whose
    pmid is falsy, which is the intended outcome here.
    """
    from ..subagents.retrieval_knowledge.agent import _to_record

    record = _to_record(
        {"title": "Wolf genomics", "doi": "10.x/y", "year": 2024, "abstract": "..."}
    )
    assert record["pmid"] is None
    assert record["doi"] == "10.x/y"


def test_retrieval_skips_rather_than_blocking_on_a_cold_model_cache(
    monkeypatch,
) -> None:
    """A cold cache must fail fast, not hang the request.

    The first SentenceTransformer(...) call downloads ~500MB and retries a
    stalled transfer for minutes. Discovery sits on the path of a user's
    question, so blocking there is worse than falling back.
    """
    from ..subagents.retrieval_knowledge import agent as rk_agent

    monkeypatch.setattr(rk_agent, "embedding_model_is_cached", lambda: False)
    result = rk_agent.search_literature("wolf genomics")

    assert result["papers"] == []
    assert result["source"] == "retrieval_model_not_cached"


# ---------------------------------------------------------------------------
# writing -> publication_support
# ---------------------------------------------------------------------------


def test_publication_support_prefers_the_retrieval_pipeline(monkeypatch) -> None:
    """Retrieved journals win over the LLM's recollection."""
    agent = PublicationSupportAgent()
    monkeypatch.setattr(
        PublicationSupportAgent,
        "_retrieve",
        lambda self, instruction, draft: [
            {"name": "Molecular Ecology", "publisher": "Wiley", "is_oa": False}
        ],
    )
    monkeypatch.setattr(
        PublicationSupportAgent,
        "_ask_llm",
        staticmethod(lambda msg: pytest.fail("LLM fallback must not run")),
    )

    result = agent.run(AgentRequest(instruction="where do I publish this?", context={}))

    assert result.status == AgentStatus.COMPLETED
    assert result.output["retrieval_backed"] is True
    assert result.output["journals"][0]["name"] == "Molecular Ecology"
    # The frontend passes this through asString(); it has to stay a string.
    assert isinstance(result.output["recommended_journals"], str)
    assert "Molecular Ecology" in result.output["recommended_journals"]


def test_publication_support_falls_back_to_the_llm(monkeypatch) -> None:
    """An unconfigured or failing pipeline must degrade, not raise."""
    agent = PublicationSupportAgent()
    monkeypatch.setattr(
        PublicationSupportAgent, "_retrieve", lambda self, instruction, draft: []
    )
    monkeypatch.setattr(
        PublicationSupportAgent,
        "_ask_llm",
        staticmethod(lambda msg: "- **Journal of Wolves**"),
    )

    result = agent.run(AgentRequest(instruction="where do I publish this?", context={}))

    assert result.status == AgentStatus.COMPLETED
    assert result.output["retrieval_backed"] is False
    assert result.output["journals"] == []


def test_publication_retrieve_swallows_pipeline_errors(monkeypatch) -> None:
    """`_retrieve` is the boundary: it returns [] rather than propagating.

    The module object is patched directly rather than by dotted-string path:
    `_retrieve` imports the pipeline inside the method, so the name it binds
    is read off this module at call time either way, and patching the object
    avoids re-importing the package through a second path.
    """
    from ..subagents.publication_support.ranking import query_interpreter

    agent = PublicationSupportAgent()

    def boom(*args, **kwargs):
        raise RuntimeError("qdrant is down")

    monkeypatch.setattr(query_interpreter, "interpret_query", boom)
    assert agent._retrieve("where do I publish?", "") == []
