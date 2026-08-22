from __future__ import annotations

from unittest.mock import patch

import pytest

from ..workflows.nodes.species_resolver_node import species_resolver_node
from ..workflows.state import GenomeAgentState

# NOTE: this replaces an earlier test (test_orchestrator_exception.py) that
# targeted `GenomeAgentOrchestrator.run(...)` directly. That hand-rolled
# asyncio.gather orchestrator was superseded by the LangGraph rewrite
# (build_genome_graph / GenomeAgentLangGraphOrchestrator in orchestrator.py),
# which moved this exact behavior into species_resolver_node. The scenario
# and assertions are unchanged — only the entry point being tested is.


@pytest.mark.asyncio
async def test_species_resolver_node_zero_candidates_is_fatal():
    """When resolve_species_llm returns None (zero candidates after
    reformulation), the node must surface this as a fatal error with
    assembly_id=None, not silently continue."""
    state = GenomeAgentState(user_question="genome of the xyzzy123", species_name="definitely not a real species xyzzy123")

    with patch(
        "genome_agent.workflows.nodes.species_resolver_node.resolve_species_llm",
        return_value=None,
    ):
        with patch(
            "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
            return_value={"assembly_id": None, "scientific_name": None, "common_name": None, "confidence": 0.0},
        ):
            result = await species_resolver_node(state)

    assert result["assembly_id"] is None
    assert len(result["errors"]) == 1
    assert "could not be resolved to a genome assembly" in result["errors"][0]


@pytest.mark.asyncio
async def test_species_resolver_node_exception_handling():
    """When resolve_species raises, the node must not crash and must return
    a well-shaped error dict with assembly_id=None and an error message
    referencing the raised exception."""
    state = GenomeAgentState(user_question="genome of the tiger", species_name="tiger")

    with patch(
        "genome_agent.workflows.nodes.species_resolver_node.resolve_species_llm",
        return_value=None,
    ):
        with patch(
            "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
            side_effect=Exception("Simulated NCBI failure"),
        ):
            result = await species_resolver_node(state)

    assert result["assembly_id"] is None
    assert len(result["errors"]) == 1
    assert "Simulated NCBI failure" in result["errors"][0]
