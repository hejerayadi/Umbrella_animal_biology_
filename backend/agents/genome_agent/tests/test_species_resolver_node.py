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
async def test_species_resolver_node_exception_handling():
    """When resolve_species raises, the node must not crash and must return
    a well-shaped error dict with assembly_id=None and an error message
    referencing the raised exception."""
    state = GenomeAgentState(user_question="genome of the tiger", species_name="tiger")

    with patch(
        "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
        side_effect=Exception("Simulated NCBI failure"),
    ):
        result = await species_resolver_node(state)

    assert result["assembly_id"] is None
    assert len(result["errors"]) == 1
    assert "Simulated NCBI failure" in result["errors"][0]
