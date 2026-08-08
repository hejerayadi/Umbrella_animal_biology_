from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from backend.agents.genome_agent.orchestrator import GenomeAgentOrchestrator


@pytest.mark.asyncio
async def test_species_resolver_exception_handling():
    """When resolve_species raises, the orchestrator must not crash and must
    return a well-shaped error dict with assembly_id=None, confidence=0.0,
    and an error message referencing the raised exception."""
    orch = GenomeAgentOrchestrator()

    with patch(
        "backend.agents.genome_agent.orchestrator.resolve_species",
        side_effect=Exception("Simulated NCBI failure"),
    ):
        result = await orch.run("tiger", visualization_scope="chromosome_map")

    assert isinstance(result["species"], dict)
    assert result["species"]["assembly_id"] is None
    assert result["species"]["confidence"] == 0.0
    assert result["metadata"] is None
    assert result["annotation"] is None
    assert result["visualization"] is None
    assert len(result["errors"]) == 1
    assert "Simulated NCBI failure" in result["errors"][0]
