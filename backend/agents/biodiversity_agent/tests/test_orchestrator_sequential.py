"""Sequential path - single feature -> single worker."""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentStatus,
    BiodiversityFeature,
)


@pytest.mark.asyncio
async def test_species_distribution_end_to_end(orchestrator) -> None:
    request = AgentRequest(
        instruction="Where do African elephants live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="African elephant",  # common name -> normalized via taxonomy
    )
    result = await orchestrator.run(request)

    assert result.status is AgentStatus.COMPLETED
    assert result.map_url
    assert result.observation_count and result.observation_count > 0
    assert "Species Distribution Agent" in result.source_agents


@pytest.mark.asyncio
async def test_hotspots_does_not_require_species(orchestrator) -> None:
    request = AgentRequest(
        instruction="Show biodiversity hotspots.",
        context={},
        feature=BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value,
        region="global",
    )
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.COMPLETED
    assert result.hotspots and len(result.hotspots) > 0


@pytest.mark.asyncio
async def test_no_feature_fails_cleanly(orchestrator) -> None:
    request = AgentRequest(instruction="anything", context={})
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.FAILED
    assert "no valid feature" in str(result.output).lower()
