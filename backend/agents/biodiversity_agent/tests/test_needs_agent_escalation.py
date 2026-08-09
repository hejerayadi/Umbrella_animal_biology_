"""Escalation path - NEEDS_AGENT bubbles up unchanged."""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator
from backend.agents.biodiversity_agent.orchestrator.services.qdrant_client import (
    SpeciesTaxonomyService,
    _OfflineBackend,
)
from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    BiodiversityFeature,
)
from backend.agents.biodiversity_agent.workers.habitat.mock import HabitatMock
from backend.agents.biodiversity_agent.workers.hotspots.mock import HotspotsMock
from backend.agents.biodiversity_agent.workers.migration.mock import MigrationMock


class _NeedsAgentDistributionMock:
    """Stand-in worker that always escalates - simulates the case where
    a Biodiversity worker discovers it needs the Trait Discovery Agent
    to answer a trait-driven distribution query."""

    def run(self, request: AgentRequest) -> AgentResult:
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent="Trait Discovery Agent",
            prompt_to_target_agent=(
                f"Resolve traits driving the distribution of "
                f"'{request.species_name}'."
            ),
            source_agents=["Species Distribution Agent"],
        )


@pytest.fixture
def escalating_orchestrator() -> BiodiversityOrchestrator:
    workers = {
        BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: _NeedsAgentDistributionMock(),
        BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
        BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsMock(),
        BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
    }
    taxonomy = SpeciesTaxonomyService(_OfflineBackend())
    return BiodiversityOrchestrator(workers=workers, taxonomy=taxonomy)


@pytest.mark.asyncio
async def test_needs_agent_bubbles_up_to_global(escalating_orchestrator) -> None:
    request = AgentRequest(
        instruction="Where do African elephants live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="Loxodonta africana",
    )
    result = await escalating_orchestrator.run(request)

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Trait Discovery Agent"
    assert "Loxodonta africana" in (result.prompt_to_target_agent or "")
