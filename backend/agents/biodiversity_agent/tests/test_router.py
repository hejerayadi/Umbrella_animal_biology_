"""Router-level tests - the smallest unit of the orchestrator."""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.orchestrator.router import Router
from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    BiodiversityFeature,
)
from backend.agents.biodiversity_agent.workers.habitat.mock import HabitatMock
from backend.agents.biodiversity_agent.workers.hotspots.mock import HotspotsMock
from backend.agents.biodiversity_agent.workers.migration.mock import MigrationMock
from backend.agents.biodiversity_agent.workers.species_distribution.mock import (
    SpeciesDistributionMock,
)


@pytest.fixture
def router() -> Router:
    return Router(
        {
            BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock(),
            BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
            BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsMock(),
            BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
        }
    )


def test_pick_routes_species_distribution(router: Router) -> None:
    request = AgentRequest(
        instruction="Where do African elephants live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="Loxodonta africana",
    )
    feature, worker = router.pick(request)
    assert feature is BiodiversityFeature.SPECIES_DISTRIBUTION_MAP
    result = worker.run(request)
    assert isinstance(result, AgentResult)
    assert result.status is AgentStatus.COMPLETED


def test_pick_raises_on_missing_feature(router: Router) -> None:
    request = AgentRequest(instruction="anything", context={})
    with pytest.raises(ValueError, match="feature is required"):
        router.pick(request)


def test_pick_raises_on_unknown_feature(router: Router) -> None:
    request = AgentRequest(instruction="anything", context={}, feature="not_a_feature")
    with pytest.raises(ValueError, match="Unknown biodiversity feature"):
        router.pick(request)


def test_router_rejects_incomplete_worker_map() -> None:
    with pytest.raises(ValueError, match="missing worker"):
        Router({BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock()})


def test_pick_many_returns_ordered_workers(router: Router) -> None:
    features = [
        BiodiversityFeature.SPECIES_DISTRIBUTION_MAP,
        BiodiversityFeature.MIGRATION_ANALYSIS,
    ]
    picked = router.pick_many(features)
    assert [f for f, _ in picked] == features
