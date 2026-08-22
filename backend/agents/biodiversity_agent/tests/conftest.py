"""Shared pytest fixtures for the Biodiversity Agent test suite."""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator
from backend.agents.biodiversity_agent.orchestrator.services.qdrant_client import (
    SpeciesTaxonomyService,
    _OfflineBackend,
)
from backend.agents.biodiversity_agent.schema import BiodiversityFeature
from backend.agents.biodiversity_agent.workers.habitat.mock import HabitatMock
from backend.agents.biodiversity_agent.workers.hotspots.mock import HotspotsMock
from backend.agents.biodiversity_agent.workers.migration.mock import MigrationMock
from backend.agents.biodiversity_agent.workers.species_distribution.mock import (
    SpeciesDistributionMock,
)


@pytest.fixture
def orchestrator() -> BiodiversityOrchestrator:
    """Orchestrator wired to the offline taxonomy backend AND to the
    deterministic mock workers, so the tests never hit Qdrant or GBIF.

    The production default (see ``BiodiversityOrchestrator.__init__``)
    uses ``SpeciesDistributionWorker`` which calls real GBIF - that
    behaviour is exercised through manual smoke tests, not unit tests.
    """

    workers = {
        BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock(),
        BiodiversityFeature.HABITAT_VISUALIZATION:    HabitatMock(),
        BiodiversityFeature.BIODIVERSITY_HOTSPOTS:    HotspotsMock(),
        BiodiversityFeature.MIGRATION_ANALYSIS:       MigrationMock(),
    }
    taxonomy = SpeciesTaxonomyService(_OfflineBackend())
    return BiodiversityOrchestrator(workers=workers, taxonomy=taxonomy)
