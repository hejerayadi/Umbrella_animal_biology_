"""Shared pytest fixtures for the Biodiversity Agent test suite."""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator
from backend.agents.biodiversity_agent.orchestrator.services.qdrant_client import (
    SpeciesTaxonomyService,
    _OfflineBackend,
)


@pytest.fixture
def orchestrator() -> BiodiversityOrchestrator:
    """Orchestrator wired to the offline taxonomy backend so the tests
    never hit Qdrant."""

    taxonomy = SpeciesTaxonomyService(_OfflineBackend())
    return BiodiversityOrchestrator(taxonomy=taxonomy)
