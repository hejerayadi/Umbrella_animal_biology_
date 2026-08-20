"""Shared pytest fixtures for the Evolution Agent Sprint 2 test suite."""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.orchestrator import EvolutionOrchestrator
from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
    MolecularComparisonMock,
)
from backend.agents.evolution_agent.workers.phylogenetic_tree.mock import (
    PhylogeneticTreeMock,
)
from backend.agents.evolution_agent.orchestrator.services.species_resolver import (
    SpeciesResolverService,
    _OfflineBackend,
)


def _offline_orchestrator(**kwargs) -> EvolutionOrchestrator:
    """Return an orchestrator wired to the offline species resolver.

    ``**kwargs`` are forwarded to EvolutionOrchestrator so individual tests
    can inject custom mc_worker / phylo_worker stubs.
    """
    resolver = SpeciesResolverService(_OfflineBackend())
    kwargs.setdefault("mc_worker", MolecularComparisonMock())
    kwargs.setdefault("phylo_worker", PhylogeneticTreeMock())
    return EvolutionOrchestrator(resolver=resolver, **kwargs)


@pytest.fixture
def orchestrator() -> EvolutionOrchestrator:
    """Test orchestrator with deterministic worker doubles."""
    return _offline_orchestrator()


@pytest.fixture
def make_orchestrator():
    """Factory fixture — lets tests inject stub workers."""
    return _offline_orchestrator
