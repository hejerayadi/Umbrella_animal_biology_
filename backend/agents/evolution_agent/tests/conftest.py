"""Shared pytest fixtures for the Evolution Agent Sprint 2 test suite."""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.orchestrator import EvolutionOrchestrator
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
    return EvolutionOrchestrator(resolver=resolver, **kwargs)


@pytest.fixture
def orchestrator() -> EvolutionOrchestrator:
    """Default orchestrator — both real mocks, offline resolver."""
    return _offline_orchestrator()


@pytest.fixture
def make_orchestrator():
    """Factory fixture — lets tests inject stub workers."""
    return _offline_orchestrator
