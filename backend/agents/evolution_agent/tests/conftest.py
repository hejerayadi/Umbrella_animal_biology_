"""Shared pytest fixtures for the Evolution Agent test suite."""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.orchestrator import EvolutionOrchestrator
from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
    MolecularComparisonMock,
)
from backend.agents.evolution_agent.workers.phylogenetic_tree.worker import (
    PhylogeneticTreeWorker,
)
from backend.agents.evolution_agent.orchestrator.services.species_resolver import (
    SpeciesResolverService,
    _OfflineBackend,
)


async def _offline_explainer(**_kwargs) -> str:
    """Deterministic stand-in for LLM #2 — keeps the suite off the network.

    Tests that care about the Explainer inject their own double; this only
    stops every *other* test from reaching a real backend once the explain
    node was wired into the graph.
    """
    return "Offline test interpretation."


def _offline_orchestrator(**kwargs) -> EvolutionOrchestrator:
    """Return an orchestrator wired to the offline species resolver."""
    resolver = SpeciesResolverService(_OfflineBackend())
    kwargs.setdefault("mc_worker", MolecularComparisonMock())
    kwargs.setdefault("phylo_worker", PhylogeneticTreeWorker())
    kwargs.setdefault("explainer", _offline_explainer)
    return EvolutionOrchestrator(resolver=resolver, **kwargs)


@pytest.fixture
def orchestrator() -> EvolutionOrchestrator:
    """Test orchestrator with deterministic worker doubles."""
    return _offline_orchestrator()


@pytest.fixture
def make_orchestrator():
    """Factory fixture — lets tests inject stub workers."""
    return _offline_orchestrator
