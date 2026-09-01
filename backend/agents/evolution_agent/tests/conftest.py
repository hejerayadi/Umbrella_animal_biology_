"""Shared pytest fixtures for the Evolution Agent test suite."""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.tools import iqtree as _iqtree_mod
from backend.agents.evolution_agent.workers.phylogenetic_tree import worker as _phylo_mod

from backend.agents.evolution_agent.orchestrator import EvolutionOrchestrator
from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
    MolecularComparisonMock,
)
from backend.agents.evolution_agent.workers.phylogenetic_tree.worker import (
    _SEQUENCES,
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


def _offline_fetch(species: str, _gene_candidates: list[str]) -> str:
    """Stand-in for the UniProt fetch — serves the offline catalogue.

    The phylogenetic worker fetches from UniProt like the molecular one, so
    without this the suite would depend on a live network call. Returning
    the catalogue sequence directly (rather than letting the worker fall
    back to it) also keeps runs free of the ``offline_sequence_fallback``
    warning, which only belongs on a genuine fetch failure.
    """
    key = species.strip().lower()
    if key in _SEQUENCES:
        return _SEQUENCES[key]
    raise ValueError(f"no offline sequence for {species!r}")


def offline_phylo_worker(**kwargs) -> PhylogeneticTreeWorker:
    """A PhylogeneticTreeWorker that never touches the network."""
    kwargs.setdefault("fetch_fn", _offline_fetch)
    return PhylogeneticTreeWorker(**kwargs)


def _offline_orchestrator(**kwargs) -> EvolutionOrchestrator:
    """Return an orchestrator wired to the offline species resolver."""
    resolver = SpeciesResolverService(_OfflineBackend())
    kwargs.setdefault("mc_worker", MolecularComparisonMock())
    kwargs.setdefault("phylo_worker", offline_phylo_worker())
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


@pytest.fixture(autouse=True)
def _no_network_sequence_fetch(monkeypatch):
    """Keep the whole suite off UniProt.

    The phylogenetic worker fetches sequences the same way the molecular
    one does, so every plain ``PhylogeneticTreeWorker()`` in these tests
    would otherwise make a live HTTP call. Patching the module-level
    default covers direct instantiations too, not just the fixtures.

    A test that wants to exercise fetch failure injects its own
    ``fetch_fn``; that still wins, because ``__init__`` prefers the
    argument over this default.
    """
    monkeypatch.setattr(_phylo_mod, "fetch_uniprot_sequence", _offline_fetch)


# ---------------------------------------------------------------------------
# External bioinformatics tools
# ---------------------------------------------------------------------------
#
# MAFFT and IQ-TREE are real executables that must be installed separately.
# The suite stubs them the same way tests/test_tool_contracts.py already
# does, so orchestrator, branch and explainer tests exercise the worker's
# own logic -- id mapping, taxon-loss detection, UFBoot semantics, warning
# propagation -- without depending on a local toolchain.
#
# The wrappers around the real binaries are covered separately in
# tests/test_tool_contracts.py. A test that needs different tool behaviour
# monkeypatches these names itself; that wins, because it is applied after
# this fixture.


def _fake_mafft(sequences, *_args, **_kwargs) -> str:
    """Deterministic 'alignment': right-pad every sequence to equal length."""
    width = max((len(s) for s in sequences.values()), default=0)
    return "".join(
        f">{seq_id}\n{seq.ljust(width, '-')}\n" for seq_id, seq in sequences.items()
    )


def _fake_iqtree(alignment, model="MFP", bootstrap=1000, **_kwargs):
    """Deterministic ladder tree over the alignment's taxa.

    Mirrors the real client's contract: support values only exist when
    UFBoot actually ran, and ``overall_confidence`` is ``None`` otherwise
    rather than a fabricated default.
    """
    ids = list(alignment)
    support: dict[str, int] = {}
    node = f"{ids[0]}:0.1"
    for index, taxon in enumerate(ids[1:]):
        if bootstrap > 0:
            support[f"node_{index}"] = 95
            node = f"({node},{taxon}:0.1)95:0.1"
        else:
            node = f"({node},{taxon}:0.1):0.1"
    newick = node.rsplit(":", 1)[0] + ";"

    confidence = {k: v / 100.0 for k, v in support.items()}
    overall = (
        round(sum(confidence.values()) / len(confidence), 4) if confidence else None
    )
    return _iqtree_mod.PhyloResult(
        newick_tree=newick,
        model="LG+G4" if model == "MFP" else model,
        bootstrap_support=support,
        confidence_values=confidence,
        overall_confidence=overall,
        ufboot_run=bool(support),
    )


@pytest.fixture(autouse=True)
def _stub_external_tools(monkeypatch):
    monkeypatch.setattr(_phylo_mod, "mafft_align", _fake_mafft)
    monkeypatch.setattr(_phylo_mod, "iqtree_build", _fake_iqtree)
