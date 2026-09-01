"""Unit tests for the Molecular Comparison mock worker.

Tests the MC mock in isolation — no orchestrator, no LangGraph, no resolver.
Verifies the data contracts the orchestrator depends on: output type, required
fields, failure conditions.
"""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentStatus,
    MolecularComparisonResult,
    SimilarityEdge,
    SpeciesGroup,
)
from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
    MolecularComparisonMock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(species: list[str], context: dict | None = None) -> AgentRequest:
    return AgentRequest(
        instruction="test",
        context=context or {},
        species_list=species,
    )


# ---------------------------------------------------------------------------
# MolecularComparisonMock
# ---------------------------------------------------------------------------

class TestMolecularComparisonMock:
    mc = MolecularComparisonMock()

    def test_returns_mc_result_in_output(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes"]))
        assert r.status is AgentStatus.COMPLETED
        assert isinstance(r.output, MolecularComparisonResult)

    def test_result_carries_no_alignment(self) -> None:
        """MAFFT is exclusive to the phylogenetic branch.

        Gap characters would corrupt ESM-2 embeddings, so this sub-agent
        never produces an alignment — see MolecularComparisonResult.
        """
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert not hasattr(r.output, "alignment")
        assert not hasattr(r.output, "alignment_url")

    def test_similarity_scores_are_similarity_edge_objects(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "mus musculus"]))
        assert all(isinstance(e, SimilarityEdge) for e in r.output.similarity_scores)

    def test_correct_number_of_pairwise_scores(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert len(r.output.similarity_scores) == 3

    def test_human_chimp_score_is_0_98(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes"]))
        edge = r.output.similarity_scores[0]
        assert edge.score == 0.98

    def test_species_groups_are_species_group_objects(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert all(isinstance(g, SpeciesGroup) for g in r.output.species_groups)

    def test_human_chimp_mouse_forms_one_group_above_threshold(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert len(r.output.species_groups) == 1
        assert len(r.output.species_groups[0].species) == 3

    def test_zebrafish_forms_its_own_group(self) -> None:
        r = self.mc.run(
            _req(["homo sapiens", "pan troglodytes", "mus musculus", "danio rerio"])
        )
        group_sizes = sorted(len(g.species) for g in r.output.species_groups)
        assert 1 in group_sizes
        assert 3 in group_sizes

    def test_similarity_network_nodes_match_species(self) -> None:
        """The network is nx.node_link_data output, not an adjacency dict."""
        species = ["homo sapiens", "mus musculus"]
        r = self.mc.run(_req(species))
        net = r.output.similarity_network
        assert {n["id"] for n in net["nodes"]} == set(species)

    def test_similarity_network_is_undirected(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "mus musculus"]))
        net = r.output.similarity_network
        assert net["directed"] is False
        pairs = {frozenset({e["source"], e["target"]}) for e in net["edges"]}
        assert frozenset({"homo sapiens", "mus musculus"}) in pairs

    def test_fails_with_one_species(self) -> None:
        r = self.mc.run(_req(["homo sapiens"]))
        assert r.status is AgentStatus.FAILED
        assert "at least 2" in r.output.lower()

    def test_fails_with_no_species(self) -> None:
        r = self.mc.run(_req([]))
        assert r.status is AgentStatus.FAILED

    def test_fails_for_unknown_species(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "draco magicus"]))
        assert r.status is AgentStatus.FAILED
        assert "draco magicus" in r.output.lower()

    def test_confidence_equals_mean_score(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes"]))
        assert r.confidence == 0.98
