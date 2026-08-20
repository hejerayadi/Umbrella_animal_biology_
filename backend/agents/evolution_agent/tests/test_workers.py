"""Unit tests for the two Sprint 2 worker mocks.

These tests exercise each worker in isolation — no orchestrator, no
LangGraph, no resolver.  They verify the data contracts the orchestrator
depends on: output type, required fields, failure conditions.
"""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentStatus,
    MolecularComparisonResult,
    PhylogeneticResult,
    SimilarityEdge,
    SpeciesGroup,
)
from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
    MolecularComparisonMock,
)
from backend.agents.evolution_agent.workers.phylogenetic_tree.mock import (
    PhylogeneticTreeMock,
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

    def test_fasta_alignment_contains_all_species(self) -> None:
        species = ["homo sapiens", "pan troglodytes", "mus musculus"]
        r = self.mc.run(_req(species))
        alignment = r.output.alignment
        for sp in species:
            assert sp.replace(" ", "_") in alignment

    def test_fasta_alignment_has_correct_header_count(self) -> None:
        species = ["homo sapiens", "mus musculus", "gallus gallus"]
        r = self.mc.run(_req(species))
        headers = [l for l in r.output.alignment.splitlines() if l.startswith(">")]
        assert len(headers) == 3

    def test_similarity_scores_are_similarity_edge_objects(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "mus musculus"]))
        assert all(isinstance(e, SimilarityEdge) for e in r.output.similarity_scores)

    def test_correct_number_of_pairwise_scores(self) -> None:
        # n=3 species → 3 pairs
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
        """human(0.98)-chimp + human(0.85)-mouse + chimp(0.84)-mouse all ≥ 0.75
        → all three in one connected component."""
        r = self.mc.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert len(r.output.species_groups) == 1
        assert len(r.output.species_groups[0].species) == 3

    def test_zebrafish_forms_its_own_group(self) -> None:
        """zebrafish scores < 0.75 against everything → isolated node."""
        r = self.mc.run(
            _req(["homo sapiens", "pan troglodytes", "mus musculus", "danio rerio"])
        )
        group_sizes = sorted(len(g.species) for g in r.output.species_groups)
        assert 1 in group_sizes      # danio rerio alone
        assert 3 in group_sizes      # the three mammals

    def test_similarity_network_keys_match_species(self) -> None:
        species = ["homo sapiens", "mus musculus"]
        r = self.mc.run(_req(species))
        assert set(r.output.similarity_network.keys()) == set(species)

    def test_similarity_network_is_symmetric(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "mus musculus"]))
        net = r.output.similarity_network
        # human lists mouse as neighbour
        human_neighbours = {e["neighbour"] for e in net["homo sapiens"]}
        mouse_neighbours = {e["neighbour"] for e in net["mus musculus"]}
        assert "mus musculus"  in human_neighbours
        assert "homo sapiens"  in mouse_neighbours

    def test_alignment_url_is_returned(self) -> None:
        r = self.mc.run(_req(["homo sapiens", "danio rerio"]))
        assert r.output.alignment_url.startswith("https://")
        assert r.alignment_url == r.output.alignment_url

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


# ---------------------------------------------------------------------------
# PhylogeneticTreeMock
# ---------------------------------------------------------------------------

class TestPhylogeneticTreeMock:
    pt = PhylogeneticTreeMock()

    def test_returns_phylogenetic_result_in_output(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert r.status is AgentStatus.COMPLETED
        assert isinstance(r.output, PhylogeneticResult)

    def test_newick_tree_contains_all_species(self) -> None:
        species = ["homo sapiens", "pan troglodytes", "mus musculus"]
        r = self.pt.run(_req(species))
        for sp in species:
            assert sp in r.output.newick_tree

    def test_newick_tree_ends_with_semicolon(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert r.output.newick_tree.endswith(";")

    def test_model_is_lg_g4(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert r.output.model == "LG+G4"

    def test_bootstrap_support_is_dict_of_ints(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        bs = r.output.bootstrap_support
        assert isinstance(bs, dict)
        assert all(isinstance(v, int) for v in bs.values())

    def test_bootstrap_values_in_0_100(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        for v in r.output.bootstrap_support.values():
            assert 0 <= v <= 100

    def test_confidence_values_keyed_by_species(self) -> None:
        species = ["homo sapiens", "pan troglodytes", "mus musculus"]
        r = self.pt.run(_req(species))
        assert set(r.output.confidence_values.keys()) == set(species)

    def test_confidence_values_in_0_1(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        for v in r.output.confidence_values.values():
            assert 0.0 <= v <= 1.0

    def test_overall_confidence_is_mean_bootstrap(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        bs = r.output.bootstrap_support
        expected = round(sum(bs.values()) / len(bs) / 100, 4)
        assert r.output.overall_confidence == expected

    def test_accepts_alignment_from_context(self) -> None:
        """Phylo worker must not fail when context['alignment'] is present."""
        r = self.pt.run(
            _req(
                ["homo sapiens", "pan troglodytes", "mus musculus"],
                context={"alignment": ">homo_sapiens\nMTNIRKSHP\n>pan_troglodytes\nMTNIRKSHP"},
            )
        )
        assert r.status is AgentStatus.COMPLETED

    def test_tree_url_is_returned(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes", "mus musculus"]))
        assert r.output.tree_url.startswith("https://")
        assert r.tree_url == r.output.tree_url

    def test_fails_with_two_species(self) -> None:
        r = self.pt.run(_req(["homo sapiens", "pan troglodytes"]))
        assert r.status is AgentStatus.FAILED
        assert "at least 3" in r.output.lower()

    def test_fails_with_unknown_species(self) -> None:
        r = self.pt.run(
            _req(["homo sapiens", "pan troglodytes", "draco magicus"])
        )
        assert r.status is AgentStatus.FAILED

    def test_five_species_full_newick(self) -> None:
        all_five = [
            "homo sapiens", "pan troglodytes", "mus musculus",
            "gallus gallus", "danio rerio",
        ]
        r = self.pt.run(_req(all_five))
        assert r.status is AgentStatus.COMPLETED
        assert "danio rerio" in r.output.newick_tree
