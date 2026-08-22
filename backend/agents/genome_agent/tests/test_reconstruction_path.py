"""
tests/test_reconstruction_path.py
==================================
Unit + integration tests for the genome-reconstruction escalation path.

What is covered
---------------
1.  State dataclass         – reconstruction_need field exists and defaults to None.
2.  Metadata node (unit)    – scaffold / contig assembly levels write the correct
                              reconstruction_need dict; chromosome / complete levels
                              leave it None; missing assembly_level is treated as
                              complete (no false positives).
3.  Reconstruction resolver – resolve_capability / resolve_capability_fallback are
                              called; target_agent is set when a match is found; a
                              "none" decision writes FAILED and doesn't crash.
4.  Graph routing (integration) – end-to-end with every external I/O call mocked:
        a. scaffold assembly → reconstruction_resolver runs → adapter returns
           NEEDS_AGENT with target_agent == "Reconstruction Agent".
        b. complete assembly → reconstruction_resolver is skipped → adapter
           returns COMPLETED (no regression).
5.  Adapter (unit)          – to_result() orders the reconstruction check before
                              the visualization check so both can be present
                              simultaneously without the visualization check
                              winning by accident.

Run
---
    pytest tests/test_reconstruction_path.py -v
    pytest tests/test_reconstruction_path.py -v -k "scaffold"   # one sub-set
"""
from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers – let the tests import gracefully even when the package layout
# hasn't been installed yet (e.g. raw `pytest` from the repo root).
# ---------------------------------------------------------------------------
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genome_agent.schemas import AgentStatus


# ===========================================================================
# 1. STATE DATACLASS
# ===========================================================================

class TestGenomeAgentStateReconstructionField:
    """The reconstruction_need field must be present and default to None."""

    def test_field_exists_and_defaults_to_none(self):
        from genome_agent.workflows.state import GenomeAgentState

        state = GenomeAgentState(user_question="test")
        assert hasattr(state, "reconstruction_need"), (
            "GenomeAgentState is missing the reconstruction_need field. "
            "Add `reconstruction_need: dict | None = None` to the dataclass."
        )
        assert state.reconstruction_need is None

    def test_field_accepts_dict(self):
        from genome_agent.workflows.state import GenomeAgentState

        payload = {"status": "NEEDS_AGENT", "target_agent": None, "prompt_to_target_agent": "do stuff"}
        state = GenomeAgentState(user_question="test", reconstruction_need=payload)
        assert state.reconstruction_need == payload


# ===========================================================================
# 2. METADATA NODE – _INCOMPLETE_LEVELS detection
# ===========================================================================

_GOOD_METADATA = {
    "genome_size_bp": 2_900_000_000,
    "assembly_level": "Chromosome",
    "species_name": "Homo sapiens",
}

_SCAFFOLD_METADATA = {**_GOOD_METADATA, "assembly_level": "Scaffold"}
_CONTIG_METADATA   = {**_GOOD_METADATA, "assembly_level": "Contig"}
_COMPLETE_METADATA = {**_GOOD_METADATA, "assembly_level": "Complete Genome"}
_NONE_LEVEL_META   = {**_GOOD_METADATA, "assembly_level": None}


class TestMetadataNodeReconstructionDetection:
    """
    get_genome_metadata_node must set reconstruction_need when assembly_level
    is 'Scaffold' or 'Contig', and leave it None for every other level.
    """

    @pytest.fixture()
    def base_state(self):
        from genome_agent.workflows.state import GenomeAgentState
        return GenomeAgentState(
            user_question="test",
            needs_metadata=True,
            assembly_id="GCF_000001405.40",
        )

    # -- helpers --

    def _run_node(self, metadata_return, base_state):
        from genome_agent.workflows.nodes.genome_data_nodes import get_genome_metadata_node

        with patch(
            "genome_agent.workflows.nodes.genome_data_nodes.get_genome_metadata",
            new=AsyncMock(return_value=metadata_return),
        ):
            return asyncio.run(
                get_genome_metadata_node(base_state)
            )

    # -- scaffold --

    def test_scaffold_sets_reconstruction_need(self, base_state):
        result = self._run_node(_SCAFFOLD_METADATA, base_state)
        need = result.get("reconstruction_need")
        assert need is not None, "Expected reconstruction_need to be set for Scaffold assembly"
        assert need["status"] == "NEEDS_AGENT"
        assert need["target_agent"] is None          # resolver fills this in later
        assert "Scaffold" in need["prompt_to_target_agent"]
        assert "gap" in need["prompt_to_target_agent"].lower() or "reconstruct" in need["prompt_to_target_agent"].lower()

    def test_scaffold_case_insensitive(self, base_state):
        """Node must handle NCBI returning mixed-case strings."""
        meta = {**_SCAFFOLD_METADATA, "assembly_level": "SCAFFOLD"}
        result = self._run_node(meta, base_state)
        assert result.get("reconstruction_need") is not None

    # -- contig --

    def test_contig_sets_reconstruction_need(self, base_state):
        result = self._run_node(_CONTIG_METADATA, base_state)
        need = result.get("reconstruction_need")
        assert need is not None, "Expected reconstruction_need to be set for Contig assembly"
        assert need["status"] == "NEEDS_AGENT"
        assert "Contig" in need["prompt_to_target_agent"]

    # -- complete / chromosome (no flag) --

    def test_chromosome_leaves_reconstruction_need_none(self, base_state):
        result = self._run_node(_GOOD_METADATA, base_state)
        assert result.get("reconstruction_need") is None

    def test_complete_genome_leaves_reconstruction_need_none(self, base_state):
        result = self._run_node(_COMPLETE_METADATA, base_state)
        assert result.get("reconstruction_need") is None

    def test_none_assembly_level_leaves_reconstruction_need_none(self, base_state):
        """A missing assembly_level should NOT trigger reconstruction — no false positives."""
        result = self._run_node(_NONE_LEVEL_META, base_state)
        assert result.get("reconstruction_need") is None

    # -- prompt content --

    def test_prompt_contains_assembly_id(self, base_state):
        result = self._run_node(_SCAFFOLD_METADATA, base_state)
        need = result["reconstruction_need"]
        assert base_state.assembly_id in need["prompt_to_target_agent"]

    # -- metadata still returned alongside the flag --

    def test_metadata_also_returned_with_flag(self, base_state):
        result = self._run_node(_SCAFFOLD_METADATA, base_state)
        assert result.get("metadata") is not None, (
            "Metadata must still be included in the return dict even when "
            "reconstruction_need is flagged — downstream nodes need it."
        )


# ===========================================================================
# 3. RECONSTRUCTION RESOLVER NODE
# ===========================================================================

def _make_decision(target_agent: str, handoff_message: str = "please reconstruct"):
    decision = MagicMock()
    decision.target_agent = target_agent
    decision.handoff_message = handoff_message
    return decision


class TestReconstructionResolverNode:
    """reconstruction_resolver_node must fill in target_agent and forward the prompt."""

    @pytest.fixture()
    def incomplete_state(self):
        from genome_agent.workflows.state import GenomeAgentState
        return GenomeAgentState(
            user_question="reconstruct axolotl",
            assembly_id="GCF_testid",
            reconstruction_need={
                "status": "NEEDS_AGENT",
                "target_agent": None,
                "prompt_to_target_agent": "Genome assembly GCF_testid is at 'Scaffold' level with gaps.",
            },
        )

    def _run_resolver(self, state, resolve_return=None, fallback_return=None):
        from genome_agent.workflows.nodes.reconstruction_resolver_node import (
            reconstruction_resolver_node,
        )

        resolve_rv   = resolve_return  or _make_decision("Reconstruction Agent", "reconstruct this")
        fallback_rv  = fallback_return or _make_decision("Reconstruction Agent", "reconstruct this (fallback)")

        with (
            patch(
                "genome_agent.workflows.nodes.reconstruction_resolver_node.resolve_capability",
                return_value=resolve_rv,
            ),
            patch(
                "genome_agent.workflows.nodes.reconstruction_resolver_node.resolve_capability_fallback",
                return_value=fallback_rv,
            ),
        ):
            return asyncio.run(
                reconstruction_resolver_node(state)
            )

    def test_resolver_sets_target_agent(self, incomplete_state):
        result = self._run_resolver(incomplete_state)
        need = result["reconstruction_need"]
        assert need["target_agent"] == "Reconstruction Agent"
        assert need["status"] == "NEEDS_AGENT"

    def test_resolver_sets_handoff_message(self, incomplete_state):
        result = self._run_resolver(incomplete_state)
        need = result["reconstruction_need"]
        assert need["prompt_to_target_agent"] is not None
        assert len(need["prompt_to_target_agent"]) > 0

    def test_resolver_uses_fallback_when_primary_returns_none(self, incomplete_state):
        """If resolve_capability returns None, fallback must be used."""
        fallback = _make_decision("Reconstruction Agent", "fallback message")
        result = self._run_resolver(incomplete_state, resolve_return=None, fallback_return=fallback)
        assert result["reconstruction_need"]["target_agent"] == "Reconstruction Agent"

    def test_resolver_writes_failed_when_no_agent_found(self, incomplete_state):
        """If both resolver and fallback return 'none', status must be FAILED."""
        no_agent = _make_decision("none")
        result = self._run_resolver(incomplete_state, resolve_return=no_agent, fallback_return=no_agent)
        need = result["reconstruction_need"]
        assert need["status"] == "FAILED"
        assert need["target_agent"] is None
        assert len(result.get("errors", [])) > 0

    def test_resolver_propagates_existing_context(self, incomplete_state):
        """resolver must not wipe reconstruction_need fields it didn't change."""
        result = self._run_resolver(incomplete_state)
        # 'status' key must still be present
        assert "status" in result["reconstruction_need"]


# ===========================================================================
# 4. GRAPH ROUTING – end-to-end integration (all I/O mocked)
# ===========================================================================

_SCAFFOLD_SPECIES = {"taxon_id": "8296", "scientific_name": "Ambystoma mexicanum"}
_SCAFFOLD_ASSEMBLY = "GCF_002915635.1"

_COMPLETE_SPECIES  = {"taxon_id": "9606", "scientific_name": "Homo sapiens"}
_COMPLETE_ASSEMBLY = "GCF_000001405.40"


def _make_annotation():
    return {
        "gene_count": 20_000,
        "gene_list": [{"gene_id": "BRCA1", "symbol": "BRCA1"}],
    }


class TestGraphReconstructionRouting:
    """
    End-to-end graph run with all external calls mocked.
    Validates that the routing table sends scaffold assemblies through
    reconstruction_resolver and that the adapter surfaces NEEDS_AGENT.
    """

    @pytest.fixture(autouse=True)
    def _patch_all_io(self):
        """
        Patch the three external calls (species resolver, metadata, annotation)
        and the two capability resolver functions.  Each test can override the
        metadata mock to change the assembly_level.
        """
        self._metadata_mock   = AsyncMock()
        self._annotation_mock = AsyncMock(return_value=_make_annotation())

        with (
            patch(
                "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
                new=AsyncMock(return_value={**_SCAFFOLD_SPECIES, "assembly_id": _SCAFFOLD_ASSEMBLY}),
            ),
            patch(
                "genome_agent.workflows.nodes.genome_data_nodes.get_genome_metadata",
                new=self._metadata_mock,
            ),
            patch(
                "genome_agent.workflows.nodes.genome_data_nodes.get_gene_annotation",
                new=self._annotation_mock,
            ),
            patch(
                "genome_agent.workflows.nodes.reconstruction_resolver_node.resolve_capability",
                return_value=_make_decision("Reconstruction Agent", "Please reconstruct axolotl genome."),
            ),
            patch(
                "genome_agent.workflows.nodes.reconstruction_resolver_node.resolve_capability_fallback",
                return_value=_make_decision("Reconstruction Agent", "Please reconstruct axolotl genome (fallback)."),
            ),
            # explanation_writer — keep it fast with a no-op
            patch(
                "genome_agent.workflows.nodes.explanation_writer_node.write_explanation",
                new=AsyncMock(return_value="Assembly is incomplete; handing off to Reconstruction Agent."),
            ),
        ):
            yield

    def _run(self, species_name: str, visualization_scope: str = "none") -> Any:
        from genome_agent.orchestrator import GenomeAgentLangGraphOrchestrator
        from genome_agent.orchestrator_adapter import to_result

        orch = GenomeAgentLangGraphOrchestrator()
        state = asyncio.run(
            orch.run(
                user_question=f"Reconstruct the {species_name} genome.",
                species_name=species_name,
                visualization_scope=visualization_scope,
            )
        )
        # GenomeAgentLangGraphOrchestrator.run() returns the raw internal
        # GenomeAgentState (see test_orchestrator_adapter.py) — to_result()
        # is what maps reconstruction_need/visualization onto the platform's
        # AgentResult(status, target_agent, prompt_to_target_agent, output).
        return to_result(state)

    # ── 4a. Scaffold → NEEDS_AGENT ─────────────────────────────────────────

    def test_scaffold_assembly_returns_needs_agent(self):
        self._metadata_mock.return_value = _SCAFFOLD_METADATA
        result = self._run("axolotl")
        assert result.status == AgentStatus.NEEDS_AGENT, (
            f"Expected NEEDS_AGENT for scaffold assembly, got {result.status}"
        )

    def test_scaffold_assembly_target_is_reconstruction_agent(self):
        self._metadata_mock.return_value = _SCAFFOLD_METADATA
        result = self._run("axolotl")
        assert result.target_agent == "Reconstruction Agent", (
            f"target_agent should be 'Reconstruction Agent', got {result.target_agent!r}"
        )

    def test_scaffold_prompt_not_empty(self):
        self._metadata_mock.return_value = _SCAFFOLD_METADATA
        result = self._run("axolotl")
        assert result.prompt_to_target_agent, "prompt_to_target_agent must not be empty"

    def test_scaffold_output_still_contains_metadata(self):
        """Even when escalating, the partial output should carry whatever data was fetched."""
        self._metadata_mock.return_value = _SCAFFOLD_METADATA
        result = self._run("axolotl")
        assert result.output.get("genome_metadata") is not None, (
            "output['genome_metadata'] should be populated even when escalating to Reconstruction Agent"
        )

    # ── contig (second incomplete level) ──────────────────────────────────

    def test_contig_assembly_returns_needs_agent(self):
        self._metadata_mock.return_value = _CONTIG_METADATA
        with patch(
            "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
            new=AsyncMock(return_value={**_SCAFFOLD_SPECIES, "assembly_id": _SCAFFOLD_ASSEMBLY}),
        ):
            result = self._run("coelacanth")
        assert result.status == AgentStatus.NEEDS_AGENT
        assert result.target_agent == "Reconstruction Agent"

    # ── 4b. Complete assembly → NO reconstruction (regression guard) ────────

    def test_complete_assembly_does_not_trigger_reconstruction(self):
        self._metadata_mock.return_value = _COMPLETE_METADATA
        with patch(
            "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
            new=AsyncMock(return_value={**_COMPLETE_SPECIES, "assembly_id": _COMPLETE_ASSEMBLY}),
        ):
            result = self._run("human")
        assert result.status != AgentStatus.NEEDS_AGENT or result.target_agent != "Reconstruction Agent", (
            "A Chromosome-level assembly must NOT trigger reconstruction escalation."
        )

    # ── visualization path not accidentally blocked ────────────────────────

    def test_scaffold_with_visualization_scope_still_escalates_reconstruction(self):
        """
        If assembly is incomplete, reconstruction takes priority over visualization —
        the graph should go to reconstruction_resolver, not generate_visualization.
        """
        self._metadata_mock.return_value = _SCAFFOLD_METADATA
        result = self._run("axolotl", visualization_scope="chromosome_map")
        assert result.status == AgentStatus.NEEDS_AGENT
        assert result.target_agent == "Reconstruction Agent"


# ===========================================================================
# 5. ADAPTER – ordering of NEEDS_AGENT checks
# ===========================================================================

class TestAdapterReconstructionCheck:
    """
    to_result() must check reconstruction_need BEFORE visualization so that
    both can be simultaneously present without the wrong one winning.
    """

    def _make_state(self, reconstruction_need=None, visualization=None):
        from genome_agent.workflows.state import GenomeAgentState

        return GenomeAgentState(
            user_question="test",
            assembly_id="GCF_test",
            species={"scientific_name": "Test species"},
            metadata=_SCAFFOLD_METADATA,
            annotation=_make_annotation(),
            visualization=visualization,
            reconstruction_need=reconstruction_need,
            explanation="Some explanation text.",
        )

    def test_reconstruction_need_wins_over_visualization_needs_agent(self):
        """When both flags are set, the reconstruction escalation must be returned."""
        from genome_agent.orchestrator_adapter import to_result
        from genome_agent.schemas import AgentStatus

        state = self._make_state(
            reconstruction_need={
                "status": "NEEDS_AGENT",
                "target_agent": "Reconstruction Agent",
                "prompt_to_target_agent": "Please reconstruct.",
            },
            visualization={
                "status": "NEEDS_AGENT",
                "target_agent": "Protein Structure Visualization Agent",
                "prompt_to_target_agent": "Please visualize.",
            },
        )
        result = to_result(state)
        assert result.status == AgentStatus.NEEDS_AGENT
        assert result.target_agent == "Reconstruction Agent", (
            "Reconstruction escalation must take priority over visualization escalation."
        )

    def test_no_reconstruction_no_viz_returns_completed(self):
        from genome_agent.orchestrator_adapter import to_result
        from genome_agent.schemas import AgentStatus

        state = self._make_state(
            reconstruction_need=None,
            visualization={"status": "COMPLETED", "chart": {}},
        )
        result = to_result(state)
        assert result.status == AgentStatus.COMPLETED

    def test_reconstruction_failed_does_not_escalate(self):
        """A FAILED reconstruction_need (no agent found) must not produce NEEDS_AGENT."""
        from genome_agent.orchestrator_adapter import to_result
        from genome_agent.schemas import AgentStatus

        state = self._make_state(
            reconstruction_need={
                "status": "FAILED",
                "target_agent": None,
                "prompt_to_target_agent": None,
            },
            visualization={"status": "COMPLETED", "chart": {}},
        )
        result = to_result(state)
        # Status must be COMPLETED (visualization succeeded) or ERROR — not NEEDS_AGENT
        assert result.status != AgentStatus.NEEDS_AGENT or result.target_agent != "Reconstruction Agent"


# ===========================================================================
# 6. CAPABILITY RESOLVER KEYWORD FALLBACK
# ===========================================================================

class TestCapabilityResolverKeywords:
    """
    The keyword fallback must route gap/scaffold/contig/reconstruct/incomplete
    to 'Reconstruction Agent' even when the LLM is unavailable.
    """

    @pytest.mark.parametrize("keyword", [
        "gap", "incomplete", "scaffold", "contig", "reconstruct",
    ])
    def test_keyword_routes_to_reconstruction_agent(self, keyword):
        from genome_agent.workflows.capability_resolver import resolve_capability_fallback

        prompt = f"The assembly has {keyword} regions that need fixing."
        decision = resolve_capability_fallback(
            current_agent="genome_metadata",
            prompt_to_target_agent=prompt,
        )
        assert decision.target_agent == "Reconstruction Agent", (
            f"Keyword '{keyword}' in prompt should route to 'Reconstruction Agent', "
            f"got {decision.target_agent!r}. Add it to capability_keywords in "
            "genome_agent/workflows/capability_resolver.py."
        )

    def test_protein_keyword_still_routes_to_protein_agent(self):
        """Sanity check: protein keywords must not be accidentally remapped."""
        from genome_agent.workflows.capability_resolver import resolve_capability_fallback

        decision = resolve_capability_fallback(
            current_agent="genome_metadata",
            prompt_to_target_agent="Please predict the 3D protein structure.",
        )
        assert decision.target_agent == "Protein Structure Visualization Agent"