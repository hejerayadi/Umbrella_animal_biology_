"""Tests for the HTTP <-> orchestrator translation layer.

Narrow on purpose: the pure mapping functions only, with a stubbed
orchestrator. No NCBI calls, no LLM, no HTTP server - so these stay fast and
cannot fail for reasons unrelated to the mapping.

The orchestrator's own behaviour is covered by the other test modules, which
know nothing about this layer.
"""

from __future__ import annotations

import json

import pytest

from ..orchestrator_adapter import (
    OrchestratorGenomeAgent,
    resolve_species_name,
    to_result,
)
from ..schema import AgentRequest, AgentStatus
from ..workflows.state import GenomeAgentState


def _resolved_state(**overrides) -> GenomeAgentState:
    """A state as it looks after a successful run."""
    base = dict(
        user_question="What is the genome of the tiger?",
        species_name="tiger",
        assembly_id="GCF_000464555.1",
        species={
            "assembly_id": "GCF_000464555.1",
            "scientific_name": "Panthera tigris",
            "common_name": "Tiger",
            "confidence": 1.0,
        },
        metadata={"genome_size_bp": 2489000000, "chromosome_count": 19},
        annotation={
            "gene_list": ["TP53", "BRCA1"],
            "gene_table": [{"gene_name": "TP53", "gene_id": "7157"}],
        },
    )
    base.update(overrides)
    return GenomeAgentState(**base)


# --------------------------------------------------------------------------
# species resolution
# --------------------------------------------------------------------------
def test_context_species_is_preferred():
    """The Global Orchestrator's extractor is purpose-built for this."""
    name = resolve_species_name(
        AgentRequest(instruction="tell me about the tiger", context={"species": "Panthera tigris"})
    )
    assert name == "Panthera tigris"


def test_species_name_key_also_accepted():
    name = resolve_species_name(AgentRequest(instruction="x", context={"species_name": "tiger"}))
    assert name == "tiger"


def test_falls_back_to_the_instruction():
    """Called directly, with no extractor upstream, the sentence is all we have."""
    name = resolve_species_name(AgentRequest(instruction="tiger", context={}))
    assert name == "tiger"


# --------------------------------------------------------------------------
# result mapping
# --------------------------------------------------------------------------
def test_unresolved_species_fails_with_the_name_it_searched():
    result = to_result(GenomeAgentState(species_name="notaspecies", assembly_id=None))
    assert result.status is AgentStatus.FAILED
    assert "notaspecies" in result.output


def test_completed_always_publishes_the_genome_key():
    """Trait, Evolution, Protein and Reconstruction all branch on this key."""
    result = to_result(_resolved_state())
    assert result.status is AgentStatus.COMPLETED
    assert "genome" in result.output
    assert "Panthera tigris" in result.output["genome"]
    assert "GCF_000464555.1" in result.output["genome"]


def test_gene_list_is_published_for_the_trait_agent():
    """The reason for running the real agent: real symbols, not a fixed list."""
    result = to_result(_resolved_state())
    assert result.output["gene_list"] == ["TP53", "BRCA1"]


def test_missing_annotation_omits_gene_list_rather_than_sending_an_empty_one():
    """An empty list reads as 'no genes exist', which is not what happened."""
    result = to_result(_resolved_state(annotation=None))
    assert "gene_list" not in result.output
    assert "genome" in result.output


def test_output_is_json_serialisable():
    """Output is merged into the shared context and sent over HTTP."""
    result = to_result(_resolved_state())
    json.dumps(result.output)  # raises if bytes or a dataclass leaked in


def test_chart_bytes_never_reach_the_output():
    """A real SVG comes back as bytes, which JSON cannot carry."""
    result = to_result(
        _resolved_state(
            visualization={
                "status": "COMPLETED",
                "chart_data": b"<svg>...</svg>",
                "format": "svg",
                "comparisons": [{"common_name": "Tiger", "is_queried_species": True}],
            }
        )
    )
    json.dumps(result.output)
    assert result.output["visualization"]["available"] is True
    assert result.output["visualization"]["format"] == "svg"
    # The numbers behind the chart still travel; the bytes do not.
    assert result.output["visualization"]["comparisons"]
    assert "chart_data" not in result.output["visualization"]


def test_protein_structure_request_escalates():
    result = to_result(
        _resolved_state(
            visualization={
                "status": "NEEDS_AGENT",
                "target_agent": None,
                "prompt_to_target_agent": "Render an interactive, labeled 3D structure.",
                "chart_data": None,
                "format": None,
            }
        )
    )
    assert result.status is AgentStatus.NEEDS_AGENT
    assert "3D structure" in result.prompt_to_target_agent
    # target_agent stays None on purpose: this agent names the need, and the
    # Global Orchestrator's resolver picks who fills it from the prompt.
    assert result.target_agent is None
    # Findings gathered before the pause still travel.
    assert result.output["gene_list"] == ["TP53", "BRCA1"]


def test_errors_become_warnings_not_a_failure():
    result = to_result(_resolved_state(errors=["NCBI timed out for chromosome count"]))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["warnings"] == ["NCBI timed out for chromosome count"]


# --------------------------------------------------------------------------
# the agent wrapper
# --------------------------------------------------------------------------
class _StubOrchestrator:
    def __init__(self, state: GenomeAgentState):
        self._state = state
        self.calls: list[dict] = []

    async def run(self, user_question, species_name, visualization_scope=""):
        self.calls.append(
            {
                "user_question": user_question,
                "species_name": species_name,
                "visualization_scope": visualization_scope,
            }
        )
        return self._state


@pytest.mark.asyncio
async def test_run_passes_the_question_and_lets_the_router_pick_the_scope():
    stub = _StubOrchestrator(_resolved_state())
    agent = OrchestratorGenomeAgent(orchestrator=stub)

    await agent.run(
        AgentRequest(instruction="Show me the tiger genome", context={"species": "tiger"})
    )

    assert stub.calls[0]["user_question"] == "Show me the tiger genome"
    assert stub.calls[0]["species_name"] == "tiger"
    # "" means "router, you decide" - any other value would be treated as an
    # explicit caller request and would override the router's inference.
    assert stub.calls[0]["visualization_scope"] == ""


@pytest.mark.asyncio
async def test_run_without_a_species_fails_before_calling_ncbi():
    stub = _StubOrchestrator(_resolved_state())
    agent = OrchestratorGenomeAgent(orchestrator=stub)

    result = await agent.run(AgentRequest(instruction="   ", context={}))

    assert result.status is AgentStatus.FAILED
    assert stub.calls == []  # no network call attempted
