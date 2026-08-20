"""Tests for the HTTP<->workflow translation layer.

Deliberately narrow: these cover the two pure mapping functions only. No HTTP
server, no NIM key, no compiled graph - so they stay fast and cannot fail for
reasons that have nothing to do with the mapping.

The workflow's own behaviour is covered by test_workflows.py and
test_orchestrator_scenarios.py, which know nothing about this layer.
"""
import pytest

from schemas.common import AgentStatus as WorkflowStatus
from schemas.outputs import GOAnnotation, LiteratureRecord, PathwayEntry, ProteinEntry

from schema import AgentRequest, AgentStatus
from workflow_adapter import to_result, to_state


# --------------------------------------------------------------------------
# to_state - orchestrator request -> workflow input
# --------------------------------------------------------------------------
def test_to_state_reads_species_and_trait_from_context():
    state = to_state(AgentRequest(
        instruction="Which genes cause fur growth?",
        context={"species": "mouse", "trait_name": "fur growth"},
    ))

    assert state.species_name == "mouse"
    assert state.trait_name == "fur growth"
    assert state.instruction == "Which genes cause fur growth?"


def test_to_state_lifts_gene_list_out_of_context():
    """The state's __post_init__ owns this; the adapter must not duplicate it."""
    state = to_state(AgentRequest(
        instruction="Which genes cause fur growth?",
        context={"species": "mouse", "gene_list": ["FGF5", "KRT71"]},
    ))

    assert state.gene_list == ["FGF5", "KRT71"]


def test_to_state_falls_back_when_no_trait_named():
    """A broad question seeds a species but no trait - that must still run."""
    state = to_state(AgentRequest(
        instruction="Explain the traits of the African elephant.",
        context={"species": "Loxodonta africana"},
    ))

    assert state.species_name == "Loxodonta africana"
    assert state.trait_name  # some usable string, never empty
    assert state.gene_list == []


def test_to_state_accepts_species_name_as_well_as_species():
    state = to_state(AgentRequest(instruction="x", context={"species_name": "mouse"}))
    assert state.species_name == "mouse"


def test_to_state_survives_empty_context():
    state = to_state(AgentRequest(instruction="anything", context={}))
    assert state.trait_name and state.species_name
    assert state.gene_list == []


# --------------------------------------------------------------------------
# to_result - workflow final state -> orchestrator result
# --------------------------------------------------------------------------
def _completed_state():
    return {
        "status": WorkflowStatus.COMPLETED,
        "go_annotations": [GOAnnotation(gene_symbol="FGF5", go_id="GO:1", go_name="hair")],
        "pathway_data": [PathwayEntry(pathway_id="mmu04010", pathway_name="MAPK")],
        "protein_data": [ProteinEntry(gene_symbol="FGF5", protein_name="FGF-5", function_summary="s")],
        "evidence": [LiteratureRecord(pmid="1", title="t", year=2008, short_summary="s")],
        "explanation": "FGF5 drives fur growth.",
    }


def test_to_result_completed_publishes_traits_key():
    """Protein and ImageGeneration branch on `traits`; losing it strands them."""
    result = to_result(_completed_state())

    assert result.status is AgentStatus.COMPLETED
    assert result.output["traits"] == ["FGF5"]
    assert result.output["explanation"] == "FGF5 drives fur growth."


def test_to_result_output_is_json_serialisable():
    """Output is merged into the shared context and serialised - no dataclasses."""
    import json

    result = to_result(_completed_state())
    json.dumps(result.output)  # raises if a dataclass leaked through

    assert result.output["go_annotations"][0]["gene_symbol"] == "FGF5"
    assert result.output["pathways"][0]["pathway_name"] == "MAPK"
    assert result.output["proteins"][0]["protein_name"] == "FGF-5"
    assert result.output["evidence"][0]["pmid"] == "1"


def test_to_result_needs_agent_carries_escalation_and_partial_findings():
    result = to_result({
        "status": WorkflowStatus.NEEDS_AGENT,
        "target_agent": "Genome Agent",
        "prompt_to_target_agent": "Resolve a candidate gene list.",
        "go_annotations": [],
    })

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Genome Agent"
    assert "gene list" in result.prompt_to_target_agent
    assert result.output is not None  # partial findings still travel


def test_to_result_failed_names_the_step_that_failed():
    result = to_result({
        "status": WorkflowStatus.FAILED,
        "gene_mapper_status": WorkflowStatus.FAILED,
        "gene_list": ["ZZZTOP"],
    })

    assert result.status is AgentStatus.FAILED
    assert "ZZZTOP" in result.output


def test_to_result_missing_status_is_a_failure_not_a_crash():
    result = to_result({})
    assert result.status is AgentStatus.FAILED


@pytest.mark.parametrize("workflow_status", list(WorkflowStatus))
def test_to_result_converts_every_status_by_value(workflow_status):
    """The two AgentStatus enums are different classes with the same values."""
    result = to_result({"status": workflow_status, "go_annotations": []})
    assert result.status.value == workflow_status.value
