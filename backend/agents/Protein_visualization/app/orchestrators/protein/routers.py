"""Conditional edges.

This is where the workflow stops being a fixed chain: each function reads the
state and names the next node. All of them are pure and synchronously testable.
"""

from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, PreferredSource, ValidationStatus
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.mapping import mapping_required
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import (
    BUILD_EVIDENCE,
    COMPLETE,
    FETCH_ANNOTATIONS,
    GENERATE_EXPLANATION,
    MAP_RESIDUES,
    NEEDS_CLARIFICATION,
    RESOLVE_IDENTITY,
    RETRIEVE_KNOWLEDGE,
    RETURN_PARTIAL,
    RUN_CRITIC,
    SCIENTIFIC_ABSTAIN,
    SEARCH_ALPHAFOLD,
    SEARCH_EXPERIMENTAL,
    SELECT_STRUCTURE,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState

PARALLEL_RETRIEVAL = [SEARCH_EXPERIMENTAL, FETCH_ANNOTATIONS, RETRIEVE_KNOWLEDGE]


def route_after_validation(state: ProteinWorkflowState) -> str:
    if state["current_status"] is AnalysisStatus.needs_clarification:
        return NEEDS_CLARIFICATION
    return RESOLVE_IDENTITY


def route_after_identity(state: ProteinWorkflowState) -> str | list[str]:
    """Identity is the anchor: without it nothing downstream may run.

    Once confirmed, structures, annotations and knowledge are fetched in parallel.
    """
    if state.get("resolved_protein") is None:
        return SCIENTIFIC_ABSTAIN
    return list(PARALLEL_RETRIEVAL)


def route_after_pdb_evaluation(state: ProteinWorkflowState) -> str:
    """AlphaFold is a fallback, never a parallel default."""
    task = state["task"]
    if task.preferred_source is PreferredSource.pdb:
        return SELECT_STRUCTURE
    if task.preferred_source is PreferredSource.alphafold:
        return SEARCH_ALPHAFOLD
    return SELECT_STRUCTURE if state["valid_pdb_candidates"] else SEARCH_ALPHAFOLD


def route_mapping_required(state: ProteinWorkflowState) -> str:
    if state.get("selected_structure") is None:
        return BUILD_EVIDENCE
    return MAP_RESIDUES if mapping_required(state["task"]) else BUILD_EVIDENCE


def route_explanation(state: ProteinWorkflowState) -> str:
    return GENERATE_EXPLANATION if state["task"].include_explanation else RUN_CRITIC


def route_after_critic(state: ProteinWorkflowState) -> str:
    verdict = state.get("validation_status", ValidationStatus.abstain)
    if verdict is ValidationStatus.abstain:
        return SCIENTIFIC_ABSTAIN
    if verdict is ValidationStatus.revise or state.get("warnings"):
        return RETURN_PARTIAL
    return COMPLETE
