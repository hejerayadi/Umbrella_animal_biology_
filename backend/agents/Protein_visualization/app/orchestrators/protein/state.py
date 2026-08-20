"""LangGraph state for the protein workflow.

Parallel branches (structures, annotations, retrieval) run concurrently, so every
key several branches may write needs a reducer. Branch-owned keys are written by
exactly one node and need none.
"""

import operator
from typing import Annotated, Any, TypedDict
from uuid import UUID

from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, ValidationStatus
from backend.agents.Protein_visualization.app.domain.models import (
    Annotation,
    EvidencePack,
    EvidenceRef,
    Explanation,
    KnowledgeHit,
    LlmUsage,
    ProteinStructureRequest,
    ResidueMapping,
    ResolvedProtein,
    StructureCandidate,
)


def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged


class ProteinWorkflowState(TypedDict, total=False):
    # Input, fixed for the whole run
    task: ProteinStructureRequest
    analysis_id: UUID

    # Progress
    current_status: AnalysisStatus
    validation_status: ValidationStatus

    # Branch-owned results
    resolved_protein: ResolvedProtein | None
    pdb_candidates: list[StructureCandidate]
    valid_pdb_candidates: list[StructureCandidate]
    alphafold_candidates: list[StructureCandidate]
    selected_structure: StructureCandidate | None
    alternative_structures: list[StructureCandidate]
    annotations: list[Annotation]
    residue_mappings: list[ResidueMapping]
    retrieved_documents: list[KnowledgeHit]
    evidence_pack: EvidencePack | None
    molstar_config: dict[str, Any]
    explanation: Explanation | None
    clarification: str | None

    # Shared across parallel branches
    evidence: Annotated[list[EvidenceRef], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    executed_nodes: Annotated[set[str], operator.or_]
    retry_counts: Annotated[dict[str, int], merge_counts]
    # Explanation and the critic's audit each make at most one Azure call; both
    # write here, so this accumulates rather than overwrites like the rest of
    # the branch-owned keys above.
    llm_usage: Annotated[list[LlmUsage], operator.add]


def initial_state(task: ProteinStructureRequest, analysis_id: UUID) -> ProteinWorkflowState:
    return ProteinWorkflowState(
        task=task,
        analysis_id=analysis_id,
        current_status=AnalysisStatus.received,
        validation_status=ValidationStatus.abstain,
        resolved_protein=None,
        pdb_candidates=[],
        valid_pdb_candidates=[],
        alphafold_candidates=[],
        selected_structure=None,
        alternative_structures=[],
        annotations=[],
        residue_mappings=[],
        retrieved_documents=[],
        evidence_pack=None,
        molstar_config={},
        explanation=None,
        clarification=None,
        evidence=[],
        warnings=[],
        errors=[],
        executed_nodes=set(),
        retry_counts={},
        llm_usage=[],
    )
