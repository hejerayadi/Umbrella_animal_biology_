from uuid import uuid4

import pytest

from app.contracts.agent_result import AgentResult, AgentStatus
from app.contracts.protein_response import (
    ProteinAnalysisResponse,
    ProteinSummary,
    StructureResponse,
)
from app.domain.enums import AnalysisStatus, ValidationStatus
from app.orchestrators.protein.result_policy import to_agent_result
from tests.factories import agent_task


def analysis_response(
    *,
    status: AnalysisStatus = AnalysisStatus.completed,
    validation_status: ValidationStatus = ValidationStatus.accept,
    with_protein: bool = True,
    with_structure: bool = True,
    warnings: list[str] | None = None,
) -> ProteinAnalysisResponse:
    return ProteinAnalysisResponse(
        analysis_id=uuid4(),
        task_id=uuid4(),
        status=status,
        validation_status=validation_status,
        protein=(
            ProteinSummary(
                uniprot_accession="P04637",
                gene_symbol="TP53",
                scientific_name="Homo sapiens",
                taxon_id=9606,
            )
            if with_protein
            else None
        ),
        selected_structure=(
            StructureResponse(
                source="RCSB_PDB",
                external_id="1TUP",
                structure_type="EXPERIMENTAL",
                chain_id="A",
                experimental_method="X-RAY DIFFRACTION",
                resolution_angstrom=2.2,
                sequence_coverage=0.72,
                file_format="MMCIF",
                file_url="https://files.rcsb.org/download/1TUP.cif",
                selection_score=0.732,
            )
            if with_structure
            else None
        ),
        warnings=warnings or [],
    )


def test_needs_agent_requires_a_target_and_prompt() -> None:
    with pytest.raises(ValueError, match="requires target_agent"):
        AgentResult(status=AgentStatus.NEEDS_AGENT)


def test_usable_partial_analysis_is_completed_at_the_routing_level() -> None:
    response = analysis_response(
        status=AnalysisStatus.partial,
        validation_status=ValidationStatus.revise,
        warnings=["ALPHAFOLD_FALLBACK: predicted structure selected"],
    )

    result = to_agent_result(response)

    assert result.status is AgentStatus.COMPLETED
    assert result.output["status"] == "PARTIAL"
    assert result.target_agent is None


def test_clarification_continues_with_the_same_agent_after_caller_input() -> None:
    response = analysis_response(
        status=AnalysisStatus.needs_clarification,
        validation_status=ValidationStatus.abstain,
        with_protein=False,
        with_structure=False,
    )

    result = to_agent_result(response)

    assert result.status is AgentStatus.CONTINUE
    assert result.retryable is False
    assert "resubmit" in (result.continuation_reason or "")


def test_temporary_identity_provider_failure_is_retryable_continue() -> None:
    response = analysis_response(
        status=AnalysisStatus.failed,
        validation_status=ValidationStatus.abstain,
        with_protein=False,
        with_structure=False,
        warnings=["UNIPROT_TIMEOUT: request timed out"],
    )

    result = to_agent_result(response)

    assert result.status is AgentStatus.CONTINUE
    assert result.retryable is True
    assert result.target_agent is None


def test_unresolved_identity_is_delegated_to_genome_agent() -> None:
    response = analysis_response(
        status=AnalysisStatus.failed,
        validation_status=ValidationStatus.abstain,
        with_protein=False,
        with_structure=False,
        warnings=["IDENTITY_UNRESOLVED: Protein identity is ambiguous"],
    )

    result = to_agent_result(response, agent_task())

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "genome_agent"
    assert "canonical protein identity" in (result.prompt_to_target_agent or "")
    assert "TP53" in (result.prompt_to_target_agent or "")


def test_temporary_structure_provider_failure_continues_before_delegating() -> None:
    response = analysis_response(
        status=AnalysisStatus.no_structure_found,
        validation_status=ValidationStatus.abstain,
        with_structure=False,
        warnings=["PDB_TIMEOUT: request timed out"],
    )

    result = to_agent_result(response)

    assert result.status is AgentStatus.CONTINUE
    assert result.retryable is True
    assert result.target_agent is None


def test_missing_structural_evidence_is_delegated_to_literature_agent() -> None:
    response = analysis_response(
        status=AnalysisStatus.no_structure_found,
        validation_status=ValidationStatus.abstain,
        with_structure=False,
    )

    result = to_agent_result(response)

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "literature_agent"
    assert "P04637" in (result.prompt_to_target_agent or "")


def test_unclassified_terminal_failure_is_failed() -> None:
    response = analysis_response(
        status=AnalysisStatus.failed,
        validation_status=ValidationStatus.abstain,
        with_protein=False,
        with_structure=False,
    )

    result = to_agent_result(response)

    assert result.status is AgentStatus.FAILED
    assert result.error
