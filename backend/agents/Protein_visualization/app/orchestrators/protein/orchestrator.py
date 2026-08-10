"""Protein sub-orchestrator: owns the workflow and the contract with the Grand Orchestrator."""

import logging
from typing import Any, cast
from uuid import UUID, uuid4

from langgraph.graph.state import CompiledStateGraph

from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.contracts.agent_result import AgentResult
from backend.agents.Protein_visualization.app.contracts.protein_request import ProteinAnalysisRequest
from backend.agents.Protein_visualization.app.contracts.protein_response import (
    ExplanationResponse,
    LlmUsageResponse,
    ProteinAnalysisResponse,
    ProteinSummary,
    StructureResponse,
)
from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, ValidationStatus
from backend.agents.Protein_visualization.app.domain.models import LlmUsage, StructureCandidate
from backend.agents.Protein_visualization.app.observability.context import log_context
from backend.agents.Protein_visualization.app.observability.logging import log_event
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import WORKFLOW_SEQUENCE
from backend.agents.Protein_visualization.app.orchestrators.protein.result_policy import to_agent_result
from backend.agents.Protein_visualization.app.orchestrators.protein.state import (
    ProteinWorkflowState,
    initial_state,
)

logger = logging.getLogger("app.orchestrator")


def _llm_usage(usage: LlmUsage) -> LlmUsageResponse:
    settings = get_settings()
    cost = None
    if (
        settings.azure_input_price_per_1k_usd is not None
        and settings.azure_output_price_per_1k_usd is not None
        and usage.input_tokens is not None
        and usage.output_tokens is not None
    ):
        cost = round(
            usage.input_tokens / 1000 * settings.azure_input_price_per_1k_usd
            + usage.output_tokens / 1000 * settings.azure_output_price_per_1k_usd,
            6,
        )
    return LlmUsageResponse(
        node=usage.node,
        model=usage.model,
        duration_ms=usage.duration_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        estimated_cost_usd=cost,
    )


def _structure(candidate: StructureCandidate) -> StructureResponse:
    return StructureResponse(
        source=candidate.source.value,
        external_id=candidate.external_id,
        structure_type=candidate.structure_type,
        chain_id=candidate.chain_id,
        experimental_method=candidate.experimental_method,
        resolution_angstrom=candidate.resolution_angstrom,
        sequence_coverage=candidate.sequence_coverage,
        mean_plddt=candidate.mean_plddt,
        file_format=candidate.file_format,
        file_url=candidate.file_url,
        selection_score=candidate.selection_score,
        warnings=candidate.warnings,
    )


class ProteinOrchestrator:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    async def analyze(self, task: ProteinAnalysisRequest) -> ProteinAnalysisResponse:
        analysis_id = uuid4()
        request = task.to_domain()
        with log_context(analysis_id=str(analysis_id), task_id=str(task.task_id)):
            log_event(
                logger,
                "workflow.started",
                gene=request.resolved_gene_id,
                accession=request.uniprot_accession,
                taxon_id=request.species.taxon_id,
                preferred_source=request.preferred_source.value,
            )
            result = await self.graph.ainvoke(
                initial_state(request, analysis_id),
                config={"configurable": {"thread_id": str(analysis_id)}},
            )
            final = cast(ProteinWorkflowState, result)
            log_event(
                logger,
                "workflow.finished",
                status=final["current_status"].value,
                validation_status=final["validation_status"].value,
                nodes=len(final.get("executed_nodes", set())),
                warnings=len(final.get("warnings", [])),
            )
        return self.to_response(task.task_id, analysis_id, final)

    async def execute(self, task: ProteinAnalysisRequest) -> AgentResult:
        """Run the scientific workflow and return its inter-agent routing result."""
        response = await self.analyze(task)
        return to_agent_result(response, task)

    @staticmethod
    def to_response(task_id: UUID, analysis_id: UUID, state: ProteinWorkflowState) -> ProteinAnalysisResponse:
        protein = state.get("resolved_protein")
        structure = state.get("selected_structure")
        explanation = state.get("explanation")

        annotations: list[dict[str, Any]] = [
            {
                "source": item.source,
                "accession": item.accession,
                "kind": item.kind,
                "label": item.label,
                "start": item.start,
                "end": item.end,
            }
            for item in state.get("annotations", [])
        ]
        residue_mappings: list[dict[str, Any]] = [
            {
                "uniprot_accession": item.uniprot_accession,
                "uniprot_position": item.uniprot_position,
                "pdb_id": item.pdb_id,
                "chain_id": item.chain_id,
                "pdb_residue_number": item.pdb_residue_number,
                "is_observed": item.is_observed,
                "mapping_source": item.mapping_source,
            }
            for item in state.get("residue_mappings", [])
        ]
        evidence: list[dict[str, Any]] = [
            {
                "provider": item.provider,
                "external_id": item.external_id,
                "retrieved_at": item.retrieved_at,
                "source_url": item.source_url,
                "checksum": item.checksum,
            }
            for item in state.get("evidence", [])
        ]

        return ProteinAnalysisResponse(
            analysis_id=analysis_id,
            task_id=task_id,
            status=state.get("current_status", AnalysisStatus.failed),
            validation_status=state.get("validation_status", ValidationStatus.abstain),
            protein=(
                ProteinSummary(
                    uniprot_accession=protein.uniprot_accession,
                    gene_symbol=protein.gene_symbol,
                    scientific_name=protein.scientific_name,
                    taxon_id=protein.taxon_id,
                )
                if protein
                else None
            ),
            selected_structure=_structure(structure) if structure else None,
            alternative_structures=[
                _structure(candidate) for candidate in state.get("alternative_structures", [])
            ],
            annotations=annotations,
            residue_mappings=residue_mappings,
            molstar_config=state.get("molstar_config", {}),
            explanation=(
                ExplanationResponse(
                    summary=explanation.summary,
                    limitations=list(explanation.limitations),
                )
                if explanation
                else None
            ),
            warnings=list(state.get("warnings", [])),
            evidence=evidence,
            executed_nodes=[node for node in WORKFLOW_SEQUENCE if node in state.get("executed_nodes", set())],
            errors=list(state.get("errors", [])),
            retry_counts=dict(state.get("retry_counts", {})),
            llm_usage=[_llm_usage(usage) for usage in state.get("llm_usage", [])],
        )
