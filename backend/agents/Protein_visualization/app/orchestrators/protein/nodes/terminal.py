"""Terminal nodes. Each one fixes the final status; none of them fetches data."""

from typing import Any

from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, ValidationStatus
from backend.agents.Protein_visualization.app.observability.logging import log_event
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import executed, logger
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import (
    COMPLETE,
    NEEDS_CLARIFICATION,
    RETURN_PARTIAL,
    SCIENTIFIC_ABSTAIN,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


def _final_status(state: ProteinWorkflowState, default: AnalysisStatus) -> AnalysisStatus:
    """`NO_STRUCTURE_FOUND` is a result in its own right and outranks the default."""
    current = state.get("current_status")
    return current if current is AnalysisStatus.no_structure_found else default


async def complete(state: ProteinWorkflowState) -> dict[str, Any]:
    log_event(logger, "workflow.completed", node=COMPLETE, status="completed")
    return executed(
        COMPLETE,
        current_status=AnalysisStatus.completed,
        validation_status=state.get("validation_status", ValidationStatus.accept),
    )


async def return_partial(state: ProteinWorkflowState) -> dict[str, Any]:
    log_event(
        logger,
        "workflow.partial",
        node=RETURN_PARTIAL,
        status="partial",
        warnings=len(state.get("warnings", [])),
    )
    validation_status = state.get("validation_status", ValidationStatus.revise)
    if validation_status is ValidationStatus.accept:
        validation_status = ValidationStatus.revise
    return executed(
        RETURN_PARTIAL,
        current_status=_final_status(state, AnalysisStatus.partial),
        validation_status=validation_status,
    )


async def return_needs_clarification(state: ProteinWorkflowState) -> dict[str, Any]:
    log_event(logger, "workflow.needs_clarification", node=NEEDS_CLARIFICATION, status="rejected")
    return executed(
        NEEDS_CLARIFICATION,
        current_status=AnalysisStatus.needs_clarification,
        validation_status=ValidationStatus.abstain,
        warnings=[f"NEEDS_CLARIFICATION: {state.get('clarification') or 'The task input is incomplete.'}"],
    )


async def scientific_abstain(state: ProteinWorkflowState) -> dict[str, Any]:
    """Stop rather than produce an unsupported structural claim."""
    log_event(logger, "workflow.abstained", node=SCIENTIFIC_ABSTAIN, status="abstained")
    identity_missing = state.get("resolved_protein") is None
    default = AnalysisStatus.failed if identity_missing else AnalysisStatus.partial
    return executed(
        SCIENTIFIC_ABSTAIN,
        current_status=_final_status(state, default),
        validation_status=ValidationStatus.abstain,
    )
