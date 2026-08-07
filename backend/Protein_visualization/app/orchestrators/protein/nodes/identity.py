"""Canonical UniProt identity. A failure here stops the workflow."""

from typing import Any

from app.capabilities.identity import IdentityCapability
from app.domain.enums import AnalysisStatus
from app.domain.exceptions import ProteinAgentError, ProteinNotFoundError
from app.observability.logging import log_stage
from app.orchestrators.protein.nodes._common import executed, failure_code, logger
from app.orchestrators.protein.nodes.names import RESOLVE_IDENTITY
from app.orchestrators.protein.state import ProteinWorkflowState


class IdentityNode:
    def __init__(self, capability: IdentityCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        try:
            with log_stage(logger, RESOLVE_IDENTITY, node=RESOLVE_IDENTITY) as outcome:
                protein, evidence = await self.capability.resolve(state["task"])
                outcome["accession"] = protein.uniprot_accession
        except ProteinNotFoundError as exc:
            # Ambiguous or mismatched identity: never guess, never select a structure.
            return executed(
                RESOLVE_IDENTITY,
                current_status=AnalysisStatus.failed,
                warnings=[f"IDENTITY_UNRESOLVED: {exc}"],
                errors=[f"{RESOLVE_IDENTITY}: {type(exc).__name__}"],
                resolved_protein=None,
            )
        except ProteinAgentError as exc:
            return executed(
                RESOLVE_IDENTITY,
                current_status=AnalysisStatus.failed,
                warnings=[f"{failure_code(exc, 'UNIPROT_UNAVAILABLE', 'UNIPROT_TIMEOUT')}: {exc}"],
                errors=[f"{RESOLVE_IDENTITY}: {type(exc).__name__}"],
                resolved_protein=None,
            )

        return executed(
            RESOLVE_IDENTITY,
            current_status=AnalysisStatus.running,
            resolved_protein=protein,
            evidence=[evidence],
        )
