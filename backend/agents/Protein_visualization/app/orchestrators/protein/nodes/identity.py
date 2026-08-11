"""Canonical UniProt identity. A failure here stops the workflow."""

from typing import Any

from backend.agents.Protein_visualization.app.capabilities.identity import IdentityCapability
from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinAgentError, ProteinNotFoundError
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import (
    executed,
    failed,
    failure_code,
    logger,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import RESOLVE_IDENTITY
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


class IdentityNode:
    def __init__(self, capability: IdentityCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        try:
            with log_stage(
                logger,
                f"protein.node.{RESOLVE_IDENTITY}",
                node=RESOLVE_IDENTITY,
                capability="identity",
            ) as outcome:
                protein, evidence = await self.capability.resolve(state["task"])
                outcome["accession"] = protein.uniprot_accession
        except ProteinNotFoundError as exc:
            # Ambiguous or mismatched identity: never guess, never select a structure.
            return failed(
                RESOLVE_IDENTITY,
                "IDENTITY_UNRESOLVED",
                exc,
                current_status=AnalysisStatus.failed,
                resolved_protein=None,
            )
        except ProteinAgentError as exc:
            code = failure_code(exc, "UNIPROT_UNAVAILABLE", "UNIPROT_TIMEOUT")
            return failed(
                RESOLVE_IDENTITY,
                code,
                exc,
                current_status=AnalysisStatus.failed,
                resolved_protein=None,
            )

        return executed(
            RESOLVE_IDENTITY,
            current_status=AnalysisStatus.running,
            resolved_protein=protein,
            evidence=[evidence],
        )
