"""InterPro domains and functional regions. Optional: failure degrades, never stops."""

from datetime import UTC, datetime
from typing import Any

from backend.agents.Protein_visualization.app.capabilities.annotations import AnnotationCapability
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinAgentError
from backend.agents.Protein_visualization.app.domain.models import EvidenceRef
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import (
    degraded,
    executed,
    failure_code,
    logger,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import FETCH_ANNOTATIONS
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


class AnnotationNode:
    def __init__(self, capability: AnnotationCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        protein = state["resolved_protein"]
        assert protein is not None
        try:
            with log_stage(
                logger,
                f"protein.node.{FETCH_ANNOTATIONS}",
                node=FETCH_ANNOTATIONS,
                capability="annotation",
            ) as outcome:
                annotations = await self.capability.annotate(protein)
                outcome["annotations"] = len(annotations)
        except ProteinAgentError as exc:
            return degraded(
                FETCH_ANNOTATIONS,
                failure_code(exc, "ANNOTATIONS_UNAVAILABLE", "ANNOTATIONS_TIMEOUT"),
                exc,
                annotations=[],
            )

        retrieved_at = datetime.now(UTC).isoformat()
        evidence = [
            EvidenceRef(
                provider="InterPro",
                external_id=accession,
                retrieved_at=retrieved_at,
                source_url=f"https://www.ebi.ac.uk/interpro/entry/InterPro/{accession}/",
            )
            for accession in dict.fromkeys(item.accession for item in annotations)
        ]
        return executed(FETCH_ANNOTATIONS, annotations=annotations, evidence=evidence)
