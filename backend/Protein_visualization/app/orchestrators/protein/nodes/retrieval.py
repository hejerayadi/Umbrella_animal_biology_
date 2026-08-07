"""Qdrant knowledge retrieval. Optional: failure only limits the explanation."""

from datetime import UTC, datetime
from typing import Any

from app.capabilities.retrieval import RetrievalCapability
from app.domain.models import EvidenceRef
from app.observability.logging import log_stage
from app.orchestrators.protein.nodes._common import degraded, executed, failure_code, logger
from app.orchestrators.protein.nodes.names import RETRIEVE_KNOWLEDGE
from app.orchestrators.protein.state import ProteinWorkflowState


class RetrievalNode:
    def __init__(self, capability: RetrievalCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        protein = state["resolved_protein"]
        assert protein is not None
        try:
            with log_stage(logger, RETRIEVE_KNOWLEDGE, node=RETRIEVE_KNOWLEDGE) as outcome:
                hits = await self.capability.retrieve(protein)
                outcome["hits"] = len(hits)
        except Exception as exc:  # the vector store is infrastructure, not a domain source
            return degraded(
                RETRIEVE_KNOWLEDGE,
                failure_code(exc, "RETRIEVAL_UNAVAILABLE", "RETRIEVAL_TIMEOUT"),
                exc,
                retrieved_documents=[],
            )

        retrieved_at = datetime.now(UTC).isoformat()
        evidence = [
            EvidenceRef(
                provider=str(hit.metadata.get("source", "protein_knowledge")),
                external_id=hit.id,
                retrieved_at=retrieved_at,
                source_url=hit.metadata.get("source_url"),
                checksum=hit.metadata.get("content_sha256"),
            )
            for hit in hits
        ]
        return executed(RETRIEVE_KNOWLEDGE, retrieved_documents=hits, evidence=evidence)
