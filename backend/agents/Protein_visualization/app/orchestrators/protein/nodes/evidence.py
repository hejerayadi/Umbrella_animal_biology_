"""Assembles the EvidencePack that bounds everything the LLM may say."""

from typing import Any

from backend.agents.Protein_visualization.app.capabilities.evidence import EvidenceCapability
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import executed, logger
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import BUILD_EVIDENCE
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


class EvidenceNode:
    def __init__(self, capability: EvidenceCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        with log_stage(logger, BUILD_EVIDENCE, node=BUILD_EVIDENCE) as outcome:
            pack = self.capability.build(
                request=state["task"],
                protein=state["resolved_protein"],
                structure=state["selected_structure"],
                annotations=state["annotations"],
                mappings=state["residue_mappings"],
                knowledge=state["retrieved_documents"],
                evidence=state["evidence"],
                warnings=state["warnings"],
            )
            outcome["facts"] = len(pack.facts)
            outcome["limitations"] = len(pack.limitations)
        return executed(BUILD_EVIDENCE, evidence_pack=pack)
