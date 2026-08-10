"""Evidence-bound explanation. The LLM only ever sees the EvidencePack."""

from typing import Any

from backend.agents.Protein_visualization.app.capabilities.explanation import ExplanationCapability
from backend.agents.Protein_visualization.app.domain.models import EvidencePack
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import executed, logger
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import GENERATE_EXPLANATION
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


class ExplanationNode:
    def __init__(self, capability: ExplanationCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        pack = state["evidence_pack"] or EvidencePack()
        with log_stage(logger, GENERATE_EXPLANATION, node=GENERATE_EXPLANATION) as outcome:
            explanation, usage = await self.capability.explain(pack, GENERATE_EXPLANATION)
            outcome["generated"] = self.capability.llm is not None and self.capability.llm.enabled
            outcome["characters"] = len(explanation.summary)
            if usage:
                outcome["tokens"] = usage.total_tokens

        warnings = []
        if self.capability.llm is not None and self.capability.llm.enabled and not explanation.generated:
            warnings.append("LLM_UNAVAILABLE: the evidence summary was assembled without the model.")
        return executed(
            GENERATE_EXPLANATION,
            explanation=explanation,
            warnings=warnings,
            llm_usage=[usage] if usage else [],
        )
