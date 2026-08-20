"""Scientific critic node: the last gate before the response leaves the agent."""

from typing import Any

from backend.agents.Protein_visualization.app.capabilities.critic import CriticCapability
from backend.agents.Protein_visualization.app.capabilities.explanation.service import LanguageModel
from backend.agents.Protein_visualization.app.domain.enums import ValidationStatus
from backend.agents.Protein_visualization.app.domain.models import EvidencePack
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import executed, logger
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import RUN_CRITIC
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


class CriticNode:
    def __init__(self, capability: CriticCapability, llm: LanguageModel | None = None) -> None:
        self.capability = capability
        self.llm = llm

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        with log_stage(
            logger,
            f"protein.node.{RUN_CRITIC}",
            node=RUN_CRITIC,
            capability="scientific_critic",
        ) as outcome:
            report = self.capability.review(
                state["task"],
                state["resolved_protein"],
                state["selected_structure"],
                state["residue_mappings"],
                state.get("warnings", []),
            )
            audited, usage = await self.capability.audit(
                report, state["evidence_pack"] or EvidencePack(), self.llm, RUN_CRITIC
            )
            outcome["deterministic_verdict"] = report.verdict
            outcome["verdict"] = audited.verdict
            if usage:
                outcome["tokens"] = usage.total_tokens
                outcome["model"] = usage.model

        verdict = ValidationStatus(audited.verdict)
        warnings = (
            [f"CRITIC_{verdict.value}: {'; '.join(audited.reasons)}"]
            if verdict is not ValidationStatus.accept
            else []
        )
        return executed(
            RUN_CRITIC,
            validation_status=verdict,
            warnings=warnings,
            llm_usage=[usage] if usage else [],
        )
