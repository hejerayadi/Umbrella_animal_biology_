"""Scientific critic node: the last gate before the response leaves the agent."""

from typing import Any

from app.capabilities.critic import CriticCapability
from app.capabilities.explanation.service import LanguageModel
from app.domain.enums import ValidationStatus
from app.domain.models import EvidencePack
from app.observability.logging import log_stage
from app.orchestrators.protein.nodes._common import executed, logger
from app.orchestrators.protein.nodes.names import RUN_CRITIC
from app.orchestrators.protein.state import ProteinWorkflowState


class CriticNode:
    def __init__(self, capability: CriticCapability, llm: LanguageModel | None = None) -> None:
        self.capability = capability
        self.llm = llm

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        with log_stage(logger, RUN_CRITIC, node=RUN_CRITIC) as outcome:
            report = self.capability.review(
                state["task"],
                state["resolved_protein"],
                state["selected_structure"],
                state["residue_mappings"],
            )
            audited = await self.capability.audit(report, state["evidence_pack"] or EvidencePack(), self.llm)
            outcome["deterministic_verdict"] = report.verdict
            outcome["verdict"] = audited.verdict

        verdict = ValidationStatus(audited.verdict)
        warnings = (
            [f"CRITIC_{verdict.value}: {'; '.join(audited.reasons)}"]
            if verdict is not ValidationStatus.accept
            else []
        )
        return executed(RUN_CRITIC, validation_status=verdict, warnings=warnings)
