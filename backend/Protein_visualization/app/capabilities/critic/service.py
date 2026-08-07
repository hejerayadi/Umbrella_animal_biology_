"""Scientific critic.

The deterministic verdict is computed from the workflow facts alone. A language
model may then confirm it or make it stricter — never more permissive.
"""

import logging

from app.capabilities.explanation.service import LanguageModel, evidence_context
from app.domain.enums import ValidationStatus
from app.domain.models import (
    CriticReport,
    EvidencePack,
    ProteinStructureRequest,
    ResidueMapping,
    ResolvedProtein,
    StructureCandidate,
)

logger = logging.getLogger("app.critic")

SEVERITY = {
    ValidationStatus.accept: 0,
    ValidationStatus.revise: 1,
    ValidationStatus.abstain: 2,
}


def _strictest(*statuses: ValidationStatus) -> ValidationStatus:
    return max(statuses, key=lambda status: SEVERITY[status])


class CriticCapability:
    def review(
        self,
        request: ProteinStructureRequest,
        protein: ResolvedProtein | None,
        structure: StructureCandidate | None,
        mappings: list[ResidueMapping],
    ) -> CriticReport:
        if protein is None:
            return CriticReport(
                verdict=ValidationStatus.abstain.value,
                reasons=("Protein identity was not confirmed; no structural claim can be made.",),
            )
        if structure is None:
            return CriticReport(
                verdict=ValidationStatus.abstain.value,
                reasons=("No structure satisfied the selection rules.",),
            )

        verdict = ValidationStatus.accept
        reasons: list[str] = []

        if (request.residue_position is not None or request.mutation) and not any(
            mapping.is_observed for mapping in mappings
        ):
            verdict = _strictest(verdict, ValidationStatus.revise)
            reasons.append(
                "A residue or mutation was requested but no observed SIFTS mapping was obtained, "
                "so the position is not highlighted."
            )
        if structure.structure_type == "PREDICTED":
            verdict = _strictest(verdict, ValidationStatus.revise)
            reasons.append(
                "The selected model is an AlphaFold prediction and must be presented as predicted."
            )
        if not reasons:
            reasons.append("Identity, selected structure and evidence are mutually consistent.")
        return CriticReport(verdict=verdict.value, reasons=tuple(reasons))

    async def audit(
        self,
        deterministic: CriticReport,
        evidence: EvidencePack,
        llm: LanguageModel | None = None,
    ) -> CriticReport:
        """Let the model tighten the verdict; a proposed upgrade is discarded."""
        if llm is None or not llm.enabled:
            return deterministic

        try:
            output = await llm.critique(evidence_context(evidence))
        except Exception as exc:
            logger.warning("critic_llm_failed", exc_info=exc)
            return deterministic

        proposed = ValidationStatus(output.verdict)
        current = ValidationStatus(deterministic.verdict)
        if SEVERITY[proposed] <= SEVERITY[current]:
            return deterministic
        return CriticReport(
            verdict=proposed.value,
            reasons=tuple(dict.fromkeys((*deterministic.reasons, *output.reasons))),
        )
