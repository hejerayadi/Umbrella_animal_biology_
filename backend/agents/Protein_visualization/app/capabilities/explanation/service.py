import logging
from typing import Protocol

from backend.agents.Protein_visualization.app.domain.models import EvidencePack, Explanation
from backend.agents.Protein_visualization.app.llm.schemas import CriticOutput, ExplanationOutput

logger = logging.getLogger("app.explanation")


class LanguageModel(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def explain(self, context: dict[str, object]) -> ExplanationOutput: ...

    async def critique(self, context: dict[str, object]) -> CriticOutput: ...


def evidence_context(evidence: EvidencePack) -> dict[str, object]:
    """The only payload a language model is allowed to see."""
    return {
        "facts": list(evidence.facts),
        "limitations": list(evidence.limitations),
        "sources": [
            {"provider": item.provider, "external_id": item.external_id, "url": item.source_url}
            for item in evidence.evidence
        ],
    }


class ExplanationCapability:
    def __init__(self, llm: LanguageModel | None = None) -> None:
        self.llm = llm

    async def explain(self, evidence: EvidencePack) -> Explanation:
        if self.llm is not None and self.llm.enabled:
            try:
                output = await self.llm.explain(evidence_context(evidence))
            except Exception as exc:
                # A model failure must never cost the caller its structured evidence.
                logger.warning("explanation_llm_failed", exc_info=exc)
            else:
                limitations = tuple(dict.fromkeys((*evidence.limitations, *output.limitations)))
                return Explanation(summary=output.summary, limitations=limitations, generated=True)

        summary = " ".join(evidence.facts) if evidence.facts else "Insufficient evidence for an explanation."
        return Explanation(summary=summary, limitations=evidence.limitations, generated=False)
