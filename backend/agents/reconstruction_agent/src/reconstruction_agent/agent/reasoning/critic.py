"""Reviews a reconstruction before it is reported.

The critic is what makes this an agent rather than a pipeline: it can reject
its own output and send the loop back for more evidence. Its findings feed the
next planning round through `state["critiques"]`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ...configuration.logging import get_logger
from ...contracts.output import GapReconstruction, ReconstructionStatus
from ...domain.models import GapContext
from ...infrastructure.llm.client import LLMClient, Message
from ..prompts import critic as prompts

_log = get_logger(__name__)

# Below this identity, bases read across a gap are not evidence about the
# target - matches the reference ranker's usability floor.
_WEAK_IDENTITY = 0.7
_THIN_EVIDENCE_REFERENCES = 2


@dataclass(frozen=True, slots=True)
class Critique:
    """The verdict on one reconstruction."""

    gap_id: str
    acceptable: bool
    problems: list[str] = field(default_factory=list)
    suggestion: str | None = None

    def as_note(self) -> str:
        """One line for the planner's next round."""
        if self.acceptable:
            return f"{self.gap_id}: accepted."
        problems = "; ".join(self.problems) or "unspecified problem"
        suggestion = f" Suggested: {self.suggestion}" if self.suggestion else ""
        return f"{self.gap_id}: {problems}.{suggestion}"


class Critic:
    """Checks a reconstruction against its evidence.

    The deterministic checks always run; the LLM review is additive when one is
    configured. Doing it that way means the safety-relevant checks - thin
    evidence, weak identity - cannot be lost to a model that answers badly.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def review(
        self, context: GapContext, reconstruction: GapReconstruction
    ) -> Critique:
        problems = self._deterministic_problems(reconstruction)

        if self._llm.available and reconstruction.reconstructed_sequence:
            try:
                problems.extend(await self._llm_problems(context, reconstruction))
            except Exception as error:  # noqa: BLE001 - review must not abort a run
                _log.warning("Critic LLM failed for %s (%s); using local checks only.",
                             context.identifier, error)

        return Critique(
            gap_id=reconstruction.gap_id,
            acceptable=not problems,
            problems=problems,
        )

    def _deterministic_problems(self, reconstruction: GapReconstruction) -> list[str]:
        """Checks that hold regardless of whether an LLM is available."""
        problems: list[str] = []

        if reconstruction.status is ReconstructionStatus.UNRESOLVED:
            problems.append("No reconstruction was produced from the available evidence.")
            return problems

        if len(reconstruction.evidence) < _THIN_EVIDENCE_REFERENCES:
            problems.append(
                f"Only {len(reconstruction.evidence)} reference(s) support this "
                "reconstruction; a consensus needs more independent evidence."
            )

        identities = [
            item.identity for item in reconstruction.evidence if item.identity is not None
        ]
        if identities and max(identities) < _WEAK_IDENTITY:
            problems.append(
                f"Best flanking identity is only {max(identities):.0%}, too divergent to "
                "read bases across the gap from."
            )

        if reconstruction.status is ReconstructionStatus.LOW_CONFIDENCE:
            problems.append(
                f"Confidence {reconstruction.confidence:.0%} is below the reporting "
                "threshold."
            )

        return problems

    async def _llm_problems(
        self, context: GapContext, reconstruction: GapReconstruction
    ) -> list[str]:
        reply = await self._llm.complete(
            [
                Message("system", prompts.SYSTEM),
                Message(
                    "user",
                    prompts.user_prompt(
                        gap={
                            "gap_id": context.identifier,
                            "length": context.gap.length,
                            "has_both_flanks": context.has_both_flanks,
                        },
                        candidate={
                            "sequence": reconstruction.reconstructed_sequence,
                            "length": len(reconstruction.reconstructed_sequence or ""),
                            "confidence": reconstruction.confidence,
                        },
                        evidence=[item.model_dump() for item in reconstruction.evidence],
                    ),
                ),
            ]
        )

        try:
            payload = json.loads(reply.strip().removeprefix("```json").removeprefix("```").
                                 removesuffix("```").strip())
        except json.JSONDecodeError:
            _log.warning("Critic LLM returned non-JSON; ignoring its review.")
            return []

        if not isinstance(payload, dict) or payload.get("acceptable", True):
            return []
        return [str(problem) for problem in payload.get("problems", [])]
