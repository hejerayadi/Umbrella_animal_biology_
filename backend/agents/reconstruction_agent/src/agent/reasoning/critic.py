"""Reviews a reconstruction before it is reported, and returns a verdict.

The critic is what makes this an agent rather than a pipeline: it can reject
its own output and send the loop back for more evidence. It answers with one of
three verdicts, and the difference between the last two is the whole point:

- **ACCEPT**  - the evidence supports this; finalise it.
- **REVISE**  - something specific is missing that another tool call could
                supply; re-plan with the critique attached.
- **ABSTAIN** - the evidence is insufficient and no further call would help;
                stop and report the gap as unresolved.

ABSTAIN is a scientific result, not a failure. Saying "these gaps cannot be
reconstructed from the references available" is a real answer to the
orchestrator's question, and it is reported as COMPLETED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.prompts import critic_system, critic_user
from configuration.logging import get_logger
from contracts.output import GapReconstruction, ReconstructionStatus
from domain.models import GapContext
from infrastructure.llm.client import LLMClient, Message

_log = get_logger(__name__)

# Below this identity, bases read across a gap are not evidence about the
# target - matches the reference ranker's usability floor.
_WEAK_IDENTITY = 0.7
_THIN_EVIDENCE_REFERENCES = 2


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class Critique:
    """The verdict on one reconstruction, with the reasoning behind it."""

    gap_id: str
    verdict: Verdict
    problems: list[str] = field(default_factory=list)
    suggestion: str | None = None

    @property
    def acceptable(self) -> bool:
        return self.verdict is Verdict.ACCEPT

    def as_note(self) -> str:
        """One line for the planner's next round."""
        if self.verdict is Verdict.ACCEPT:
            return f"{self.gap_id}: accepted."
        problems = "; ".join(self.problems) or "unspecified problem"
        suggestion = f" Suggested: {self.suggestion}" if self.suggestion else ""
        return f"{self.gap_id} [{self.verdict.value}]: {problems}.{suggestion}"


class Critic:
    """Checks a reconstruction against its evidence.

    The deterministic checks always run; the LLM review is additive when one is
    configured. Doing it that way means the safety-relevant checks - thin
    evidence, weak identity - cannot be lost to a model that answers badly, and
    a model cannot manufacture an ACCEPT over them.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Tokens spent since the caller last collected them. The budget is
        # metered per iteration, so this is drained rather than accumulated.
        self._tokens = 0

    def take_tokens(self) -> int:
        """Tokens spent since the last call, resetting the counter."""
        spent, self._tokens = self._tokens, 0
        return spent

    async def review(
        self,
        context: GapContext,
        reconstruction: GapReconstruction,
        *,
        more_evidence_possible: bool = True,
    ) -> Critique:
        """Judge one reconstruction.

        `more_evidence_possible` is what separates REVISE from ABSTAIN: with
        budget spent or every tool already tried for this gap, re-planning
        cannot help however good the suggestion sounds.
        """
        problems = self._deterministic_problems(reconstruction)
        suggestion: str | None = None

        if self._llm.available and reconstruction.reconstructed_sequence:
            try:
                extra, suggestion = await self._llm_review(context, reconstruction)
                problems.extend(extra)
            except Exception as error:  # noqa: BLE001 - review must not abort a run
                _log.warning(
                    "critic_llm_failed", gap_id=context.identifier, error=str(error)
                )

        if not problems:
            return Critique(reconstruction.gap_id, Verdict.ACCEPT)

        verdict = Verdict.REVISE if more_evidence_possible else Verdict.ABSTAIN
        return Critique(
            gap_id=reconstruction.gap_id,
            verdict=verdict,
            problems=problems,
            suggestion=suggestion,
        )

    def _deterministic_problems(self, reconstruction: GapReconstruction) -> list[str]:
        """Checks that hold regardless of whether an LLM is available."""
        problems: list[str] = []

        if reconstruction.status is ReconstructionStatus.UNRESOLVED:
            return ["No reconstruction was produced from the available evidence."]

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

    async def _llm_review(
        self, context: GapContext, reconstruction: GapReconstruction
    ) -> tuple[list[str], str | None]:
        completion = await self._llm.complete_with_usage(
            [
                Message("system", critic_system()),
                Message(
                    "user",
                    critic_user(
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

        self._tokens += completion.total_tokens
        payload = _parse_json(completion.text)
        if payload is None:
            _log.warning("critic_llm_non_json")
            return [], None

        # A model saying "acceptable" does not clear the deterministic problems
        # already found; it can only add to them.
        if payload.get("acceptable", True):
            return [], None

        problems = [str(problem) for problem in payload.get("problems", [])]
        suggestion = payload.get("suggestion")
        return problems, str(suggestion) if suggestion else None


def _parse_json(reply: str) -> dict[str, Any] | None:
    """Read a JSON object out of a reply, tolerating a ```json fence."""
    text = reply.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        body = lines[1:]
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        text = "\n".join(body)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
