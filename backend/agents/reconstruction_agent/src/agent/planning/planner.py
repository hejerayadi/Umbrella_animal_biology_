"""Decides which tools to run for the current state.

Two modes, and both are first-class. With an LLM the planner adapts to what
the previous iteration found; without one it runs the standard reconstruction
pipeline. The deterministic path is not a degraded stub - it is the correct
plan for the common case, and it is what the tests run against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent import prompts
from agent.state.state import ReconstructionState, has_usable_alignment
from configuration.logging import get_logger
from infrastructure.llm.client import LLMClient, Message

_log = get_logger(__name__)


#: Structured-output schema for the plan. Sent as `response_format` so the
#: model is constrained rather than merely asked; `_parse` still validates,
#: because not every deployment or API version honours it and the client
#: silently retries without it when rejected.
PLAN_SCHEMA: dict[str, Any] = {
    "name": "reconstruction_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["tool", "gap_id", "reason"],
                    "properties": {
                        "tool": {"type": "string"},
                        "gap_id": {"type": ["string", "null"]},
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    },
}


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One tool invocation the planner wants performed."""

    tool: str
    gap_id: str | None = None
    reason: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "gap_id": self.gap_id,
            "reason": self.reason,
            "arguments": self.arguments,
        }


class Planner:
    """Produces the next plan from the current state."""

    def __init__(self, llm: LLMClient, available_tools: list[str]) -> None:
        self._llm = llm
        self._available = set(available_tools)
        # Tokens spent since the caller last collected them. The budget is
        # metered per iteration, so this is drained rather than accumulated.
        self._tokens = 0

    def take_tokens(self) -> int:
        """Tokens spent since the last call, resetting the counter."""
        spent, self._tokens = self._tokens, 0
        return spent

    async def plan(
        self, state: ReconstructionState, catalogue: list[dict[str, Any]]
    ) -> list[PlanStep]:
        """The steps to run this iteration.

        Falls back to the deterministic plan whenever the LLM is unavailable or
        answers unusably - a malformed plan must not stall a run that the
        standard pipeline could have completed.
        """
        if not self._llm.available:
            return self.deterministic_plan(state)

        try:
            completion = await self._llm.complete_with_usage(
                [
                    Message("system", prompts.planner_system()),
                    Message(
                        "user",
                        prompts.planner_user(
                            instruction=state.get("instruction", ""),
                            organism=state.get("organism"),
                            gaps=self._describe_gaps(state),
                            tools=catalogue,
                            prior_critiques=state.get("critiques") or [],
                        ),
                    ),
                ],
                json_schema=PLAN_SCHEMA,
            )
            self._tokens += completion.total_tokens
            steps = self._parse(completion.text)
            if steps:
                return steps
            _log.warning(
                "planner_llm_empty", detail="No usable steps; using the deterministic plan."
            )
        except Exception as error:  # noqa: BLE001 - planning must never abort a run
            _log.warning("planner_llm_failed", error=str(error))

        return self.deterministic_plan(state)

    def deterministic_plan(self, state: ReconstructionState) -> list[PlanStep]:
        """The standard reconstruction pipeline, per unresolved gap.

        Order is forced by data dependency: references must exist before they
        can be ranked or aligned, and the alignment must exist before a
        candidate can be read out of it. Each gap advances one stage per
        iteration, so a gap that already has references goes straight to
        alignment on the next pass.

        This path must be able to reach every outcome the LLM path can,
        escalation included. An agent whose delegation only works when Azure is
        reachable would silently lose that capability on the runs where the
        model is down - exactly when a fallback matters.
        """
        steps: list[PlanStep] = []
        skipped = state.get("skipped") or {}
        references = state.get("references") or {}
        alignments = state.get("alignments") or {}
        resolved = state.get("reconstructions") or {}

        for context in state.get("gap_contexts") or []:
            gap_id = context.identifier
            if gap_id in skipped or _is_final(resolved.get(gap_id)):
                continue

            if not references.get(gap_id):
                steps.append(
                    PlanStep(
                        tool="blast_search",
                        gap_id=gap_id,
                        reason="No references yet; find sequences homologous to the flanks.",
                    )
                )
            elif not _has_usable_alignment(alignments.get(gap_id)):
                # An alignment that does not span the gap is not a satisfied
                # precondition - nothing can be read out of it. Treating its
                # mere presence as "done" is what stopped the retry from ever
                # being planned.
                steps.append(
                    PlanStep(
                        tool="mafft_align",
                        gap_id=gap_id,
                        reason="References found; align them to locate the gap's columns.",
                    )
                )

        if self._relatedness_unresolved(state):
            # Deliberately last: it costs nothing, and its answer only matters
            # once there are references to rank.
            steps.append(
                PlanStep(
                    tool="evolutionary_context",
                    gap_id=None,
                    reason=(
                        "References were found but none carry a relatedness score, so they "
                        "cannot be ranked by phylogenetic proximity."
                    ),
                )
            )

        return steps

    @staticmethod
    def _relatedness_unresolved(state: ReconstructionState) -> bool:
        """Whether references exist that nothing has scored for relatedness yet.

        The reference ranker weighs `relatedness` alongside alignment identity,
        and an unscored reference contributes zero on that axis - so leaving it
        unscored quietly biases ranking toward whatever aligned best, which is
        the failure mode phylogenetic proximity exists to correct.
        """
        if not state.get("organism"):
            return False

        references = [
            reference
            for gap_references in (state.get("references") or {}).values()
            for reference in gap_references
        ]
        if not references:
            return False

        # Already asked this run - the tool is deterministic, so asking twice
        # cannot produce a different answer.
        already_run = any(
            observation.tool == "evolutionary_context"
            for observation in state.get("observations") or []
        )
        if already_run:
            return False

        return any(reference.relatedness is None for reference in references)

    @staticmethod
    def _describe_gaps(state: ReconstructionState) -> list[dict[str, Any]]:
        """Gap summaries for the prompt - shape, not sequence.

        The residues themselves are deliberately withheld: the planner selects
        tools, and giving it sequence invites it to propose bases instead.
        """
        skipped = state.get("skipped") or {}
        resolved = state.get("reconstructions") or {}
        references = state.get("references") or {}

        return [
            {
                "gap_id": context.identifier,
                "length": context.gap.length,
                "left_flank_length": len(context.left_flank),
                "right_flank_length": len(context.right_flank),
                "has_both_flanks": context.has_both_flanks,
                "reference_count": len(references.get(context.identifier, [])),
                "status": (
                    "skipped"
                    if context.identifier in skipped
                    else "resolved"
                    if context.identifier in resolved
                    else "open"
                ),
            }
            for context in state.get("gap_contexts") or []
        ]

    def _parse(self, reply: str) -> list[PlanStep]:
        """Read plan steps out of the model's JSON reply.

        Unknown tool names are dropped rather than passed to the registry: a
        hallucinated tool is a planning error to correct, not a run to fail.
        """
        try:
            payload = json.loads(_strip_code_fence(reply))
        except json.JSONDecodeError:
            return []

        raw_steps = payload.get("steps") if isinstance(payload, dict) else None
        if not isinstance(raw_steps, list):
            return []

        steps: list[PlanStep] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            tool = str(raw.get("tool", ""))
            if tool not in self._available:
                _log.warning("planner_unknown_tool", tool=tool)
                continue
            steps.append(
                PlanStep(
                    tool=tool,
                    gap_id=raw.get("gap_id"),
                    reason=str(raw.get("reason", "")),
                    arguments=raw.get("arguments") or {},
                )
            )
        return steps


def _is_final(reconstruction: object) -> bool:
    """Whether a gap's outcome is settled enough to stop planning for it.

    Only a RECONSTRUCTED entry is settled here. An UNRESOLVED or
    LOW_CONFIDENCE one is this iteration's best answer, not a verdict, and more
    evidence can still overturn it - so the planner keeps working on it.
    """
    from contracts.output import ReconstructionStatus

    return (
        reconstruction is not None
        and getattr(reconstruction, "status", None) is ReconstructionStatus.RECONSTRUCTED
    )


def _has_usable_alignment(alignment: object) -> bool:
    return has_usable_alignment(alignment)


def _strip_code_fence(text: str) -> str:
    """Remove a ```json fence if the model wrapped its answer in one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body)
