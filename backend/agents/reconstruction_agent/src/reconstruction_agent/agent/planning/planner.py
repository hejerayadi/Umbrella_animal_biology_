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

from ...configuration.logging import get_logger
from ...infrastructure.llm.client import LLMClient, Message
from ..prompts import planner as prompts
from ..state.state import ReconstructionState

_log = get_logger(__name__)


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
            reply = await self._llm.complete(
                [
                    Message("system", prompts.SYSTEM),
                    Message(
                        "user",
                        prompts.user_prompt(
                            instruction=state.get("instruction", ""),
                            organism=state.get("organism"),
                            gaps=self._describe_gaps(state),
                            tools=catalogue,
                            prior_critiques=state.get("critiques") or [],
                        ),
                    ),
                ]
            )
            steps = self._parse(reply)
            if steps:
                return steps
            _log.warning("Planner LLM returned no usable steps; using the deterministic plan.")
        except Exception as error:  # noqa: BLE001 - planning must never abort a run
            _log.warning("Planner LLM failed (%s); using the deterministic plan.", error)

        return self.deterministic_plan(state)

    def deterministic_plan(self, state: ReconstructionState) -> list[PlanStep]:
        """The standard reconstruction pipeline, per unresolved gap.

        Order is forced by data dependency: references must exist before they
        can be ranked or aligned, and the alignment must exist before a
        candidate can be read out of it. Each gap advances one stage per
        iteration, so a gap that already has references goes straight to
        alignment on the next pass.
        """
        steps: list[PlanStep] = []
        skipped = state.get("skipped") or {}
        references = state.get("references") or {}
        alignments = state.get("alignments") or {}
        resolved = state.get("reconstructions") or {}

        for context in state.get("gap_contexts") or []:
            gap_id = context.identifier
            if gap_id in skipped or gap_id in resolved:
                continue

            if not references.get(gap_id):
                steps.append(
                    PlanStep(
                        tool="blast_search",
                        gap_id=gap_id,
                        reason="No references yet; find sequences homologous to the flanks.",
                    )
                )
            elif gap_id not in alignments:
                steps.append(
                    PlanStep(
                        tool="mafft_align",
                        gap_id=gap_id,
                        reason="References found; align them to locate the gap's columns.",
                    )
                )

        return steps

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
                _log.warning("Planner proposed unknown tool %r; dropping that step.", tool)
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
