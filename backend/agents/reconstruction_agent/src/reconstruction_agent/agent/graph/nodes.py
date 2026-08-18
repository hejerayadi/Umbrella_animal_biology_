"""The graph's nodes: one function per step of the agent loop.

Each takes the whole state and returns only what it changed. Nodes hold no
state of their own - `ReconstructionNodes` carries the collaborators, not run
data - so the same instance serves every concurrent run.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from ...configuration.logging import get_logger
from ...contracts.events import EventType
from ...domain.models import Alignment, Reference
from ...domain.policies.validation_policy import ValidationPolicy
from ...domain.services import ContextExtractor, GapDetector
from ...observability.events import EventEmitter
from ...tools.registry import ToolRegistry
from ..planning.planner import Planner
from ..planning.stop_policy import StopPolicy
from ..planning.tool_selector import ToolSelector
from ..reasoning.critic import Critic
from ..reasoning.reasoner import Reasoner
from ..state import transitions
from ..state.state import ReconstructionState

_log = get_logger(__name__)


@dataclass(slots=True)
class ReconstructionNodes:
    """The collaborators every node needs, wired once at startup."""

    detector: GapDetector
    extractor: ContextExtractor
    policy: ValidationPolicy
    planner: Planner
    selector: ToolSelector
    registry: ToolRegistry
    reasoner: Reasoner
    critic: Critic
    stop_policy: StopPolicy
    events: EventEmitter

    # --- 1. Find the work ---------------------------------------------------

    async def detect_gaps(self, state: ReconstructionState) -> ReconstructionState:
        """Locate the gaps and decide which are worth attempting."""
        target = state["target"]
        gaps = self.detector.detect(target)
        contexts = self.extractor.extract_all(target, gaps)

        update: dict[str, Any] = dict(transitions.with_gaps(contexts))
        skipped: dict[str, str] = {}

        for context in contexts:
            attempt, reason = self.policy.should_attempt(context)
            if not attempt and reason:
                skipped[context.identifier] = reason

        if skipped:
            update["skipped"] = skipped

        self.events.emit(
            EventType.GAPS_DETECTED,
            state["run_id"],
            f"Found {len(gaps)} gap(s); {len(skipped)} will not be attempted.",
            {"gap_count": len(gaps), "skipped": len(skipped)},
        )
        return cast(ReconstructionState, update)

    # --- 2. Decide what to do ----------------------------------------------

    async def plan(self, state: ReconstructionState) -> ReconstructionState:
        """Choose this iteration's tool calls."""
        steps = await self.planner.plan(state, self.registry.catalogue())

        self.events.emit(
            EventType.PLAN_CREATED,
            state["run_id"],
            f"Planned {len(steps)} step(s).",
            {"steps": [step.as_dict() for step in steps]},
        )
        return transitions.with_plan([step.as_dict() for step in steps])

    # --- 3. Do it -----------------------------------------------------------

    async def execute_tools(self, state: ReconstructionState) -> ReconstructionState:
        """Run the planned steps, concurrently where they are independent.

        Steps for different gaps do not interact, so they are gathered rather
        than awaited in sequence - the difference is minutes when several gaps
        each need a BLAST job.
        """
        from ..planning.planner import PlanStep

        invocations = []
        for raw in state.get("plan") or []:
            step = PlanStep(
                tool=raw["tool"],
                gap_id=raw.get("gap_id"),
                reason=raw.get("reason", ""),
                arguments=raw.get("arguments") or {},
            )
            invocation = self.selector.build(step, state)
            if invocation is not None:
                invocations.append(invocation)

        if not invocations:
            return transitions.with_warning("The plan produced no runnable tool calls.")

        results = await asyncio.gather(
            *(self._run_one(invocation, state) for invocation in invocations),
            return_exceptions=True,
        )

        merged: dict[str, Any] = {
            "references": {},
            "alignments": {},
            "tool_calls": [],
            "errors": [],
        }
        for result in results:
            if isinstance(result, BaseException):
                merged["errors"].append(str(result))
                continue
            for key, value in result.items():
                if key in ("references", "alignments"):
                    merged[key].update(value)
                elif key in ("tool_calls", "errors"):
                    merged[key].extend(value)

        return cast(ReconstructionState, {key: value for key, value in merged.items() if value})

    async def _run_one(self, invocation: Any, state: ReconstructionState) -> dict[str, Any]:
        """Execute one tool and fold its output into a partial state."""
        run_id = state["run_id"]
        self.events.emit(
            EventType.TOOL_STARTED,
            run_id,
            f"Running {invocation.tool}.",
            {"tool": invocation.tool, "gap_id": invocation.gap_id},
        )

        try:
            output = await self.registry.run(invocation.tool, invocation.payload)
        except Exception as error:  # noqa: BLE001 - recorded, never fatal
            self.events.emit(
                EventType.TOOL_FAILED, run_id, str(error), {"tool": invocation.tool}
            )
            return {
                "tool_calls": [
                    {"tool": invocation.tool, "gap_id": invocation.gap_id, "succeeded": False}
                ],
                "errors": [f"{invocation.tool}: {error}"],
            }

        update: dict[str, Any] = {
            "tool_calls": [
                {
                    "tool": invocation.tool,
                    "gap_id": invocation.gap_id,
                    "succeeded": bool(getattr(output, "succeeded", True)),
                }
            ],
            "errors": [],
        }

        if not getattr(output, "succeeded", True):
            update["errors"] = [f"{invocation.tool}: {getattr(output, 'error', 'failed')}"]
            self.events.emit(EventType.TOOL_FAILED, run_id, str(update["errors"][0]))
            return update

        references: list[Reference] = list(getattr(output, "references", []) or [])
        if references and invocation.gap_id:
            update["references"] = {invocation.gap_id: references}

        alignment = getattr(output, "alignment", None)
        if isinstance(alignment, Alignment) and invocation.gap_id:
            update["alignments"] = {invocation.gap_id: alignment}

        self.events.emit(
            EventType.TOOL_COMPLETED,
            run_id,
            f"{invocation.tool} completed.",
            {"tool": invocation.tool, "references": len(references)},
        )
        return update

    # --- 4. Make sense of it ------------------------------------------------

    async def reason(self, state: ReconstructionState) -> ReconstructionState:
        """Build and score candidates from every alignment gathered so far."""
        alignments = state.get("alignments") or {}
        references = state.get("references") or {}
        resolved = state.get("reconstructions") or {}

        candidates: dict[str, Any] = {}
        reconstructions: dict[str, Any] = {}

        for context in state.get("gap_contexts") or []:
            gap_id = context.identifier
            if gap_id in resolved or gap_id not in alignments:
                continue

            built = self.reasoner.build_candidates(
                context, alignments[gap_id], references.get(gap_id, [])
            )
            candidates[gap_id] = built
            reconstructions[gap_id] = self.reasoner.finalise(context, built)

            self.events.emit(
                EventType.CANDIDATE_PROPOSED,
                state["run_id"],
                f"Proposed a reconstruction for {gap_id}.",
                {"gap_id": gap_id, "candidates": len(built)},
            )

        update: dict[str, Any] = {}
        if candidates:
            update["candidates"] = candidates
        if reconstructions:
            update["reconstructions"] = reconstructions
        return cast(ReconstructionState, update)

    # --- 5. Check our work --------------------------------------------------

    async def critique(self, state: ReconstructionState) -> ReconstructionState:
        """Review this iteration's reconstructions and record the findings."""
        contexts = {context.identifier: context for context in state.get("gap_contexts") or []}
        notes: list[str] = []

        for gap_id, reconstruction in (state.get("reconstructions") or {}).items():
            context = contexts.get(gap_id)
            if context is None:
                continue
            critique = await self.critic.review(context, reconstruction)
            if not critique.acceptable:
                notes.append(critique.as_note())

        if notes:
            self.events.emit(
                EventType.CRITIQUE_ISSUED,
                state["run_id"],
                f"{len(notes)} reconstruction(s) need more evidence.",
                {"critiques": notes},
            )

        iteration = state.get("iteration", 0) + 1
        self.events.emit(
            EventType.ITERATION_COMPLETED, state["run_id"], f"Iteration {iteration} complete."
        )

        update: dict[str, Any] = {"iteration": iteration}
        if notes:
            update["critiques"] = notes

        should_stop, reason = self.stop_policy.should_stop(
            cast(ReconstructionState, {**state, **update})
        )
        if should_stop and reason is not None:
            update["should_continue"] = False
            update["stop_reason"] = reason.value

        return cast(ReconstructionState, update)
