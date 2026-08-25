"""Named state changes, so nodes do not hand-build partial dicts.

Every node returns a partial `ReconstructionState`. Routing those through
named helpers keeps the state's shape in one place: when a field is added,
this file changes rather than every node that touches it.
"""
from __future__ import annotations

from typing import Any

from agent.state.state import ReconstructionState
from contracts.output import GapReconstruction
from domain.models import Candidate, GapContext, Reference


def with_gaps(contexts: list[GapContext]) -> ReconstructionState:
    return ReconstructionState(gap_contexts=contexts)


def with_skipped(gap_id: str, reason: str) -> ReconstructionState:
    return ReconstructionState(skipped={gap_id: reason})


def with_references(gap_id: str, references: list[Reference]) -> ReconstructionState:
    return ReconstructionState(references={gap_id: references})


def with_alignment(gap_id: str, alignment: Any) -> ReconstructionState:
    return ReconstructionState(alignments={gap_id: alignment})


def with_candidates(gap_id: str, candidates: list[Candidate]) -> ReconstructionState:
    return ReconstructionState(candidates={gap_id: candidates})


def with_reconstruction(gap_id: str, reconstruction: GapReconstruction) -> ReconstructionState:
    return ReconstructionState(reconstructions={gap_id: reconstruction})


def with_plan(plan: list[dict[str, Any]]) -> ReconstructionState:
    return ReconstructionState(plan=plan)


def with_tool_call(
    tool: str, *, gap_id: str | None = None, succeeded: bool, detail: str | None = None
) -> ReconstructionState:
    """Record one tool invocation in the run's audit trail."""
    return ReconstructionState(
        tool_calls=[
            {"tool": tool, "gap_id": gap_id, "succeeded": succeeded, "detail": detail}
        ]
    )


def with_critique(critique: str) -> ReconstructionState:
    return ReconstructionState(critiques=[critique])


def with_warning(message: str) -> ReconstructionState:
    return ReconstructionState(warnings=[message])


def with_error(message: str) -> ReconstructionState:
    return ReconstructionState(errors=[message])


def advance_iteration(current: int) -> ReconstructionState:
    return ReconstructionState(iteration=current + 1)


def halt(reason: str) -> ReconstructionState:
    """Stop the agent loop, recording why.

    The reason is always set alongside the flag: a run that stopped without
    saying why is the hardest kind to debug.
    """
    return ReconstructionState(should_continue=False, stop_reason=reason)


def delegate_to(agent: str, prompt: str) -> ReconstructionState:
    """Hand the question to another agent via the orchestrator."""
    return ReconstructionState(
        needs_agent=agent,
        prompt_to_target_agent=prompt,
        should_continue=False,
        stop_reason=f"Delegating to {agent}.",
    )
