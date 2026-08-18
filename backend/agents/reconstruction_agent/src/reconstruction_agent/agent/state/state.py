"""The state object every graph node reads and writes.

LangGraph merges each node's returned partial state into this one using the
reducers declared in the annotations - see `reducers.py` for why the list
fields append rather than overwrite.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ...contracts.output import GapReconstruction
from ...domain.models import Candidate, GapContext, Reference, Sequence
from .reducers import merge_by_gap, replace


class ReconstructionState(TypedDict, total=False):
    """Everything one reconstruction run knows about itself.

    `total=False` because nodes return partial updates; only the keys a node
    actually changed are present in what it returns.
    """

    # --- Immutable run inputs, set once at entry ---------------------------
    run_id: str
    instruction: str
    target: Sequence
    organism: str | None
    requested_organisms: list[str]

    # --- Discovered structure ---------------------------------------------
    gap_contexts: Annotated[list[GapContext], replace]
    # Gaps the validation policy declined to attempt, with the reason, so they
    # can be reported as SKIPPED rather than vanishing.
    skipped: Annotated[dict[str, str], merge_by_gap]

    # --- Evidence, accumulated across iterations --------------------------
    # Keyed by gap id: each gap gathers its own references and candidates, and
    # nodes may process gaps in any order.
    references: Annotated[dict[str, list[Reference]], merge_by_gap]
    alignments: Annotated[dict[str, Any], merge_by_gap]
    candidates: Annotated[dict[str, list[Candidate]], merge_by_gap]
    reconstructions: Annotated[dict[str, GapReconstruction], merge_by_gap]

    # --- Agent loop control -----------------------------------------------
    plan: Annotated[list[dict[str, Any]], replace]
    # Appended to, never replaced: the full trace of what ran is the audit
    # trail, and a node that overwrote it would erase earlier iterations.
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    critiques: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    should_continue: bool
    stop_reason: str | None

    # --- Outcome -----------------------------------------------------------
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    # Set when the agent concludes another agent must answer first; becomes a
    # NEEDS_AGENT result for the orchestrator.
    needs_agent: str | None
    prompt_to_target_agent: str | None


def initial_state(
    *,
    run_id: str,
    instruction: str,
    target: Sequence,
    organism: str | None,
    requested_organisms: list[str],
    max_iterations: int,
) -> ReconstructionState:
    """A fully-populated starting state.

    Every collection is initialised here rather than defaulted at read time, so
    no node has to guard against a missing key.
    """
    return ReconstructionState(
        run_id=run_id,
        instruction=instruction,
        target=target,
        organism=organism,
        requested_organisms=requested_organisms,
        gap_contexts=[],
        skipped={},
        references={},
        alignments={},
        candidates={},
        reconstructions={},
        plan=[],
        tool_calls=[],
        critiques=[],
        iteration=0,
        max_iterations=max_iterations,
        should_continue=True,
        stop_reason=None,
        warnings=[],
        errors=[],
        needs_agent=None,
        prompt_to_target_agent=None,
    )
