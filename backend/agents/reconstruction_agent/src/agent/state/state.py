"""The state object every graph node reads and writes.

LangGraph merges each node's returned partial state into this one using the
reducers declared in the annotations - see `reducers.py` for why the list
fields append rather than overwrite.

This state is **checkpointed and resumed across HTTP calls**. The orchestrator
allows 120 s per call and three CONTINUE retries, so one reconstruction is
spread over up to four slices, each picking up where the last stopped. Two
consequences shape what may live here:

- Everything must survive a round trip through the checkpointer, so values are
  dataclasses and plain types, never open clients or coroutines.
- Anything that must not be redone on resume (tool calls already paid for,
  evidence already gathered) has to be *in* the state, not in a local variable.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from agent.state.reducers import merge_by_gap, replace
from contracts.observation import Observation
from contracts.output import GapReconstruction
from domain.models import Candidate, GapContext, Reference, Sequence


class ReconstructionState(TypedDict, total=False):
    """Everything one reconstruction run knows about itself.

    `total=False` because nodes return partial updates; only the keys a node
    actually changed are present in what it returns.
    """

    # --- Immutable run inputs, set once at entry ---------------------------
    run_id: str
    #: The orchestrator's run-wide correlation id. Stable across CONTINUE
    #: retries, which is what makes it the checkpoint key - unlike `run_id`,
    #: which is regenerated per HTTP call.
    trace_id: str
    instruction: str
    target: Sequence
    organism: str | None
    requested_organisms: list[str]

    # --- Discovered structure ---------------------------------------------
    gap_contexts: Annotated[list[GapContext], replace]
    # Gaps the validation policy declined to attempt, with the reason, so they
    # can be reported as SKIPPED rather than vanishing.
    skipped: Annotated[dict[str, str], merge_by_gap]

    # --- Evidence, accumulated across iterations and slices ---------------
    # Keyed by gap id: each gap gathers its own references and candidates, and
    # nodes may process gaps in any order.
    references: Annotated[dict[str, list[Reference]], merge_by_gap]
    alignments: Annotated[dict[str, Any], merge_by_gap]
    candidates: Annotated[dict[str, list[Candidate]], merge_by_gap]
    reconstructions: Annotated[dict[str, GapReconstruction], merge_by_gap]

    # --- Agent loop control -----------------------------------------------
    plan: Annotated[list[dict[str, Any]], replace]
    #: The invocations `select_tools` resolved from the plan, waiting to run.
    #: Separate from `plan` so selection is inspectable before execution.
    pending_invocations: Annotated[list[dict[str, Any]], replace]
    # Appended to, never replaced: the full trace of what ran is the audit
    # trail, and a node that overwrote it would erase earlier iterations.
    observations: Annotated[list[Observation], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    critiques: Annotated[list[str], operator.add]
    #: Per-gap verdict from the last critique round: accept | revise | abstain.
    verdicts: Annotated[dict[str, str], merge_by_gap]
    #: How many times each (tool, gap) pair has been attempted, so a semantic
    #: failure is retried once with relaxed parameters and then abandoned.
    attempts: Annotated[dict[str, int], merge_by_gap]

    iteration: int
    max_iterations: int
    should_continue: bool
    stop_reason: str | None

    # --- Budgets and slicing ----------------------------------------------
    #: Which HTTP slice this is, 0-based. The orchestrator grants four; on the
    #: last one the agent must finish rather than ask for another.
    slice_index: int
    max_slices: int
    #: `time.monotonic()` when the current slice opened. Per-slice, because the
    #: 120 s timeout it guards is per HTTP call, not per run.
    slice_started_at: float | None
    budget_tool_calls: int
    budget_tool_calls_by_name: Annotated[dict[str, int], merge_by_gap]
    budget_llm_tokens: int
    #: Set when the slice ran out of wall clock. Becomes a CONTINUE.
    yielded: bool

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
    trace_id: str,
    instruction: str,
    target: Sequence,
    organism: str | None,
    requested_organisms: list[str],
    max_iterations: int,
    max_slices: int,
) -> ReconstructionState:
    """A fully-populated starting state.

    Every collection is initialised here rather than defaulted at read time, so
    no node has to guard against a missing key.
    """
    return ReconstructionState(
        run_id=run_id,
        trace_id=trace_id,
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
        pending_invocations=[],
        observations=[],
        tool_calls=[],
        critiques=[],
        verdicts={},
        attempts={},
        iteration=0,
        max_iterations=max_iterations,
        should_continue=True,
        stop_reason=None,
        slice_index=0,
        max_slices=max_slices,
        slice_started_at=None,
        budget_tool_calls=0,
        budget_tool_calls_by_name={},
        budget_llm_tokens=0,
        yielded=False,
        warnings=[],
        errors=[],
        needs_agent=None,
        prompt_to_target_agent=None,
    )


def attempt_key(tool: str, gap_id: str | None) -> str:
    """The `attempts` key for one tool/gap pair."""
    return f"{tool}:{gap_id or '-'}"


def is_last_slice(state: ReconstructionState) -> bool:
    """Whether this is the final slice the orchestrator will grant.

    On the last slice the agent must return COMPLETED with whatever it has:
    another CONTINUE is converted to FAILED by the orchestrator's worker node,
    which would discard every finding gathered so far.
    """
    return state.get("slice_index", 0) >= state.get("max_slices", 4) - 1
