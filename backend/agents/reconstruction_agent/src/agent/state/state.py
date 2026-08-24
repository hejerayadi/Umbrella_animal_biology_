"""The state object every graph node reads and writes.

LangGraph merges each node's returned partial state into this one using the
reducers declared in the annotations - see `reducers.py` for why the list
fields append rather than overwrite.

This state is **checkpointed and resumed across HTTP calls**. The orchestrator
allows 600 s per call and three CONTINUE retries, so one reconstruction is
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

from agent.state.reducers import accumulate_references, merge_by_gap, replace
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
    # Accumulated, not replaced: a later round must add to a gap's evidence
    # rather than discard what earlier rounds paid for.
    references: Annotated[dict[str, list[Reference]], accumulate_references]
    alignments: Annotated[dict[str, Any], merge_by_gap]
    candidates: Annotated[dict[str, list[Candidate]], merge_by_gap]
    #: Evo 2's arbitration for a contested gap: candidate id -> plausibility,
    #: plus which it preferred. Keyed by gap so a resumed slice does not pay
    #: for the same generation again.
    plausibility: Annotated[dict[str, Any], merge_by_gap]
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
    #: Why each revised gap failed, as a `RevisionReason` value. This is what
    #: lets the next plan change strategy instead of reissuing the same call.
    revision_reasons: Annotated[dict[str, str], merge_by_gap]
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
    #: 600 s timeout it guards is per HTTP call, not per run.
    slice_started_at: float | None
    budget_tool_calls: int
    budget_tool_calls_by_name: Annotated[dict[str, int], merge_by_gap]
    budget_llm_tokens: int
    #: Set when the slice ran out of wall clock. Becomes a CONTINUE.
    yielded: bool
    #: EMBL-EBI job ids for calls a slice deadline cut short, keyed the same
    #: way as `attempts`. A BLAST job takes far longer than one slice grants,
    #: so the next slice polls the job already running instead of paying for a
    #: new one; without this the work of every slice was thrown away and the
    #: run could never finish. Cleared as soon as a call returns.
    pending_jobs: Annotated[dict[str, str], merge_by_gap]

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
        plausibility={},
        reconstructions={},
        plan=[],
        pending_invocations=[],
        observations=[],
        tool_calls=[],
        critiques=[],
        verdicts={},
        revision_reasons={},
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
        pending_jobs={},
        warnings=[],
        errors=[],
        needs_agent=None,
        prompt_to_target_agent=None,
    )


def attempt_key(tool: str, gap_id: str | None) -> str:
    """The `attempts` key for one tool/gap pair."""
    return f"{tool}:{gap_id or '-'}"


def has_usable_alignment(alignment: object) -> bool:
    """Whether an alignment actually locates the gap's columns.

    An alignment that aligned the flanks but inserted nothing across the gap is
    a real answer about that gap, and it is stored - but it is *not* a
    satisfied precondition for reasoning, because there is nothing to read a
    candidate out of. Treating its presence as "alignment done" is what stops a
    retry from ever being planned.
    """
    return bool(alignment is not None and getattr(alignment, "spans_gap", False))


def final_gap_outcomes(state: ReconstructionState) -> dict[str, str]:
    """Every gap whose outcome will not change again, and what that outcome is.

    A gap is final when it has been skipped, reconstructed, or abandoned by the
    critic. A gap merely *present* in `reconstructions` is not final: an
    UNRESOLVED or LOW_CONFIDENCE entry is this iteration's best answer, and
    another round of evidence can still overturn it.

    Values are one of: ``skipped``, ``reconstructed``, ``abstained``.
    """
    from contracts.output import ReconstructionStatus

    outcomes: dict[str, str] = {}
    reconstructions = state.get("reconstructions") or {}
    verdicts = state.get("verdicts") or {}

    for gap_id in state.get("skipped") or {}:
        outcomes[gap_id] = "skipped"

    for gap_id, reconstruction in reconstructions.items():
        if gap_id in outcomes:
            continue
        # `getattr` rather than attribute access: this reads state that may
        # have come back through a checkpoint, and one malformed entry should
        # not take down the halting decision for every other gap.
        if getattr(reconstruction, "status", None) is ReconstructionStatus.RECONSTRUCTED:
            outcomes[gap_id] = "reconstructed"

    # The critic's verdict is the authority on abandonment, and it settles a
    # gap whether or not a candidate was ever produced for it.
    for gap_id, verdict in verdicts.items():
        if verdict == "abstain" and gap_id not in outcomes:
            outcomes[gap_id] = "abstained"

    return outcomes


def open_gap_ids(state: ReconstructionState) -> set[str]:
    """Gaps still in play: neither skipped, nor reconstructed, nor abandoned.

    This is what "is there work left?" means everywhere in the loop, and it is
    deliberately not "absent from `reconstructions`" - see `final_gap_outcomes`.
    """
    final = set(final_gap_outcomes(state))
    return {
        context.identifier
        for context in state.get("gap_contexts") or []
        if context.identifier not in final
    }


def unreconstructed_gap_ids(state: ReconstructionState) -> set[str]:
    """Gaps this agent was asked to fix and did not.

    Broader than `open_gap_ids`: it also covers gaps the critic has already
    abandoned. That is deliberate, and it is what escalation must ask about -
    "I gave up on this one" is precisely the case where another agent's help
    is worth requesting, and keying escalation on *open* gaps alone meant the
    critic's abstain silently cancelled the request for help.

    Skipped gaps are excluded: they were declined on their own shape (too
    long, no usable flank), and no amount of phylogeny changes that.
    """
    skipped = set(state.get("skipped") or {})
    outcomes = final_gap_outcomes(state)
    return {
        context.identifier
        for context in state.get("gap_contexts") or []
        if context.identifier not in skipped
        and outcomes.get(context.identifier) != "reconstructed"
    }


def is_last_slice(state: ReconstructionState) -> bool:
    """Whether this is the final slice the orchestrator will grant.

    On the last slice the agent must return COMPLETED with whatever it has:
    another CONTINUE is converted to FAILED by the orchestrator's worker node,
    which would discard every finding gathered so far.
    """
    return state.get("slice_index", 0) >= state.get("max_slices", 4) - 1
