"""What one reconstruction is allowed to spend, and what it has spent.

The agent runs inside someone else's HTTP request. The orchestrator allows
600 s per call and only three CONTINUE retries, so an unbounded loop does not
merely cost money - it gets the whole run force-failed and every finding
discarded. Budgets are therefore a correctness concern, not a cost control.

`Budgets` is the immutable allowance; `BudgetUsage` is the running tally that
lives in graph state and is reported back to the caller. Keeping them apart
means the allowance can be reasoned about without carrying consumption around,
and the tally serialises into a checkpoint as plain numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class BudgetKind(str, Enum):
    """Which allowance ran out. Reported so a run that stopped early can say why."""

    TOOL_CALLS = "tool_calls"
    TOOL_CALLS_FOR_TOOL = "tool_calls_for_tool"
    LLM_TOKENS = "llm_tokens"
    WALL_CLOCK = "wall_clock"


@dataclass(frozen=True, slots=True)
class Budgets:
    """The allowance for one run, across every slice."""

    max_tool_calls: int = 24
    #: Per-tool caps for the expensive submit-and-poll tools. A tool absent
    #: from this map is limited only by `max_tool_calls`.
    per_tool: dict[str, int] = field(default_factory=dict)
    max_llm_tokens: int = 60_000
    #: Yield back to the orchestrator after this long, well inside its 600 s
    #: read timeout. Kept in step with `ContinuationSettings.
    #: yield_after_seconds`, which is what the running agent actually uses;
    #: this default only applies to a `Budgets()` built by hand, in tests.
    yield_after_seconds: float = 480.0


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """What has been spent so far.

    Immutable and rebuilt from checkpointed counters on every read, rather than
    held as mutable state on a node: the graph's nodes are shared across
    concurrent runs, and the tally has to survive a checkpoint round trip
    between slices. State is the single source of truth; this is a view of it.

    `elapsed_seconds` is deliberately per-slice, not cumulative. The wall-clock
    budget exists to return control before one HTTP call times out, and time
    spent in an earlier slice is not part of this call's 600 s.
    """

    tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    llm_tokens: int = 0
    #: `time.monotonic()` when the current slice began; `None` outside a slice.
    slice_started_at: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        if self.slice_started_at is None:
            return 0.0
        return time.monotonic() - self.slice_started_at

    def as_dict(self) -> dict[str, object]:
        """The shape reported in the result payload."""
        return {
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "llm_tokens": self.llm_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Answers "may I?" and "should I stop?" against an allowance."""

    budgets: Budgets

    def may_run(self, tool: str, usage: BudgetUsage) -> tuple[bool, BudgetKind | None]:
        """Whether one more call to `tool` is permitted.

        Checked before the call rather than after, because the point is to
        avoid starting work that cannot be paid for.
        """
        if usage.tool_calls >= self.budgets.max_tool_calls:
            return False, BudgetKind.TOOL_CALLS

        cap = self.budgets.per_tool.get(tool)
        if cap is not None and usage.tool_calls_by_name.get(tool, 0) >= cap:
            return False, BudgetKind.TOOL_CALLS_FOR_TOOL

        return True, None

    def exhausted(self, usage: BudgetUsage) -> BudgetKind | None:
        """Which allowance, if any, is spent. Checked between iterations."""
        if usage.tool_calls >= self.budgets.max_tool_calls:
            return BudgetKind.TOOL_CALLS
        if self.budgets.max_llm_tokens and usage.llm_tokens >= self.budgets.max_llm_tokens:
            return BudgetKind.LLM_TOKENS
        return None

    def should_yield(self, usage: BudgetUsage) -> bool:
        """Whether to checkpoint and hand control back to the orchestrator.

        Distinct from `exhausted`: the work is not over, this slice is. The
        caller turns this into a CONTINUE the orchestrator will retry.
        """
        return usage.elapsed_seconds >= self.budgets.yield_after_seconds

    def remaining_seconds(self, usage: BudgetUsage) -> float:
        """Wall clock left in this slice, never below zero.

        This is what one tool call may be allowed to take. The yield check
        happens between graph nodes, so without bounding the call itself a
        single submit-and-poll tool - MAFFT at EMBL-EBI polls for up to 600 s -
        holds the slice open long past the orchestrator's 600 s read timeout.
        The orchestrator then records the agent as unreachable and every
        finding in the slice is lost, which is strictly worse than yielding.
        """
        return max(0.0, self.budgets.yield_after_seconds - usage.elapsed_seconds)

    def remaining_tool_calls(self, usage: BudgetUsage) -> int:
        return max(0, self.budgets.max_tool_calls - usage.tool_calls)
