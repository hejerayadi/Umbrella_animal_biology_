"""Counting what a run is allowed to spend, outside the language model.

Budget enforcement is deterministic and lives here rather than in a prompt. A
model asked to respect a limit will sometimes respect it; a ledger always does,
and the failure mode of the model getting it wrong is an unbounded loop against
metered external services.

The important property is that slots are **reserved before work is dispatched,
never counted afterwards**. A round that fires five concurrent searches asks
for five slots up front and receives however many remain; accounting after the
fact would let all five run and discover the overrun once the money is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reconstruction_agent.domain.enums import ToolName

#: Tools whose cost is tracked individually because they dominate the run.
_BLAST_TOOLS = frozenset({ToolName.SEARCH_HOMOLOGS})
_MAFFT_TOOLS = frozenset({ToolName.ALIGN_HOMOLOGS})
_EVO2_TOOLS = frozenset({ToolName.EVALUATE_WITH_EVO2})


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """The ceilings for one run."""

    max_iterations: int = 6
    max_tool_calls: int = 24
    max_blast_calls: int = 8
    max_mafft_calls: int = 8
    max_evo2_calls: int = 4
    max_llm_calls: int = 12


@dataclass(slots=True)
class BudgetLedger:
    """Tracks consumption against the limits, and hands out reservations."""

    limits: BudgetLimits = field(default_factory=BudgetLimits)

    iterations: int = 0
    tool_calls: int = 0
    blast_calls: int = 0
    mafft_calls: int = 0
    evo2_calls: int = 0
    llm_calls: int = 0

    def start_iteration(self) -> bool:
        """Begin a loop iteration, or report that the allowance is spent."""
        if self.iterations >= self.limits.max_iterations:
            return False
        self.iterations += 1
        return True

    def reserve(self, tool: ToolName, count: int = 1) -> int:
        """Claim up to `count` calls of `tool`, returning how many were granted.

        A partial grant is normal and deliberate: asking for three searches with
        two slots left should run two searches, not refuse the round outright.
        Callers dispatch exactly what they are granted.
        """
        if count <= 0:
            return 0

        granted = min(count, self._headroom(tool))
        if granted <= 0:
            return 0

        self.tool_calls += granted
        if tool in _BLAST_TOOLS:
            self.blast_calls += granted
        elif tool in _MAFFT_TOOLS:
            self.mafft_calls += granted
        elif tool in _EVO2_TOOLS:
            self.evo2_calls += granted
        return granted

    def reserve_llm(self, count: int = 1) -> int:
        """Claim up to `count` model calls, returning how many were granted."""
        granted = max(min(count, self.limits.max_llm_calls - self.llm_calls), 0)
        self.llm_calls += granted
        return granted

    def _headroom(self, tool: ToolName) -> int:
        """Remaining calls of `tool`, respecting both its own cap and the total."""
        overall = self.limits.max_tool_calls - self.tool_calls
        if tool in _BLAST_TOOLS:
            specific = self.limits.max_blast_calls - self.blast_calls
        elif tool in _MAFFT_TOOLS:
            specific = self.limits.max_mafft_calls - self.mafft_calls
        elif tool in _EVO2_TOOLS:
            specific = self.limits.max_evo2_calls - self.evo2_calls
        else:
            specific = overall
        return max(min(overall, specific), 0)

    def can_afford(self, tool: ToolName) -> bool:
        """Whether at least one call of `tool` is still available."""
        return self._headroom(tool) > 0

    @property
    def exhausted(self) -> bool:
        """Whether no further tool work is possible at all."""
        return self.tool_calls >= self.limits.max_tool_calls

    def snapshot(self) -> dict[str, int | bool]:
        """The consumption report carried in the API response `meta`."""
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "blast_calls": self.blast_calls,
            "mafft_calls": self.mafft_calls,
            "evo2_calls": self.evo2_calls,
            "llm_calls": self.llm_calls,
            "budget_exhausted": self.exhausted,
        }
