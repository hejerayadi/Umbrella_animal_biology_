"""What every tool is, and what every tool promises.

A tool is a typed shell over a service. It owns no biology: it validates its
arguments, reserves budget, calls one service, and reports what happened in a
shape the graph can merge into state. Keeping that boundary thin is what lets
the same service be exercised by a test, by the deterministic path and by an
LLM-selected plan without three versions of the logic.

Two promises hold for every tool:

**A tool never raises for a scientific outcome.** "No homologue crossed the
gap" is a result, not a failure, and it comes back as `ok=False` with a reason.
Only a genuine transport or programming fault escapes - and `ToolRegistry`
catches even those, because a node that raises kills a run that may already
hold resolvable evidence for other gaps.

**A tool reports its own cost.** Duration and budget consumption are recorded
by the registry rather than by each tool, so the accounting cannot drift
between tools and an unrecorded call is impossible by construction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.evidence import EvidenceContribution


class ToolOutcome[OutputT: BaseModel](BaseModel):
    """What one tool call produced, successful or not."""

    model_config = ConfigDict(frozen=True)

    tool: ToolName
    ok: bool
    #: Present when `ok`; absent otherwise.
    data: OutputT | None = None
    #: Why the call produced nothing usable. Written for the critic to read,
    #: so it names the deficit rather than describing the failure.
    reason: str = ""
    duration_seconds: float = 0.0
    #: Set when the call could not run at all, as opposed to running and
    #: finding nothing. The distinction drives retry-versus-replan.
    transport_error: bool = False
    #: What this call measured, in the form the run's audit trail is built
    #: from. Filled by `execute` from the tool's own `contribute`, so no
    #: caller has to remember to record it.
    evidence: EvidenceContribution = EvidenceContribution()

    def failed(self) -> bool:
        return not self.ok


class ToolRecord(BaseModel):
    """One line of the run's tool history, for observability and the critic."""

    model_config = ConfigDict(frozen=True)

    tool: ToolName
    gap_id: str = ""
    ok: bool = False
    reason: str = ""
    duration_seconds: float = 0.0
    #: Small, JSON-safe summary of what the call saw. Never the sequences
    #: themselves - a history entry is read by a human and by a prompt, and
    #: both are ruined by a thousand bases of DNA.
    summary: dict[str, Any] = Field(default_factory=dict)


class Tool[InputT: BaseModel, OutputT: BaseModel](ABC):
    """One typed capability the graph may invoke."""

    #: The name the planner selects by. Must be a `ToolName` member so an
    #: invalid plan is caught by enum validation rather than at dispatch.
    name: ToolName
    #: One line, read by the planner prompt. Written for a reader deciding
    #: whether this is the right next action.
    description: str = ""
    input_model: type[InputT]
    #: True for the tool that produces the answer. Budget limits exist to stop
    #: a run spending the clock on evidence it will not finish gathering; they
    #: must never stop it *reporting* what it already has. A finalisation
    #: refused for budget makes the whole run return nothing - observed on a
    #: scaffold with 675 unresolved regions, where the eighth gap finished with
    #: no result at all.
    budget_exempt: bool = False

    @abstractmethod
    async def run(self, request: InputT) -> ToolOutcome[OutputT]:
        """Execute the tool. Implementations must not raise for a scientific outcome."""

    def summarise(self, outcome: ToolOutcome[OutputT]) -> dict[str, Any]:
        """The small JSON-safe digest recorded in the run history.

        Overridden per tool. The default carries nothing, which is honest for
        a tool that has no measurement worth reporting.
        """
        return {}

    def contribute(self, outcome: ToolOutcome[OutputT]) -> EvidenceContribution:
        """What this call adds to the run's evidence.

        Part of the tool contract rather than something the graph derives, so
        a tool that measures something and reports nothing is visible as an
        empty contribution instead of as a silent hole in the audit trail.

        Called for failed outcomes too: a search that found hits but none
        crossing the gap has measured something the critic needs, and a run
        that reached a service has consulted it whatever the answer was.
        """
        return EvidenceContribution()

    async def execute(self, request: InputT) -> tuple[ToolOutcome[OutputT], ToolRecord]:
        """Run the tool and time it, producing the history entry alongside."""
        started = time.monotonic()
        outcome = await self.run(request)
        elapsed = round(time.monotonic() - started, 3)
        timed = outcome.model_copy(
            update={"duration_seconds": elapsed, "evidence": self.contribute(outcome)}
        )
        record = ToolRecord(
            tool=self.name,
            gap_id=getattr(request, "gap_id", "") or "",
            ok=timed.ok,
            reason=timed.reason,
            duration_seconds=elapsed,
            summary=self.summarise(timed),
        )
        return timed, record
