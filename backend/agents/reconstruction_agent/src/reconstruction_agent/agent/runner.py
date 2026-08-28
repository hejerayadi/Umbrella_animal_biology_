"""Building and running the agent loop for one gap.

The seam between the request-level service - which resolves the record, triages
gaps and assembles the final result - and the per-gap graph that gathers
evidence and decides.

A graph is built per run rather than per process. The `AgentContext` it closes
over holds that run's `Deadline` and `BudgetLedger`, and those are ledgers, not
configuration: a graph compiled once and shared would hand every request the
first request's clock.
"""

from __future__ import annotations

from reconstruction_agent.agent.critic import Critic
from reconstruction_agent.agent.evaluator import Evaluator
from reconstruction_agent.agent.graph import DEFAULT_MAX_REPLANS, ReconstructionGraph
from reconstruction_agent.agent.planner import Planner
from reconstruction_agent.agent.replanner import Replanner
from reconstruction_agent.agent.state import AgentContext, GapState, initial_state
from reconstruction_agent.domain.enums import GapStatus, UnresolvedReason
from reconstruction_agent.domain.models.result import GapReconstruction
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.integrations.llm.base import LLMClient
from reconstruction_agent.observability.logger import get_logger
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.orchestration.deadline import Deadline, PhaseBudget
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.registry import ToolRegistry

_log = get_logger(__name__)


class GapRunner:
    """Runs the agent loop for one gap and returns its reconstruction."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        llm: LLMClient,
        engine: ConfidenceEngine,
        phases: PhaseBudget,
        max_replans: int = DEFAULT_MAX_REPLANS,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._engine = engine
        self._phases = phases
        self._max_replans = max_replans

    async def run(
        self,
        gap: Gap,
        *,
        request_id: str,
        deadline: Deadline,
        budget: BudgetLedger,
        accession: str | None = None,
        residues: str | None = None,
        scientific_name: str | None = None,
    ) -> GapReconstruction:
        """Gather evidence for one gap and decide what it supports."""
        graph = self._build(deadline, budget)
        state = initial_state(
            gap=gap,
            request_id=request_id,
            accession=accession,
            residues=residues,
            scientific_name=scientific_name,
        )

        final = await graph.run(state)
        result = final.get("result")
        if result is not None:
            return result

        # The graph ended without finalising. Only reachable if the loop was
        # cut short by something outside its own control, so the honest report
        # is that the run ran out of time, with the coordinates preserved.
        _log.warning("gap_run_produced_no_result", gap_id=gap.gap_id)
        return GapReconstruction(
            gap=gap,
            status=GapStatus.UNRESOLVED,
            unresolved_reason=UnresolvedReason.DEADLINE_EXCEEDED,
            explanation="The run ended before this gap could be decided.",
        )

    def _build(self, deadline: Deadline, budget: BudgetLedger) -> ReconstructionGraph:
        context = AgentContext(
            registry=self._registry,
            llm=self._llm,
            deadline=deadline,
            budget=budget,
            phases=self._phases,
        )
        return ReconstructionGraph(
            context,
            planner=Planner(self._llm),
            evaluator=Evaluator(self._engine),
            critic=Critic(self._llm),
            replanner=Replanner(),
            max_replans=self._max_replans,
        )


__all__ = ["GapRunner", "GapState"]
