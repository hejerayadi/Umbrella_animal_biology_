"""The loop itself, driven end to end over faked tools.

Two things are worth proving here and nowhere else.

**Evidence survives the merge.** A result computed by one node and dropped
during a state merge looks exactly like a result that was never computed, and
no happy-path test catches it. So the tests assert on what reached the final
state, not just on what the tools returned.

**The loop terminates.** Every path out - resolved, refused, replanned to
exhaustion, cut off by the clock - is exercised, because an agent that loops is
an agent that returns nothing at all.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.agent.critic import Critic
from reconstruction_agent.agent.evaluator import Evaluator
from reconstruction_agent.agent.graph import ReconstructionGraph
from reconstruction_agent.agent.planner import Planner
from reconstruction_agent.agent.replanner import Replanner
from reconstruction_agent.agent.state import AgentContext, initial_state
from reconstruction_agent.domain.enums import GapStatus, ToolName
from reconstruction_agent.domain.models.result import GapReconstruction
from reconstruction_agent.domain.models.sequence import Gap, GapContext, SequenceRecord
from reconstruction_agent.integrations.llm.null import NullLLMClient
from reconstruction_agent.orchestration.budget import BudgetLedger, BudgetLimits
from reconstruction_agent.orchestration.deadline import Deadline, PhaseBudget
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.base import Tool, ToolOutcome
from reconstruction_agent.tools.registry import ToolRegistry

_GAP = Gap(gap_id="gap_1", start=20, end=65)
_RECORD = SequenceRecord(accession="NC_TEST.1", residues="ACGT" * 5 + "N" * 45 + "TGCA" * 5)
_CONTEXT = GapContext(
    gap=_GAP, source_accession="NC_TEST.1", left_flank="ACGT" * 5, right_flank="TGCA" * 5
)


class _Any(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    gap_id: str = "gap_1"


class _Fake(Tool[_Any, BaseModel]):
    """A tool that returns whatever it was handed, and records that it ran."""

    input_model = _Any

    def __init__(
        self, name: ToolName, payload: BaseModel | None, *, ok: bool = True, reason: str = ""
    ) -> None:
        self.name = name
        self.description = name.value
        self._payload = payload
        self._ok = ok
        self._reason = reason
        self.calls = 0

    async def run(self, request: _Any) -> ToolOutcome[BaseModel]:
        self.calls += 1
        return ToolOutcome(tool=self.name, ok=self._ok, data=self._payload, reason=self._reason)


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    record: SequenceRecord | None = None
    context: GapContext | None = None
    reconstruction: GapReconstruction | None = None


def _graph(*tools: Tool[Any, Any], deadline: Deadline | None = None) -> ReconstructionGraph:
    context = AgentContext(
        registry=ToolRegistry(tools),
        llm=NullLLMClient(),
        deadline=deadline or Deadline(total_seconds=300.0),
        budget=BudgetLedger(limits=BudgetLimits()),
        phases=PhaseBudget(),
    )
    return ReconstructionGraph(
        context,
        planner=Planner(NullLLMClient()),
        evaluator=Evaluator(ConfidenceEngine()),
        critic=Critic(NullLLMClient()),
        replanner=Replanner(),
    )


def _state() -> Any:
    return initial_state(
        gap=_GAP,
        request_id="rec_test",
        accession="NC_TEST.1",
        residues=None,
        scientific_name="Ursus maritimus",
    )


def _resolved() -> GapReconstruction:
    return GapReconstruction(gap=_GAP, status=GapStatus.RESOLVED, explanation="45 bases recovered")


class TestTheLoopAlwaysProducesAResult:
    async def test_a_run_with_no_evidence_still_finalises(self) -> None:
        """Refusal is a result: the gap comes back with its coordinates intact."""
        finalize = _Fake(
            ToolName.FINALIZE_RESULT,
            _Payload(
                reconstruction=GapReconstruction(
                    gap=_GAP, status=GapStatus.UNRESOLVED, explanation="nothing was measured"
                )
            ),
            ok=False,
        )
        final = await _graph(finalize).run(_state())

        assert final["result"] is not None
        assert final["result"].status is GapStatus.UNRESOLVED
        assert final["result"].gap == _GAP
        assert finalize.calls == 1

    async def test_a_closed_window_finalises_without_dispatching_work(self) -> None:
        """The deadline check must precede dispatch, not follow it."""
        search = _Fake(ToolName.SEARCH_HOMOLOGS, None)
        finalize = _Fake(ToolName.FINALIZE_RESULT, _Payload(reconstruction=_resolved()))
        expired = Deadline(total_seconds=0.0, reserve_seconds=0.0)

        final = await _graph(search, finalize, deadline=expired).run(_state())

        assert search.calls == 0
        assert final["result"] is not None


class TestEvidenceSurvivesTheMerge:
    async def test_a_tool_result_reaches_the_final_state(self) -> None:
        """The failure this guards is a computed result lost during a merge -
        indistinguishable from one never computed."""
        context_tool = _Fake(
            ToolName.GET_SEQUENCE_CONTEXT, _Payload(record=_RECORD, context=_CONTEXT)
        )
        finalize = _Fake(ToolName.FINALIZE_RESULT, _Payload(reconstruction=_resolved()))

        final = await _graph(context_tool, finalize).run(_state())

        assert final["record"] == _RECORD
        assert final["context"] == _CONTEXT

    async def test_the_tool_history_accumulates_rather_than_overwriting(self) -> None:
        """Two nodes writing the same channel must append; the reducer is what
        makes that true, and it is only true if it is declared."""
        context_tool = _Fake(ToolName.GET_SEQUENCE_CONTEXT, _Payload(record=_RECORD))
        finalize = _Fake(ToolName.FINALIZE_RESULT, _Payload(reconstruction=_resolved()))

        final = await _graph(context_tool, finalize).run(_state())
        tools_run = [record.tool for record in final["tool_history"]]

        assert ToolName.GET_SEQUENCE_CONTEXT in tools_run
        assert ToolName.FINALIZE_RESULT in tools_run
        assert len(final["tool_history"]) >= 2

    async def test_a_failed_call_is_recorded_with_its_reason(self) -> None:
        """The critic reads these; a failure with no reason is undiagnosable."""
        context_tool = _Fake(
            ToolName.GET_SEQUENCE_CONTEXT,
            None,
            ok=False,
            reason="the gap has no usable flanking sequence",
        )
        finalize = _Fake(ToolName.FINALIZE_RESULT, _Payload(reconstruction=_resolved()))

        final = await _graph(context_tool, finalize).run(_state())

        assert any("no usable flanking" in note for note in final["observations"])


class TestTheLoopTerminates:
    async def test_replanning_stops_at_its_allowance(self) -> None:
        """An agent that replans forever returns nothing at all."""
        search = _Fake(ToolName.SEARCH_HOMOLOGS, None, ok=False, reason="nothing found")
        finalize = _Fake(
            ToolName.FINALIZE_RESULT,
            _Payload(reconstruction=GapReconstruction(gap=_GAP, status=GapStatus.UNRESOLVED)),
            ok=False,
        )

        final = await _graph(search, finalize).run(_state())

        assert final["result"] is not None
        assert final["replan_count"] <= 2

    async def test_a_budget_ceiling_ends_the_run(self) -> None:
        context = AgentContext(
            registry=ToolRegistry(
                (
                    _Fake(ToolName.GET_SEQUENCE_CONTEXT, _Payload(record=_RECORD)),
                    _Fake(ToolName.FINALIZE_RESULT, _Payload(reconstruction=_resolved())),
                )
            ),
            llm=NullLLMClient(),
            deadline=Deadline(total_seconds=300.0),
            budget=BudgetLedger(limits=BudgetLimits(max_tool_calls=1)),
            phases=PhaseBudget(),
        )
        graph = ReconstructionGraph(
            context,
            planner=Planner(NullLLMClient()),
            evaluator=Evaluator(ConfidenceEngine()),
            critic=Critic(NullLLMClient()),
        )

        final = await graph.run(_state())

        assert final["tool_history"]
