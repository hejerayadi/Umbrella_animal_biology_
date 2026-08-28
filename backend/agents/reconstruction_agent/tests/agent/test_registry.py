"""The registry's promise: nothing a tool call can do kills the run.

A node that raises discards evidence already gathered for the other gaps in the
same request, so every failure mode - an unknown name, bad arguments, an
exhausted budget, a typed error, an unexpected crash - has to come back as a
readable outcome instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.orchestration.budget import BudgetLedger, BudgetLimits
from reconstruction_agent.tools.base import Tool, ToolOutcome
from reconstruction_agent.tools.registry import ToolRegistry


class _Input(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str = "gap_1"
    value: int = 1


class _Output(BaseModel):
    model_config = ConfigDict(frozen=True)

    doubled: int


class _Doubling(Tool[_Input, _Output]):
    name = ToolName.GENERATE_CANDIDATES
    description = "double a number"
    input_model = _Input

    async def run(self, request: _Input) -> ToolOutcome[_Output]:
        return ToolOutcome(tool=self.name, ok=True, data=_Output(doubled=request.value * 2))


class _Typed(Tool[_Input, _Output]):
    name = ToolName.SEARCH_HOMOLOGS
    description = "always fails with a typed error"
    input_model = _Input

    async def run(self, request: _Input) -> ToolOutcome[_Output]:
        raise ReconstructionError("the upstream service refused")


class _Crashing(Tool[_Input, _Output]):
    name = ToolName.ALIGN_HOMOLOGS
    description = "always crashes"
    input_model = _Input

    async def run(self, request: _Input) -> ToolOutcome[_Output]:
        raise RuntimeError("undefined behaviour")


def _registry() -> ToolRegistry:
    return ToolRegistry((_Doubling(), _Typed(), _Crashing()))


def _budget(**limits: int) -> BudgetLedger:
    return BudgetLedger(limits=BudgetLimits(**limits))


class TestDispatch:
    async def test_a_registered_tool_runs_and_is_timed(self) -> None:
        outcome, record = await _registry().invoke(
            ToolName.GENERATE_CANDIDATES, {"value": 21}, budget=_budget()
        )

        assert outcome.ok
        assert outcome.data is not None and outcome.data.doubled == 42
        assert record.duration_seconds >= 0.0

    async def test_an_already_typed_request_is_passed_through(self) -> None:
        outcome, _ = await _registry().invoke(
            ToolName.GENERATE_CANDIDATES, _Input(value=5), budget=_budget()
        )

        assert outcome.data is not None and outcome.data.doubled == 10


class TestNothingEscapes:
    async def test_an_unregistered_name_is_a_result_not_a_crash(self) -> None:
        """An LLM-authored plan can name a tool that does not exist."""
        outcome, record = await _registry().invoke(
            ToolName.EVALUATE_WITH_EVO2, {}, budget=_budget()
        )

        assert outcome.failed()
        assert "not a registered tool" in outcome.reason
        assert record.ok is False

    async def test_invalid_arguments_are_reported_in_planning_terms(self) -> None:
        outcome, _ = await _registry().invoke(
            ToolName.GENERATE_CANDIDATES, {"value": "not a number"}, budget=_budget()
        )

        assert outcome.failed()
        assert "invalid arguments" in outcome.reason

    async def test_a_typed_error_becomes_a_transport_failure(self) -> None:
        outcome, _ = await _registry().invoke(ToolName.SEARCH_HOMOLOGS, {}, budget=_budget())

        assert outcome.failed()
        assert outcome.transport_error is True
        assert "refused" in outcome.reason

    async def test_an_unexpected_crash_still_returns_an_outcome(self) -> None:
        outcome, _ = await _registry().invoke(ToolName.ALIGN_HOMOLOGS, {}, budget=_budget())

        assert outcome.failed()
        assert outcome.transport_error is True


class TestBudgetIsReservedBeforeDispatch:
    async def test_a_refused_reservation_short_circuits_the_call(self) -> None:
        """A tool that ran and was then found to be over budget has already
        spent the wall-clock time."""
        budget = _budget(max_tool_calls=1)
        first, _ = await _registry().invoke(
            ToolName.GENERATE_CANDIDATES, {"value": 1}, budget=budget
        )
        second, _ = await _registry().invoke(
            ToolName.GENERATE_CANDIDATES, {"value": 1}, budget=budget
        )

        assert first.ok
        assert second.failed()
        assert "Budget exhausted" in second.reason

    async def test_a_failing_tool_still_consumes_its_reservation(self) -> None:
        """Otherwise a tool that fails repeatedly is free and loops forever."""
        budget = _budget(max_tool_calls=10)
        await _registry().invoke(ToolName.SEARCH_HOMOLOGS, {}, budget=budget)

        assert budget.tool_calls == 1


class TestTheCatalogueDescribesTheSurface:
    def test_every_registered_tool_is_listed_with_a_description(self) -> None:
        catalogue = dict(_registry().catalogue())

        assert set(catalogue) == {
            ToolName.GENERATE_CANDIDATES,
            ToolName.SEARCH_HOMOLOGS,
            ToolName.ALIGN_HOMOLOGS,
        }
        assert all(description for description in catalogue.values())


class TestFinalisationIsNeverRefusedForBudget:
    """Budget limits stop a run spending the clock on evidence it will not
    finish gathering. They must never stop it reporting what it already has -
    observed on a scaffold with 675 unresolved regions, where the eighth gap
    exhausted the tool budget and finished with no result at all."""

    async def test_an_exempt_tool_runs_past_an_exhausted_budget(self) -> None:
        class _Exempt(_Doubling):
            name = ToolName.FINALIZE_RESULT
            budget_exempt = True

        registry = ToolRegistry((_Doubling(), _Exempt()))
        budget = _budget(max_tool_calls=1)

        await registry.invoke(ToolName.GENERATE_CANDIDATES, {"value": 1}, budget=budget)
        assert budget.exhausted

        outcome, _ = await registry.invoke(ToolName.FINALIZE_RESULT, {"value": 2}, budget=budget)

        assert outcome.ok
        assert outcome.data is not None and outcome.data.doubled == 4

    async def test_a_non_exempt_tool_is_still_refused(self) -> None:
        """The exemption is for the answer, not a way around the budget."""
        budget = _budget(max_tool_calls=1)
        await _registry().invoke(ToolName.GENERATE_CANDIDATES, {"value": 1}, budget=budget)
        outcome, _ = await _registry().invoke(
            ToolName.GENERATE_CANDIDATES, {"value": 1}, budget=budget
        )

        assert outcome.failed()

    def test_only_finalisation_is_exempt(self) -> None:
        """A second exempt tool would make the budget advisory."""
        exempt = [tool.name.value for tool in _all_tools() if tool.budget_exempt]
        assert exempt == [ToolName.FINALIZE_RESULT.value]


def _all_tools() -> tuple[Any, ...]:
    from reconstruction_agent.tools.alignment import AlignHomologsTool, AnalyzeAlignmentTool
    from reconstruction_agent.tools.candidate import GenerateCandidatesTool, ScoreCandidateTool
    from reconstruction_agent.tools.finalize import FinalizeResultTool
    from reconstruction_agent.tools.homology import GetHomologSequencesTool, SearchHomologsTool
    from reconstruction_agent.tools.plausibility import EvaluateWithEvo2Tool
    from reconstruction_agent.tools.reconstruction import ReconstructGapTool
    from reconstruction_agent.tools.sequence import (
        GetAssemblyMetadataTool,
        GetSequenceContextTool,
    )
    from reconstruction_agent.tools.validation import ValidateCandidateTool

    return (
        GetSequenceContextTool,
        GetAssemblyMetadataTool,
        SearchHomologsTool,
        GetHomologSequencesTool,
        AlignHomologsTool,
        AnalyzeAlignmentTool,
        GenerateCandidatesTool,
        EvaluateWithEvo2Tool,
        ScoreCandidateTool,
        ValidateCandidateTool,
        ReconstructGapTool,
        FinalizeResultTool,
    )
