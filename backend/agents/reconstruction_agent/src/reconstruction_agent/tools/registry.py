"""The tool surface, assembled once and dispatched by name.

The registry is the only place a `ToolName` becomes a running call. That
matters for three reasons the graph depends on:

**An unknown name is a result, not a crash.** A planner - especially an
LLM-backed one - can select a tool that is not registered. That comes back as a
failed outcome the critic can read, so a bad plan costs one iteration instead
of the run.

**Budget is reserved before dispatch, never after.** A tool that ran and was
then found to be over budget has already spent the wall-clock time, so the
ledger is consulted first and a refused reservation short-circuits the call.

**Nothing escapes.** Even a genuine fault is converted to a failed outcome
here, because a node that raises discards evidence already gathered for the
other gaps in the same run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.observability.logger import get_logger
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.tools.base import Tool, ToolOutcome, ToolRecord

_log = get_logger(__name__)


class ToolRegistry:
    """Every tool the agent can invoke, keyed by the name the planner uses."""

    def __init__(self, tools: tuple[Tool[Any, Any], ...]) -> None:
        self._tools: dict[ToolName, Tool[Any, Any]] = {tool.name: tool for tool in tools}

    def __contains__(self, name: ToolName) -> bool:
        return name in self._tools

    @property
    def names(self) -> tuple[ToolName, ...]:
        return tuple(self._tools)

    def get(self, name: ToolName) -> Tool[Any, Any] | None:
        return self._tools.get(name)

    def catalogue(self) -> tuple[tuple[ToolName, str], ...]:
        """Name and one-line description of each tool, for the planner prompt."""
        return tuple((tool.name, tool.description) for tool in self._tools.values())

    async def invoke(
        self,
        name: ToolName,
        arguments: dict[str, Any] | BaseModel,
        *,
        budget: BudgetLedger,
        gap_id: str = "",
    ) -> tuple[ToolOutcome[Any], ToolRecord]:
        """Run one tool by name, converting every failure into an outcome."""
        tool = self._tools.get(name)
        if tool is None:
            return _refused(
                name, gap_id, f"{name.value} is not a registered tool.", transport=False
            )

        try:
            request = (
                arguments
                if isinstance(arguments, tool.input_model)
                else tool.input_model.model_validate(
                    arguments.model_dump() if isinstance(arguments, BaseModel) else arguments
                )
            )
        except ValidationError as error:
            # A malformed call is a planning fault, so it is reported in terms
            # the replanner can act on rather than as a stack trace.
            return _refused(
                name, gap_id, f"{name.value} was called with invalid arguments: {error}", False
            )

        # Reserved before dispatch, except for the tool that reports the
        # answer: see `Tool.budget_exempt`. The reservation is still recorded
        # for an exempt tool so the accounting stays honest - it simply cannot
        # refuse the call.
        if budget.reserve(name) <= 0 and not tool.budget_exempt:
            return _refused(
                name, gap_id, f"Budget exhausted before {name.value} could run.", transport=False
            )

        try:
            return await tool.execute(request)
        except ReconstructionError as error:
            _log.warning("tool_failed", tool=name.value, gap_id=gap_id, error=str(error))
            return _refused(name, gap_id, str(error), transport=True)
        except Exception as error:  # noqa: BLE001 - a node must never kill the run
            _log.exception("tool_crashed", tool=name.value, gap_id=gap_id, error=str(error))
            return _refused(name, gap_id, f"{name.value} failed unexpectedly: {error}", True)


def _refused(
    name: ToolName, gap_id: str, reason: str, transport: bool
) -> tuple[ToolOutcome[Any], ToolRecord]:
    """A call that produced nothing, shaped exactly like one that did."""
    outcome: ToolOutcome[Any] = ToolOutcome(
        tool=name, ok=False, reason=reason, transport_error=transport
    )
    return outcome, ToolRecord(tool=name, gap_id=gap_id, ok=False, reason=reason)
