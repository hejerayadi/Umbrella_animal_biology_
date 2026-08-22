"""The shape every tool in this agent conforms to.

The planner picks tools by name from the registry and the graph runs them
without knowing what they do, so they all have to look the same from outside:
a name, a description the planner can reason over, and an async `run` taking
and returning Pydantic models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolInput(BaseModel):
    """Base for tool inputs."""


class ToolOutput(BaseModel):
    """Base for tool outputs.

    Every tool reports `succeeded` rather than only raising: a failed external
    call is normal, and the graph needs to record it as evidence and carry on
    with the other gaps instead of aborting the run.
    """

    succeeded: bool = True
    error: str | None = None
    # Free-form detail for the run log - timings, job ids, hit counts.
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC, Generic[InputT, OutputT]):
    """One capability the agent can invoke."""

    #: Stable identifier the planner uses to select this tool.
    name: str
    #: What this tool does and when to use it. Goes into the planner prompt, so
    #: it is written for a model to act on, not just for a human to read.
    description: str
    #: Rough cost hint (seconds) used by the planner to prefer cheap tools when
    #: several would answer the same question.
    estimated_seconds: float = 1.0

    @abstractmethod
    async def run(self, payload: InputT) -> OutputT:
        """Execute the tool. Should not raise for expected external failures."""

    def describe(self) -> dict[str, Any]:
        """The planner-facing summary of this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "estimated_seconds": self.estimated_seconds,
        }
