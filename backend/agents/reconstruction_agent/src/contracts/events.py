"""Progress events emitted while a reconstruction runs.

A reconstruction can take minutes - BLAST and MAFFT are submit-then-poll jobs.
These events let the graph report what it is doing without the API layer
reaching into agent internals to find out.

Consumers subscribe through `observability.events`. Emitting is fire-and-forget:
no consumer, no cost, and a failing consumer never breaks a run.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    GAPS_DETECTED = "gaps_detected"
    PLAN_CREATED = "plan_created"

    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    CANDIDATE_PROPOSED = "candidate_proposed"
    CRITIQUE_ISSUED = "critique_issued"
    ITERATION_COMPLETED = "iteration_completed"


class AgentEvent(BaseModel):
    """One point-in-time fact about a run."""

    model_config = ConfigDict(frozen=True)

    type: EventType
    run_id: str
    message: str = ""
    # Event-specific detail. Kept loose on purpose: these are for humans and
    # dashboards, never for control flow.
    data: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
