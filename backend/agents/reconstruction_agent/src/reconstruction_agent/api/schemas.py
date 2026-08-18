"""The HTTP wire shapes.

`AgentRequest` and `AgentResult` reproduce the repo-wide agent contract that
`backend/orchestrator/schema.py` parses. They are duplicated here rather than
imported because this agent runs in its own `.venv` and must not depend on the
`backend` package - that isolation is what lets it pin its own dependencies.

Changing these means changing the orchestrator too.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Mirrors `backend/orchestrator/schema.py`."""

    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


class AgentRequest(BaseModel):
    """What the orchestrator POSTs to /execute."""

    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """What every /execute call answers with, success or failure.

    Deliberately the only response shape: the orchestrator's router expects one
    schema back every time and already handles a FAILED status, so errors are
    reported in this envelope rather than as an HTTP error body.
    """

    status: AgentStatus
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    output: Any | None = None


class HealthResponse(BaseModel):
    """Liveness and configuration report."""

    status: str = "ok"
    agent: str = "reconstruction_agent"
    version: str = "0.1.0"
    # What the agent can actually do right now, so a misconfigured deployment
    # is visible before a query fails rather than after.
    tools: list[str] = Field(default_factory=list)
    llm_enabled: bool = False
    external_services_configured: dict[str, bool] = Field(default_factory=dict)
