"""The Umbrella orchestrator contract, duplicated here on purpose.

Every agent in this repository defines these three types locally rather than
importing them from `backend/`. That is deliberate: each agent runs in its own
virtual environment, and importing shared Python would make one agent unable to
start because another one changed a dependency.

The consequence is that these names are a wire contract, not a shared class.
The orchestrator compares `status` by its string value, so the values below may
never be renamed, and every field it reads must keep its exact spelling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """What the orchestrator should do next.

    This is a routing decision, not a scientific one. A run that reconstructed
    nothing because the evidence did not support it is still COMPLETED - the
    agent did its job and the answer was "no". FAILED means the agent could not
    do its job at all, and the orchestrator reports a breakdown to the user.
    """

    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


class AgentRequest(BaseModel):
    """What the orchestrator sends.

    `context` is shared across the whole run and every agent contributes to it,
    so it is read defensively and never assumed to hold any particular key.
    """

    instruction: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """What the orchestrator parses back, whatever happened.

    Always returned with HTTP 200. The orchestrator expects exactly one schema
    and already knows how to handle a FAILED status; an HTTP error body would
    break that and turn a reported failure into an unparseable response.
    """

    status: AgentStatus
    #: Merged into the shared context when it is a dict, so the key names here
    #: are a cross-agent contract and must match `card.json`.
    output: Any | None = None
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    continuation_reason: str | None = None
    #: A CONTINUE without this is converted straight to FAILED by the router.
    retryable: bool = False
    error: str | None = None
