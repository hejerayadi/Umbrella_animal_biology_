"""Inter-agent result returned by the Protein Agent to the Grand Orchestrator.

``AgentStatus`` is a routing status.  The scientific workflow status remains in
``output["status"]`` so the Grand Orchestrator does not confuse a usable partial
analysis with an unfinished agent hand-off.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AgentResult:
    status: AgentStatus
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    continuation_reason: str | None = None
    retryable: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is AgentStatus.NEEDS_AGENT:
            if not self.target_agent or not self.prompt_to_target_agent:
                raise ValueError("NEEDS_AGENT requires target_agent and prompt_to_target_agent")
        elif self.target_agent is not None or self.prompt_to_target_agent is not None:
            raise ValueError("Only NEEDS_AGENT may target another agent")

        if self.status is AgentStatus.CONTINUE and not self.continuation_reason:
            raise ValueError("CONTINUE requires continuation_reason")
        if self.status is AgentStatus.FAILED and not self.error:
            raise ValueError("FAILED requires error")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation for scripts and message buses."""
        return {
            "status": self.status.value,
            "target_agent": self.target_agent,
            "prompt_to_target_agent": self.prompt_to_target_agent,
            "output": self.output,
            "continuation_reason": self.continuation_reason,
            "retryable": self.retryable,
            "error": self.error,
        }
