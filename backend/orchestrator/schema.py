from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    """The orchestrator's canonical view of an agent's status.

    Each agent still defines its own identical enum locally - they are
    independent services and share no Python code. This is the one the
    orchestrator parses agent responses into, so every node downstream of
    `worker_node` compares against a single class.
    """

    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


@dataclass
class AgentResult:
    """An agent's `POST /execute` response, parsed back into an object.

    Keeping this as a dataclass rather than letting the raw JSON dict flow
    into `WorkflowState` is what lets `router.py`, `resolver_node.py` and
    `answer_nodes.py` stay unaware that agents are now called over HTTP.
    """

    status: AgentStatus
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    output: Any | None = None


@dataclass
class OrchestratorInput:
    user_query: str
    initial_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorOutput:
    final_answer: str
    context: dict[str, Any] = field(default_factory=dict)
    execution_history: list[str] = field(default_factory=list)
