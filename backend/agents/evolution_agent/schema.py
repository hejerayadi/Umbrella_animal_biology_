from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


@dataclass
class AgentRequest:
    instruction: str
    context: dict[str, Any]


@dataclass
class AgentResult:

    status: AgentStatus

    target_agent: str | None = None

    prompt_to_target_agent: str | None = None

    output: Any | None = None