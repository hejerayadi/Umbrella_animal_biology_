from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
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
    output: Any = None
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
