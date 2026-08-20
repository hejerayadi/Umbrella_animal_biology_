from enum import Enum


class AgentStatus(Enum):
    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"