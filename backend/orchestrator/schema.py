from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorInput:
    user_query: str
    initial_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorOutput:
    final_answer: str
    context: dict[str, Any] = field(default_factory=dict)
    execution_history: list[str] = field(default_factory=list)
