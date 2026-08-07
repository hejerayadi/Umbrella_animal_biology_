"""State object for the toy LangGraph agent (Sprint 2 task 2).

The whole point of this graph is to prove LangGraph works end-to-end
before we plug it into the real biodiversity orchestrator. The state
therefore stays deliberately minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToyState:
    question: str
    answer: str = ""
    needs_escalation: bool = False
    escalation_target: Optional[str] = None
    parallel_result_a: str = ""
    parallel_result_b: str = ""
    merged_result: str = ""
