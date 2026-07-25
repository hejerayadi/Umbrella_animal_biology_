from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    """Mutable state shared across the orchestrator workflow.

    This object acts as the in-memory record of the user's request and the
    intermediate facts gathered by each agent. Individual fields store the
    domain-specific outputs that agents may produce, while `execution_history`
    records the order of agent activity for traceability and debugging.
    """

    # The original user request that started the workflow.
    user_query: str = ""

    # Optional biological context that downstream agents can refine or reuse.
    species: str | None = None
    image: str | None = None
    genome: str | None = None

    # Collection fields are created per instance so concurrent workflows do
    # not accidentally share mutable defaults.
    genes: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    protein: str | None = None
    papers: list[str] = field(default_factory=list)
    biodiversity: dict[str, Any] = field(default_factory=dict)

    # Keeps a simple chronological trace of agent decisions and results.
    execution_history: list[str] = field(default_factory=list)

    def add_history(self, entry: str) -> None:
        """Append a single workflow event to the execution trace."""
        self.execution_history.append(entry)
