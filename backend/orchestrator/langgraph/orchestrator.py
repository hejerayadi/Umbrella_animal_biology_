"""Public entry point for running the orchestrator graph."""
from __future__ import annotations

from typing import Any

from ..state import WorkflowState
from .graph import build_orchestrator_graph


class GlobalOrchestrator:
    """Public entry point: runs the compiled LangGraph workflow for a query.

    This is the single object the rest of the app (e.g. main.py, or an API
    endpoint) is meant to use. It hides all the graph-building details in
    `graph.py` behind one simple method: `run(user_query)`.
    """

    def __init__(self) -> None:
        # Build the graph once when the orchestrator is created, not on
        # every single query - building it is the "expensive" one-time setup.
        self._graph = build_orchestrator_graph()

    def run(
        self,
        user_query: str,
        initial_context: dict[str, Any] | None = None,
        *,
        has_image: bool = False,
    ) -> WorkflowState:
        """Execute the full plan -> worker -> resolver loop until COMPLETED or FAILED.

        `has_image` tells the planner an image was attached to this message.
        The image itself never enters the state - see `WorkflowState.has_image`.
        """

        # Start with a fresh clipboard: just the user's question and any
        # extra known facts (e.g. {"species": "woolly mammoth"}).
        initial_state = WorkflowState(
            user_query=user_query,
            context=dict(initial_context or {}),
            has_image=has_image,
        )

        # Hand it to LangGraph, which runs planner -> workers -> resolver ->
        # workers -> ... automatically until the graph reaches END.
        final = self._graph.invoke(initial_state)

        # Depending on the LangGraph version, the result may come back as a
        # plain dict instead of a WorkflowState object - normalize it here so
        # callers always get a real WorkflowState either way.
        return final if isinstance(final, WorkflowState) else WorkflowState(**final)
