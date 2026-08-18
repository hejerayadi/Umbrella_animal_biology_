"""Drives one pass of the agent graph.

Sits between the service (which knows about requests and results) and the
graph (which knows about state), so neither has to know about the other.
"""
from __future__ import annotations

from typing import Any, cast

from ..agent.graph.builder import build_graph, build_nodes
from ..agent.graph.conditions import has_work, should_continue
from ..agent.state.state import ReconstructionState, initial_state
from ..configuration.logging import get_logger
from ..configuration.settings import Settings
from ..contracts.events import EventType
from ..domain.models import Sequence
from ..observability.events import EventEmitter
from ..tools.registry import ToolRegistry

_log = get_logger(__name__)


class AgentRunner:
    """Runs the reconstruction graph for one request.

    Compiles the graph once and reuses it: compilation is not free, and the
    graph holds no per-run state - that all lives in `ReconstructionState`.
    """

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        events: EventEmitter | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._events = events or EventEmitter()
        self._graph: Any | None = None
        self._nodes = build_nodes(settings, registry, self._events)

    def _compiled(self) -> Any:
        if self._graph is None:
            self._graph = build_graph(self._settings, self._registry, self._events)
        return self._graph

    async def run(
        self,
        *,
        run_id: str,
        instruction: str,
        target: Sequence,
        organism: str | None,
        requested_organisms: list[str],
        max_iterations: int | None = None,
    ) -> ReconstructionState:
        """Execute the graph and return its final state."""
        state = initial_state(
            run_id=run_id,
            instruction=instruction,
            target=target,
            organism=organism,
            requested_organisms=requested_organisms,
            max_iterations=max_iterations or self._settings.max_iterations,
        )

        self._events.emit(
            EventType.RUN_STARTED,
            run_id,
            f"Reconstructing {target.identifier} ({len(target)} bases).",
            {"sequence_id": target.identifier, "length": len(target)},
        )

        try:
            final = await self._compiled().ainvoke(
                state,
                # The thread id scopes the checkpointer to this run; without it
                # concurrent runs would share checkpoint history.
                config={"configurable": {"thread_id": run_id}},
            )
            self._events.emit(EventType.RUN_COMPLETED, run_id, "Run finished.")
            return cast(ReconstructionState, final)

        except ImportError:
            # LangGraph absent: fall back to driving the same nodes by hand so
            # the agent still works in a minimal environment.
            _log.warning("LangGraph unavailable; running the nodes directly.")
            return await self._run_without_langgraph(state)

        except Exception as error:  # noqa: BLE001 - surfaced as a FAILED result
            self._events.emit(EventType.RUN_FAILED, run_id, str(error))
            raise

    async def _run_without_langgraph(self, state: ReconstructionState) -> ReconstructionState:
        """The same loop as the graph, driven imperatively.

        Kept faithful to `graph/edges.py` on purpose: it exists so a missing
        optional dependency degrades the deployment, not the science. If the
        graph's topology changes, this changes with it.
        """
        state = cast(ReconstructionState, {**state, **await self._nodes.detect_gaps(state)})

        if not has_work(state):
            return state

        while should_continue(state):
            for step in (
                self._nodes.plan,
                self._nodes.execute_tools,
                self._nodes.reason,
                self._nodes.critique,
            ):
                update = await step(state)
                state = _merge(state, update)

        return state


def _merge(state: ReconstructionState, update: ReconstructionState) -> ReconstructionState:
    """Apply a node's partial update, mirroring the state's reducers.

    Dict fields merge and list fields append, exactly as the `Annotated`
    reducers in `state.py` specify - otherwise the fallback path would lose
    evidence the graph path keeps.
    """
    merged: dict[str, Any] = dict(state)

    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        elif key in ("tool_calls", "critiques", "warnings", "errors") and isinstance(value, list):
            merged[key] = list(merged.get(key) or []) + value
        else:
            merged[key] = value

    return cast(ReconstructionState, merged)
