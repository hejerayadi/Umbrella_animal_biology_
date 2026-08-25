"""Drives one slice of the agent graph.

Sits between the service (which knows about requests and results) and the graph
(which knows about state), so neither has to know about the other.

The unit of work here is a **slice**, not a run. The orchestrator allows 600 s
per HTTP call and retries a CONTINUE three times, so one reconstruction may span
four slices. Each is a fresh `/execute` call that resumes the checkpoint written
by the previous one, keyed by the orchestrator's trace id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agent.graph.builder import build_graph, build_nodes
from agent.graph.conditions import has_work, should_continue
from agent.state.state import ReconstructionState, initial_state, is_last_slice
from configuration.logging import get_logger
from configuration.settings import Settings
from contracts.events import EventType
from domain.models import Sequence
from observability.events import EventEmitter
from tools.registry import ToolRegistry

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SliceOutcome:
    """What one slice produced, and whether the run is over.

    `finished` False is what becomes a CONTINUE for the orchestrator. The
    distinction is not derivable from the state alone - a run that stopped
    because its wall clock expired looks much like one that stopped because it
    resolved everything, and only this says which.
    """

    state: ReconstructionState
    finished: bool
    continuation_reason: str | None = None


class AgentRunner:
    """Runs one slice of the reconstruction graph."""

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        events: EventEmitter | None = None,
        *,
        checkpointer: Any | None = None,
        databases: Any | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._events = events or EventEmitter()
        self._checkpointer = checkpointer
        self._graph: Any | None = None
        # `databases` resolves which BLAST database to search. Optional: with
        # none, the planner's own choice stands alone and a run that cannot
        # name one reports that rather than guessing a taxonomic division.
        self._nodes = build_nodes(settings, registry, self._events, databases=databases)

    def _compiled(self) -> Any:
        if self._graph is None:
            self._graph = build_graph(
                self._settings,
                self._registry,
                self._events,
                checkpointer=self._checkpointer,
                # The same instance the fallback path uses. Two sets would mean
                # two LLM clients and two of every collaborator, and anything
                # patched onto one would be invisible to the other.
                nodes=self._nodes,
            )
        return self._graph

    async def run_slice(
        self,
        *,
        run_id: str,
        trace_id: str,
        instruction: str,
        target: Sequence,
        organism: str | None,
        requested_organisms: list[str],
    ) -> SliceOutcome:
        """Execute one slice, resuming an earlier one when a checkpoint exists."""
        graph = self._compiled()
        # The thread id is the orchestrator's trace id, not this call's run id:
        # it is the only identifier stable across CONTINUE retries, and keying
        # on run_id would start every slice from nothing.
        config = {"configurable": {"thread_id": trace_id}}

        resumed = await self._resume(graph, config)

        if resumed is None:
            state = initial_state(
                run_id=run_id,
                trace_id=trace_id,
                instruction=instruction,
                target=target,
                organism=organism,
                requested_organisms=requested_organisms,
                max_iterations=self._settings.max_iterations,
                max_slices=self._settings.continuation.max_slices,
            )
            self._events.emit(
                EventType.RUN_STARTED,
                run_id,
                f"Reconstructing {target.identifier} ({len(target)} bases).",
                {"sequence_id": target.identifier, "length": len(target)},
            )
        else:
            # Carry this call's run id in so the slice's own logs correlate,
            # while everything gathered before is preserved.
            state = cast(ReconstructionState, {**resumed, "run_id": run_id})
            _log.info(
                "run_resumed",
                trace_id=trace_id,
                slice_index=state.get("slice_index", 0),
                resolved=len(state.get("reconstructions") or {}),
            )

        try:
            final = cast(ReconstructionState, await graph.ainvoke(state, config=config))
        except ImportError:
            # LangGraph absent: drive the same nodes by hand so the agent still
            # works in a minimal environment.
            _log.warning("langgraph_unavailable", detail="Running the nodes directly.")
            final = await self._run_without_langgraph(state)
        except Exception as error:  # noqa: BLE001 - surfaced as a FAILED result
            self._events.emit(EventType.RUN_FAILED, run_id, str(error))
            raise

        return self._outcome(final)

    async def _resume(self, graph: Any, config: dict[str, Any]) -> ReconstructionState | None:
        """The checkpointed state for this trace id, if any.

        Returns None on any failure rather than raising: an unreadable
        checkpoint should cost a restart, not the whole request.
        """
        if self._checkpointer is None:
            return None

        try:
            snapshot = await graph.aget_state(config)
        except Exception as error:  # noqa: BLE001
            _log.warning("checkpoint_read_failed", error=str(error))
            return None

        values = getattr(snapshot, "values", None)
        if not values or not values.get("target"):
            return None
        return cast(ReconstructionState, dict(values))

    def _outcome(self, state: ReconstructionState) -> SliceOutcome:
        """Whether the run is over, and what to tell the orchestrator if not.

        A yield is only honoured when another slice is actually available. On
        the last one the orchestrator converts a CONTINUE into FAILED and
        discards everything gathered, so the agent finishes with partial
        results instead.
        """
        if not state.get("yielded"):
            return SliceOutcome(state=state, finished=True)

        if is_last_slice(state):
            _log.info(
                "yield_suppressed",
                detail="Last slice; returning partial results rather than CONTINUE.",
            )
            return SliceOutcome(state=state, finished=True)

        contexts = state.get("gap_contexts") or []
        resolved = set(state.get("reconstructions") or {})
        skipped = set(state.get("skipped") or {})
        outstanding = [
            context.identifier
            for context in contexts
            if context.identifier not in resolved and context.identifier not in skipped
        ]

        return SliceOutcome(
            state=state,
            finished=False,
            continuation_reason=(
                f"Reconstruction in progress: {len(resolved)} of {len(contexts)} gap(s) "
                f"resolved, still working on {', '.join(outstanding[:5]) or 'remaining gaps'}."
            ),
        )

    async def _run_without_langgraph(self, state: ReconstructionState) -> ReconstructionState:
        """The same loop as the graph, driven imperatively.

        Kept faithful to `graph/edges.py` on purpose: it exists so a missing
        optional dependency degrades the deployment, not the science. If the
        graph's topology changes, this changes with it.
        """
        state = _merge(state, await self._nodes.load_or_init(state))
        state = _merge(state, await self._nodes.detect_gaps(state))

        if not has_work(state):
            return _merge(state, await self._nodes.finalize(state))

        while should_continue(state):
            for step in (
                self._nodes.plan,
                self._nodes.select_tools,
                self._nodes.execute_tools,
                self._nodes.observe,
                self._nodes.reason,
                self._nodes.validate,
                self._nodes.critique,
                self._nodes.decide,
            ):
                state = _merge(state, await step(state))

        return _merge(state, await self._nodes.finalize(state))


def _merge(state: ReconstructionState, update: ReconstructionState) -> ReconstructionState:
    """Apply a node's partial update, mirroring the state's reducers.

    Dict fields merge and list fields append, exactly as the `Annotated`
    reducers in `state.py` specify - otherwise the fallback path would lose
    evidence the graph path keeps.
    """
    merged: dict[str, Any] = dict(state)

    for key, value in update.items():
        if key in ("observations", "tool_calls", "critiques", "warnings", "errors") and isinstance(
            value, list
        ):
            merged[key] = list(merged.get(key) or []) + value
        elif key in ("plan", "pending_invocations", "gap_contexts"):
            # `replace` reducers: a later value is a correction, not an addition.
            merged[key] = value
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return cast(ReconstructionState, merged)
