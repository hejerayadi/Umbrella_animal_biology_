"""Live progress events emitted while the orchestrator graph runs.

The graph is synchronous: `GlobalOrchestrator.run()` returns only once the
whole planner -> workers -> responder loop has finished, which for a real
research question is anywhere from ten seconds to several minutes. Until this
module existed, the UI learnt what happened from `execution_history` in that
final reply - so the user watched a spinner for four minutes and then received
the entire story at once, after it stopped being useful.

Nodes call `step_started` / `step_finished` / `thought` / `answer_delta` as
they work. Whether anyone is listening is not their problem: with no emitter
bound - every existing caller, every test, `python -m backend.main` - each of
these is a no-op and the graph behaves exactly as before.

The emitter is held in a `ContextVar` rather than threaded through every node
signature. The graph builds its nodes once at import time and LangGraph owns
the call stack in between, so there is no parameter to pass an emitter down;
a context variable is the one channel that reaches a node without changing the
shape of the graph. It is set per run (see `emitting_to`), and because
`asyncio.to_thread` hands the worker thread a *copy* of the context, two
concurrent chat requests cannot see each other's sink.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_logger = logging.getLogger(__name__)

# What a listener receives: one JSON-serialisable dict per event.
Emitter = Callable[[dict[str, Any]], None]

_emitter: ContextVar[Emitter | None] = ContextVar("orchestrator_emitter", default=None)

# Numbers each step within one run, so an agent that runs twice produces two
# rows in the UI instead of the second one overwriting the first.
_step_ids: ContextVar[Iterator[int] | None] = ContextVar("orchestrator_step_ids", default=None)


@contextmanager
def emitting_to(sink: Emitter) -> Iterator[None]:
    """Send every event raised inside this block to `sink`.

    `sink` is called from whatever thread the graph runs on, so it has to be
    cheap and thread-safe - putting the event on a queue, not writing to a
    socket.
    """

    emitter_token = _emitter.set(sink)
    counter_token = _step_ids.set(itertools.count(1))
    try:
        yield
    finally:
        _emitter.reset(emitter_token)
        _step_ids.reset(counter_token)


def _emit(event: dict[str, Any]) -> None:
    sink = _emitter.get()
    if sink is None:
        return
    try:
        sink(event)
    except Exception:  # pragma: no cover - a broken listener must not stop the run
        # A dead listener is a disconnected browser tab, which is normal. The
        # research work is worth more than the commentary on it, so the graph
        # carries on and the event is dropped.
        _logger.debug("Dropped orchestrator event %r", event.get("type"), exc_info=True)


def step_started(agent: str, text: str) -> str:
    """Announce a step that is now running. Returns its id.

    Hold on to the id and pass it to `step_finished`, so the UI updates that
    line in place rather than appending a second one.
    """

    counter = _step_ids.get()
    step_id = f"step-{next(counter)}" if counter is not None else "step-0"
    _emit({"type": "step", "id": step_id, "agent": agent, "text": text, "state": "running"})
    return step_id


def step_finished(step_id: str, agent: str, text: str, *, failed: bool = False) -> None:
    """Mark a started step as over, replacing its text with the outcome."""

    _emit(
        {
            "type": "step",
            "id": step_id,
            "agent": agent,
            "text": text,
            "state": "failed" if failed else "done",
        }
    )


def thought(step_id: str, text: str) -> None:
    """Attach the model's own words about a decision to the step that made it.

    Kept separate from the step text because it is the LLM speaking, not the
    orchestrator describing itself - the UI renders it in the quieter voice.
    """

    if not text:
        return
    _emit({"type": "thought", "id": step_id, "text": text})


def answer_delta(delta: str) -> None:
    """One chunk of the final answer, as the responder writes it."""

    if not delta:
        return
    _emit({"type": "answer", "delta": delta})


def agent_label(agent_name: str) -> str:
    """The registry name of an agent, spelled for a human reader.

    Registry keys are single words so they can be node names and dict keys
    ("ImageGeneration"); the user should still read "Image Generation agent".
    """

    spaced = "".join(
        f" {char}" if index and char.isupper() and not agent_name[index - 1].isupper() else char
        for index, char in enumerate(agent_name)
    )
    return f"{spaced.strip()} agent"
