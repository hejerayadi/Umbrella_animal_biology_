"""In-process publication of `AgentEvent`s.

Subscribers are optional and cheap: with none registered, emitting is a log
line and nothing more. That is what lets nodes emit freely without weighing up
whether anyone is listening.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..configuration.logging import get_logger
from ..contracts.events import AgentEvent, EventType

_log = get_logger(__name__)

Subscriber = Callable[[AgentEvent], None]


class EventEmitter:
    """Fan-out to registered subscribers.

    A failing subscriber is logged and skipped, never propagated: observability
    must not be able to fail a reconstruction.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def emit(
        self,
        event_type: EventType,
        run_id: str,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            type=event_type, run_id=run_id, message=message, data=data or {}
        )

        _log.info(message or event_type.value, extra={"run_id": run_id, "event": event_type.value})

        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as error:  # noqa: BLE001 - never fail a run for a listener
                _log.warning("Event subscriber raised %s; continuing.", error)

        return event


class CollectingEmitter(EventEmitter):
    """An emitter that also retains every event.

    Used by the tests and by `scripts/smoke_test.py` to assert on what a run
    actually did, rather than only on what it returned.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[AgentEvent] = []

    def emit(
        self,
        event_type: EventType,
        run_id: str,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> AgentEvent:
        event = super().emit(event_type, run_id, message, data)
        self.events.append(event)
        return event

    def of_type(self, event_type: EventType) -> list[AgentEvent]:
        return [event for event in self.events if event.type is event_type]
