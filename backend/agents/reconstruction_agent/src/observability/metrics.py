"""Counters and timings for one run.

Intentionally in-process and dependency-free. The agent is a stateless worker
behind the orchestrator, so there is no metrics backend to own here; what
matters is that each run can report what it cost, which lands in the result's
diagnostics and the logs.
"""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    """What one reconstruction run consumed."""

    run_id: str
    started_at: float = field(default_factory=time.monotonic)

    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def record(self, name: str, seconds: float) -> None:
        """Accumulate a duration. Repeated names sum, so five BLAST calls
        report their total rather than only the last one."""
        self.timings[name] += seconds

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        """Time a block, recording it even when the block raises.

        External calls fail often enough that excluding failures would make the
        timings systematically understate what a run really cost.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - started)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "counters": dict(self.counters),
            "timings": {name: round(value, 3) for name, value in self.timings.items()},
        }
