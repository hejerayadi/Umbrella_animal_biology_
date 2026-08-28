"""The run clock.

This agent answers inside a single HTTP call. The orchestrator allows 600
seconds and discards the whole response if that passes, so the run holds itself
to a much tighter budget and stops while it can still speak.

Expiry is therefore not an error. It is the moment to finalise with whatever
evidence is in hand: gaps already resolved are returned, gaps still open come
back UNRESOLVED with the reason recorded, and the orchestrator receives a real
partial answer instead of a timeout. A reserve is held back so that
finalisation itself always has room to run - scoring and serialising a result
after the clock has already run out would produce nothing at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class Deadline:
    """How much wall-clock time the run has left."""

    total_seconds: float
    #: Time held back for scoring, validation and serialisation.
    reserve_seconds: float = 30.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining(self) -> float:
        """Seconds before the hard deadline, never negative."""
        return max(self.total_seconds - self.elapsed, 0.0)

    def remaining_for_work(self) -> float:
        """Seconds available for tools, keeping the finalisation reserve intact.

        Everything that calls an external service asks for this rather than
        `remaining`, which is what guarantees there is still time to build a
        result when the last tool returns.
        """
        return max(self.remaining() - self.reserve_seconds, 0.0)

    def expired(self) -> bool:
        """Whether the working window has closed and finalisation must begin."""
        return self.remaining_for_work() <= 0.0

    def allows(self, seconds: float) -> bool:
        """Whether an operation expected to take `seconds` still fits."""
        return seconds <= self.remaining_for_work()

    def budget_for(self, requested: float) -> float:
        """`requested` seconds, clipped to what is actually left."""
        return min(requested, self.remaining_for_work())


@dataclass(frozen=True, slots=True)
class PhaseBudget:
    """How the working window is apportioned across the phases of a run.

    Advisory rather than enforced: a phase that finishes early leaves its
    unused time to the next one, because the deadline is the real constraint
    and a phase limit that idles the clock helps nobody.
    """

    homology_seconds: float = 200.0
    alignment_seconds: float = 40.0
    arbitration_seconds: float = 30.0

    def for_homology(self, deadline: Deadline) -> float:
        return deadline.budget_for(self.homology_seconds)

    def for_alignment(self, deadline: Deadline) -> float:
        return deadline.budget_for(self.alignment_seconds)

    def for_arbitration(self, deadline: Deadline) -> float:
        return deadline.budget_for(self.arbitration_seconds)
