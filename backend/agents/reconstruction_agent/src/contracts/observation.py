"""What came back from running one tool.

The missing step in the old loop: tool outputs were merged into state
anonymously, so nothing downstream could answer "what did we actually try, and
what did it cost?". An `Observation` is that record - one per attempt,
including failures and retries, because a tool that returned nothing is
evidence about the gap just as much as one that returned hits.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ObservationStatus(str, Enum):
    OK = "ok"
    #: The tool ran and reported failure - a service error, a rejected query.
    FAILED = "failed"
    #: The tool ran cleanly but produced nothing usable: BLAST with no hits, an
    #: alignment that does not span the gap. Not an error, but not progress.
    EMPTY = "empty"
    #: Refused before running because a budget was spent.
    SKIPPED = "skipped"


class Observation(BaseModel):
    """One tool attempt, with enough detail to explain the run afterwards."""

    model_config = ConfigDict(frozen=True)

    tool: str
    gap_id: str | None = None
    status: ObservationStatus
    #: Which attempt this was for this tool/gap pair. A semantic failure is
    #: retried once with relaxed parameters; both attempts are recorded.
    attempt: int = 1
    #: Which plan/act/critique round produced this, 0-based. The stop policy
    #: needs "did the *last* round achieve anything?", and accumulated state
    #: cannot answer that - once any round succeeds, a state-wide check stays
    #: true for the rest of the run.
    iteration: int = 0
    #: True when this attempt ran with relaxed parameters after an earlier one
    #: found nothing. Recorded so a retry is distinguishable from a first try
    #: in the audit trail.
    relaxed: bool = False
    duration_seconds: float = 0.0
    #: How much new evidence this produced - references found, rows aligned.
    evidence_added: int = 0
    detail: str | None = None
    #: Tool-specific diagnostics, for the run log only. Never control flow.
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def made_progress(self) -> bool:
        return self.status is ObservationStatus.OK and self.evidence_added > 0
