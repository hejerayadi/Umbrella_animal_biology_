"""What this agent sends back.

The orchestrator only ever sees the repo-wide `AgentResult` shape
(status/target_agent/prompt_to_target_agent/output). These models are what we
put *inside* `output`, so a caller that wants structure gets it while the
envelope stays exactly what `backend/orchestrator/schema.py` parses.

`result_builder` is what turns these into the envelope; nothing here imports
the orchestrator.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contracts.observation import Observation


class ReconstructionStatus(str, Enum):
    """How a single gap's reconstruction turned out."""

    RECONSTRUCTED = "reconstructed"
    # A candidate existed but scored below the confidence floor. Reported, not
    # silently dropped - a low-confidence answer is still evidence.
    LOW_CONFIDENCE = "low_confidence"
    # No usable reference alignment was found for this gap.
    UNRESOLVED = "unresolved"
    # Gap exceeded the configured length cap and was not attempted.
    SKIPPED = "skipped"


class EvidenceItem(BaseModel):
    """One traceable reason behind a reconstruction.

    Every reconstructed base must be attributable. This is what makes the
    output reviewable by a biologist instead of an opaque model guess.
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Tool or database, e.g. 'ncbi', 'blast', 'mafft'.")
    reference_id: str | None = Field(default=None, description="Accession backing this claim.")
    organism: str | None = None
    identity: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Alignment identity over the flanking context."
    )
    note: str | None = None


class GapReconstruction(BaseModel):
    """The outcome for one gap in the target sequence."""

    model_config = ConfigDict(frozen=True)

    gap_id: str
    start: int = Field(description="0-based inclusive start offset in the target sequence.")
    end: int = Field(description="0-based exclusive end offset in the target sequence.")
    length: int

    status: ReconstructionStatus
    reconstructed_sequence: str | None = Field(
        default=None, description="Proposed bases; None unless status is RECONSTRUCTED."
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation: str | None = Field(
        default=None, description="Why these bases, in language a biologist can check."
    )


class ReconstructionResult(BaseModel):
    """The full payload placed in `AgentResult.output`."""

    model_config = ConfigDict(frozen=True)

    sequence_id: str
    organism: str | None = None

    original_length: int
    # The target sequence with every RECONSTRUCTED gap filled in. Gaps left
    # UNRESOLVED/SKIPPED keep their original N runs, so length is preserved
    # and offsets stay comparable to the input.
    reconstructed_sequence: str | None = None

    gaps: list[GapReconstruction] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="", description="Human-readable account of what was done.")

    iterations: int = Field(default=0, description="Plan/act/critique cycles the graph ran.")
    tools_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    #: Why the loop stopped: all_gaps_resolved, abstained, budget_exhausted, ...
    #: Reported so a partial answer explains itself rather than looking truncated.
    stop_reason: str | None = None
    #: HTTP slices this run took. Greater than 1 means the agent yielded a
    #: CONTINUE and the orchestrator resumed it.
    slices: int = Field(default=1, ge=1)
    #: What the run consumed: tool calls, per-tool counts, LLM tokens.
    budget: dict[str, Any] = Field(default_factory=dict)
    #: Every tool attempt, including the ones that found nothing. The audit
    #: trail for "why did this gap come back unresolved?".
    observations: list[Observation] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def reconstructed_count(self) -> int:
        return sum(1 for gap in self.gaps if gap.status is ReconstructionStatus.RECONSTRUCTED)
