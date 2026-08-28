"""The finished answer: per gap, and for the request as a whole.

Every reconstruction carries its provenance. A returned sequence that cannot
say where it came from is indistinguishable from a fabricated one, and the
whole point of this agent is that the difference is checkable.

Refusal is a first-class outcome. An unresolved gap keeps its coordinates,
names its reason, and returns no sequence at all - which is a better answer
than a confident guess, and is why `UNRESOLVED` sits alongside `RESOLVED`
rather than under an error.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import (
    GapStatus,
    ReconstructionStatus,
    UnresolvedReason,
)
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.domain.models.taxonomy import TargetProfile


class Provenance(BaseModel):
    """Where the evidence for one gap came from."""

    model_config = ConfigDict(frozen=True)

    #: External services actually consulted, e.g. NCBI, EMBL-EBI, NVIDIA.
    providers: tuple[str, ...] = ()
    #: Reference collections searched, as opaque provider identifiers. Recorded
    #: for audit; no code reads these values back.
    databases_searched: tuple[str, ...] = ()
    #: Why those collections were chosen, in plain language. This is the
    #: audit trail for the selection decision, so a wrong choice is visible
    #: after the fact instead of having to be inferred from poor results.
    database_rationale: str = ""
    tool_calls: int = 0


class HomologyEvidence(BaseModel):
    """Summary of the homology search behind one gap."""

    model_config = ConfigDict(frozen=True)

    hits_examined: int = 0
    gap_spanning_hits: int = 0
    best_identity: float = 0.0
    closest_organism: str | None = None


class AlignmentEvidence(BaseModel):
    """Summary of what the alignment showed for one gap."""

    model_config = ConfigDict(frozen=True)

    references_aligned: int = 0
    references_spanning_gap: int = 0
    conservation: float = 0.0
    competing_fills: int = 0


class Evo2Evidence(BaseModel):
    """Summary of the Evo 2 arbitration, when one took place."""

    model_config = ConfigDict(frozen=True)

    consulted: bool = False
    #: Agreement between the candidate and the Evo 2 continuation, 0..1.
    agreement: float | None = None
    mean_model_confidence: float | None = None
    #: Populated when Evo 2 was wanted but unavailable; the run continues
    #: without it rather than failing, so this records why it is missing.
    unavailable_reason: str | None = None


class GapEvidence(BaseModel):
    """Everything that was learned about one gap."""

    model_config = ConfigDict(frozen=True)

    homology: HomologyEvidence = Field(default_factory=HomologyEvidence)
    alignment: AlignmentEvidence = Field(default_factory=AlignmentEvidence)
    evo2: Evo2Evidence = Field(default_factory=Evo2Evidence)
    validation_checks: tuple[str, ...] = ()
    validation_failures: tuple[str, ...] = ()


class GapReconstruction(BaseModel):
    """The outcome for one requested gap."""

    model_config = ConfigDict(frozen=True)

    gap: Gap
    status: GapStatus

    #: Present only when status is RESOLVED.
    selected_candidate: Candidate | None = None
    #: Runners-up, kept so a reviewer can see what was rejected and why.
    alternatives: tuple[Candidate, ...] = ()

    #: Required whenever the status is not RESOLVED.
    unresolved_reason: UnresolvedReason | None = None
    explanation: str = ""

    evidence: GapEvidence = Field(default_factory=GapEvidence)
    provenance: Provenance = Field(default_factory=Provenance)
    warnings: tuple[str, ...] = ()

    @property
    def gap_id(self) -> str:
        return self.gap.gap_id

    @property
    def is_resolved(self) -> bool:
        return self.status is GapStatus.RESOLVED and self.selected_candidate is not None

    @property
    def sequence(self) -> str | None:
        return self.selected_candidate.sequence if self.selected_candidate else None

    @property
    def confidence(self) -> float | None:
        return self.selected_candidate.final_confidence if self.selected_candidate else None


class ReconstructionResult(BaseModel):
    """The answer to one reconstruction request."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    sequence_accession: str | None = None
    assembly_id: str | None = None
    target_profile: TargetProfile | None = None

    reconstructions: tuple[GapReconstruction, ...] = ()
    warnings: tuple[str, ...] = ()
    #: Non-fatal failures worth reporting, e.g. an optional provider that was
    #: unreachable. A run that produced results despite these is not a failure.
    errors: tuple[str, ...] = ()

    iteration_count: int = 0

    @property
    def requested_gaps(self) -> int:
        return len(self.reconstructions)

    @property
    def resolved_gaps(self) -> int:
        return sum(1 for item in self.reconstructions if item.is_resolved)

    @property
    def unresolved_gaps(self) -> int:
        return self.requested_gaps - self.resolved_gaps

    @property
    def status(self) -> ReconstructionStatus:
        """The overall status, derived rather than stored.

        Deriving it means it can never disagree with the per-gap outcomes it
        summarises. Note that a partly successful run reports
        PARTIALLY_COMPLETED: eight reconstructions out of ten is a result, and
        reporting it as a failure would discard all eight.
        """
        if not self.reconstructions:
            return ReconstructionStatus.UNRESOLVED
        if self.resolved_gaps == self.requested_gaps:
            return ReconstructionStatus.COMPLETED
        if self.resolved_gaps == 0:
            return ReconstructionStatus.UNRESOLVED
        return ReconstructionStatus.PARTIALLY_COMPLETED

    def summary_line(self) -> str:
        """One sentence for the orchestrator to hand to the responder."""
        if not self.reconstructions:
            return "No unresolved regions were reconstructed."
        organism = self.target_profile.scientific_name if self.target_profile else "the target"
        return (
            f"Reconstructed {self.resolved_gaps} of {self.requested_gaps} "
            f"unresolved regions in {organism}."
        )
