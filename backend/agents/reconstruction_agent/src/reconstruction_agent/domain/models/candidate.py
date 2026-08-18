"""A proposed filling for one gap, with the evidence that produced it."""
from __future__ import annotations

from dataclasses import dataclass, field

from ...contracts.output import EvidenceItem


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposed reconstruction of one gap.

    Several candidates normally compete for the same gap - one per reference,
    or one per method. `candidate_ranker` picks between them; nothing here
    decides which wins.

    `confidence` stays unset (None) until scoring runs, so an unscored
    candidate cannot be mistaken for a zero-confidence one.
    """

    gap_id: str
    sequence: str
    method: str = "alignment_consensus"

    confidence: float | None = None
    # Fraction of contributing references that agreed on each base, averaged.
    # High identity with low support means one reference is carrying the
    # answer alone, which is weaker evidence than the identity suggests.
    support: float | None = None
    supporting_references: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    explanation: str | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence {self.confidence} is outside 0..1.")

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def is_scored(self) -> bool:
        return self.confidence is not None

    def scored(
        self,
        confidence: float,
        *,
        support: float | None = None,
        explanation: str | None = None,
    ) -> Candidate:
        """Copy of this candidate carrying a score."""
        return Candidate(
            gap_id=self.gap_id,
            sequence=self.sequence,
            method=self.method,
            confidence=confidence,
            support=support if support is not None else self.support,
            supporting_references=list(self.supporting_references),
            evidence=list(self.evidence),
            explanation=explanation or self.explanation,
            warnings=list(self.warnings),
        )
