"""Competing reconstruction hypotheses and their deterministic scores.

Two rules shape everything here.

The LLM never produces a number. Every score below is computed by a service
from measurable evidence, so the same evidence always yields the same
confidence and a reviewer can reproduce it.

Ambiguity is preserved. When references disagree, that is a finding, not noise
to be averaged away - so each distinct proposal becomes its own candidate and
carries its own support, and the choice between them is made from scores, in
the open.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import CandidateOrigin, ConfidenceLevel


class ValidationVerdict(BaseModel):
    """The outcome of one deterministic biological check."""

    model_config = ConfigDict(frozen=True)

    check: str
    passed: bool
    #: Why, in terms a biologist would accept. Always populated on failure.
    reason: str = ""
    #: Contribution to the validation component, 0..1.
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    #: Whether failing this check disqualifies the candidate outright.
    #:
    #: Stated rather than inferred from a zero score. An advisory check can
    #: legitimately reach zero - a fill twice the length of its gap scores
    #: nothing on length - and treating that as disqualifying would silently
    #: reject candidates the check was only ever meant to caution about.
    fatal: bool = False


class CandidateScores(BaseModel):
    """The independent evidence signals behind one candidate confidence value.

    Kept separate rather than pre-combined so that a low final confidence can
    always be explained by which signal was weak, and so that re-weighting does
    not require re-running any external service.
    """

    model_config = ConfigDict(frozen=True)

    #: Identity, coverage and depth of the supporting homologues.
    homology: float = Field(default=0.0, ge=0.0, le=1.0)
    #: How well the alignment supports this exact fill.
    alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Agreement among spanning references across the gap columns.
    conservation: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Taxonomic closeness of the supporting organisms to the target.
    evolutionary: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Agreement with the Evo 2 continuation. None when Evo 2 was not
    #: consulted, which is the normal case - it is called to break a tie, not
    #: on every candidate. None and 0.0 mean different things, and the
    #: confidence engine must not confuse them.
    evo2: float | None = None
    #: Aggregate of the deterministic biological validators.
    validation: float = Field(default=1.0, ge=0.0, le=1.0)


class Candidate(BaseModel):
    """One proposed sequence for one gap, with everything that justifies it."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    gap_id: str
    #: Evidence or prediction. Read by the confidence engine, which scores the
    #: two on different terms, and surfaced in the response so a caller is
    #: never left to infer it from an empty supporting_hits list.
    origin: CandidateOrigin = CandidateOrigin.HOMOLOGY
    #: The proposed bases. Never contains an ambiguity character: an ambiguous
    #: base is not an answer to the question that was asked.
    sequence: str

    #: Accessions of the homologues proposing this exact fill.
    supporting_hits: tuple[str, ...] = ()
    #: Distinct organisms behind those hits.
    supporting_organisms: tuple[str, ...] = ()

    scores: CandidateScores = Field(default_factory=CandidateScores)
    validation_results: tuple[ValidationVerdict, ...] = ()

    #: Computed by the confidence engine from the scores above. Never set by
    #: a language model.
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW

    #: Why this candidate scored as it did, in plain language, for the response.
    rationale: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def support_count(self) -> int:
        return len(self.supporting_hits)

    @property
    def is_model_generated(self) -> bool:
        """True when no organism is known to carry this sequence."""
        return self.origin is CandidateOrigin.MODEL

    @property
    def passed_validation(self) -> bool:
        return all(verdict.passed for verdict in self.validation_results)

    def failed_checks(self) -> tuple[str, ...]:
        return tuple(v.check for v in self.validation_results if not v.passed)
