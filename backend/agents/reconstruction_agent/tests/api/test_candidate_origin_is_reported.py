"""A model prediction must be marked as one in the response payload.

The domain carries `origin` on every candidate and the confidence engine scores
the two kinds on different, capped paths. None of that helps a caller if the
field never reaches the wire - and for a while it did not: a live run filled a
real N-run with an Evo 2 continuation and reported it with `origin: None`.

`supporting_hits: []` is a hint a reader has to interpret. `origin: MODEL` is a
statement. These tests pin the statement.
"""

from __future__ import annotations

from reconstruction_agent.api.v1.mappers.reconstruction_mapper import to_agent_output, to_data
from reconstruction_agent.domain.enums import (
    CandidateOrigin,
    ConfidenceLevel,
    GapStatus,
)
from reconstruction_agent.domain.models.candidate import Candidate, CandidateScores
from reconstruction_agent.domain.models.result import GapReconstruction, ReconstructionResult
from reconstruction_agent.domain.models.sequence import Gap

_GAP = Gap(gap_id="gap_1", start=100, end=110)


def _candidate(origin: CandidateOrigin) -> Candidate:
    model = origin is CandidateOrigin.MODEL
    return Candidate(
        candidate_id="gap_1_cand_1",
        gap_id="gap_1",
        origin=origin,
        sequence="GGGAGGGGGC",
        supporting_hits=() if model else ("NC_003428", "AP012595"),
        supporting_organisms=() if model else ("Ursus maritimus",),
        scores=CandidateScores(evo2=0.64) if model else CandidateScores(homology=0.95),
        final_confidence=0.48 if model else 0.95,
        confidence_level=ConfidenceLevel.MEDIUM if model else ConfidenceLevel.HIGH,
        rationale="an Evo 2 prediction" if model else "twelve references agree",
    )


def _result(origin: CandidateOrigin) -> ReconstructionResult:
    return ReconstructionResult(
        request_id="rec_test",
        sequence_accession="NW_007907101.1",
        reconstructions=(
            GapReconstruction(
                gap=_GAP, status=GapStatus.RESOLVED, selected_candidate=_candidate(origin)
            ),
        ),
    )


def _reported(origin: CandidateOrigin) -> dict[str, object]:
    payload = to_data(_result(origin))
    candidate = payload["reconstructions"][0]["selected_candidate"]  # type: ignore[index]
    assert isinstance(candidate, dict)
    return candidate


class TestOriginReachesTheWire:
    def test_a_model_fill_is_marked_as_a_prediction(self) -> None:
        candidate = _reported(CandidateOrigin.MODEL)

        assert candidate["origin"] == "MODEL"
        assert candidate["is_model_generated"] is True

    def test_an_observed_fill_is_marked_as_one(self) -> None:
        candidate = _reported(CandidateOrigin.HOMOLOGY)

        assert candidate["origin"] == "HOMOLOGY"
        assert candidate["is_model_generated"] is False

    def test_origin_is_never_absent_or_null(self) -> None:
        """The defect: the field was dropped entirely, and a prediction arrived
        indistinguishable from an observation."""
        for origin in CandidateOrigin:
            candidate = _reported(origin)
            assert candidate.get("origin"), f"{origin} reported no origin"


class TestTheOrchestratorContextCarriesItToo:
    def test_the_shared_context_payload_marks_a_prediction(self) -> None:
        """Other agents read `reconstruction` out of the shared context; a
        prediction must not reach them looking like measured evidence."""
        output = to_agent_output(_result(CandidateOrigin.MODEL))
        reconstructions = output["reconstruction"]["reconstructions"]  # type: ignore[index]
        candidate = reconstructions[0]["selected_candidate"]

        assert candidate["origin"] == "MODEL"

    def test_the_rationale_still_says_it_in_words(self) -> None:
        """For a caller reading the summary rather than the schema."""
        candidate = _reported(CandidateOrigin.MODEL)
        assert "prediction" in str(candidate["rationale"]).lower()
