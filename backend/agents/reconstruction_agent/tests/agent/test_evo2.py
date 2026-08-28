"""Arbitration: when it runs, what it measures, and what it must never do.

Evo 2 has no endpoint that scores an existing sequence, so the agreement score
is a comparison against the model's own continuation. Everything here protects
the consequences of that: an agreement is an opinion, it is weighted least, it
is asked for only when there is a tie, and its absence is never a failure.

The single most important assertion in the file is that a computed agreement
survives into the confidence. A result calculated and then dropped costs a call
and looks exactly like a call that was never made.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.domain.enums import ConfidenceLevel, ErrorCode
from reconstruction_agent.domain.exceptions import ExternalServiceError
from reconstruction_agent.domain.models.candidate import Candidate, CandidateScores
from reconstruction_agent.domain.models.sequence import Gap, GapContext
from reconstruction_agent.integrations.evo2.models import Evo2Generation
from reconstruction_agent.services.plausibility.evo2_service import (
    Evo2Service,
    agreement_score,
)
from reconstruction_agent.services.plausibility.plausibility_policy import ArbitrationPolicy
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.candidate.score_candidate import (
    ScoreCandidateInput,
    ScoreCandidateTool,
)
from reconstruction_agent.tools.plausibility.evaluate_with_evo2 import (
    EvaluateWithEvo2Tool,
    Evo2Input,
)

_FILL = "ACGTACGTAC"
_GAP = Gap(gap_id="gap_1", start=20, end=30)
_CONTEXT = GapContext(gap=_GAP, left_flank="ACGT" * 30, right_flank="TGCA" * 30)


class _FakeClient:
    """Stands in for NIM, with a scripted continuation."""

    def __init__(
        self,
        generation: Evo2Generation | None = None,
        *,
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self._generation = generation or Evo2Generation()
        self._available = available
        self._error = error
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    async def generate(self, prompt: str, *, num_tokens: int) -> Evo2Generation:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._generation


def _candidate(candidate_id: str, sequence: str, confidence: float) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        gap_id="gap_1",
        sequence=sequence,
        scores=CandidateScores(homology=confidence, alignment=confidence),
        final_confidence=confidence,
        confidence_level=ConfidenceLevel.MEDIUM,
    )


class TestTheAgreementScore:
    def test_an_exact_match_scores_one(self) -> None:
        assert agreement_score(_FILL, Evo2Generation(sequence=_FILL)) == 1.0

    def test_a_complete_mismatch_scores_zero(self) -> None:
        assert agreement_score("AAAAAAAAAA", Evo2Generation(sequence="CCCCCCCCCC")) == 0.0

    def test_it_is_monotonic_in_the_number_of_matching_bases(self) -> None:
        near = agreement_score("ACGTACGTAA", Evo2Generation(sequence=_FILL))
        far = agreement_score("ACGTAAAAAA", Evo2Generation(sequence=_FILL))
        assert 1.0 > near > far > 0.0

    def test_a_low_certainty_disagreement_counts_for_little(self) -> None:
        """Otherwise a base the model emitted at 0.26 argues as loudly against a
        candidate as one it emitted at 0.99."""
        unsure = Evo2Generation(sequence="AC", sampled_probs=(1.0, 0.01))
        confident = Evo2Generation(sequence="AC", sampled_probs=(1.0, 1.0))

        assert agreement_score("AG", unsure) > agreement_score("AG", confident)

    def test_a_shorter_candidate_cannot_score_well_by_matching_a_prefix(self) -> None:
        assert agreement_score("ACGTA", Evo2Generation(sequence=_FILL)) < 0.75

    def test_an_empty_generation_scores_zero_rather_than_raising(self) -> None:
        assert agreement_score(_FILL, Evo2Generation()) == 0.0


class TestArbitrationIsConditional:
    def test_a_settled_candidate_is_not_arbitrated(self) -> None:
        """Running Evo 2 on a fill twelve conspecific genomes agree on cannot
        improve the answer and can only introduce a disagreement."""
        decision = ArbitrationPolicy().decide((_candidate("c1", _FILL, 0.96),), gap_length=10)
        assert not decision.arbitrate
        assert "settled" in decision.reason

    def test_two_close_candidates_are_arbitrated(self) -> None:
        decision = ArbitrationPolicy().decide(
            (_candidate("c1", _FILL, 0.62), _candidate("c2", "TTTTTTTTTT", 0.60)),
            gap_length=10,
        )
        assert decision.arbitrate
        assert "margin" in decision.reason

    def test_a_borderline_lone_candidate_is_arbitrated(self) -> None:
        decision = ArbitrationPolicy().decide((_candidate("c1", _FILL, 0.35),), gap_length=10)
        assert decision.arbitrate
        assert "borderline" in decision.reason

    def test_a_region_too_long_to_generate_is_not_arbitrated(self) -> None:
        decision = ArbitrationPolicy().decide((_candidate("c1", _FILL, 0.35),), gap_length=5000)
        assert not decision.arbitrate

    def test_arbitration_is_never_repeated(self) -> None:
        """Near-greedy decoding means a second call returns the same answer."""
        already = _candidate("c1", _FILL, 0.35).model_copy(
            update={"scores": CandidateScores(evo2=0.8)}
        )
        decision = ArbitrationPolicy().decide((already,), gap_length=10)
        assert not decision.arbitrate
        assert "already" in decision.reason


class TestUnavailabilityIsNeverAFailureOfTheGap:
    async def test_an_unconfigured_key_reports_the_reason(self) -> None:
        service = Evo2Service(_FakeClient(available=False))  # type: ignore[arg-type]
        result = await service.arbitrate(
            (_candidate("c1", _FILL, 0.35),), left_flank="ACGT" * 30, gap_length=10
        )

        assert not result.consulted
        assert result.unavailable_reason is not None
        assert "NVIDIA_API_KEY" in result.unavailable_reason

    async def test_a_service_error_is_absorbed(self) -> None:
        client = _FakeClient(
            error=ExternalServiceError("evo2", "NIM is down", code=ErrorCode.EVO2_UNAVAILABLE)
        )
        service = Evo2Service(client)  # type: ignore[arg-type]
        result = await service.arbitrate(
            (_candidate("c1", _FILL, 0.35),), left_flank="ACGT" * 30, gap_length=10
        )

        assert not result.consulted
        assert result.unavailable_reason is not None

    async def test_the_tool_reports_unavailability_as_a_successful_outcome(self) -> None:
        """The gap keeps every other score; only the tiebreaker is lost."""
        tool = EvaluateWithEvo2Tool(
            Evo2Service(_FakeClient(available=False)),  # type: ignore[arg-type]
            ConfidenceEngine(),
        )
        outcome = await tool.run(
            Evo2Input(
                gap_id="gap_1",
                candidates=(_candidate("c1", _FILL, 0.35),),
                context=_CONTEXT,
                left_flank="ACGT" * 30,
                gap_length=10,
            )
        )

        assert outcome.ok
        assert outcome.data is not None and outcome.data.consulted is False
        assert outcome.data.reason

    async def test_unavailability_is_still_recorded_as_evidence(self) -> None:
        """A gap whose evidence says nothing about Evo 2 is indistinguishable
        from one where arbitration silently failed."""
        tool = EvaluateWithEvo2Tool(
            Evo2Service(_FakeClient(available=False)),  # type: ignore[arg-type]
            ConfidenceEngine(),
        )
        outcome = await tool.run(
            Evo2Input(
                gap_id="gap_1",
                candidates=(_candidate("c1", _FILL, 0.35),),
                context=_CONTEXT,
                left_flank="ACGT" * 30,
                gap_length=10,
            )
        )
        contribution = tool.contribute(outcome)

        assert contribution.evo2 is not None
        assert contribution.evo2.consulted is False
        assert contribution.evo2.unavailable_reason


class TestAComputedAgreementIsNeverLost:
    """The failure mode the design singles out: a call spent on a number that
    nothing reads."""

    async def test_arbitration_changes_the_confidence(self) -> None:
        engine = ConfidenceEngine()
        candidate = _candidate("c1", _FILL, 0.60)
        before = candidate.final_confidence

        outcome = await ScoreCandidateTool(engine).run(
            ScoreCandidateInput(gap_id="gap_1", candidates=(candidate,), evo2_agreement={"c1": 1.0})
        )

        assert outcome.data is not None
        rescored = outcome.data.candidates[0]
        assert rescored.scores.evo2 == 1.0
        assert rescored.final_confidence != before

    async def test_arbitration_can_change_which_candidate_leads(self) -> None:
        """Otherwise asking was pointless."""
        outcome = await ScoreCandidateTool(ConfidenceEngine()).run(
            ScoreCandidateInput(
                gap_id="gap_1",
                candidates=(
                    _candidate("c1", _FILL, 0.61),
                    _candidate("c2", "TTTTTTTTTT", 0.60),
                ),
                evo2_agreement={"c1": 0.0, "c2": 1.0},
            )
        )

        assert outcome.data is not None
        assert outcome.data.candidates[0].candidate_id == "c2"
        assert outcome.data.order_changed is True

    async def test_a_candidate_arbitration_skipped_keeps_none(self) -> None:
        """`None` and `0.0` mean different things: not asked, versus asked and
        disagreed. Scoring the first as the second penalises a candidate for
        the tool's own coverage."""
        outcome = await ScoreCandidateTool(ConfidenceEngine()).run(
            ScoreCandidateInput(
                gap_id="gap_1",
                candidates=(_candidate("c1", _FILL, 0.6), _candidate("c2", "TTTT", 0.5)),
                evo2_agreement={"c1": 0.9},
            )
        )

        assert outcome.data is not None
        by_id = {item.candidate_id: item for item in outcome.data.candidates}
        assert by_id["c1"].scores.evo2 == 0.9
        assert by_id["c2"].scores.evo2 is None

    async def test_rescoring_without_an_agreement_is_a_no_op(self) -> None:
        candidates = (_candidate("c1", _FILL, 0.6),)
        outcome = await ScoreCandidateTool(ConfidenceEngine()).run(
            ScoreCandidateInput(gap_id="gap_1", candidates=candidates, evo2_agreement={})
        )

        assert outcome.ok
        assert outcome.data is not None
        assert outcome.data.candidates == candidates


class TestEvo2NeverOutvotesHomology:
    def test_total_disagreement_cannot_sink_a_well_supported_candidate(self) -> None:
        """Evo 2 is an opinion about what a model expects, not an observation of
        what any organism has. It carries the least weight for that reason."""
        engine = ConfidenceEngine()
        strong = CandidateScores(homology=0.96, alignment=1.0, conservation=1.0, evolutionary=1.0)
        without = engine.confidence(strong)
        against = engine.confidence(strong.model_copy(update={"evo2": 0.0}))

        assert against < without
        assert engine.is_returnable(against)

    def test_agreement_cannot_rescue_a_candidate_that_failed_validation(self) -> None:
        """Validation is a gate, not a signal."""
        engine = ConfidenceEngine()
        invalid = CandidateScores(homology=0.9, alignment=0.9, evo2=1.0, validation=0.0)

        assert engine.confidence(invalid) == 0.0


def test_a_generation_reports_the_models_own_certainty() -> None:
    generation = Evo2Generation(sequence="ACGT", sampled_probs=(0.9, 0.8, 0.7, 0.6))
    assert generation.mean_confidence is not None
    assert 0.7 < generation.mean_confidence < 0.8


def test_a_generation_without_probabilities_reports_none(monkeypatch: Any) -> None:
    """Absent probabilities means unweighted agreement, not zero certainty."""
    assert Evo2Generation(sequence="ACGT").mean_confidence is None
