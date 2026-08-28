"""Evo 2 as a generator: filling the gaps homology cannot span.

A gap with no gap-spanning homologue used to come back UNRESOLVED. A model
trained on genomes still has something to say about what belongs between two
flanks, so it is now asked - unconditionally, because there is no other source
of an answer for that gap.

Everything else in this file exists to keep the resulting sequence honest. A
generated fill and an observed one are both plausible strings of the same
length, and nothing about the bases distinguishes them. So the distinction is
carried explicitly, scored on a separate capped path, and stated in words:

- `origin` is MODEL, and `supporting_hits` is empty and truthfully so.
- The confidence can never reach the band a homology-backed fill reaches.
- An observed fill always outranks a predicted one, whatever the model's
  certainty.
"""

from __future__ import annotations

from reconstruction_agent.domain.enums import CandidateOrigin, ConfidenceLevel
from reconstruction_agent.domain.models.candidate import Candidate, CandidateScores
from reconstruction_agent.domain.models.sequence import Gap, GapContext
from reconstruction_agent.integrations.evo2.models import Evo2Generation
from reconstruction_agent.services.candidate.model_candidate import build_model_candidate
from reconstruction_agent.services.plausibility.evo2_service import Evo2Service
from reconstruction_agent.services.plausibility.plausibility_policy import ArbitrationPolicy
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.plausibility.evaluate_with_evo2 import (
    EvaluateWithEvo2Tool,
    Evo2Input,
)

_GAP = Gap(gap_id="gap_1", start=20, end=30)
_CONTEXT = GapContext(
    gap=_GAP, source_accession="NC_TEST.1", left_flank="ACGT" * 30, right_flank="TGCA" * 30
)
_FILL = "ACGTACGTAC"


class _FakeClient:
    def __init__(self, generation: Evo2Generation, *, available: bool = True) -> None:
        self._generation = generation
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    async def generate(self, prompt: str, *, num_tokens: int) -> Evo2Generation:
        self.calls += 1
        return self._generation


def _tool(generation: Evo2Generation) -> EvaluateWithEvo2Tool:
    return EvaluateWithEvo2Tool(
        Evo2Service(_FakeClient(generation)),  # type: ignore[arg-type]
        ConfidenceEngine(),
    )


def _request(candidates: tuple[Candidate, ...] = ()) -> Evo2Input:
    return Evo2Input(
        gap_id="gap_1",
        candidates=candidates,
        context=_CONTEXT,
        left_flank=_CONTEXT.left_flank,
        gap_length=_GAP.length,
    )


class TestAGapWithNoHomologueIsStillAnswered:
    def test_the_policy_always_calls_when_there_are_no_candidates(self) -> None:
        """The whole point: refusing to ask leaves the region empty for want of
        a question."""
        decision = ArbitrationPolicy().decide((), gap_length=10)

        assert decision.arbitrate
        assert "no homologue spans" in decision.reason.lower()

    async def test_the_model_produces_a_usable_fill(self) -> None:
        outcome = await _tool(Evo2Generation(sequence=_FILL, sampled_probs=(0.9,) * 10)).run(
            _request()
        )

        assert outcome.ok
        assert outcome.data is not None
        generated = outcome.data.generated_candidate
        assert generated is not None
        assert generated.sequence == _FILL
        assert len(generated.sequence) == _GAP.length

    async def test_a_region_too_long_to_generate_is_still_refused(self) -> None:
        """Past the ceiling the continuation stops being coherent, and a long
        fabricated fill is worse than an honest refusal."""
        decision = ArbitrationPolicy().decide((), gap_length=5000)

        assert not decision.arbitrate


class TestAPredictionIsNeverMistakenForAnObservation:
    def _generated(self, certainty: float = 0.95) -> Candidate:
        candidate = build_model_candidate(
            _FILL, _CONTEXT, engine=ConfidenceEngine(), model_certainty=certainty
        )
        assert candidate is not None
        return candidate

    def test_it_is_marked_as_model_generated(self) -> None:
        candidate = self._generated()

        assert candidate.origin is CandidateOrigin.MODEL
        assert candidate.is_model_generated

    def test_it_claims_no_supporting_organism(self) -> None:
        """Truthfully empty: no sequenced organism is known to carry it."""
        candidate = self._generated()

        assert candidate.supporting_hits == ()
        assert candidate.supporting_organisms == ()

    def test_its_rationale_says_so_in_words(self) -> None:
        """A caller reading a summary rather than a schema still has to be told."""
        candidate = self._generated()

        assert "prediction" in candidate.rationale.lower()
        assert "no sequenced organism" in candidate.rationale.lower()

    def test_its_confidence_is_capped_below_the_high_band(self) -> None:
        """However sure the model was, a prediction is never reported as
        strongly as a fill a dozen relatives agree on."""
        engine = ConfidenceEngine()
        candidate = self._generated(certainty=1.0)

        assert candidate.final_confidence <= engine.thresholds.model_ceiling
        assert candidate.confidence_level is not ConfidenceLevel.HIGH

    def test_it_is_returnable_rather_than_inert(self) -> None:
        """Scoring the unmeasurable homology terms as zero would put every
        prediction below the floor and make the capability pointless."""
        engine = ConfidenceEngine()
        candidate = self._generated(certainty=0.9)

        assert engine.is_returnable(candidate.final_confidence)


class TestObservationAlwaysOutranksPrediction:
    def test_a_homology_backed_fill_scores_higher_than_a_certain_prediction(self) -> None:
        engine = ConfidenceEngine()
        observed = engine.confidence(
            CandidateScores(homology=0.9, alignment=0.9, conservation=0.9, evolutionary=0.9),
            CandidateOrigin.HOMOLOGY,
        )
        predicted = engine.confidence(
            CandidateScores(evo2=1.0, validation=1.0), CandidateOrigin.MODEL
        )

        assert observed > predicted

    def test_a_prediction_that_fails_validation_is_not_returnable(self) -> None:
        """Validation is a gate for a model fill exactly as for an observed one -
        a model is perfectly capable of writing a low-complexity run confidently."""
        engine = ConfidenceEngine()
        confidence = engine.confidence(
            CandidateScores(evo2=1.0, validation=0.0), CandidateOrigin.MODEL
        )

        assert not engine.is_returnable(confidence)


class TestTheGeneratedFillMustFitTheGap:
    def test_a_continuation_of_the_wrong_length_is_discarded(self) -> None:
        """Offering it as a low-scoring option invites it to be taken, and it
        would shift every coordinate after the region."""
        assert (
            build_model_candidate("ACGT", _CONTEXT, engine=ConfidenceEngine(), model_certainty=0.9)
            is None
        )

    def test_an_empty_continuation_is_discarded(self) -> None:
        assert (
            build_model_candidate("", _CONTEXT, engine=ConfidenceEngine(), model_certainty=0.9)
            is None
        )

    async def test_the_tool_returns_no_candidate_when_the_fill_does_not_fit(self) -> None:
        outcome = await _tool(Evo2Generation(sequence="ACGT")).run(_request())

        assert outcome.ok
        assert outcome.data is not None
        assert outcome.data.generated_candidate is None


class TestArbitrationStillDeclinesWhereHomologyHasDecided:
    async def test_a_settled_gap_does_not_call_the_model(self) -> None:
        """Running Evo 2 on a fill twelve conspecific genomes agree on cannot
        improve the answer and costs the deadline."""
        client = _FakeClient(Evo2Generation(sequence=_FILL))
        tool = EvaluateWithEvo2Tool(
            Evo2Service(client),  # type: ignore[arg-type]
            ConfidenceEngine(),
        )
        settled = Candidate(
            candidate_id="c1",
            gap_id="gap_1",
            sequence=_FILL,
            scores=CandidateScores(homology=0.96, alignment=1.0),
            final_confidence=0.96,
        )

        outcome = await tool.run(_request((settled,)))

        assert outcome.ok
        assert client.calls == 0
        assert outcome.data is not None and outcome.data.consulted is False
