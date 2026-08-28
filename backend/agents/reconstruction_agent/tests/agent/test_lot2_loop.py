"""The Lot 2 loop: writing a fill, replanning onto arbitration, counting right.

Three things are proved here that nothing else covers.

**Substitution refuses rather than corrupts.** `gap_replacer` is the one place
in the agent that modifies a sequence, and each of its refusals guards a failure
that nothing downstream could detect.

**Replanning escalates in the right order.** Another observed genome is
evidence; Evo 2 is a modelling opinion. Spending the opinion first would let a
model settle a question more data would have settled properly.

**RETRY and REPLAN are counted separately.** A flaky network must not exhaust
the agent's thinking budget, which is exactly what happens if a transport
failure is allowed to look like a scientific deficit.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.agent.critic import diagnose_by_rule
from reconstruction_agent.agent.replanner import Replanner
from reconstruction_agent.domain.enums import ConfidenceLevel, CriticDeficit, ToolName
from reconstruction_agent.domain.models.candidate import Candidate, CandidateScores
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.services.reconstruction.gap_replacer import apply_fill
from reconstruction_agent.tools.reconstruction.reconstruct_gap import (
    ReconstructGapInput,
    ReconstructGapTool,
)

_GAP = Gap(gap_id="gap_1", start=4, end=14)
#: 4 called bases, a 10-base hole, 4 more called bases.
_RECORD = "ACGT" + "N" * 10 + "TGCA"
_FILL = "ACGTACGTAC"


def _candidate(candidate_id: str, sequence: str, confidence: float) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        gap_id="gap_1",
        sequence=sequence,
        scores=CandidateScores(homology=confidence),
        final_confidence=confidence,
        confidence_level=ConfidenceLevel.MEDIUM,
    )


def _state(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "gap_id": "gap_1",
        "gap": _GAP,
        "overrides": {},
        "exhausted_scopes": frozenset(),
        "measured_scopes": (),
        "hits": (),
        "candidates": (),
        "evo2_agreement": {},
    }
    return {**base, **overrides}


class TestSubstitutionRefusesRatherThanCorrupts:
    def test_a_well_formed_fill_is_written(self) -> None:
        result = apply_fill(_RECORD, _GAP, _FILL)

        assert result.applied
        assert result.residues == "ACGT" + _FILL + "TGCA"
        assert "N" not in result.residues

    def test_a_fill_of_the_wrong_length_is_refused(self) -> None:
        """It would shift every coordinate after this region, undetectably."""
        result = apply_fill(_RECORD, _GAP, "ACGT")

        assert not result.applied
        assert "shift" in result.reason

    def test_an_ambiguous_fill_is_refused(self) -> None:
        """Marking a region repaired without resolving it is worse than leaving
        it open, because the record then claims something untrue."""
        result = apply_fill(_RECORD, _GAP, "ACGTNNNNAC")

        assert not result.applied
        assert "unambiguous" in result.reason

    def test_overwriting_called_bases_is_refused(self) -> None:
        """That is data loss, not repair."""
        called = "ACGT" + "ACGTACGTAC" + "TGCA"
        result = apply_fill(called, _GAP, _FILL)

        assert not result.applied
        assert "data loss" in result.reason

    def test_a_gap_past_the_end_of_the_record_is_refused(self) -> None:
        result = apply_fill("ACGT", _GAP, _FILL)

        assert not result.applied
        assert "bases long" in result.reason

    async def test_the_tool_returns_the_region_not_the_whole_record(self) -> None:
        """A whole scaffold in the shared context becomes megabytes in every
        downstream agent's prompt."""
        outcome = await ReconstructGapTool().run(
            ReconstructGapInput(
                gap_id="gap_1",
                gap=_GAP,
                candidate=_candidate("c1", _FILL, 0.9),
                residues=_RECORD,
            )
        )

        assert outcome.ok
        assert outcome.data is not None
        assert outcome.data.filled_sequence == _FILL
        assert len(outcome.data.repaired_window) <= len(_RECORD)

    async def test_the_tool_refuses_without_raising(self) -> None:
        outcome = await ReconstructGapTool().run(
            ReconstructGapInput(
                gap_id="gap_1",
                gap=_GAP,
                candidate=_candidate("c1", "ACGT", 0.9),
                residues=_RECORD,
            )
        )

        assert not outcome.ok
        assert outcome.reason


class TestCompetingCandidatesEscalateInTheRightOrder:
    def _competing(self, **extra: Any) -> Any:
        return _state(
            candidates=(_candidate("c1", _FILL, 0.61), _candidate("c2", "TTTTTTTTTT", 0.60)),
            **extra,
        )

    def test_more_references_are_tried_before_arbitration(self) -> None:
        """Another observed genome is evidence; Evo 2 is an opinion."""
        strategy = Replanner().plan_for(CriticDeficit.COMPETING_CANDIDATES, self._competing())

        assert ToolName.GET_HOMOLOG_SEQUENCES in strategy.actions
        assert ToolName.EVALUATE_WITH_EVO2 not in strategy.actions

    def test_arbitration_follows_when_more_references_did_not_help(self) -> None:
        strategy = Replanner().plan_for(
            CriticDeficit.COMPETING_CANDIDATES,
            self._competing(overrides={ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": 25}}),
        )

        assert ToolName.EVALUATE_WITH_EVO2 in strategy.actions
        # Rescoring must follow, or the agreement is computed and never read.
        assert ToolName.SCORE_CANDIDATE in strategy.actions
        assert strategy.overrides[ToolName.EVALUATE_WITH_EVO2.value]["force"] is True

    def test_arbitration_is_not_repeated(self) -> None:
        """Near-greedy decoding means the second answer is the first answer."""
        strategy = Replanner().plan_for(
            CriticDeficit.COMPETING_CANDIDATES,
            self._competing(
                overrides={ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": 25}},
                evo2_agreement={"c1": 0.8, "c2": 0.2},
            ),
        )

        assert not strategy.actionable
        assert "both are reported" in strategy.rationale

    def test_disagreement_is_reported_never_resolved_for_the_model(self) -> None:
        """A genome model preferring something else is not evidence that a
        dozen sequenced relatives are wrong."""
        strategy = Replanner().plan_for(CriticDeficit.EVO2_DISAGREEMENT, _state())

        assert not strategy.actionable
        assert "not resolved in the model" in strategy.rationale


class TestValidationFailureTriesTheRunnersUp:
    def test_a_second_candidate_is_revalidated(self) -> None:
        """Candidates are ranked by confidence, not by validity, so a failing
        leader can sit above an alternative that passes."""
        strategy = Replanner().plan_for(
            CriticDeficit.BIOLOGICAL_VALIDATION_FAILED,
            _state(candidates=(_candidate("c1", _FILL, 0.6), _candidate("c2", "ACGT", 0.5))),
        )

        assert ToolName.VALIDATE_CANDIDATE in strategy.actions

    def test_a_lone_failing_candidate_is_not_retried(self) -> None:
        strategy = Replanner().plan_for(
            CriticDeficit.BIOLOGICAL_VALIDATION_FAILED,
            _state(candidates=(_candidate("c1", _FILL, 0.6),)),
        )

        assert not strategy.actionable


class TestRetryAndReplanAreCountedSeparately:
    """A flaky network must not exhaust the agent's thinking budget."""

    def test_a_transport_failure_produces_no_deficit(self) -> None:
        state = _state(
            measured_scopes=(
                HomologySearchOutcome(
                    database_code="scope@a", database_label="a", error="timed out"
                ),
            )
        )
        assert diagnose_by_rule(state) is None

    def test_a_measured_absence_does_produce_one(self) -> None:
        state = _state(
            measured_scopes=(HomologySearchOutcome(database_code="scope@a", database_label="a"),)
        )
        assert diagnose_by_rule(state) is CriticDeficit.NO_HOMOLOGS

    def test_a_scoping_failure_replans_onto_a_scope_never_invented(self) -> None:
        """The scopes come from the target's own lineage; the replanner adds
        none of its own."""
        state = _state(
            measured_scopes=(
                HomologySearchOutcome(
                    database_code="scope@txid1",
                    database_label="a",
                    hits=(HomologHit(accession="X1", identity=0.67),),
                    gap_spanning_hits=0,
                ),
            ),
            hits=(HomologHit(accession="X1", identity=0.67),),
        )
        deficit = diagnose_by_rule(state)
        strategy = Replanner().plan_for(deficit or CriticDeficit.DATABASE_MISMATCH, state)

        assert deficit is CriticDeficit.DATABASE_MISMATCH
        assert strategy.exhaust_scopes == frozenset({"scope@txid1"})
        assert not any("txid" in code for code in strategy.overrides)
