"""The agentic loop: verdicts, routing, observations and retries.

What separates this from a pipeline is that the critic can send it round again,
and that a step refused for budget is recorded rather than silently dropped.
"""
from __future__ import annotations

import pytest

from agent.graph.conditions import (
    CRITIQUE,
    EXECUTE_TOOLS,
    FINALIZE,
    PLAN,
    route_after_decision,
    route_after_detection,
    route_after_selection,
)
from agent.reasoning.critic import Critic, Verdict
from contracts.observation import Observation, ObservationStatus
from contracts.output import EvidenceItem, GapReconstruction, ReconstructionStatus
from domain.models import Gap, GapContext
from infrastructure.llm.client import NullLLMClient


def make_context(gap_id: str = "gap_1") -> GapContext:
    return GapContext(
        gap=Gap(gap_id, 100, 140), left_flank="A" * 100, right_flank="C" * 100
    )


def solid(gap_id: str = "gap_1") -> GapReconstruction:
    """A reconstruction the deterministic checks accept."""
    return GapReconstruction(
        gap_id=gap_id,
        start=100,
        end=140,
        length=40,
        status=ReconstructionStatus.RECONSTRUCTED,
        reconstructed_sequence="ACGT" * 10,
        confidence=0.9,
        evidence=[
            EvidenceItem(source="mafft", reference_id="REF_1", identity=0.95),
            EvidenceItem(source="mafft", reference_id="REF_2", identity=0.93),
        ],
    )


class TestVerdicts:
    @pytest.fixture
    def critic(self) -> Critic:
        return Critic(NullLLMClient())

    async def test_solid_evidence_is_accepted(self, critic: Critic) -> None:
        critique = await critic.review(make_context(), solid())

        assert critique.verdict is Verdict.ACCEPT
        assert critique.acceptable

    async def test_thin_evidence_asks_for_a_revision(self, critic: Critic) -> None:
        thin = solid().model_copy(
            update={"evidence": [EvidenceItem(source="mafft", reference_id="REF_1")]}
        )

        critique = await critic.review(make_context(), thin, more_evidence_possible=True)

        assert critique.verdict is Verdict.REVISE
        assert any("reference" in problem for problem in critique.problems)

    async def test_the_same_problem_abstains_when_nothing_more_can_be_done(
        self, critic: Critic
    ) -> None:
        """REVISE would be a promise the loop cannot keep once budget is spent."""
        thin = solid().model_copy(
            update={"evidence": [EvidenceItem(source="mafft", reference_id="REF_1")]}
        )

        critique = await critic.review(make_context(), thin, more_evidence_possible=False)

        assert critique.verdict is Verdict.ABSTAIN

    async def test_weak_identity_is_a_problem(self, critic: Critic) -> None:
        weak = solid().model_copy(
            update={
                "evidence": [
                    EvidenceItem(source="mafft", reference_id="REF_1", identity=0.4),
                    EvidenceItem(source="mafft", reference_id="REF_2", identity=0.5),
                ]
            }
        )

        critique = await critic.review(make_context(), weak)

        assert critique.verdict is not Verdict.ACCEPT
        assert any("identity" in problem for problem in critique.problems)

    async def test_an_unresolved_gap_is_never_accepted(self, critic: Critic) -> None:
        unresolved = GapReconstruction(
            gap_id="gap_1",
            start=100,
            end=140,
            length=40,
            status=ReconstructionStatus.UNRESOLVED,
        )

        critique = await critic.review(make_context(), unresolved)

        assert critique.verdict is not Verdict.ACCEPT

    async def test_the_note_names_the_verdict_for_the_next_plan(
        self, critic: Critic
    ) -> None:
        thin = solid().model_copy(
            update={"evidence": [EvidenceItem(source="mafft", reference_id="REF_1")]}
        )

        note = (await critic.review(make_context(), thin)).as_note()

        assert "gap_1" in note
        assert "revise" in note


class TestRouting:
    def test_no_work_skips_straight_to_the_end(self) -> None:
        assert route_after_detection({"gap_contexts": []}) == FINALIZE

    def test_every_gap_skipped_counts_as_no_work(self) -> None:
        state = {"gap_contexts": [make_context()], "skipped": {"gap_1": "too long"}}

        assert route_after_detection(state) == FINALIZE

    def test_work_present_goes_to_planning(self) -> None:
        state = {"gap_contexts": [make_context()], "skipped": {}}

        assert route_after_detection(state) == PLAN

    def test_a_plan_with_nothing_runnable_skips_execution(self) -> None:
        """Falling through to a no-op tool round would waste an iteration."""
        assert route_after_selection({"pending_invocations": []}) == CRITIQUE

    def test_a_runnable_plan_executes(self) -> None:
        state = {"pending_invocations": [{"tool": "blast_search"}]}

        assert route_after_selection(state) == EXECUTE_TOOLS

    def test_revise_loops_back_to_planning(self) -> None:
        state = {"should_continue": True, "iteration": 1, "max_iterations": 6}

        assert route_after_decision(state) == PLAN

    def test_a_stopped_loop_finalises(self) -> None:
        state = {"should_continue": False, "iteration": 1, "max_iterations": 6}

        assert route_after_decision(state) == FINALIZE

    def test_the_iteration_ceiling_finalises(self) -> None:
        state = {"should_continue": True, "iteration": 6, "max_iterations": 6}

        assert route_after_decision(state) == FINALIZE


class TestObservations:
    def test_an_empty_result_is_recorded_not_treated_as_failure(self) -> None:
        """BLAST with no hits is a real answer about the gap, not an error."""
        observation = Observation(
            tool="blast_search", gap_id="gap_1", status=ObservationStatus.EMPTY
        )

        assert not observation.made_progress
        assert observation.status is not ObservationStatus.FAILED

    def test_progress_needs_evidence_not_just_success(self) -> None:
        succeeded_empty = Observation(
            tool="ncbi_search", status=ObservationStatus.OK, evidence_added=0
        )
        succeeded_useful = Observation(
            tool="ncbi_search", status=ObservationStatus.OK, evidence_added=3
        )

        assert not succeeded_empty.made_progress
        assert succeeded_useful.made_progress

    def test_a_refused_call_records_why(self) -> None:
        observation = Observation(
            tool="blast_search",
            status=ObservationStatus.SKIPPED,
            detail="Budget exhausted: tool_calls.",
        )

        assert observation.detail is not None
        assert "Budget" in observation.detail
