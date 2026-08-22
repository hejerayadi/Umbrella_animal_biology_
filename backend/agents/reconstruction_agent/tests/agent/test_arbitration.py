"""Settling a 50/50 consensus with an independent read of the sequence.

The chain under test:

    contested columns -> candidates A and B -> Evo 2 scores both
        -> the better-predicted fill wins -> confidence reflects the doubt

Before this existed the agent produced one plurality string per gap, so a
contested column had no second hypothesis, `evo2_plausibility` had nothing to
compare, and the tool was unreachable anyway - `ToolSelector` had no branch to
build its payload and dropped every planned step with `tool_payload_unbuildable`.
"""
from __future__ import annotations

from agent.planning.planner import Planner
from agent.planning.tool_selector import ToolSelector
from agent.reasoning.reasoner import Reasoner
from agent.state.state import initial_state
from domain.models import AlignedPair, Alignment, Candidate, Gap, GapContext, Sequence
from domain.policies import ConfidencePolicy
from domain.services import CandidateRanker, ReconstructionValidator

FLANK = "ACGTTGCA" * 12
GAP = 8


def context_for() -> GapContext:
    return GapContext(
        gap=Gap("gap_1", len(FLANK), len(FLANK) + GAP),
        left_flank=FLANK,
        right_flank=FLANK,
    )


def split_alignment(first: str, second: str, *, each: int = 1) -> Alignment:
    """An alignment where the references split evenly over the gap columns."""
    target = FLANK + "-" * GAP + FLANK
    pairs = []
    for index in range(each):
        pairs.append(AlignedPair("target", f"REF_A{index}", target, FLANK + first + FLANK))
        pairs.append(AlignedPair("target", f"REF_B{index}", target, FLANK + second + FLANK))

    return Alignment(
        gap_id="gap_1",
        pairs=pairs,
        gap_column_start=len(FLANK),
        gap_column_end=len(FLANK) + GAP,
    )


def make_reasoner() -> Reasoner:
    return Reasoner(
        ranker=CandidateRanker(),
        validator=ReconstructionValidator(),
        confidence=ConfidencePolicy(minimum_confidence=0.25),
    )


class TestCompetingCandidates:
    def test_a_split_vote_produces_two_hypotheses(self) -> None:
        """The losing references are a competing answer, not noise."""
        candidates = make_reasoner().build_candidates(
            context_for(), split_alignment("AAAAAAAA", "TTTTTTTT"), []
        )

        assert len(candidates) == 2
        assert {c.sequence for c in candidates} == {"AAAAAAAA", "TTTTTTTT"}

    def test_a_clean_consensus_produces_one(self) -> None:
        """No second hypothesis exists, and paying Evo 2 to confirm it is waste."""
        alignment = split_alignment("AAAAAAAA", "AAAAAAAA")

        candidates = make_reasoner().build_candidates(context_for(), alignment, [])

        assert len(candidates) == 1

    def test_a_decisive_majority_produces_one(self) -> None:
        """Three against one is settled; only a near-tie leaves a real alternative."""
        target = FLANK + "-" * GAP + FLANK
        pairs = [
            AlignedPair("target", f"REF_{i}", target, FLANK + "AAAAAAAA" + FLANK)
            for i in range(3)
        ]
        pairs.append(AlignedPair("target", "REF_X", target, FLANK + "TTTTTTTT" + FLANK))
        alignment = Alignment(
            gap_id="gap_1",
            pairs=pairs,
            gap_column_start=len(FLANK),
            gap_column_end=len(FLANK) + GAP,
        )

        candidates = make_reasoner().build_candidates(context_for(), alignment, [])

        assert len(candidates) == 1
        assert candidates[0].sequence == "AAAAAAAA"


class TestEvo2Arbitrates:
    def test_the_better_predicted_fill_wins(self) -> None:
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")
        candidates = reasoner.build_candidates(context, alignment, [])
        loser = candidates[1].sequence

        settled = reasoner.arbitrate(
            context,
            candidates,
            alignment,
            [],
            # Evo 2 expects the second candidate, not the first.
            scores={"candidate_0": 0.20, "candidate_1": 0.80},
        )

        assert settled[0].sequence == loser, "the model's preference did not decide"

    def test_the_rejected_fill_keeps_a_lower_score(self) -> None:
        """A tie-break has to move the scores apart, not merely reorder them."""
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")

        settled = reasoner.arbitrate(
            context,
            reasoner.build_candidates(context, alignment, []),
            alignment,
            [],
            scores={"candidate_0": 0.80, "candidate_1": 0.20},
        )

        assert (settled[0].confidence or 0) > (settled[1].confidence or 0)

    def test_close_scores_still_separate(self) -> None:
        """0.8 against 0.7 is the ordinary case, and must not be a no-op."""
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")

        settled = reasoner.arbitrate(
            context,
            reasoner.build_candidates(context, alignment, []),
            alignment,
            [],
            scores={"candidate_0": 0.80, "candidate_1": 0.70},
        )

        assert (settled[0].confidence or 0) != (settled[1].confidence or 0)

    def test_it_cannot_promote_a_fill_above_unarbitrated_confidence(self) -> None:
        """A genome model may doubt the alignment; it may not overrule it."""
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")
        before = reasoner.build_candidates(context, alignment, [])

        after = reasoner.arbitrate(
            context, before, alignment, [], scores={"candidate_0": 1.0, "candidate_1": 1.0}
        )

        assert (after[0].confidence or 0) <= (before[0].confidence or 0) + 1e-9

    def test_hypotheses_are_whole_sequences_not_column_mixtures(self) -> None:
        """Caught live: two split references produced neither of their fills.

        Picking a winner column by column mixes the references into a chimera -
        a sequence no homologue carries. Evo 2 was being asked to choose between
        two fills that did not exist, and it duly rejected both.
        """
        candidates = make_reasoner().build_candidates(
            context_for(), split_alignment("AAAACCCC", "TTTTGGGG"), []
        )

        assert {c.sequence for c in candidates} == {"AAAACCCC", "TTTTGGGG"}

    def test_settling_a_tie_lifts_the_answer_above_the_threshold(self) -> None:
        """Otherwise arbitration finds the right fill and then rejects it.

        The ambiguity penalty exists because a split vote says nothing about
        which base is right. Once something outside the vote answers that, the
        penalty has to relax - on a live run the model predicted the true fill
        exactly and the reconstruction still scored 0.10 against a 0.25 floor.
        """
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")
        before = reasoner.build_candidates(context, alignment, [])

        settled = reasoner.arbitrate(
            context, before, alignment, [], scores={"candidate_0": 1.0, "candidate_1": 0.0}
        )

        assert (before[0].confidence or 0) < 0.25
        assert (settled[0].confidence or 0) >= 0.25

    def test_relief_is_proportional_to_how_cleanly_the_model_chose(self) -> None:
        """A model that is itself undecided settles nothing and earns nothing."""
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")
        candidates = reasoner.build_candidates(context, alignment, [])

        decisive = reasoner.arbitrate(
            context, candidates, alignment, [], scores={"candidate_0": 1.0, "candidate_1": 0.0}
        )
        undecided = reasoner.arbitrate(
            context, candidates, alignment, [], scores={"candidate_0": 0.51, "candidate_1": 0.49}
        )

        assert (decisive[0].confidence or 0) > (undecided[0].confidence or 0)

    def test_relief_cannot_exceed_a_decisive_vote(self) -> None:
        """Arbitration lifts a gap to "settled" and no further."""
        reasoner = make_reasoner()
        context = context_for()
        split = split_alignment("AAAAAAAA", "TTTTTTTT")

        arbitrated = reasoner.arbitrate(
            context,
            reasoner.build_candidates(context, split, []),
            split,
            [],
            scores={"candidate_0": 1.0, "candidate_1": 0.0},
        )
        unanimous = reasoner.build_candidates(
            context, split_alignment("AAAAAAAA", "AAAAAAAA"), []
        )

        assert (arbitrated[0].confidence or 0) <= (unanimous[0].confidence or 0)

    def test_no_scores_leaves_the_candidates_alone(self) -> None:
        reasoner = make_reasoner()
        context = context_for()
        alignment = split_alignment("AAAAAAAA", "TTTTTTTT")
        candidates = reasoner.build_candidates(context, alignment, [])

        assert reasoner.arbitrate(context, candidates, alignment, [], scores={}) == candidates


def state_with(candidates: dict, **overrides: object) -> dict:
    state = initial_state(
        run_id="run-1",
        trace_id="trace-1",
        instruction="Reconstruct.",
        target=Sequence.parse("seq", FLANK + "N" * GAP + FLANK),
        organism="Testus organismus",
        requested_organisms=[],
        max_iterations=6,
        max_slices=4,
    )
    state["gap_contexts"] = [context_for()]
    state["candidates"] = candidates
    state.update(overrides)  # type: ignore[typeddict-item]
    return dict(state)


class TestTheLoopReachesEvo2:
    def two_candidates(self) -> list[Candidate]:
        return [
            Candidate(gap_id="gap_1", sequence="AAAAAAAA", support=0.05),
            Candidate(gap_id="gap_1", sequence="TTTTTTTT", support=0.05),
        ]

    def test_a_contested_gap_is_planned_for_arbitration(self) -> None:
        state = state_with({"gap_1": self.two_candidates()})

        tools = [s.tool for s in Planner(None, []).deterministic_plan(state)]  # type: ignore[arg-type]

        assert "evo2_plausibility" in tools

    def test_it_is_not_planned_twice_for_the_same_gap(self) -> None:
        """The verdict is checkpointed; paying for it again is pure waste."""
        state = state_with(
            {"gap_1": self.two_candidates()},
            plausibility={"gap_1": {"scores": {"candidate_0": 0.5}}},
        )

        tools = [s.tool for s in Planner(None, []).deterministic_plan(state)]  # type: ignore[arg-type]

        assert "evo2_plausibility" not in tools

    def test_a_settled_gap_is_never_sent(self) -> None:
        state = state_with({"gap_1": [Candidate(gap_id="gap_1", sequence="AAAAAAAA")]})

        tools = [s.tool for s in Planner(None, []).deterministic_plan(state)]  # type: ignore[arg-type]

        assert "evo2_plausibility" not in tools

    def test_the_payload_carries_both_candidates(self) -> None:
        """The branch whose absence made the tool unreachable."""
        from agent.planning.planner import PlanStep

        state = state_with({"gap_1": self.two_candidates()})
        invocation = ToolSelector().build(
            PlanStep(tool="evo2_plausibility", gap_id="gap_1", reason=""),
            state,  # type: ignore[arg-type]
        )

        assert invocation is not None
        assert invocation.payload.candidates == {
            "candidate_0": "AAAAAAAA",
            "candidate_1": "TTTTTTTT",
        }
        assert invocation.payload.left_flank == FLANK

    def test_a_lone_candidate_builds_no_call(self) -> None:
        from agent.planning.planner import PlanStep

        state = state_with({"gap_1": [Candidate(gap_id="gap_1", sequence="AAAAAAAA")]})

        assert (
            ToolSelector().build(
                PlanStep(tool="evo2_plausibility", gap_id="gap_1", reason=""),
                state,  # type: ignore[arg-type]
            )
            is None
        )
