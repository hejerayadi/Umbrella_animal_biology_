"""Choosing which gaps a run can actually afford to attempt.

Measured on `NW_007907101`, the polar bear scaffold the Genome Agent hands over:
34 N-runs of 10+ bases in the first megabase alone, so roughly 500 across its
15.9 Mb. The BLAST budget is eight calls. Attempting every gap does not give a
worse answer, it gives none - the planner emits a step per gap and the run
spends its slices refusing them for budget.
"""
from __future__ import annotations

from domain.models import Gap, GapContext
from domain.services.gap_priority import priority, select


def context(identifier: str, start: int, length: int, *, left: str = "A" * 200,
            right: str = "C" * 200) -> GapContext:
    return GapContext(
        gap=Gap(identifier=identifier, start=start, end=start + length),
        left_flank=left,
        right_flank=right,
    )


class TestPriority:
    def test_a_gap_with_one_flank_ranks_last(self) -> None:
        """The query is the flanks joined and the carrier test asks which
        references cross the junction between them. With one flank there is no
        junction, so no hit can ever be measured as carrying the gap."""
        both = context("gap_1", 5000, 500)
        one = context("gap_2", 10, 20, left="")
        assert priority(one) > priority(both)

    def test_shorter_gaps_rank_first(self) -> None:
        """A short gap is likelier to sit inside one homologous block. A 3.3 kb
        hole - the largest in that scaffold's first megabase - is the least
        recoverable thing in the file."""
        assert priority(context("s", 100, 20)) < priority(context("l", 100, 3325))

    def test_the_order_is_stable(self) -> None:
        """A rerun must attempt the same gaps, or results are not comparable."""
        gaps = [context(f"gap_{i}", i * 1000, 50) for i in range(5)]
        assert [c.identifier for c in sorted(gaps, key=priority)] == [
            c.identifier for c in sorted(gaps, key=priority)
        ]


class TestSelect:
    def test_everything_is_attempted_when_it_fits(self) -> None:
        gaps = [context(f"gap_{i}", i * 1000, 50) for i in range(5)]
        chosen, skipped = select(gaps, limit=12)
        assert len(chosen) == 5
        assert skipped == {}

    def test_a_scaffold_sized_run_commits_to_what_it_can_finish(self) -> None:
        gaps = [context(f"gap_{i}", i * 1000, 50) for i in range(500)]
        chosen, skipped = select(gaps, limit=12)
        assert len(chosen) == 12
        assert len(skipped) == 488

    def test_the_gaps_it_cannot_reach_say_so(self) -> None:
        """Not silently dropped: reporting 8 of 500 examined is honest, while
        appearing to have considered all 500 and resolved none is not."""
        gaps = [context(f"gap_{i}", i * 1000, 50) for i in range(50)]
        _, skipped = select(gaps, limit=2)
        reason = next(iter(skipped.values()))
        assert "50 gaps" in reason and "covers 2" in reason

    def test_a_policy_skip_keeps_its_own_reason(self) -> None:
        """"Too long to attempt" is more specific than "ranked below the cut"."""
        gaps = [context(f"gap_{i}", i * 1000, 50) for i in range(5)]
        chosen, skipped = select(gaps, limit=12, skipped={"gap_0": "Gap exceeds 5000 bases."})
        assert skipped["gap_0"] == "Gap exceeds 5000 bases."
        assert "gap_0" not in {c.identifier for c in chosen}

    def test_selection_is_returned_in_sequence_order(self) -> None:
        """Ranking decides *which* gaps; reporting reads along the sequence."""
        gaps = [context("late", 9000, 10), context("early", 100, 500)]
        chosen, _ = select(gaps, limit=2)
        assert [c.identifier for c in chosen] == ["late", "early"]
