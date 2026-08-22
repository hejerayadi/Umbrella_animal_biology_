"""Regression guards on the end-to-end pipeline and on the confidence score.

Two claims are protected here, both of which took a rebuild to earn and either
of which a future tuning pass could silently undo:

1. **Every retrieval stage reaches the aligner.** BLAST hits used to arrive as
   metadata with no residues, so `ToolSelector._mafft` filtered all of them out
   and the default plan could not align anything. `retrieval_recall` is the
   number that would have been 0.

2. **The confidence score ranks correct reconstructions above incorrect ones.**
   It measured 0.575 before - barely above the 0.5 that means no signal - which
   is why the threshold had to be set so high that recall collapsed to 0.289.

A small slice of the benchmark grid, so this stays fast enough to run on every
commit. `scripts/benchmark_pipeline.py` runs the full grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from benchmark_pipeline import Case, report, run_case  # noqa: E402

pytestmark = [pytest.mark.evaluation, pytest.mark.asyncio]

#: Enough of the grid to cover the ways evidence goes wrong - divergence,
#: pseudogenes, minus-strand hits, thin reference sets - without the runtime of
#: the full sweep.
CASES = [
    Case(30, 4, 1.00, 0.00, 0.00, False),
    Case(30, 4, 0.75, 0.05, 0.25, False),
    Case(30, 2, 1.00, 0.00, 0.00, True),
    Case(120, 8, 0.75, 0.05, 0.00, False),
    Case(120, 4, 1.00, 0.05, 0.25, True),
    Case(120, 8, 1.00, 0.00, 0.00, False),
    # Evenly split evidence, which is where every confidently-wrong answer came
    # from. Included so the ranking check has both populations to compare - and
    # so a change that stops crushing coin flips is caught here.
    Case(30, 2, 0.50, 0.00, 0.00, False),
    Case(120, 2, 0.50, 0.05, 0.00, False),
]

THRESHOLD = 0.25


@pytest.fixture(scope="module")
async def traces() -> list:
    return [await run_case(case, THRESHOLD, seed=20260819) for case in CASES]


class TestEvidencePathReachesTheAligner:
    async def test_every_hit_is_retrieved_with_its_residues(self, traces: list) -> None:
        """The defect that made the default plan unable to align anything."""
        assert all(trace.retrieved for trace in traces)

    async def test_alignments_locate_the_gap(self, traces: list) -> None:
        assert all(trace.aligned for trace in traces)

    async def test_a_candidate_is_produced_for_every_case(self, traces: list) -> None:
        assert all(trace.candidate for trace in traces)


class TestQuality:
    async def test_recall_stays_well_above_the_old_pipeline(self, traces: list) -> None:
        """0.451 end to end before the rebuild, at perfect precision."""
        metrics = report(list(traces), THRESHOLD)

        assert metrics["recall"] >= 0.70

    async def test_precision_stays_within_the_stated_floor(self, traces: list) -> None:
        """Recall was bought deliberately, but only down to a floor."""
        metrics = report(list(traces), THRESHOLD)

        assert metrics["precision"] >= 0.90

    async def test_nothing_is_silently_withheld(self, traces: list) -> None:
        """Coverage counts answers produced, accepted or not."""
        metrics = report(list(traces), THRESHOLD)

        assert metrics["coverage"] == 1.0


class TestConfidenceCarriesSignal:
    """Without this, a threshold cannot trade recall for precision at all."""

    async def test_correct_reconstructions_outrank_incorrect_ones(
        self, traces: list
    ) -> None:
        right = [t.confidence for t in traces if t.correct]
        wrong = [t.confidence for t in traces if t.produced and not t.correct]

        if not right or not wrong:
            pytest.skip("this slice of the grid produced only one population")

        wins = sum(
            1.0 if r > w else 0.5 if r == w else 0.0 for r in right for w in wrong
        )
        ranking = wins / (len(right) * len(wrong))

        assert ranking >= 0.90, f"the score ranked correct above incorrect at only {ranking:.3f}"
