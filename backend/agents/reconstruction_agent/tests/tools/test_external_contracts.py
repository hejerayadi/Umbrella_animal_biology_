"""Guards on the exact wire shapes the external services demand.

Both bugs below reached a live 400/404 before being caught, because nothing
offline exercised the request shape. These tests are cheap and would have
caught either one.
"""
from __future__ import annotations

import pytest

from infrastructure.embl_ebi.blast_client import (
    _EXPECT_VALUES,
    _HIT_COUNTS,
    _snap_expect,
    _snap_hit_count,
)
from infrastructure.nvidia.client import Continuation
from tools.blast.schemas import BlastSearchInput
from tools.evo.tool import Evo2PlausibilityTool


class TestBlastParameterSnapping:
    """EMBL-EBI validates these against fixed lists and 400s on anything else."""

    def test_python_float_formatting_does_not_leak_into_exp(self) -> None:
        """The original bug: str(1e-5) is '1e-05', which EBI rejects."""
        assert _snap_expect(1e-5) == "1e-5"
        assert str(1e-5) not in {value for _, value in _EXPECT_VALUES}

    @pytest.mark.parametrize("expect", [1e-200, 1e-100, 1e-50, 1e-10, 1e-5, 1e-3, 1.0, 10.0])
    def test_every_snapped_expect_is_an_allowed_value(self, expect: float) -> None:
        assert _snap_expect(expect) in {value for _, value in _EXPECT_VALUES}

    def test_arbitrary_expect_snaps_to_the_nearest_allowed(self) -> None:
        """Compared in log space - these span 200 orders of magnitude."""
        assert _snap_expect(2e-6) == "1e-5"
        assert _snap_expect(3e-11) == "1e-10"

    def test_zero_expect_snaps_to_the_strictest_rather_than_crashing(self) -> None:
        assert _snap_expect(0.0) == "1e-200"

    @pytest.mark.parametrize("requested", [1, 5, 37, 50, 99, 1000, 5000])
    def test_every_snapped_hit_count_is_allowed(self, requested: int) -> None:
        assert _snap_hit_count(requested) in _HIT_COUNTS

    def test_hit_count_rounds_up_so_callers_never_get_fewer(self) -> None:
        assert _snap_hit_count(37) == 50
        assert _snap_hit_count(6) == 10

    def test_default_database_is_a_real_ena_partition(self) -> None:
        """'em_rel' is not a database; EBI partitions ENA by division."""
        assert BlastSearchInput(sequence="ACGT").database != "em_rel"
        assert BlastSearchInput(sequence="ACGT").database.startswith("em_")


class TestEvo2Agreement:
    """Evo 2 has no scoring endpoint; agreement is measured against /generate."""

    def test_identical_candidate_scores_one(self) -> None:
        continuation = Continuation(sequence="ACGT", sampled_probs=[0.9, 0.9, 0.9, 0.9])

        result = Evo2PlausibilityTool._compare("c1", "ACGT", continuation)

        assert result.agreement == 1.0
        assert result.weighted_agreement == 1.0

    def test_fully_divergent_candidate_scores_zero(self) -> None:
        continuation = Continuation(sequence="ACGT", sampled_probs=[0.9, 0.9, 0.9, 0.9])

        result = Evo2PlausibilityTool._compare("c1", "TGCA", continuation)

        assert result.agreement == 0.0
        assert result.weighted_agreement == 0.0

    def test_a_mismatch_the_model_was_unsure_about_costs_less(self) -> None:
        """That weighting is the whole point of asking for sampled_probs."""
        confident = Continuation(sequence="AC", sampled_probs=[0.99, 0.99])
        unsure = Continuation(sequence="AC", sampled_probs=[0.99, 0.30])

        # Both candidates match position 0 and miss position 1.
        confident_score = Evo2PlausibilityTool._compare("c", "AG", confident).weighted_agreement
        unsure_score = Evo2PlausibilityTool._compare("c", "AG", unsure).weighted_agreement

        assert unsure_score > confident_score

    def test_missing_probabilities_do_not_crash_the_comparison(self) -> None:
        continuation = Continuation(sequence="ACGT", sampled_probs=[])

        result = Evo2PlausibilityTool._compare("c1", "ACGT", continuation)

        assert result.agreement == 1.0

    def test_no_overlap_scores_zero_rather_than_dividing_by_zero(self) -> None:
        continuation = Continuation(sequence="", sampled_probs=[])

        result = Evo2PlausibilityTool._compare("c1", "ACGT", continuation)

        assert result.agreement == 0.0

    def test_mean_confidence_of_an_empty_continuation_is_zero(self) -> None:
        assert Continuation(sequence="", sampled_probs=[]).mean_confidence == 0.0
