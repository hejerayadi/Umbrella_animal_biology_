"""The confidence gate."""
from __future__ import annotations

import pytest

from ..config import ThresholdConfig
from ..domain.confidence import clarification_for, decide
from .conftest import candidate

THRESHOLDS = ThresholdConfig(
    identified_min_score=0.75, identified_min_margin=0.08, uncertain_min_score=0.45
)


def test_no_candidates_is_not_identified():
    assert decide([], None, "neutral", THRESHOLDS) == "not_identified"


def test_clear_winner_is_identified():
    candidates = [candidate("panthera_leo", 0.92), candidate("panthera_tigris", 0.60)]
    assert decide(candidates, 0.32, "neutral", THRESHOLDS) == "identified"


def test_close_candidates_are_uncertain():
    candidates = [candidate("panthera_leo", 0.88), candidate("panthera_tigris", 0.86)]
    assert decide(candidates, 0.02, "neutral", THRESHOLDS) == "uncertain"


def test_weak_top_score_is_not_identified():
    candidates = [candidate("panthera_leo", 0.30)]
    assert decide(candidates, 1.0, "neutral", THRESHOLDS) == "not_identified"


def test_middling_score_is_uncertain():
    candidates = [candidate("panthera_leo", 0.60)]
    assert decide(candidates, 1.0, "neutral", THRESHOLDS) == "uncertain"


def test_conflict_can_never_produce_identified():
    """Even a perfect retrieval must not be called identified when the text
    names a different species."""
    candidates = [candidate("panthera_leo", 0.99), candidate("panthera_tigris", 0.10)]
    assert decide(candidates, 0.89, "conflict", THRESHOLDS) == "uncertain"


def test_agreement_does_not_promote_a_weak_result():
    """Text can hold a result back. It can never push one up."""
    candidates = [candidate("panthera_leo", 0.50)]
    assert decide(candidates, 1.0, "agree", THRESHOLDS) == "uncertain"
    assert decide(candidates, 1.0, "neutral", THRESHOLDS) == "uncertain"


def test_agreement_does_not_change_the_score():
    candidates = [candidate("panthera_leo", 0.92)]
    before = candidates[0].similarity_score
    decide(candidates, 1.0, "agree", THRESHOLDS)
    assert candidates[0].similarity_score == before


def test_thresholds_are_configurable():
    strict = ThresholdConfig(
        identified_min_score=0.95, identified_min_margin=0.5, uncertain_min_score=0.9
    )
    candidates = [candidate("panthera_leo", 0.92)]

    # Same evidence, different boundaries: 0.92 clears the default bar but not
    # the strict one, where it stays above the floor and so lands on uncertain.
    assert decide(candidates, 1.0, "neutral", THRESHOLDS) == "identified"
    assert decide(candidates, 1.0, "neutral", strict) == "uncertain"

    # Below the strict floor there is nothing worth reporting at all.
    assert decide([candidate("panthera_leo", 0.80)], 1.0, "neutral", strict) == "not_identified"


def test_margin_alone_can_block_identification():
    """A high score with a crowded field is not an identification."""
    candidates = [candidate("panthera_leo", 0.95), candidate("panthera_tigris", 0.94)]
    assert decide(candidates, 0.01, "neutral", THRESHOLDS) == "uncertain"


@pytest.mark.parametrize("outcome", ["uncertain", "not_identified"])
def test_inconclusive_outcomes_get_a_clarification_question(outcome):
    question = clarification_for(outcome, [candidate("panthera_leo", 0.6)])
    assert question and question.endswith("?")


def test_identified_needs_no_clarification():
    assert clarification_for("identified", [candidate("panthera_leo", 0.9)]) is None
