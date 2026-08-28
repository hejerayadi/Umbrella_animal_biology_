"""Plan validation, and the rule that a bad plan costs an iteration and no more.

The planner is the one place a language model's output reaches control flow, so
the boundary is tested from both sides: what a valid plan may look like, and
that everything else is replaced rather than repaired.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from reconstruction_agent.agent.planner import (
    CANONICAL_PLAN,
    DEFAULT_PLAN,
    Planner,
    ProposedPlan,
    validate,
)
from reconstruction_agent.domain.enums import ToolName


class _Model:
    """An LLM double that answers with whatever it was constructed with."""

    def __init__(self, reply: BaseModel | None, *, available: bool = True) -> None:
        self._reply = reply
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    async def structured(self, **kwargs: Any) -> BaseModel | None:
        self.calls += 1
        return self._reply


class TestWhatCountsAsAnExecutablePlan:
    def test_the_canonical_order_is_valid(self) -> None:
        assert validate([action.value for action in CANONICAL_PLAN]) == CANONICAL_PLAN

    def test_the_default_plan_is_a_valid_subsequence_of_it(self) -> None:
        """Arbitration, rescoring and revalidation are replan responses to a
        measured deficit, not steps to schedule before anything has failed."""
        assert validate([action.value for action in DEFAULT_PLAN]) == DEFAULT_PLAN
        assert set(DEFAULT_PLAN) < set(CANONICAL_PLAN)

    def test_a_shorter_subsequence_is_valid(self) -> None:
        """Skipping taxonomy when the organism is already known is the kind of
        shortening a model is genuinely useful for - provided the required
        actions survive it."""
        actions = validate(
            [
                "get_sequence_context",
                "search_homologs",
                "evaluate_with_evo2",
                "score_candidate",
                "finalize_result",
            ]
        )
        assert actions is not None
        assert ToolName.GET_ASSEMBLY_METADATA not in actions
        assert ToolName.SEARCH_HOMOLOGS in actions

    def test_a_plan_dropping_evo2_is_rejected(self) -> None:
        """Observed in a live run: the model produced a plan without it, which
        was a legal subsequence and silently removed the only way a gap with no
        gap-spanning homologue ever gets filled."""
        assert (
            validate(
                [
                    "get_sequence_context",
                    "search_homologs",
                    "generate_candidates",
                    "finalize_result",
                ]
            )
            is None
        )

    def test_an_out_of_order_plan_is_rejected(self) -> None:
        """Aligning before fetching would dispatch a tool whose input does not exist."""
        assert (
            validate(
                ["get_sequence_context", "align_homologs", "search_homologs", "finalize_result"]
            )
            is None
        )

    def test_a_plan_without_finalisation_is_rejected(self) -> None:
        """A gap that is never committed to or refused is a gap the run cannot report."""
        assert validate(["get_sequence_context", "search_homologs"]) is None

    def test_a_plan_without_the_first_action_is_rejected(self) -> None:
        assert validate(["search_homologs", "finalize_result"]) is None

    def test_a_repeated_action_is_rejected(self) -> None:
        """More of something is a replan decided from evidence, not a scheduled repeat."""
        assert validate(["get_sequence_context", "get_sequence_context", "finalize_result"]) is None

    def test_an_unknown_action_is_rejected(self) -> None:
        assert validate(["get_sequence_context", "consult_oracle", "finalize_result"]) is None

    def test_an_empty_plan_is_rejected(self) -> None:
        assert validate([]) is None


class TestTheDeterministicPathIsAlwaysAvailable:
    async def test_no_model_gives_the_canonical_plan(self) -> None:
        planner = Planner(_Model(None, available=False))
        plan = await planner.plan(gap_id="gap_1", gap_length=45)

        assert plan.actions == DEFAULT_PLAN
        assert plan.model_authored is False

    async def test_an_unavailable_model_is_never_called(self) -> None:
        model = _Model(None, available=False)
        await Planner(model).plan(gap_id="gap_1", gap_length=45)

        assert model.calls == 0

    async def test_an_invalid_proposal_is_replaced_not_repaired(self) -> None:
        """A half-valid plan executed hopefully is worse than a correct one
        executed plainly."""
        model = _Model(ProposedPlan(actions=["align_homologs"], rationale="straight to it"))
        plan = await Planner(model).plan(gap_id="gap_1", gap_length=45)

        assert plan.actions == DEFAULT_PLAN
        assert plan.model_authored is False

    async def test_an_unparseable_reply_falls_back(self) -> None:
        plan = await Planner(_Model(None)).plan(gap_id="gap_1", gap_length=45)
        assert plan.actions == DEFAULT_PLAN

    async def test_a_valid_proposal_is_used(self) -> None:
        model = _Model(
            ProposedPlan(
                actions=[
                    "get_sequence_context",
                    "search_homologs",
                    "evaluate_with_evo2",
                    "score_candidate",
                    "finalize_result",
                ],
                rationale="the organism is already known",
            )
        )
        plan = await Planner(model).plan(gap_id="gap_1", gap_length=45)

        assert plan.model_authored is True
        assert ToolName.GET_ASSEMBLY_METADATA not in plan.actions


@pytest.mark.parametrize(
    "action",
    [
        ToolName.GET_SEQUENCE_CONTEXT,
        ToolName.EVALUATE_WITH_EVO2,
        ToolName.SCORE_CANDIDATE,
        ToolName.FINALIZE_RESULT,
    ],
)
def test_the_required_actions_cannot_be_dropped(action: ToolName) -> None:
    proposed = [item.value for item in CANONICAL_PLAN if item is not action]
    assert validate(proposed) is None
