"""Offline validation of structured Planner and Resolver outputs."""

from typing import Any

import pytest
from pydantic import ValidationError

from backend.agent_card import AgentCard
from backend.orchestrator.capability_resolver import CapabilityResolver
from backend.orchestrator.planner import Planner


class _DictChain:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def invoke(self, _: dict[str, Any]) -> dict[str, Any]:
        return self.response


def _card(name: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} agent",
        capabilities=[name.lower()],
        required_inputs=[],
        produced_outputs=[],
    )


def test_planner_validates_dictionary_structured_output() -> None:
    planner = Planner.__new__(Planner)
    planner._agent_cards = {"Protein": _card("Protein")}
    planner._chain = _DictChain(  # type: ignore[assignment]
        {
            "reasoning": "The request concerns a protein structure.",
            "needs_agent": True,
            "initial_agent": "Protein",
        }
    )

    plan = planner.plan("Show the TP53 structure")

    assert plan.initial_agent == "Protein"
    assert plan.reasoning == "The request concerns a protein structure."


def test_resolver_validates_dictionary_structured_output() -> None:
    resolver = CapabilityResolver.__new__(CapabilityResolver)
    resolver._agent_cards = {
        "Protein": _card("Protein"),
        "Literature": _card("Literature"),
    }
    resolver._chain = _DictChain({"target_agent": "Literature"})  # type: ignore[assignment]

    target = resolver.resolve("Protein", "Find structural evidence")

    assert target == "Literature"


def test_invalid_planner_output_fails_at_the_boundary() -> None:
    planner = Planner.__new__(Planner)
    planner._agent_cards = {"Protein": _card("Protein")}
    planner._chain = _DictChain(  # type: ignore[assignment]
        {"needs_agent": True, "initial_agent": "Protein"}
    )

    with pytest.raises(ValidationError):
        planner.plan("Show the TP53 structure")


def _planner_returning(response: dict[str, Any]) -> Planner:
    planner = Planner.__new__(Planner)
    planner._agent_cards = {
        "Trait": _card("Trait"),
        "ImageGeneration": _card("ImageGeneration"),
    }
    planner._chain = _DictChain(response)  # type: ignore[assignment]
    return planner


def test_planner_carries_a_follow_up_agent() -> None:
    # "and draw it" is a second request in the same sentence, and only the
    # planner ever reads the sentence - no agent downstream can infer it.
    plan = _planner_returning(
        {
            "reasoning": "A traits question that also asks for an illustration.",
            "needs_agent": True,
            "initial_agent": "Trait",
            "follow_up_agent": "ImageGeneration",
        }
    ).plan("What traits let the Arctic fox survive the cold, and draw it")

    assert plan.initial_agent == "Trait"
    assert plan.follow_up_agent == "ImageGeneration"


def test_planner_drops_an_unknown_follow_up_instead_of_raising() -> None:
    # Unlike an unknown `initial_agent`, which is fatal: without a starting
    # agent there is no run at all, whereas losing the optional second half
    # still leaves a complete answer to the question that was asked.
    plan = _planner_returning(
        {
            "reasoning": "Traits, plus a picture.",
            "needs_agent": True,
            "initial_agent": "Trait",
            "follow_up_agent": "Illustrator",
        }
    ).plan("What traits let the Arctic fox survive the cold, and draw it")

    assert plan.initial_agent == "Trait"
    assert plan.follow_up_agent is None


def test_planner_drops_a_follow_up_that_repeats_the_initial_agent() -> None:
    # Running it twice would hand it a context it wrote itself - a wasted call
    # at best, a second image replacing the first at worst.
    plan = _planner_returning(
        {
            "reasoning": "Just a drawing.",
            "needs_agent": True,
            "initial_agent": "ImageGeneration",
            "follow_up_agent": "ImageGeneration",
        }
    ).plan("Draw an Arctic fox")

    assert plan.initial_agent == "ImageGeneration"
    assert plan.follow_up_agent is None


def test_planner_defaults_the_follow_up_to_empty() -> None:
    # An ordinary research question schedules nothing extra, so the graph
    # behaves exactly as it did before the slot existed.
    plan = _planner_returning(
        {
            "reasoning": "A plain traits question.",
            "needs_agent": True,
            "initial_agent": "Trait",
        }
    ).plan("What traits let the Arctic fox survive the cold?")

    assert plan.follow_up_agent is None
