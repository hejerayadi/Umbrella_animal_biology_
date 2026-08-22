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
