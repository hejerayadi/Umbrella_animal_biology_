"""LLM-driven capability resolver.

Given the agent that is currently waiting and the prompt describing what it
is missing, the resolver uses Azure OpenAI (via LangChain) to decide which
registered worker agent can satisfy that request. It never uses keyword
matching.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..agent_card import AgentCard
from .llm import get_llm

# The instructions we give the AI model when an agent is stuck and needs
# help. `{agent_catalog}` gets filled in with the list of available agents.
_SYSTEM_PROMPT = (
    "You are the capability resolver of a scientific multi-agent orchestrator.\n"
    "A worker agent has paused because it needs information it cannot produce itself.\n"
    "Pick exactly one agent, from the list below, able to satisfy that request.\n"
    "Use the agent name exactly as written. Never pick the waiting agent itself.\n\n"
    "Available agents:\n{agent_catalog}"
)

# Combines the system instructions with the specific "who's waiting, what do
# they need" question into one reusable template.
_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            "Waiting agent: {current_agent}\nPrompt requesting missing information: {prompt_to_target_agent}",
        ),
    ]
)


class _ResolverOutput(BaseModel):
    """Structured output requested from the LLM: just the chosen agent's name."""

    target_agent: str = Field(
        description="Exact name of the worker agent that can satisfy the missing-information request."
    )


class CapabilityResolver:
    """Maps a waiting agent's request to the worker agent that can help."""

    def __init__(self, agent_cards: dict[str, AgentCard]) -> None:
        # The full set of agents we're allowed to pick a helper from.
        self._agent_cards = agent_cards

        # Chain the prompt template into the model, telling it to answer
        # using the `_ResolverOutput` shape (one field: `target_agent`).
        self._chain = _PROMPT | get_llm().with_structured_output(_ResolverOutput)

    def resolve(self, current_agent: str, prompt_to_target_agent: str) -> str:
        """Return the name of the agent that should handle the missing request."""

        # Ask the model: "here's who's waiting and what they need - who can help?"
        # We deliberately exclude the waiting agent itself from the list of
        # options, so the model can't just pick the same agent again.
        response = self._chain.invoke(
            {
                "agent_catalog": _format_agent_catalog(self._agent_cards, exclude=current_agent),
                "current_agent": current_agent,
                "prompt_to_target_agent": prompt_to_target_agent,
            }
        )

        # Safety check #1: make sure the model picked a real, known agent.
        if response.target_agent not in self._agent_cards:
            raise ValueError(
                f"Capability resolver selected unknown agent '{response.target_agent}'; "
                f"known agents: {sorted(self._agent_cards)}"
            )

        # Safety check #2: make sure it didn't just send the waiting agent
        # back to itself (which would cause an infinite loop).
        if response.target_agent == current_agent:
            raise ValueError(
                f"Capability resolver selected the waiting agent '{current_agent}' as its own dependency"
            )

        return response.target_agent


def _format_agent_catalog(agent_cards: dict[str, AgentCard], *, exclude: str) -> str:
    """Turn the agent dictionary into a plain-text bullet list, minus one agent.

    Same idea as the planner's version of this helper, but here we skip the
    `exclude` agent (the one currently waiting) so it's never offered as its
    own solution.
    """
    return "\n".join(
        f"- {name}: {card.description} (capabilities: {', '.join(card.capabilities)})"
        for name, card in agent_cards.items()
        if name != exclude
    )
