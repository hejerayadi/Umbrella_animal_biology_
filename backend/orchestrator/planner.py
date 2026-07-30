"""LLM-driven planner: chooses which worker agent handles a request first.

The planner is not keyword based. It uses Azure OpenAI, through a LangChain
prompt template and structured output, to pick the initial agent from the
catalog in `backend/registry.py` - which is itself built only from the
stable `card.json` files, not from the agents' (still-changing) Python code.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..agent_card import AgentCard
from ..registry import AGENT_CARDS
from .llm import get_llm

# The instructions we give the AI model. `{agent_catalog}` and `{user_query}`
# are placeholders that get filled in with real values every time we ask it
# a question - this is the "prompt template" mentioned in the docstring above.
_SYSTEM_PROMPT = (
    "You are the planning module of a scientific multi-agent orchestrator.\n"
    "Choose exactly one worker agent to handle the user's request first.\n"
    "Only choose from the agents listed below, using their name exactly as written.\n\n"
    "Available agents:\n{agent_catalog}"
)

# Bundles the system instructions above with the user's actual question into
# one reusable template. Every call to the planner fills in the blanks and
# sends this to the model.
_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{user_query}"),
    ]
)


class _PlannerOutput(BaseModel):
    """Structured output requested from the LLM.

    Instead of asking the model to write a free-form sentence and then trying
    to guess what it meant, we ask it to fill in this exact shape (one field:
    `initial_agent`). That way we always get back something we can use in
    code directly, no guessing involved.
    """

    initial_agent: str = Field(
        description="Exact name of the worker agent that should handle the request first."
    )


@dataclass(frozen=True)
class ExecutionPlan:
    """The initial routing decision produced from a user's query."""

    initial_agent: str


class Planner:
    """Decides which worker agent starts the workflow for a given query."""

    def __init__(self, agent_cards: dict[str, AgentCard] | None = None) -> None:
        # Which agents we're allowed to choose from. If nothing is passed in,
        # fall back to the full list of agents the orchestrator knows about.
        self._agent_cards = agent_cards if agent_cards is not None else AGENT_CARDS

        # Wire the prompt template together with the LLM, telling the LLM to
        # answer using the `_PlannerOutput` shape defined above. The `|` here
        # is LangChain's way of chaining steps: "build the prompt, then send
        # it to the model."
        self._chain = _PROMPT | get_llm().with_structured_output(_PlannerOutput)

    def plan(self, user_query: str) -> ExecutionPlan:
        """Ask the LLM to pick the initial agent for `user_query`."""

        # Actually call the model: fill in the prompt's placeholders and get
        # back a `_PlannerOutput` object.
        response = self._chain.invoke(
            {
                "agent_catalog": _format_agent_catalog(self._agent_cards),
                "user_query": user_query,
            }
        )

        # Safety check: if the model somehow answers with an agent name that
        # doesn't actually exist, fail loudly now instead of silently
        # breaking later when we try to run a non-existent agent.
        if response.initial_agent not in self._agent_cards:
            raise ValueError(
                f"Planner selected unknown agent '{response.initial_agent}'; "
                f"known agents: {sorted(self._agent_cards)}"
            )

        return ExecutionPlan(initial_agent=response.initial_agent)


def _format_agent_catalog(agent_cards: dict[str, AgentCard]) -> str:
    """Turn the agent dictionary into a plain-text bullet list for the prompt.

    e.g. "- Genome: Retrieves genomic data (capabilities: Genome retrieval, ...)"
    This is what actually gets shown to the AI model so it knows its options.
    """
    return "\n".join(
        f"- {name}: {card.description} (capabilities: {', '.join(card.capabilities)})"
        for name, card in agent_cards.items()
    )
