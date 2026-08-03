"""LLM-driven planner: decides how a user's message should be handled.

The planner is not keyword based. It uses Azure OpenAI, through a LangChain
prompt template and structured output, to make two decisions at once:

1. Does this message actually need a research agent at all? A greeting or a
   question about the platform itself should be answered directly, not
   forced through a genomics agent.
2. If it does, which agent should START the work? Other agents get pulled in
   automatically later by the capability resolver, so the planner only picks
   the one that owns the core of the question.

The agent catalog it chooses from comes from `backend/registry.py`, which is
built only from the stable `card.json` files - not from the agents'
(still-changing) Python code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..agent_card import AgentCard
from ..registry import AGENT_CARDS
from .llm import get_llm

# Prints one line per decision to the console this process is running in
# (visible whether that's `python -m backend.main` or `uvicorn backend.api:app`).
_logger = logging.getLogger(__name__)

# The instructions we give the AI model. `{agent_catalog}` and `{user_query}`
# are placeholders that get filled in with real values every time we ask it
# a question - this is the "prompt template" mentioned in the docstring above.
_SYSTEM_PROMPT = (
    "You are the planning module of Umbrella, a scientific research platform for animal "
    "biology - genomics, biodiversity, evolution, traits, protein structures, species "
    "imagery and scientific literature.\n\n"
    "Decide how the user's message should be handled.\n\n"
    "Set `needs_agent` to false when the message is a greeting, small talk, a question "
    "about you or the platform, or anything that is not an animal-biology research "
    "request. In that case leave `initial_agent` empty - it will be answered "
    "conversationally. Never force a research agent onto a non-research message.\n\n"
    "Set `needs_agent` to true only for genuine research requests. Then pick the single "
    "agent that owns the CORE of the question as `initial_agent`. Do not try to list "
    "every agent involved: if the starting agent needs something from another agent, it "
    "will request that automatically later.\n\n"
    "Always fill in `reasoning` with one short sentence explaining your choice.\n\n"
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
    to guess what it meant, we ask it to fill in this exact shape. That way
    we always get back something we can use in code directly.

    `reasoning` is deliberately the FIRST field: the model fills the fields in
    order, so making it explain itself before committing to a choice tends to
    produce a better choice (and gives us a useful debug log line for free).
    """

    reasoning: str = Field(description="One short sentence explaining the decision.")
    needs_agent: bool = Field(
        description="True if a research worker agent is required, false for greetings, "
        "small talk, platform questions, or anything outside animal-biology research."
    )
    initial_agent: str | None = Field(
        default=None,
        description="Exact name of the agent to start with. Empty when needs_agent is false.",
    )


@dataclass(frozen=True)
class ExecutionPlan:
    """The routing decision produced from a user's message.

    `initial_agent` is `None` when no research agent is needed and the message
    should simply be answered conversationally.
    """

    initial_agent: str | None
    reasoning: str


class Planner:
    """Decides whether a message needs agents, and which one starts."""

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
        """Ask the LLM how `user_query` should be handled."""

        # Actually call the model: fill in the prompt's placeholders and get
        # back a `_PlannerOutput` object.
        response = self._chain.invoke(
            {
                "agent_catalog": _format_agent_catalog(self._agent_cards),
                "user_query": user_query,
            }
        )

        # No agent needed - this message gets a direct conversational reply.
        if not response.needs_agent or not response.initial_agent:
            _logger.info(
                "[Planner] query=%r -> no agent needed (%s)", user_query, response.reasoning
            )
            return ExecutionPlan(initial_agent=None, reasoning=response.reasoning)

        # Safety check: if the model somehow answers with an agent name that
        # doesn't actually exist, fail loudly now instead of silently
        # breaking later when we try to run a non-existent agent.
        if response.initial_agent not in self._agent_cards:
            raise ValueError(
                f"Planner selected unknown agent '{response.initial_agent}'; "
                f"known agents: {sorted(self._agent_cards)}"
            )

        _logger.info(
            "[Planner] query=%r -> %s (%s)",
            user_query,
            response.initial_agent,
            response.reasoning,
        )

        return ExecutionPlan(
            initial_agent=response.initial_agent,
            reasoning=response.reasoning,
        )


def _format_agent_catalog(agent_cards: dict[str, AgentCard]) -> str:
    """Turn the agent dictionary into a plain-text bullet list for the prompt.

    e.g. "- Genome: Retrieves genomic data (capabilities: Genome retrieval, ...)"
    This is what actually gets shown to the AI model so it knows its options.
    """
    return "\n".join(
        f"- {name}: {card.description} (capabilities: {', '.join(card.capabilities)})"
        for name, card in agent_cards.items()
    )
