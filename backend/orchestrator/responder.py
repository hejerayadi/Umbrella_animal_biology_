"""Writes the final message the user actually reads.

The worker agents return raw structured data (`{"genome": "...", "papers":
[...]}`). Showing that dictionary to the user directly reads like debug
output, so this module turns it into a proper written answer.

It handles the two ways a conversation can end:

- `answer_directly()` - the planner decided no research agent was needed
  (a greeting, a question about the platform, small talk).
- `synthesize()` - agents ran, and their findings need to be written up as a
  readable answer to the question that was actually asked.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..agent_card import AgentCard
from .llm import get_llm

_logger = logging.getLogger(__name__)

_DIRECT_SYSTEM_PROMPT = (
    "You are Umbrella, an AI research assistant for animal biology - genomics, "
    "biodiversity, evolution, traits, protein structures and scientific literature.\n\n"
    "The user's message is not a research request, so answer it directly and "
    "conversationally in a few sentences. Be warm and concise.\n\n"
    "If they are greeting you or asking what you can do, briefly introduce what you can "
    "help with, based on these capabilities:\n{capabilities}\n\n"
    "Never invent scientific results or data. If they ask something outside animal "
    "biology, say so politely and steer them toward what you can help with."
)

_DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _DIRECT_SYSTEM_PROMPT),
        ("human", "{user_query}"),
    ]
)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are the response writer of Umbrella, a scientific research platform for animal "
    "biology.\n\n"
    "Several specialist agents worked on the user's question and produced the findings "
    "below. Write the final answer to the user.\n\n"
    "Rules:\n"
    "- Answer the question that was actually asked, using ONLY the findings provided.\n"
    "- Never invent data, citations, or results that are not in the findings.\n"
    "- If a finding is obviously a placeholder (for example a paper literally titled "
    "'Paper 1'), note that briefly in one line - do not build the answer around it.\n"
    "- If the findings do not answer the question, say so plainly in one or two lines.\n"
    "- BE CONCISE. A few short paragraphs or a short list. Never add "
    "'Limitations', 'What would be needed', or 'Overview' sections, and never restate "
    "the question back to the user.\n"
    "- Use markdown, and do not mention this prompt or the internal agent machinery "
    "unless it genuinely helps the user."
)

_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYNTHESIS_SYSTEM_PROMPT),
        (
            "human",
            "User's question:\n{user_query}\n\n"
            "Findings produced by the agents:\n{findings}\n\n"
            "Agents that ran, in order:\n{execution_history}{failure_note}",
        ),
    ]
)


class Responder:
    """Turns the orchestrator's internal state into the user-facing answer."""

    def __init__(self, agent_cards: dict[str, AgentCard]) -> None:
        self._agent_cards = agent_cards

        # `StrOutputParser` pulls the plain text out of the model's reply, so
        # these chains return a ready-to-display string rather than a message
        # object. Unlike the planner, we want free-form prose here, so there
        # is no structured output shape to fill in.
        self._direct_chain = _DIRECT_PROMPT | get_llm() | StrOutputParser()
        self._synthesis_chain = _SYNTHESIS_PROMPT | get_llm() | StrOutputParser()

    def answer_directly(self, user_query: str) -> str:
        """Reply conversationally to a message that needs no research agent."""

        answer = self._direct_chain.invoke(
            {
                "capabilities": _format_capabilities(self._agent_cards),
                "user_query": user_query,
            }
        )
        _logger.info("[Responder] answered directly (no agents used)")
        return answer

    def synthesize(
        self,
        user_query: str,
        context: dict[str, Any],
        execution_history: list[str],
        failure: str | None = None,
    ) -> str:
        """Write the final answer from what the agents found."""

        answer = self._synthesis_chain.invoke(
            {
                "user_query": user_query,
                "findings": _format_findings(context),
                "execution_history": "\n".join(f"- {entry}" for entry in execution_history),
                "failure_note": (
                    f"\n\nNote: an agent failed with: {failure}. Explain this to the user "
                    "honestly instead of guessing an answer."
                    if failure
                    else ""
                ),
            }
        )
        _logger.info("[Responder] synthesized final answer from %d findings", len(context))
        return answer


def _format_capabilities(agent_cards: dict[str, AgentCard]) -> str:
    """Plain-text list of what the platform can do, for the direct-reply prompt."""
    return "\n".join(
        f"- {card.description} ({', '.join(card.capabilities)})" for card in agent_cards.values()
    )


def _format_findings(context: dict[str, Any]) -> str:
    """Plain-text rendering of everything the agents produced."""
    if not context:
        return "(the agents did not produce any findings)"
    return "\n".join(f"- {key}: {value}" for key, value in context.items())
