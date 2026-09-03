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
from collections.abc import Callable, Iterator
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
    "- Scientific quality gates are hard constraints. Preserve every PARTIAL, REVISE, "
    "ABSTAIN, and warning reported in the quality constraints. State the caveat plainly "
    "and never describe such a result as complete, clean, conclusive, or fully validated.\n"
    "- Umbrella CAN generate scientific illustrations. Never tell the user it is unable "
    "to produce, draw or attach an image, and never offer hand-drawing instructions as a "
    "substitute. If an illustration was requested and none is listed below as shown in "
    "the interface, then generating it failed on this run - say that, and say nothing "
    "about the platform lacking the ability.\n"
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
            (
                "User's question:\n{user_query}\n\n"
                "Findings produced by the agents:\n{findings}\n\n"
                "Scientific quality constraints:\n{quality_constraints}\n\n"
                "Agents that ran, in order:\n{execution_history}{failure_note}"
            ),
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

        answer = self._direct_chain.invoke(self._direct_inputs(user_query))
        _logger.info("[Responder] answered directly (no agents used)")
        return answer

    def answer_directly_stream(self, user_query: str) -> Iterator[str]:
        """`answer_directly`, yielded chunk by chunk as the model writes it.

        Same prompt, same chain, same result once joined - the only difference
        is that the caller can show the text arriving instead of waiting for
        the last token before showing the first.
        """

        yield from self._direct_chain.stream(self._direct_inputs(user_query))
        _logger.info("[Responder] streamed a direct answer (no agents used)")

    def _direct_inputs(self, user_query: str) -> dict[str, Any]:
        return {
            "capabilities": _format_capabilities(self._agent_cards),
            "user_query": user_query,
        }

    def synthesize(
        self,
        user_query: str,
        context: dict[str, Any],
        execution_history: list[str],
        failure: str | None = None,
    ) -> str:
        """Write the final answer from what the agents found."""

        answer = self._synthesis_chain.invoke(
            self._synthesis_inputs(user_query, context, execution_history, failure)
        )
        _logger.info("[Responder] synthesized final answer from %d findings", len(context))
        return answer

    def synthesize_stream(
        self,
        user_query: str,
        context: dict[str, Any],
        execution_history: list[str],
        failure: str | None = None,
    ) -> Iterator[str]:
        """`synthesize`, yielded chunk by chunk as the model writes it."""

        yield from self._synthesis_chain.stream(
            self._synthesis_inputs(user_query, context, execution_history, failure)
        )
        _logger.info("[Responder] streamed final answer from %d findings", len(context))

    def _synthesis_inputs(
        self,
        user_query: str,
        context: dict[str, Any],
        execution_history: list[str],
        failure: str | None,
    ) -> dict[str, Any]:
        return {
            "user_query": user_query,
            "findings": _format_findings(context),
            "quality_constraints": _quality_constraints(context),
            "execution_history": "\n".join(f"- {entry}" for entry in execution_history),
            "failure_note": (
                f"\n\nNote: an agent failed with: {failure}. Explain this to the user "
                "honestly instead of guessing an answer."
                if failure
                else ""
            ),
        }


def _format_capabilities(agent_cards: dict[str, AgentCard]) -> str:
    """Plain-text list of what the platform can do, for the direct-reply prompt."""
    return "\n".join(
        f"- {card.description} ({', '.join(card.capabilities)})" for card in agent_cards.values()
    )


# No single finding needs more than this to be summarised, and everything past
# it is prompt tokens spent on nothing. The cap is a backstop, not a policy:
# large values reach here by accident (a base64 image, a full sequence, a
# thousand-row result), and the failure without it is an opaque context-length
# error from the model rather than an obviously truncated finding.
_MAX_FINDING_CHARS = 4000


def _render_value(value: Any) -> str:
    text = str(value)
    if len(text) <= _MAX_FINDING_CHARS:
        return text
    return (
        f"{text[:_MAX_FINDING_CHARS]}... [truncated, {len(text)} characters total - "
        f"summarise from what is shown]"
    )


# Context keys that exist for the browser, not for the model, mapped to what
# the model should know is on screen because of them. A viewer scene is
# coordinates, colours and URLs - the frontend renders it, and there is nothing
# in it the answer could quote. Left in, the Mol* scene alone spends most of
# `_MAX_FINDING_CHARS` and then gets cut mid-structure, so the model pays for
# it and reads a broken fragment. The agent's own summary key carries what the
# answer needs to say.
#
# The note matters as much as the exclusion: told nothing, the model closes
# with "to view this structure, go to rcsb.org and search for 1KZY" while the
# rotatable structure sits directly underneath its own answer.
# A note may be a plain string, or a function of the value when what the model
# needs to be told depends on what the agent actually produced - see the
# "writing" entry, where the section written and whether its references were
# real both change the sentence.
_RENDER_ONLY_KEYS: dict[str, str | Callable[[Any], str]] = {
    "protein_viewer": (
        "An interactive 3D viewer showing this structure is displayed directly below your "
        "answer. Refer to it as already visible; never tell the user to open RCSB, AlphaFold "
        "or another viewer to see it."
    ),
    # FLUX.2-pro answers with a base64 data URI, not a link - a single 1024x1024
    # image measured ~440 KB of characters. Left in, it is the biggest "finding"
    # by far: it would consume the whole `_MAX_FINDING_CHARS` budget and hand the
    # model 4000 characters of base64 to summarise.
    #
    # The note matters as much as the exclusion. Told nothing, the model sees a
    # request to draw and no drawing, and apologises - which is exactly what the
    # Genome agent did on the Arctic fox run ("I'm not able to draw an image of
    # an Arctic fox for you").
    "image": (
        "The generated illustration is displayed directly below your answer. Refer to it as "
        "already visible - describe what it shows and the biology behind it. Never say you "
        "cannot draw or display images, and never suggest searching for one elsewhere."
    ),
    # The Literature Agent's draft is the one finding the user asked to *read*,
    # and the frontend now renders it in its own panel under the answer. Left
    # in the findings it was reproduced verbatim in the answer as well, so the
    # user got the same 200-word abstract twice on one screen - and the model
    # spent most of `_MAX_FINDING_CHARS` on text it was only meant to hand over.
    "writing": lambda value: _writing_note(value),
}


def _writing_note(value: Any) -> str:
    """What the model should say about a draft it must not reproduce.

    Returns "" when this run produced no text at all. That is not a detail:
    an empty note puts the key back into the ordinary findings, so a failed
    writing run is explained by the answer instead of being suppressed - and
    the model is not told to point at a panel that the frontend, seeing no
    draft and no journals, does not render.
    """
    if not isinstance(value, dict):
        return ""

    # The sub-orchestrator's "both" route nests the two payloads; the single
    # routes return one of them flat. Handle both rather than assuming.
    writing = value.get("writing") if isinstance(value.get("writing"), dict) else value
    publication = value.get("publication") if isinstance(value.get("publication"), dict) else value

    parts: list[str] = []

    if isinstance(writing, dict) and writing.get("draft"):
        section = str(writing.get("section") or "text").lower()
        parts.append(
            f"The requested {section} has been written and is displayed in full, in its own "
            f"panel directly below your answer. Introduce it in one or two sentences - say "
            f"what it covers and how it is structured - and then STOP. Do NOT reproduce the "
            f"{section} itself, do not quote more than a few words of it, and never say you "
            f"are unable to write it."
        )
        if writing.get("references_are_placeholder"):
            parts.append(
                "Its reference list is placeholder data, not real publications, so add one "
                "short line telling the user the text cites nothing real yet."
            )
        elif not writing.get("references_used"):
            parts.append(
                "No literature was retrieved for it, so it deliberately contains no "
                "citations. Mention that in one short line."
            )

    if isinstance(publication, dict) and publication.get("recommended_journals"):
        parts.append(
            "Suggested publication venues are listed in the same panel below your answer. "
            "Refer to them as already visible rather than repeating the list."
        )

    return " ".join(parts)


def _render_notes(context: dict[str, Any]) -> dict[str, str]:
    """The render-only keys that actually rendered something, and their notes."""
    notes: dict[str, str] = {}
    for key, note in _RENDER_ONLY_KEYS.items():
        value = context.get(key)
        if not value:
            continue
        text = note(value) if callable(note) else note
        if text:
            notes[key] = text
    return notes


def _format_findings(context: dict[str, Any]) -> str:
    """Plain-text rendering of everything the agents produced."""
    # A key is only held back from the findings when it actually put something
    # on screen. A note function that returns "" is saying "nothing rendered
    # this time" - that key stays an ordinary finding, so a run that produced
    # no draft is still explained rather than silently dropped.
    notes = _render_notes(context)
    findings = {key: value for key, value in context.items() if key not in notes}
    rendered = list(notes.values())

    if not findings and not rendered:
        return "(the agents did not produce any findings)"

    lines = [f"- {key}: {_render_value(value)}" for key, value in findings.items()]
    lines.extend(f"- (shown in the interface) {note}" for note in rendered)
    return "\n".join(lines)


def _quality_constraints(context: dict[str, Any]) -> str:
    """Surface non-clean scientific states separately from free-form findings."""
    constraints: list[str] = []
    for key, value in context.items():
        if not isinstance(value, dict):
            continue

        # `or ""` rather than a default: an agent that reports `status: None`
        # means "not stated", and `str(None)` would turn that into the string
        # "NONE", which reads as a non-clean status and would make the model
        # hedge an answer that has nothing wrong with it.
        status = str(value.get("status") or "").upper()
        validation = str(value.get("validation_status") or "").upper()
        warnings = value.get("warnings")
        parts: list[str] = []

        if status and status != "COMPLETED":
            parts.append(f"status={status}")
        if validation and validation != "ACCEPT":
            parts.append(f"validation_status={validation}")
        if isinstance(warnings, list) and warnings:
            parts.append("warnings=" + "; ".join(str(item) for item in warnings))

        if parts:
            constraints.append(f"- {key}: " + "; ".join(parts))

    return "\n".join(constraints) if constraints else "(none reported)"
