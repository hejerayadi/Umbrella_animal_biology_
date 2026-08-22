from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ...llm.client import WRITING, call_llm
from ...llm.prompts import (
    PUBLICATION_SUPPORT_PROMPT,
    SCIENTIFIC_WRITING_PROMPT,
    WRITING_SUB_ROUTING_PROMPT,
)
from ...schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

# An abstract runs several hundred tokens, and on a reasoning deployment this
# budget covers reasoning tokens as well as the visible answer - too tight a
# ceiling comes back as finish_reason="length" with empty content.
_DRAFT_TOKENS = 2000
_ROUTING_TOKENS = 256

# Returned when the LLM cannot be reached. Labelled and FAILED rather than
# quietly empty, because the Global Orchestrator merges this dict into shared
# context and the Responder writes the user's final answer from it: handed an
# unlabelled stub for a "write my abstract" request, the Responder fills the
# gap from its own memory and produces a fabricated abstract - the exact
# failure this platform exists to avoid.
UNAVAILABLE_NOTICE = (
    "The Literature Agent's writing support could not reach its language model, "
    "so no draft was produced."
)
PLACEHOLDER_REFERENCES_NOTICE = (
    "The references behind this text are placeholders - the discovery step is "
    "not yet connected to PubMed - so the draft cites nothing real."
)


def _references(context: dict) -> tuple[list, bool]:
    """The papers discovery found, and whether they are real.

    `is_placeholder` is set by `discovery/sources.py` while it is still a stub.
    It is carried through rather than dropped, so a draft built on stand-in
    papers cannot be presented as if it were grounded in literature.
    """
    discovery = context.get("discovery_output")
    if not isinstance(discovery, dict):
        return [], False
    return discovery.get("papers", []), bool(discovery.get("is_placeholder"))


def _unavailable(field: str, papers: list) -> AgentResult:
    """The honest empty answer, used whenever the model produced nothing."""
    return AgentResult(
        status=AgentStatus.FAILED,
        output={
            field: None,
            "references_used": papers,
            "is_placeholder": True,
            "notice": UNAVAILABLE_NOTICE,
        },
    )


# ---------------------------------------------------------------------------
# Leaf agents
# ---------------------------------------------------------------------------


class WritingSupportAgent:
    """Drafts scientific text (abstract, introduction, section ...) with the LLM."""

    def run(self, request: AgentRequest) -> AgentResult:
        papers, refs_are_placeholder = _references(request.context or {})

        # The reference list is handed over explicitly and the prompt forbids
        # citing anything outside it. An empty list means "write without
        # citations", never "make some up".
        reference_block = (
            "\n".join(f"- {paper}" for paper in papers)
            if papers
            else "(none - write the text without any citations)"
        )
        user_message = (
            f"{request.instruction}\n\n"
            f"References available to cite:\n{reference_block}"
        )

        try:
            draft = call_llm(
                messages=[
                    {"role": "system", "content": SCIENTIFIC_WRITING_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=_DRAFT_TOKENS,
                reasoning_effort="low",
                role=WRITING,
            ).strip()
        except Exception as exc:  # noqa: BLE001 - degrade honestly, never invent
            _logger.warning("writing support LLM unavailable: %s", exc)
            return _unavailable("draft", papers)

        # An empty completion means truncated or filtered, not "no text needed".
        # Reported rather than passed on as a blank the Responder would fill in.
        if not draft:
            _logger.warning("writing support returned an empty draft")
            return _unavailable("draft", papers)

        output = {
            "draft": draft,
            "references_used": papers,
            "style": "academic",
        }
        if refs_are_placeholder:
            output["references_are_placeholder"] = True
            output["notice"] = PLACEHOLDER_REFERENCES_NOTICE

        return AgentResult(status=AgentStatus.COMPLETED, output=output)


class PublicationSupportAgent:
    """Recommends journals / venues for a given draft, with the LLM."""

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        writing_output = context.get("writing_output")
        draft = (
            (writing_output.get("draft") or "")
            if isinstance(writing_output, dict)
            else ""
        )

        user_message = request.instruction
        if draft:
            # Recommend against the text that was actually written, not just
            # the instruction - that is the point of running writing first.
            user_message += f"\n\nThe manuscript text under consideration:\n{draft}"

        try:
            recommendations = call_llm(
                messages=[
                    {"role": "system", "content": PUBLICATION_SUPPORT_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=_DRAFT_TOKENS,
                reasoning_effort="low",
                role=WRITING,
            ).strip()
        except Exception as exc:  # noqa: BLE001 - degrade honestly, never invent
            _logger.warning("publication support LLM unavailable: %s", exc)
            recommendations = ""

        if not recommendations:
            return _unavailable("recommended_journals", [])

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "recommended_journals": recommendations,
                "based_on_draft": bool(draft),
            },
        )


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class WritingState(TypedDict):
    request: AgentRequest
    route: str                        # "writing" | "publication" | "both"
    writing_result: AgentResult | None
    publication_result: AgentResult | None
    final_result: AgentResult | None


# ---------------------------------------------------------------------------
# Sub-orchestrator
# ---------------------------------------------------------------------------


class ScientificWritingOrchestrator:
    """Scientific Writing sub-orchestrator.

    Internal LangGraph pipeline (sequential when both tasks are needed):

        classify
          |- writing     -> writing_support -> aggregate -> END
          |- publication -> publication_support -> aggregate -> END
          |- both        -> writing_support -> publication_support -> aggregate -> END
    """

    def __init__(self) -> None:
        self._writing_support = WritingSupportAgent()
        self._publication_support = PublicationSupportAgent()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(WritingState)

        graph.add_node("classify", self._classify)
        graph.add_node("writing_support", self._run_writing)
        graph.add_node("publication_support", self._run_publication)
        graph.add_node("aggregate", self._aggregate)

        graph.set_entry_point("classify")

        # After classify -> branch to first relevant node
        graph.add_conditional_edges("classify", self._route_after_classify)

        # After writing_support -> either move to publication (both) or aggregate
        graph.add_conditional_edges("writing_support", self._route_after_writing)

        # publication_support always leads to aggregate
        graph.add_edge("publication_support", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_by_keyword(instruction: str) -> str:
        """Deterministic fallback for when the LLM is unavailable."""
        lowered = instruction.lower()
        needs_pub = any(
            k in lowered
            for k in ("journal", "venue", "publication", "publish", "submit", "reviewer")
        )
        needs_write = any(
            k in lowered
            for k in ("write", "draft", "abstract", "introduction", "section", "rewrite")
        )
        if needs_write and needs_pub:
            return "both"
        if needs_write:
            return "writing"
        return "publication"

    def _classify(self, state: WritingState) -> dict:
        """Decide which writing tasks are required, preferring the LLM."""
        instruction = state["request"].instruction

        try:
            answer = (
                call_llm(
                    messages=[
                        {"role": "system", "content": WRITING_SUB_ROUTING_PROMPT},
                        {"role": "user", "content": instruction},
                    ],
                    max_completion_tokens=_ROUTING_TOKENS,
                    reasoning_effort="none",
                    role=WRITING,
                )
                .strip()
                .lower()
            )
        except Exception:  # noqa: BLE001 - routing must never fail the request
            answer = ""

        mapped = {
            "writing_support": "writing",
            "publication_support": "publication",
            "both": "both",
        }.get(answer)

        return {"route": mapped or self._classify_by_keyword(instruction)}

    def _run_writing(self, state: WritingState) -> dict:
        return {"writing_result": self._writing_support.run(state["request"])}

    def _run_publication(self, state: WritingState) -> dict:
        """Run publication support, passing the draft along when there is one."""
        request = state["request"]
        context = dict(request.context or {})
        writing = state.get("writing_result")
        if writing and writing.status == AgentStatus.COMPLETED:
            context["writing_output"] = writing.output
        result = self._publication_support.run(
            AgentRequest(instruction=request.instruction, context=context)
        )
        return {"publication_result": result}

    def _aggregate(self, state: WritingState) -> dict:
        """Merge writing + publication outputs into a single AgentResult.

        FAILED propagates: if the branch that ran could not produce anything,
        the parent must hear about it rather than receive an empty COMPLETED
        that reads like a successful blank answer.
        """
        writing = state.get("writing_result")
        publication = state.get("publication_result")
        route = state["route"]

        if route == "writing":
            final_output = writing.output if writing else {}
            status = writing.status if writing else AgentStatus.FAILED
        elif route == "publication":
            final_output = publication.output if publication else {}
            status = publication.status if publication else AgentStatus.FAILED
        else:
            final_output = {
                "writing": writing.output if writing else None,
                "publication": publication.output if publication else None,
            }
            ran = [r for r in (writing, publication) if r]
            status = (
                AgentStatus.COMPLETED
                if any(r.status == AgentStatus.COMPLETED for r in ran)
                else AgentStatus.FAILED
            )

        return {"final_result": AgentResult(status=status, output=final_output)}

    # ------------------------------------------------------------------
    # Conditional edge functions
    # ------------------------------------------------------------------

    def _route_after_classify(self, state: WritingState) -> str:
        if state["route"] == "publication":
            return "publication_support"
        return "writing_support"  # writing or both

    def _route_after_writing(self, state: WritingState) -> str:
        if state["route"] == "both":
            return "publication_support"
        return "aggregate"

    # ------------------------------------------------------------------
    # Public interface (called by the parent LiteratureOrchestrator)
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        result = self._graph.invoke(
            {
                "request": request,
                "route": "",
                "writing_result": None,
                "publication_result": None,
                "final_result": None,
            }
        )
        return result["final_result"]
