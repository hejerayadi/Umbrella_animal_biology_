from __future__ import annotations

import json
import logging
import re

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ...llm.client import WRITING, call_llm, call_llm_with_tools
from ...llm.prompts import (
    PUBLICATION_SUPPORT_PROMPT,
    SCIENTIFIC_WRITING_PROMPT,
    WRITING_SUB_ROUTING_PROMPT,
)
from ...schema import AgentRequest, AgentResult, AgentStatus
from .kb.retrieval import search_papers, search_related_work, search_citations

_logger = logging.getLogger(__name__)

_DRAFT_TOKENS = 2000
_ROUTING_TOKENS = 256
_MAX_TOOL_ROUNDS = 4  # safety cap so a confused model can't loop forever

UNAVAILABLE_NOTICE = (
    "The Literature Agent's writing support could not reach its language model, "
    "so no draft was produced."
)
PLACEHOLDER_REFERENCES_NOTICE = (
    "The references behind this text are placeholders - the discovery step is "
    "not yet connected to PubMed - so the draft cites nothing real."
)


def _references(context: dict) -> tuple[list, bool]:
    """The papers discovery found, and whether they are real."""
    discovery = context.get("discovery_output")
    if not isinstance(discovery, dict):
        return [], False
    return discovery.get("papers", []), bool(discovery.get("is_placeholder"))


def _unavailable(field: str, papers: list) -> AgentResult:
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
# KB tools -- three separate, LLM-callable retrieval tools. None of their
# output is ever a citable source: only `references_available`, built from
# real discovery output, may be cited. Each tool's own docstring/description
# (used as its function-calling description) makes this explicit to the model.
# ---------------------------------------------------------------------------


def _tool_get_abstract_examples(query: str, limit: int = 2) -> str:
    """Retrieves example abstracts (structure/tone only, never citable) from
    the knowledge base, related to `query`."""
    try:
        hits = search_papers(query, section_type="abstract", limit=limit)
        examples = [h.payload.get("target_text", "")[:500] for h in hits]
        examples = [e for e in examples if e.strip()]
        if not examples:
            return "No abstract examples found."
        return "\n\n".join(f"Example {i+1}: {e}" for i, e in enumerate(examples))
    except Exception as exc:  # noqa: BLE001 - tool failure must not crash the loop
        _logger.warning("get_abstract_examples failed: %s", exc)
        return "Abstract-example retrieval is currently unavailable."


def _tool_get_related_work_examples(query: str, limit: int = 2) -> str:
    """Retrieves example related-work syntheses (structure only, never
    citable) from the knowledge base, related to `query`."""
    try:
        hits = search_related_work(query, limit=limit)
        examples = [h.payload.get("target_text", "")[:500] for h in hits]
        examples = [e for e in examples if e.strip()]
        if not examples:
            return "No related-work examples found."
        return "\n\n".join(f"Example {i+1}: {e}" for i, e in enumerate(examples))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("get_related_work_examples failed: %s", exc)
        return "Related-work-example retrieval is currently unavailable."


def _tool_get_citation_examples(query: str, limit: int = 2) -> str:
    """Retrieves example citation phrasings (labelled by rhetorical intent --
    Background / Method / Result Comparison -- phrasing only, never citable)
    from the knowledge base, related to `query`."""
    try:
        hits = search_citations(query, limit=limit)
        examples = [
            f"[{h.payload.get('intent', 'Background')}] {h.payload.get('context', '')}"
            for h in hits
            if h.payload.get("context", "").strip()
        ]
        if not examples:
            return "No citation-phrasing examples found."
        return "\n".join(examples)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("get_citation_examples failed: %s", exc)
        return "Citation-example retrieval is currently unavailable."


_TOOL_EXECUTORS = {
    "get_abstract_examples": _tool_get_abstract_examples,
    "get_related_work_examples": _tool_get_related_work_examples,
    "get_citation_examples": _tool_get_citation_examples,
}

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_abstract_examples",
            "description": (
                "Retrieve example academic abstracts for structure and tone "
                "guidance. Use this when drafting an abstract. These examples "
                "are NEVER citable sources - only for style/structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short description of the abstract's topic.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_work_examples",
            "description": (
                "Retrieve example related-work syntheses for structure "
                "guidance. Use this when drafting a related-work section. "
                "These examples are NEVER citable sources - only for structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short description of the related-work topic.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_citation_examples",
            "description": (
                "Retrieve example citation phrasings, each labelled by "
                "rhetorical intent (Background / Method / Result Comparison). "
                "Use this when deciding how to phrase an in-text citation. "
                "These examples are NEVER citable sources - only for phrasing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short description of the citation's context.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Style tool -- grammar/coherence correction, applied automatically as a
# post-processing step once the draft exists (not an LLM-selectable tool:
# there is nothing to correct before a draft exists, so the model is never
# asked to decide whether to invoke it).
# ---------------------------------------------------------------------------


#Motif binomial simple : Genre (majuscule) + espece/sous-espece (minuscules),
# ex. "Panthera tigris", "Panthera tigris altaica". Volontairement large :
# mieux vaut proteger un peu trop de faux positifs latins que d'abimer un
# vrai nom scientifique.
_SCIENTIFIC_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+ [a-z]+(?: [a-z]+)?\b")
 
 
import re

# Motif binomial simple : Genre (majuscule) + espece/sous-espece (minuscules),
# ex. "Panthera tigris", "Panthera tigris altaica". Volontairement large :
# mieux vaut proteger un peu trop de faux positifs latins que d'abimer un
# vrai nom scientifique.
_SCIENTIFIC_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+ [a-z]+(?: [a-z]+)?\b")


def _apply_style_correction(text: str) -> tuple[str, bool]:
    """Runs LanguageTool grammar/coherence correction on `text`, while
    protecting binomial scientific names (e.g. "Panthera tigris altaica")
    from being mangled -- LanguageTool's naive `.correct()` treats unknown
    Latin species names as spelling errors and rewrites them, which is worse
    than not correcting at all for a scientific-writing agent.

    Returns (corrected_text, was_applied). Degrades gracefully - if
    LanguageTool is not installed or fails to start, the original text is
    returned unchanged rather than blocking the draft.
    """
    try:
        import language_tool_python

        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text)

        protected_spans = [
            (m.start(), m.end()) for m in _SCIENTIFIC_NAME_PATTERN.finditer(text)
        ]

        def _overlaps_protected(match) -> bool:
            start, end = match.offset, match.offset + match.error_length
            return any(start < p_end and end > p_start for p_start, p_end in protected_spans)

        filtered_matches = [m for m in matches if not _overlaps_protected(m)]

        corrected = language_tool_python.utils.correct(text, filtered_matches)
        tool.close()
        return corrected, True
    except Exception as exc:  # noqa: BLE001 - style correction is a nice-to-have
        _logger.warning("style correction unavailable: %s", exc)
        return text, False
# ---------------------------------------------------------------------------
# Leaf agents
# ---------------------------------------------------------------------------


class WritingSupportAgent:
    """Drafts scientific text (abstract, introduction, section ...) with the
    LLM, using real function-calling: the model itself decides whether and
    when to call get_abstract_examples / get_related_work_examples /
    get_citation_examples while drafting. Grammar/coherence correction runs
    automatically afterwards on the finished draft."""

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        papers, refs_are_placeholder = _references(context)

        reference_block = (
            "\n".join(f"- {paper}" for paper in papers)
            if papers
            else "(none - write the text without any citations)"
        )

        # Explicit source content the user wants condensed/expanded (e.g.
        # "write an abstract from this content: <paper body>"). Kept separate
        # from `instruction` so the model always sees clearly what is the
        # task and what is the raw material to work from.
        source_content = context.get("source_content", "")
        source_block = (
            f"\n\nSource content to work from:\n{source_content}"
            if source_content.strip()
            else ""
        )

        user_message = (
            f"{request.instruction}\n\n"
            f"References available to cite:\n{reference_block}"
            f"{source_block}"
        )

        messages = [
            {"role": "system", "content": SCIENTIFIC_WRITING_PROMPT},
            {"role": "user", "content": user_message},
        ]

        tools_used: list[str] = []

        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                message = call_llm_with_tools(
                    messages=messages,
                    tools=_TOOL_SCHEMAS,
                    max_completion_tokens=_DRAFT_TOKENS,
                    reasoning_effort="low",
                    role=WRITING,
                )

                if not message.tool_calls:
                    draft = (message.content or "").strip()
                    break

                # The model asked for one or more tools - run them and hand
                # the results back, then let it continue.
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )
                for tool_call in message.tool_calls:
                    executor = _TOOL_EXECUTORS.get(tool_call.function.name)
                    if executor is None:
                        result_text = f"Unknown tool: {tool_call.function.name}"
                    else:
                        try:
                            args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        result_text = executor(args.get("query", request.instruction))
                        tools_used.append(tool_call.function.name)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )
            else:
                # Loop exhausted _MAX_TOOL_ROUNDS without a final answer.
                draft = ""

        except Exception as exc:  # noqa: BLE001 - degrade honestly, never invent
            _logger.warning("writing support LLM unavailable: %s", exc)
            return _unavailable("draft", papers)

        if not draft:
            _logger.warning("writing support returned an empty draft")
            return _unavailable("draft", papers)

        draft, style_corrected = _apply_style_correction(draft)

        output = {
            "draft": draft,
            "references_used": papers,
            "style": "academic",
            "style_corrected": style_corrected,
            "tools_used": tools_used,
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
        except Exception as exc:  # noqa: BLE001
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
    route: str
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

    def _build_graph(self):
        graph = StateGraph(WritingState)

        graph.add_node("classify", self._classify)
        graph.add_node("writing_support", self._run_writing)
        graph.add_node("publication_support", self._run_publication)
        graph.add_node("aggregate", self._aggregate)

        graph.set_entry_point("classify")
        graph.add_conditional_edges("classify", self._route_after_classify)
        graph.add_conditional_edges("writing_support", self._route_after_writing)
        graph.add_edge("publication_support", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    @staticmethod
    def _classify_by_keyword(instruction: str) -> str:
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
        except Exception:  # noqa: BLE001
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

    def _route_after_classify(self, state: WritingState) -> str:
        if state["route"] == "publication":
            return "publication_support"
        return "writing_support"

    def _route_after_writing(self, state: WritingState) -> str:
        if state["route"] == "both":
            return "publication_support"
        return "aggregate"

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