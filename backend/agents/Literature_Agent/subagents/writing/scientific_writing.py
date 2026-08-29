from __future__ import annotations

import json
import logging
import re

from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ...llm.client import WRITING, call_llm, call_llm_with_tools
from ...llm.prompts import (
    PUBLICATION_SUPPORT_PROMPT,
    SCIENTIFIC_WRITING_PROMPT,
    WRITING_SUB_ROUTING_PROMPT,
)
from ...schema import AgentRequest, AgentResult, AgentStatus
from .kb.retrieval import search_kb_papers, search_related_work, search_citations

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


# Which section the user asked for, in the order it is tested. First hit wins,
# so the more specific phrasings ("related work" before "work") come first.
# The frontend labels the panel it renders with this, and the Responder uses it
# to introduce the draft, so it has to name what was actually written rather
# than always saying "abstract".
_SECTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Related work", ("related work", "related-work")),
    ("Literature review", ("literature review", "review section", "review of the literature")),
    ("Introduction", ("introduction", "intro section")),
    ("Discussion", ("discussion",)),
    ("Conclusion", ("conclusion", "concluding")),
    ("Methods", ("methods", "method section", "materials and methods", "methodology")),
    ("Results", ("results section", "results paragraph")),
    ("Background", ("background section",)),
    ("Abstract", ("abstract", "summary paragraph")),
)

# The system prompt's own default: "Match the section they name; if they name
# none, write an abstract." Kept identical here so the label never contradicts
# what was actually asked for.
_DEFAULT_SECTION = "Abstract"


def _section_label(instruction: str) -> str:
    """Name the piece of academic text this instruction asks for."""
    lowered = (instruction or "").lower()
    for label, keywords in _SECTION_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return label
    return _DEFAULT_SECTION


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
        hits = search_kb_papers(query, section_type="abstract", limit=limit)
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


# Motif binomial simple : Genre (majuscule) + espece/sous-espece (minuscules),
# ex. "Panthera tigris", "Panthera tigris altaica".
_SCIENTIFIC_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+) ([a-z]+)(?: ([a-z]+))?\b")

# Le motif seul est trop large : il attrape "This study focuses" en tete de
# phrase, ce qui protegeait - et donc desactivait la correction sur - les trois
# premiers mots de la plupart des phrases. On garde le biais "proteger un peu
# trop" voulu au depart, mais on ecarte les tokens qui sont manifestement de
# l'anglais courant. Un epithete latin n'est jamais dans cette liste.
_COMMON_WORDS = frozenset(
    """
    a an the this that these those it its we our us they their there here
    and or but nor so yet if then than as at by for from in into of on onto
    to with within without across among between during after before while
    when where because since despite unlike whereas although though however
    therefore thus moreover furthermore additionally similarly notably
    importantly overall finally together given using based compared
    according following is are was were be been being has have had do does
    did may might can could should would will shall must
    study studies research paper papers work works result results finding
    findings data analysis analyses method methods approach model models
    figure table section review evidence effect effects
    show shows showed shown suggest suggests suggested indicate indicates
    report reports reported reveal reveals revealed provide provides
    demonstrate demonstrates found remain remains observe observed
    increase increased decrease decreased include includes including
    decline declined declines exhibit exhibits exhibited display displays
    displayed occur occurs occurred range ranges ranged vary varies varied
    differ differs differed appear appears appeared emerge emerges emerged
    evolve evolves evolved adapt adapts adapted inhabit inhabits inhabited
    migrate migrates migrated represent represents represented
    population populations species sample samples group groups
    recent previous current present such both each most many some few one
    two three several other others same different high low
    """.split()
)

_MIN_TOKEN_LEN = 3


def _scientific_name_spans(text: str) -> list[tuple[int, int]]:
    """Les positions des noms binomiaux/trinomiaux a proteger dans `text`.

    Un token doit etre assez long et absent de `_COMMON_WORDS` pour compter
    comme latin. Le genre et l'espece doivent tous deux passer, sinon rien
    n'est protege : ca ecarte "This study focuses".

    Le troisieme token est teste separement. Le motif est gourmand, donc
    "Canis lupus declined" matche en entier ; rejeter le match complet parce
    que "declined" est un mot courant perdrait aussi le binome "Canis lupus".
    On retombe donc sur les deux premiers tokens.
    """

    def _is_latin(token: str | None) -> bool:
        return bool(
            token
            and len(token) >= _MIN_TOKEN_LEN
            and token.lower() not in _COMMON_WORDS
        )

    spans: list[tuple[int, int]] = []
    for match in _SCIENTIFIC_NAME_PATTERN.finditer(text):
        if not (_is_latin(match.group(1)) and _is_latin(match.group(2))):
            continue
        end = match.end(3) if _is_latin(match.group(3)) else match.end(2)
        spans.append((match.start(1), end))
    return spans


@lru_cache(maxsize=1)
def _language_tool():
    """The LanguageTool instance, built once.

    Constructing it starts a JVM and a local server - several seconds. Doing
    that per draft put that cost in every single request path.
    """
    import language_tool_python

    return language_tool_python.LanguageTool("en-US")


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

        matches = _language_tool().check(text)

        protected_spans = _scientific_name_spans(text)

        def _overlaps_protected(match) -> bool:
            start, end = match.offset, match.offset + match.error_length
            return any(start < p_end and end > p_start for p_start, p_end in protected_spans)

        filtered_matches = [m for m in matches if not _overlaps_protected(m)]

        # The tool is cached and shared, so it is deliberately not closed here.
        return language_tool_python.utils.correct(text, filtered_matches), True
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
        # `or ""` rather than a `.get` default: a caller that sends the key
        # explicitly set to null is normal over JSON, and `None.strip()` here
        # would raise outside the try below - escaping as an AttributeError
        # instead of degrading into a draft without source content.
        source_content = context.get("source_content") or ""
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
            "section": _section_label(request.instruction),
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
    """Recommends journals / venues for a draft.

    Two paths, in order:

    1. The Publication Support pipeline in `subagents/publication_support/`:
       the query is interpreted into a topic plus explicit constraints, BGE
       embeddings retrieve candidates from the `journals` Qdrant collection
       (ingested from OpenAlex), a hybrid semantic + topic-hierarchy score
       ranks them, an LLM re-ranks on topical fit, and the stated preferences
       are applied deterministically afterwards.

    2. Failing that, the LLM alone.

    The order matters and is not just a performance choice. Path 1 recommends
    journals it actually retrieved; path 2 recommends journals the model
    remembers, which on a platform built around never inventing scientific
    results is the weaker answer. So path 2 is the fallback, it is only taken
    when path 1 cannot run, and the output says which one produced it via
    `retrieval_backed`.
    """

    # How many journals reach the user. The pipeline re-ranks a much larger
    # candidate pool; this is the size of the final answer.
    TOP_K = 5

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

        journals = self._retrieve(request.instruction, draft)
        if journals:
            return AgentResult(
                status=AgentStatus.COMPLETED,
                output={
                    "recommended_journals": _format_journals(journals),
                    "journals": journals,
                    "based_on_draft": bool(draft),
                    "retrieval_backed": True,
                },
            )

        recommendations = self._ask_llm(user_message)
        if not recommendations:
            return _unavailable("recommended_journals", [])

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "recommended_journals": recommendations,
                "journals": [],
                "based_on_draft": bool(draft),
                "retrieval_backed": False,
            },
        )

    def _retrieve(self, instruction: str, draft: str) -> list[dict]:
        """Run the retrieval pipeline. Returns [] rather than raising.

        Imported inside the method on purpose. The pipeline pulls
        sentence-transformers and a Qdrant client, and this module is on the
        import chain behind `api.py`; an ImportError at module scope would
        stop the whole agent from starting instead of costing one route its
        better answer.
        """
        try:
            from ..publication_support.ingestion.retrieval import retrieve_journals
            from ..publication_support.ranking.llm_reranker import (
                apply_preferences,
                rerank_journals,
            )
            from ..publication_support.ranking.query_interpreter import (
                build_llm_topic,
                interpret_query,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("publication pipeline unavailable: %s", exc)
            return []

        try:
            # The draft is the better description of the work when there is
            # one, but the instruction carries the constraints ("open access",
            # "not Elsevier"), so the interpreter sees both.
            raw = f"{instruction}\n\n{draft}".strip() if draft else instruction
            interpretation = interpret_query(raw)

            ranked = retrieve_journals(
                query=interpretation["search_query"],
                open_access_only=bool(
                    interpretation.get("constraints", {}).get("open_access")
                ),
            )
            if not ranked:
                return []

            reranked = rerank_journals(
                topic=build_llm_topic(interpretation),
                candidates=ranked[:30],
            )
            return apply_preferences(
                reranked or ranked,
                interpretation.get("constraints", {}),
                top_k=self.TOP_K,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "publication pipeline failed (%s): %s", type(exc).__name__, exc
            )
            return []

    @staticmethod
    def _ask_llm(user_message: str) -> str:
        try:
            return call_llm(
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
            return ""


def _format_journals(journals: list[dict]) -> str:
    """Render the ranked journals as the markdown string the frontend reads.

    `recommended_journals` has to stay a string: the frontend passes it
    through `asString()` (orchestrator-client.ts). The structured list is
    returned alongside it under `journals` for callers that want the scores.
    """
    lines = []
    for journal in journals:
        name = journal.get("name") or "Unknown journal"
        bits = []
        if journal.get("publisher"):
            bits.append(str(journal["publisher"]))
        if journal.get("is_oa"):
            bits.append("open access")
        if journal.get("issn"):
            bits.append(f"ISSN {journal['issn']}")
        suffix = f" ({', '.join(bits)})" if bits else ""

        reason = journal.get("reasoning") or journal.get("reason") or ""
        lines.append(f"- **{name}**{suffix}" + (f" - {reason}" if reason else ""))
    return "\n".join(lines)


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
            # COMPLETED only when every branch that ran actually produced
            # something. `any()` here reported a half-failed run as a success,
            # so a draft that never got written was indistinguishable from one
            # that did. See the same reasoning in `orchestrator/graph.py`.
            ran = [r for r in (writing, publication) if r]
            status = (
                AgentStatus.COMPLETED
                if ran and all(r.status == AgentStatus.COMPLETED for r in ran)
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