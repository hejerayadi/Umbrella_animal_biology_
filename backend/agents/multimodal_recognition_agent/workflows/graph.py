"""The Recognition workflow, as a real LangGraph `StateGraph`.

    START -> validate -> plan -> embed -> retrieve -> aggregate
          -> taxonomy -> confidence -> explain -> delegate -> finalize -> END

Every stage routes conditionally: the moment a node writes `error_code` into the
state, the graph jumps straight to `finalize`. That is why `finalize` is the
only node that ever builds an `AgentResult` - success, uncertain, no match,
delegation and failure all arrive there, so the shared contract cannot hold on
some branches and not others.

Two things this graph deliberately does not have: a checkpointer and a memory
store. Nothing is persisted between requests; each `invoke` starts from a fresh
state built in `run`.

Providers are captured by the node factories rather than carried in the state -
the same pattern the Global Orchestrator uses for its worker nodes - so the
state stays a plain data object with nothing unserializable in it.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from ..adapters.bioclip import ImageEmbeddingProvider
from ..adapters.reasoning_llm import NullRecognitionLLM
from ..adapters.retrieval import RetrievalProvider
from ..adapters.taxonomy import MockTaxonomyProvider
from ..config import RecognitionConfig
from ..schema import AgentResult, AgentStatus
from ..text_analysis import RuleBasedTextAnalyzer
from . import nodes
from .state import RecognitionState

_logger = logging.getLogger(__name__)

# The order the evidence must be built in. The plan may choose which optional
# steps run; it can never reorder these.
_PIPELINE = (
    ("validate", "plan"),
    ("plan", "embed"),
    ("embed", "retrieve"),
    ("retrieve", "aggregate"),
    ("aggregate", "taxonomy"),
    ("taxonomy", "confidence"),
    ("confidence", "explain"),
    ("explain", "delegate"),
)


def _route(next_node: str):
    """Continue to `next_node`, unless a controlled failure was recorded."""

    def _router(state: RecognitionState) -> str:
        return "finalize" if state.error_code else next_node

    return _router


def make_finalize_node():
    """THE single terminal builder. Every branch of the graph ends here."""

    def _node(state: RecognitionState) -> dict:
        return {"agent_result": _build_agent_result(state)}

    return _node


def _build_agent_result(state: RecognitionState) -> AgentResult:
    # 1. A controlled refusal.
    if state.error_code:
        _logger.info("[Recognition] failed -> %s", state.error_code)
        return AgentResult(
            status=AgentStatus.FAILED,
            output={"error_code": state.error_code, "error": state.error_message},
        )

    decision = state.decision
    if decision is None:  # defensive: the graph cannot reach finalize without one
        return AgentResult(
            status=AgentStatus.FAILED,
            output={"error_code": "WORKFLOW_INCOMPLETE",
                    "error": "The recognition workflow produced no decision."},
        )

    # 2. A capability this agent does not own. The orchestrator routes it.
    if state.delegate_to is not None:
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent=state.delegate_to,
            prompt_to_target_agent=state.delegation_prompt,
            output=None,
        )

    # 3. A completed scientific outcome - including uncertain and not_identified.
    primary = decision.primary_species
    recognition: dict = {
        "decision": decision.decision,
        "text_alignment": decision.text_alignment,
        "similarity_is_probability": False,
        "explanation": decision.explanation,
        "clarification_question": decision.clarification_question,
    }
    if decision.request_better_image:
        # The one narrow follow-up the validated decisions allow. Reachable only
        # from `not_identified` - this is not a general clarification route.
        recognition["request_better_image"] = True
        recognition["better_image_reason"] = confidence_better_image_reason(decision.decision)
    if state.warnings:
        recognition["warnings"] = list(state.warnings)

    return AgentResult(
        status=AgentStatus.COMPLETED,
        output={
            "recognition": recognition,
            # `species` is the cross-agent context key Genome, Biodiversity and
            # Image Generation all read.
            "species": primary.scientific_name if primary else None,
            "species_id": primary.species_id if primary else None,
            # Never invented. Null means the mock taxonomy had no value.
            "gbif_id": primary.gbif_id if primary else None,
            "ncbi_taxid": primary.ncbi_taxid if primary else None,
            "recognition_candidates": [c.model_dump() for c in decision.candidates],
            "recognition_provenance": _provenance(state),
        },
    )


def confidence_better_image_reason(decision: str) -> str | None:
    from ..domain.confidence import better_image_request

    return better_image_request(decision)  # type: ignore[arg-type]


def _provenance(state: RecognitionState) -> dict:
    """What actually produced this answer. Every field is checkable."""
    config = state.config_snapshot
    return {
        "embedding_provider": "MockBioCLIP2Provider",
        "embedding_mode": "mock",
        "mock_provider_version": config["mock_provider_version"],
        "embedding_dimension": config["embedding_dimension"],
        "embedding_dimension_source": config["embedding_dimension_source"],
        "retrieval_provider": state.retrieval_provider,
        # "mock_local_development" or "real_minimal" - never conflated.
        "retrieval_mode": state.retrieval_mode,
        "collection": state.retrieval_collection,
        "dataset_version": state.retrieval_dataset_version,
        "qdrant_contract_frozen": config["qdrant_contract_frozen"],
        "taxonomy_mode": "mock",
        "taxonomy_sources": {"gbif": "mock", "ncbi": "mock"},
        "taxonomy_degraded": state.taxonomy_degraded,
        "taxonomy_report": state.taxonomy_report,
        "text_analysis_mode": config["text_analysis_mode"],
        "workflow_engine": "langgraph",
        "reasoning_llm_enabled": config["reasoning_llm_enabled"],
        "reasoning_llm_provider": config["reasoning_llm_provider"],
        "plan_source": state.plan_source,
        "plan_rejected": state.plan_rejected,
        "explanation_source": state.explanation_source,
        "reasoning_llm_calls": state.reasoning_llm_calls,
        "reasoning_llm_used": state.reasoning_llm_used,
        "similarity_is_probability": False,
    }


class RecognitionWorkflow:
    """Compiles the graph once, then runs one request per `run` call."""

    def __init__(
        self,
        config: RecognitionConfig,
        embedding_provider: ImageEmbeddingProvider,
        retriever: RetrievalProvider,
        taxonomy_provider: MockTaxonomyProvider,
        text_analyzer: RuleBasedTextAnalyzer,
        reasoning_llm=None,
    ) -> None:
        self._config = config
        self._embedding_provider = embedding_provider
        self._retriever = retriever
        self._taxonomy = taxonomy_provider
        self._text_analyzer = text_analyzer
        # Disabled unless one is supplied. Recognition never depends on it.
        self._reasoning_llm = reasoning_llm or NullRecognitionLLM()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RecognitionState)

        graph.add_node("validate", nodes.make_validate_node(self._config))
        graph.add_node("plan", nodes.make_plan_node(
            self._text_analyzer, self._reasoning_llm, self._config))
        graph.add_node("embed", nodes.make_embed_node(
            self._embedding_provider, self._config))
        graph.add_node("retrieve", nodes.make_retrieve_node(self._retriever, self._config))
        graph.add_node("aggregate", nodes.make_aggregate_node(self._config))
        graph.add_node("taxonomy", nodes.make_taxonomy_node(self._taxonomy))
        graph.add_node("confidence", nodes.make_confidence_node(self._config))
        graph.add_node("explain", nodes.make_explain_node(self._reasoning_llm, self._config))
        graph.add_node("delegate", nodes.make_delegation_node())
        graph.add_node("finalize", make_finalize_node())

        graph.add_edge(START, "validate")
        for source, target in _PIPELINE:
            graph.add_conditional_edges(
                source, _route(target), {target: target, "finalize": "finalize"}
            )
        graph.add_edge("delegate", "finalize")
        graph.add_edge("finalize", END)

        # No checkpointer, no store: nothing survives a request.
        return graph.compile()

    # -- entry point --------------------------------------------------------

    def run(self, instruction, context) -> AgentResult:
        initial = RecognitionState(instruction=instruction, context=context)
        # Config values the finalize node needs, snapshotted so the state stays
        # a plain data object with no provider references in it.
        initial.config_snapshot = self._config_snapshot()

        final = self._graph.invoke(initial)
        result = (
            final.agent_result if isinstance(final, RecognitionState)
            else final.get("agent_result")
        )
        if result is None:  # defensive: the graph always reaches finalize
            return AgentResult(
                status=AgentStatus.FAILED,
                output={"error_code": "WORKFLOW_INCOMPLETE",
                        "error": "The recognition workflow produced no result."},
            )
        return result

    def _config_snapshot(self) -> dict:
        config = self._config
        return {
            "mock_provider_version": config.mock_provider_version,
            "embedding_dimension": config.mock_embedding_dimension,
            "embedding_dimension_source": (
                "local_default_pending_qdrant_manifest"
                if config.mock_embedding_dimension_is_local_default
                else "configured"
            ),
            "qdrant_contract_frozen": config.qdrant.is_frozen,
            "text_analysis_mode": self._text_analyzer.mode,
            "reasoning_llm_enabled": bool(getattr(self._reasoning_llm, "enabled", False)),
            "reasoning_llm_provider": getattr(self._reasoning_llm, "name", "disabled"),
        }
