"""The Recognition workflow, as a real LangGraph `StateGraph`.

    START -> validate_image_and_text
          -> plan_or_analyze_text
          -> classify_with_mock_bioclip2
          -> evaluate_confidence
          -> validate_taxonomy_with_mock_gbif_and_ncbi
          -> explain
          -> delegate_if_needed
          -> finalize -> END

Every stage routes conditionally: the moment a node writes `error_code` into the
state, the graph jumps straight to `finalize`. That is why `finalize` is the
only node that ever builds an `AgentResult` - success, uncertain, no match,
delegation and failure all arrive there, so the shared contract cannot hold on
some branches and not others.

There is no embed node, no retrieve node and no aggregate node, because the
agent performs no vector search. Species candidates come from exactly one place:
`classify_with_mock_bioclip2`.

Two things this graph deliberately does not have: a checkpointer and a memory
store. Nothing is persisted between requests; each `invoke` starts from a fresh
state built in `run`.

Providers are captured by the node factories rather than carried in the state -
the same pattern the Global Orchestrator uses for its worker nodes - so the
state stays a plain data object with nothing unserializable in it.

--- Sprint 4 Phase 5 -------------------------------------------------------

One `RecognitionTracer` is built here, from this workflow's own config, and
passed into the node factories that make a provider or LLM call (plan,
classify, taxonomy, explain - see `nodes.py`). Disabled by default; see
`LangSmithConfig`. `run()` also records one root event per request, whatever
branch was taken, so a single trace roots the whole request without
duplicating what LangGraph's own tracing already covers node-to-node.
"""
from __future__ import annotations

import logging
import time

from langgraph.graph import END, START, StateGraph

from ..adapters.bioclip import BioCLIP2Classifier
from ..adapters.reasoning_llm import NullRecognitionLLM
from ..adapters.taxonomy import MockTaxonomyProvider
from ..config import MODEL_TARGET, RECOGNITION_MODE_MOCK_CLASSIFICATION, RecognitionConfig
from ..observability import RecognitionTracer, build_tracer
from ..schema import AgentResult, AgentStatus
from ..text_analysis import RuleBasedTextAnalyzer
from . import nodes
from .state import RecognitionState

_logger = logging.getLogger(__name__)

# Node names, and the order the evidence must be built in. The plan may choose
# which optional steps run; it can never reorder these.
VALIDATE = "validate_image_and_text"
PLAN = "plan_or_analyze_text"
CLASSIFY = "classify_with_mock_bioclip2"
CONFIDENCE = "evaluate_confidence"
TAXONOMY = "validate_taxonomy_with_mock_gbif_and_ncbi"
EXPLAIN = "explain"
DELEGATE = "delegate_if_needed"
FINALIZE = "finalize"

_PIPELINE = (
    (VALIDATE, PLAN),
    (PLAN, CLASSIFY),
    (CLASSIFY, CONFIDENCE),
    (CONFIDENCE, TAXONOMY),
    (TAXONOMY, EXPLAIN),
    (EXPLAIN, DELEGATE),
)


def _route(next_node: str):
    """Continue to `next_node`, unless a controlled failure was recorded."""

    def _router(state: RecognitionState) -> str:
        return FINALIZE if state.error_code else next_node

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
        # A classification score is a ranking value, not a probability, and in
        # Sprint 2 it is a deterministic mock value on top of that.
        "score_is_probability": False,
        "explanation": decision.explanation,
        "clarification_question": decision.clarification_question,
    }
    if decision.request_better_image:
        # The one narrow follow-up the validated decisions allow. Reachable only
        # from `not_identified` - this is not a general clarification route.
        recognition["request_better_image"] = True
        recognition["better_image_reason"] = confidence_better_image_reason(decision.decision)
    unsupported = (
        state.text_evidence.unsupported_capability if state.text_evidence else None
    )
    if unsupported:
        # Declined out loud. This agent has no similarity feature and does not
        # quietly substitute classification for one without saying so.
        recognition["unsupported_capability"] = unsupported
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
            # Never invented. Null means the mocked taxonomy had no value.
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
    """What actually produced this answer. Every field is checkable.

    Three separate mocks, named separately, so no reader can mistake one for a
    real service or assume that "mocked" applied to only some of them.
    """
    config = state.config_snapshot
    # Mock and remote execution must never be described in the same words. The
    # discriminator is the mode the provider that actually ran reports about
    # itself, not a configured value, so provenance cannot outlive a swap.
    is_mock = state.classification_mode == RECOGNITION_MODE_MOCK_CLASSIFICATION

    # Derived from what actually ran, never from configuration. These used to be
    # the literal "mock", with no real-mode branch, so a live GBIF/NCBI answer
    # was reported as mocked while the nested taxonomy_report said "real" - the
    # two halves of the same dict contradicted each other. The nested report is
    # now the source of truth, so they cannot disagree by construction.
    gbif_mode, ncbi_mode, taxonomy_executed = nodes.taxonomy_source_modes(state)

    provenance = {
        "model_target": MODEL_TARGET,
        "recognition_provider": state.classification_provider,
        # "mock_classification" or "remote_bioclip2_open_domain_species" -
        # never plain "classification".
        "recognition_mode": state.classification_mode,
        # A real model version must never be filed under a key named "mock".
        "mock_provider_version": (
            (state.classifier_version or config["mock_provider_version"])
            if is_mock else None
        ),
        "top_k_requested": state.requested_top_k,
        "gbif_mode": gbif_mode,
        "ncbi_mode": ncbi_mode,
        # Whether a lookup was actually performed. False when the classifier
        # named nothing, so there was no candidate to validate: the modes above
        # then describe the provider that was wired in, and this says plainly
        # that it was never asked anything.
        "taxonomy_executed": taxonomy_executed,
        "taxonomy_degraded": state.taxonomy_degraded,
        "taxonomy_report": state.taxonomy_report,
        # False in both modes. The remote score is a softmax over ~867k labels,
        # which is a ranking value and not calibrated confidence.
        "score_is_probability": False,
        "score_kind": (
            "deterministic_sprint2_test_score" if is_mock
            else "bioclip2_remote_zero_shot_ranking_score"
        ),
        "text_analysis_mode": config["text_analysis_mode"],
        "workflow_engine": "langgraph",
        "reasoning_llm_enabled": config["reasoning_llm_enabled"],
        "reasoning_llm_provider": config["reasoning_llm_provider"],
        "plan_source": state.plan_source,
        "plan_rejected": state.plan_rejected,
        "explanation_source": state.explanation_source,
        "reasoning_llm_calls": state.reasoning_llm_calls,
        "reasoning_llm_used": state.reasoning_llm_used,
    }

    # Additive, remote-only detail. Mock mode keeps exactly the keys it always
    # had, so every existing mock-mode assertion still describes the whole dict.
    if not is_mock:
        provenance.update(state.classification_provenance or {})

    return provenance


class RecognitionWorkflow:
    """Compiles the graph once, then runs one request per `run` call."""

    def __init__(
        self,
        config: RecognitionConfig,
        classifier: BioCLIP2Classifier,
        taxonomy_provider: MockTaxonomyProvider,
        text_analyzer: RuleBasedTextAnalyzer,
        reasoning_llm=None,
    ) -> None:
        self._config = config
        self._classifier = classifier
        self._taxonomy = taxonomy_provider
        self._text_analyzer = text_analyzer
        # Disabled unless one is supplied. Recognition never depends on it.
        self._reasoning_llm = reasoning_llm or NullRecognitionLLM()
        # One tracer for the whole workflow's lifetime, built from Recognition's
        # own config. Disabled by default - see LangSmithConfig. Passed into
        # only the nodes that make a provider or LLM call; the rest have
        # nothing a graph-level LangGraph trace does not already show.
        self._tracer: RecognitionTracer = build_tracer(self._config.langsmith)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RecognitionState)

        graph.add_node(VALIDATE, nodes.make_validate_node(self._config))
        graph.add_node(PLAN, nodes.make_plan_node(
            self._text_analyzer, self._reasoning_llm, self._config, self._tracer))
        graph.add_node(CLASSIFY, nodes.make_classify_node(
            self._classifier, self._config, self._tracer))
        graph.add_node(CONFIDENCE, nodes.make_confidence_node(self._config))
        graph.add_node(TAXONOMY, nodes.make_taxonomy_node(self._taxonomy, self._tracer))
        graph.add_node(EXPLAIN, nodes.make_explain_node(
            self._reasoning_llm, self._config, self._tracer))
        graph.add_node(DELEGATE, nodes.make_delegation_node())
        graph.add_node(FINALIZE, make_finalize_node())

        graph.add_edge(START, VALIDATE)
        for source, target in _PIPELINE:
            graph.add_conditional_edges(
                source, _route(target), {target: target, FINALIZE: FINALIZE}
            )
        graph.add_edge(DELEGATE, FINALIZE)
        graph.add_edge(FINALIZE, END)

        # No checkpointer, no store: nothing survives a request.
        return graph.compile()

    # -- entry point --------------------------------------------------------

    def run(self, instruction, context) -> AgentResult:
        started = time.perf_counter()
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
            result = AgentResult(
                status=AgentStatus.FAILED,
                output={"error_code": "WORKFLOW_INCOMPLETE",
                        "error": "The recognition workflow produced no result."},
            )

        # One root record per request, whatever branch was taken. The fixed
        # error code only - never the output dict, which may carry a
        # not-otherwise-sensitive but still request-shaped value.
        event: dict = {
            "node": "finalize",
            "operation": "recognition_request",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "status": result.status.value,
        }
        if result.status is AgentStatus.FAILED and isinstance(result.output, dict):
            event["error_code"] = result.output.get("error_code")
        elif result.status is AgentStatus.COMPLETED and isinstance(result.output, dict):
            recognition = result.output.get("recognition") or {}
            event["decision"] = recognition.get("decision")
        self._tracer.record(event)

        return result

    def _config_snapshot(self) -> dict:
        config = self._config
        return {
            "mock_provider_version": config.mock_provider_version,
            "text_analysis_mode": self._text_analyzer.mode,
            "reasoning_llm_enabled": bool(getattr(self._reasoning_llm, "enabled", False)),
            "reasoning_llm_provider": getattr(self._reasoning_llm, "name", "disabled"),
            # The mode the taxonomy provider OBJECT that was actually wired in
            # declares about itself - not the configured value, so an injected
            # or swapped provider is reported as what it really is. Used only
            # when no candidate was validated and the per-species report is
            # therefore empty; runtime evidence wins whenever it exists.
            "taxonomy_provider_mode": getattr(self._taxonomy, "mode", None),
        }
