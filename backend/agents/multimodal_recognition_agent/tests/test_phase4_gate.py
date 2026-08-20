"""Phase 4 gate: the real LangGraph workflow, driven by a fake GPT-5 mini.

Everything here runs offline: no credential, no network, no real Qdrant, no
BioCLIP-2 runtime, no GBIF or NCBI call, no real GPT-5 mini.
"""
from __future__ import annotations

import json

import pytest
from langgraph.graph.state import CompiledStateGraph

from ..adapters.qdrant_mock import MockQdrantRetriever
from ..adapters.reasoning_llm import (
    ALLOWED_PLAN_STEPS,
    MANDATORY_PLAN_STEPS,
    ExplainRequest,
    FakeGPT5MiniProvider,
    NullRecognitionLLM,
    PlanRequest,
    RecognitionPlan,
    deterministic_plan,
    explanation_is_grounded,
    sanitize_plan,
)
from ..adapters.taxonomy import MockGBIFProvider, MockNCBIProvider, MockTaxonomyProvider
from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
from ..schema import AgentRequest, AgentStatus
from ..text_analysis import RuleBasedTextAnalyzer
from ..workflows import nodes
from ..workflows.state import RecognitionState
from .conftest import StubRetriever, candidate, image_entry, make_config, png_bytes, reference

DIMENSION = 32


def build_agent(references=None, *, retriever=None, taxonomy=None, llm=None, config=None):
    return RecognitionAgent(
        config or make_config(),
        retriever=retriever if retriever is not None else StubRetriever(references or []),
        taxonomy_provider=taxonomy or MockTaxonomyProvider(),
        reasoning_llm=llm if llm is not None else FakeGPT5MiniProvider(),
    )


def request_with(instruction="Identify this animal.", extra_context=None):
    context = {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())}
    context.update(extra_context or {})
    return AgentRequest(instruction=instruction, context=context)


# ===========================================================================
# A real StateGraph
# ===========================================================================

def test_the_workflow_is_a_compiled_langgraph_stategraph():
    agent = build_agent([reference("panthera_leo", 0.96)])
    assert isinstance(agent._workflow._graph, CompiledStateGraph)


def test_the_graph_declares_every_expected_node():
    agent = build_agent([reference("panthera_leo", 0.96)])
    names = set(agent._workflow._graph.get_graph().nodes)

    for expected in ("validate", "plan", "embed", "retrieve", "aggregate",
                     "taxonomy", "confidence", "explain", "delegate", "finalize"):
        assert expected in names, expected


def test_provenance_declares_the_langgraph_engine():
    agent = build_agent([reference("panthera_leo", 0.96)])
    provenance = agent.run(request_with()).output["recognition_provenance"]
    assert provenance["workflow_engine"] == "langgraph"


def test_no_persistent_memory_or_checkpointer_is_attached():
    agent = build_agent([reference("panthera_leo", 0.96)])
    compiled = agent._workflow._graph
    assert getattr(compiled, "checkpointer", None) in (None, False)
    assert getattr(compiled, "store", None) is None


def test_two_identical_requests_share_no_state():
    """Without a checkpointer, request two cannot see request one."""
    agent = build_agent([reference("panthera_leo", 0.96, "p1", "Panthera leo")])
    first = agent.run(request_with()).output
    second = agent.run(request_with()).output
    assert first == second


# --- every node is independently testable ---------------------------------

def test_validate_node_in_isolation():
    node = nodes.make_validate_node(make_config())
    state = RecognitionState(
        instruction="Identify this animal.",
        context={RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())},
    )
    updates = node(state)
    assert updates["normalized"].media_type == "image/png"


def test_validate_node_records_a_controlled_failure_instead_of_raising():
    node = nodes.make_validate_node(make_config())
    updates = node(RecognitionState(instruction="x", context={}))
    assert updates["error_code"] == "MISSING_IMAGE"
    assert "normalized" not in updates


def test_plan_node_in_isolation():
    config = make_config()
    analyzer = RuleBasedTextAnalyzer(known_names=MockTaxonomyProvider().known_names())
    fake = FakeGPT5MiniProvider()
    node = nodes.make_plan_node(analyzer, fake, config)

    state = RecognitionState()
    state.normalized = nodes.make_validate_node(config)(
        RecognitionState(instruction="Identify this animal.",
                         context={RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())})
    )["normalized"]

    updates = node(state)
    assert updates["plan_source"] == "llm"
    assert updates["llm_plan_calls"] == 1
    assert fake.plan_calls == 1


def test_aggregate_node_in_isolation():
    node = nodes.make_aggregate_node(make_config())
    state = RecognitionState()
    state.references = [reference("panthera_leo", 0.9, "p1"), reference("panthera_leo", 0.8, "p2")]
    updates = node(state)
    assert len(updates["candidates"]) == 1
    assert updates["visual_evidence_sufficient"] is True


def test_confidence_node_in_isolation():
    from ..domain.models import TextEvidence

    node = nodes.make_confidence_node(make_config())
    state = RecognitionState()
    state.text_evidence = TextEvidence(intent="recognition")
    state.candidates = [candidate("panthera_leo", 0.95)]
    state.margin = 1.0
    updates = node(state)
    assert updates["decision"].decision == "identified"


def test_taxonomy_node_in_isolation():
    node = nodes.make_taxonomy_node(MockTaxonomyProvider())
    state = RecognitionState()
    state.candidates = [candidate("panthera_leo", 0.9)]
    updates = node(state)
    assert updates["candidates"][0].taxonomy_status == "mock_verified"
    assert updates["taxonomy_report"]["panthera_leo"]["gbif"]["available"] is True


def test_delegation_node_in_isolation_returns_only_a_hint():
    from ..domain.models import RecognitionDecision, TextEvidence

    node = nodes.make_delegation_node()
    state = RecognitionState()
    state.normalized = None
    state.text_evidence = TextEvidence(intent="scientific_follow_up",
                                       requested_capability="Evolution")
    top = candidate("panthera_leo", 0.96, name="Panthera leo")
    state.decision = RecognitionDecision(
        decision="identified", primary_species=top, candidates=[top],
        text_alignment="neutral", explanation="",
    )
    updates = node(state)
    assert updates["delegate_to"] == "Evolution"
    # A hint only - no client, no URL, no call.
    assert "Panthera leo" in updates["delegation_prompt"]


# --- one terminal builder --------------------------------------------------

@pytest.mark.parametrize(
    "references, instruction, expected_status",
    [
        ([reference("panthera_leo", 0.96, "p1", "Panthera leo")], "Identify this animal.",
         AgentStatus.COMPLETED),
        ([], "Identify this animal.", AgentStatus.COMPLETED),
        ([reference("panthera_leo", 0.96, "p1", "Panthera leo")],
         "What is the evolutionary history of this animal?", AgentStatus.NEEDS_AGENT),
    ],
)
def test_every_terminal_route_returns_the_shared_contract(
    references, instruction, expected_status
):
    result = build_agent(references).run(request_with(instruction))
    assert result.status is expected_status
    assert hasattr(result, "output") and hasattr(result, "target_agent")


def test_the_failure_route_also_returns_the_shared_contract():
    result = build_agent([]).run(AgentRequest(instruction="x", context={}))
    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == "MISSING_IMAGE"


def test_finalize_is_the_only_node_that_builds_an_agent_result():
    """Grep-level guarantee: no other node constructs the envelope."""
    import pathlib

    source = (pathlib.Path(nodes.__file__)).read_text(encoding="utf-8")
    assert "AgentResult(" not in source


# ===========================================================================
# The fake GPT-5 mini: planning, explanation, budget, rejection
# ===========================================================================

def test_the_planner_is_called_for_a_valid_request():
    fake = FakeGPT5MiniProvider()
    build_agent([reference("panthera_leo", 0.96)], llm=fake).run(request_with())
    assert fake.plan_calls == 1


def test_the_explainer_is_called_once():
    fake = FakeGPT5MiniProvider()
    build_agent([reference("panthera_leo", 0.96)], llm=fake).run(request_with())
    assert fake.explain_calls == 1


def test_the_two_call_budget_is_never_exceeded():
    fake = FakeGPT5MiniProvider()
    agent = build_agent([reference("panthera_leo", 0.96)], llm=fake)
    for _ in range(5):
        agent.run(request_with())
    # Exactly two per request, five requests.
    assert fake.plan_calls == 5
    assert fake.explain_calls == 5
    assert fake.plan_calls + fake.explain_calls == 10


def test_a_failed_request_spends_no_llm_call():
    fake = FakeGPT5MiniProvider()
    build_agent([], llm=fake).run(AgentRequest(instruction="x", context={}))
    assert fake.plan_calls == 0 and fake.explain_calls == 0


# --- strict plan validation ------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        None, "a string", 42, [],
        {},                                              # no steps
        {"steps": []},                                   # empty
        {"steps": ["embed_image", "compare"]},           # forbidden route
        {"steps": ["embed_image", "clarify"]},           # forbidden route
        {"steps": ["embed_image", "call_agent"]},        # forbidden route
        {"steps": ["embed_image", "fetch_url"]},         # forbidden route
        {"steps": ["score_confidence"]},                 # missing mandatory steps
        {"steps": list(ALLOWED_PLAN_STEPS), "intent": "hack"},
        {"steps": list(ALLOWED_PLAN_STEPS), "top_k": 0},
        {"steps": list(ALLOWED_PLAN_STEPS), "top_k": 9999},
        {"steps": list(ALLOWED_PLAN_STEPS), "top_k": "many"},
        {"steps": [1, 2, 3]},
    ],
)
def test_a_malformed_or_forbidden_plan_is_rejected(raw):
    assert sanitize_plan(raw, default_top_k=30, rule_intent="recognition") is None


def test_a_well_formed_plan_is_accepted():
    plan = sanitize_plan(
        {"steps": list(ALLOWED_PLAN_STEPS), "intent": "similarity", "top_k": 12},
        default_top_k=30, rule_intent="recognition",
    )
    assert isinstance(plan, RecognitionPlan)
    assert plan.intent == "similarity" and plan.top_k == 12 and plan.source == "llm"


def test_mandatory_steps_cannot_be_skipped():
    for mandatory in MANDATORY_PLAN_STEPS:
        steps = [s for s in ALLOWED_PLAN_STEPS if s != mandatory]
        assert sanitize_plan({"steps": steps}, default_top_k=30,
                             rule_intent="recognition") is None


def test_a_rejected_plan_falls_back_to_the_deterministic_plan():
    fake = FakeGPT5MiniProvider(plan_override={"steps": ["compare"]})
    result = build_agent([reference("panthera_leo", 0.96)], llm=fake).run(request_with())
    provenance = result.output["recognition_provenance"]

    assert provenance["plan_rejected"] is True
    assert provenance["plan_source"] == "deterministic"
    assert result.status is AgentStatus.COMPLETED


def test_a_planner_failure_falls_back_deterministically():
    fake = FakeGPT5MiniProvider(fail_plan=TimeoutError("simulated"))
    result = build_agent([reference("panthera_leo", 0.96)], llm=fake).run(request_with())

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition_provenance"]["plan_source"] == "deterministic"


def test_an_explainer_failure_falls_back_deterministically():
    fake = FakeGPT5MiniProvider(fail_explain=RuntimeError("simulated"))
    result = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")], llm=fake
    ).run(request_with())

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition_provenance"]["explanation_source"] == "deterministic"
    assert result.output["recognition"]["explanation"]


def test_the_null_brain_makes_no_call_and_still_completes():
    result = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")], llm=NullRecognitionLLM()
    ).run(request_with())

    provenance = result.output["recognition_provenance"]
    assert result.status is AgentStatus.COMPLETED
    assert provenance["reasoning_llm_calls"] == 0
    assert provenance["plan_source"] == "deterministic"


# --- the explanation is checked before it is accepted ----------------------

def _explain_request(**overrides):
    base = dict(
        decision="identified", text_alignment="neutral",
        primary_species="Panthera leo", candidate_names=("Panthera leo",),
        top_score=0.96, margin=0.5, taxonomy_status="mock_verified",
        retrieval_mode="mock_local_development", embedding_mode="mock",
        visual_evidence_sufficient=True,
    )
    base.update(overrides)
    return ExplainRequest(**base)


def test_an_explanation_naming_an_unretrieved_species_is_rejected():
    assert explanation_is_grounded(
        "This is clearly Ursus maritimus, a polar bear.", _explain_request()
    ) is False


def test_an_explanation_about_a_retrieved_species_is_accepted():
    assert explanation_is_grounded(
        "Panthera leo is the closest retrieved match.", _explain_request()
    ) is True


@pytest.mark.parametrize(
    "text",
    ["There is a 96% probability of Panthera leo.",
     "Panthera leo with a certainty of 96%.",
     "Panthera leo, 96% confident."],
)
def test_an_explanation_presenting_similarity_as_probability_is_rejected(text):
    assert explanation_is_grounded(text, _explain_request()) is False


@pytest.mark.parametrize("text", [None, 42, "", "   ", "x" * 5000])
def test_a_malformed_explanation_is_rejected(text):
    assert explanation_is_grounded(text, _explain_request()) is False


def test_an_ungrounded_explanation_never_reaches_the_output():
    fake = FakeGPT5MiniProvider(
        explanation_override="This is Ursus maritimus, weighing 400 kg."
    )
    result = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")], llm=fake
    ).run(request_with())

    explanation = result.output["recognition"]["explanation"]
    assert "Ursus maritimus" not in explanation
    assert result.output["recognition_provenance"]["explanation_source"] == "deterministic"


def test_the_safety_footer_is_present_whoever_wrote_the_explanation():
    for provider in (FakeGPT5MiniProvider(), NullRecognitionLLM()):
        result = build_agent(
            [reference("panthera_leo", 0.96, "p1", "Panthera leo")], llm=provider
        ).run(request_with())
        explanation = result.output["recognition"]["explanation"].lower()
        assert "not a probability" in explanation
        assert "mock" in explanation


# --- the model cannot touch the science ------------------------------------

def test_the_llm_cannot_add_a_species():
    fake = FakeGPT5MiniProvider(
        plan_override={"steps": list(ALLOWED_PLAN_STEPS), "taxon_hint": "polar bear"}
    )
    result = build_agent(
        [reference("panthera_leo", 0.97, "p1", "Panthera leo")], llm=fake
    ).run(request_with())

    ids = {c["species_id"] for c in result.output["recognition_candidates"]}
    assert ids == {"panthera_leo"}
    assert result.output["species"] != "Ursus maritimus"


def test_the_llm_cannot_remove_a_candidate():
    references = [
        reference("panthera_leo", 0.90, "p1", "Panthera leo"),
        reference("panthera_tigris", 0.60, "p2", "Panthera tigris"),
    ]
    with_llm = build_agent(references, llm=FakeGPT5MiniProvider()).run(request_with())
    without = build_agent(references, llm=NullRecognitionLLM()).run(request_with())

    assert (
        [c["species_id"] for c in with_llm.output["recognition_candidates"]]
        == [c["species_id"] for c in without.output["recognition_candidates"]]
    )


def test_the_llm_cannot_change_a_score_or_the_decision():
    references = [reference("panthera_leo", 0.96, "p1", "Panthera leo")]
    with_llm = build_agent(references, llm=FakeGPT5MiniProvider()).run(request_with())
    without = build_agent(references, llm=NullRecognitionLLM()).run(request_with())

    assert (
        with_llm.output["recognition_candidates"][0]["similarity_score"]
        == without.output["recognition_candidates"][0]["similarity_score"]
    )
    assert (
        with_llm.output["recognition"]["decision"]
        == without.output["recognition"]["decision"]
    )


def test_the_llm_cannot_invent_a_taxonomy_identifier():
    fake = FakeGPT5MiniProvider(explanation_override="Panthera leo, GBIF 999999999.")
    result = build_agent(
        [reference("vulpes_lagopus", 0.96, "p1", "Vulpes lagopus")], llm=fake
    ).run(request_with())

    # The fixture has no identifiers for this species; none may appear.
    assert result.output["gbif_id"] is None
    assert result.output["ncbi_taxid"] is None


def test_no_image_data_or_vector_reaches_the_planner_or_explainer():
    fake = FakeGPT5MiniProvider()
    context = {
        RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes()),
        "generated_image": {"data_url": "data:image/png;base64,AAAABBBB"},
    }
    build_agent([reference("panthera_leo", 0.96)], llm=fake).run(
        AgentRequest(instruction="Identify this animal.", context=context)
    )

    payload = json.dumps(
        [r.__dict__ for r in fake.seen_plan_requests + fake.seen_explain_requests],
        default=str,
    )
    for forbidden in ("data:image", "base64", "iVBOR", "generated_image",
                      "AZURE_OPENAI", "api_key"):
        assert forbidden not in payload, forbidden

    for seen in fake.seen_plan_requests + fake.seen_explain_requests:
        assert not hasattr(seen, "image_bytes")
        assert not hasattr(seen, "query_vector")
        assert not hasattr(seen, "context")


# ===========================================================================
# Functional outcomes
# ===========================================================================

def test_clear_identification():
    result = build_agent([
        reference("panthera_leo", 0.96, "p1", "Panthera leo"),
        reference("panthera_tigris", 0.30, "p2", "Panthera tigris"),
    ]).run(request_with())
    assert result.output["recognition"]["decision"] == "identified"


def test_close_candidates_are_uncertain():
    result = build_agent([
        reference("panthera_leo", 0.88, "p1", "Panthera leo"),
        reference("panthera_tigris", 0.87, "p2", "Panthera tigris"),
    ]).run(request_with())
    assert result.output["recognition"]["decision"] == "uncertain"


def test_no_candidate_is_not_identified():
    result = build_agent([]).run(request_with())
    assert result.output["recognition"]["decision"] == "not_identified"


def test_insufficient_visual_evidence_asks_for_a_better_image():
    result = build_agent([]).run(request_with())
    recognition = result.output["recognition"]

    assert recognition["request_better_image"] is True
    assert "image" in recognition["better_image_reason"].lower()


def test_a_good_identification_does_not_ask_for_a_better_image():
    result = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")]
    ).run(request_with())
    assert "request_better_image" not in result.output["recognition"]


def test_top_k_candidates_are_returned():
    references = [reference(f"species_{i}", 0.9 - i / 50, str(i)) for i in range(8)]
    result = build_agent(references).run(request_with())
    # top_k_species is 5 in the test config.
    assert len(result.output["recognition_candidates"]) == 5


def test_needs_agent_is_a_signal_not_a_call():
    result = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")]
    ).run(request_with("What is the evolutionary history of this animal?"))

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Evolution"
    assert result.output is None


def test_there_is_no_compare_route_and_no_general_clarify_route():
    """Asserted on what the workflow can actually execute.

    A grep would fire on the comments that document these as forbidden, which
    proves nothing. What matters is that no graph node and no plan step can
    ever be one of them.
    """
    agent = build_agent([reference("panthera_leo", 0.96)])
    node_names = {str(n).lower() for n in agent._workflow._graph.get_graph().nodes}

    for forbidden in ("compare", "clarify"):
        assert not any(forbidden in name for name in node_names), forbidden
        assert not any(forbidden in step for step in ALLOWED_PLAN_STEPS), forbidden

    # And a plan asking for one is rejected outright.
    for forbidden in ("compare", "clarify", "call_agent", "fetch_url"):
        assert sanitize_plan(
            {"steps": [*MANDATORY_PLAN_STEPS, forbidden]},
            default_top_k=30, rule_intent="recognition",
        ) is None, forbidden


def test_provenance_marks_every_mocked_component():
    provenance = build_agent(
        [reference("panthera_leo", 0.96)], retriever=MockQdrantRetriever(DIMENSION)
    ).run(request_with()).output["recognition_provenance"]

    assert provenance["embedding_mode"] == "mock"
    assert provenance["retrieval_mode"] == "mock_local_development"
    assert provenance["taxonomy_mode"] == "mock"
    assert provenance["taxonomy_sources"] == {"gbif": "mock", "ncbi": "mock"}
    assert provenance["workflow_engine"] == "langgraph"
    assert provenance["similarity_is_probability"] is False
    assert provenance["qdrant_contract_frozen"] is False


# ===========================================================================
# Mock GBIF / NCBI, as separate sources
# ===========================================================================

def test_both_sources_available():
    provider = MockTaxonomyProvider()
    enriched, report = provider.validate_candidate(candidate("panthera_leo", 0.9))
    assert report["gbif"]["available"] and report["ncbi"]["available"]
    assert enriched.taxonomy_status == "mock_verified"


def test_gbif_available_and_ncbi_unavailable():
    """The exact split the validated decisions call out."""
    provider = MockTaxonomyProvider(ncbi_unavailable=True)
    enriched, report = provider.validate_candidate(candidate("panthera_leo", 0.9))

    assert report["gbif"]["available"] is True
    assert report["ncbi"]["available"] is False
    assert enriched.gbif_id is not None
    assert enriched.ncbi_taxid is None
    assert enriched.taxonomy_status == "partial"


def test_gbif_unavailable_and_ncbi_available():
    provider = MockTaxonomyProvider(gbif_unavailable=True)
    enriched, report = provider.validate_candidate(candidate("panthera_leo", 0.9))
    assert report["gbif"]["available"] is False
    assert enriched.gbif_id is None
    assert enriched.taxonomy_status == "partial"


def test_candidate_absent_from_the_fixtures():
    provider = MockTaxonomyProvider()
    enriched, report = provider.validate_candidate(candidate("no_such_species", 0.9))
    assert report["gbif"]["matched"] is False and report["ncbi"]["matched"] is False
    assert enriched.taxonomy_status == "unverified"
    assert enriched.gbif_id is None and enriched.ncbi_taxid is None


def test_both_sources_unavailable():
    provider = MockTaxonomyProvider(simulate_unavailable=True)
    enriched, report = provider.validate_candidate(candidate("panthera_leo", 0.9))
    assert report["gbif"]["available"] is False and report["ncbi"]["available"] is False
    assert enriched.taxonomy_status == "unverified"


def test_the_sources_never_choose_a_species():
    """They annotate a candidate retrieval returned. They cannot replace it."""
    provider = MockTaxonomyProvider()
    original = candidate("panthera_leo", 0.9, name="Panthera leo")
    enriched, _ = provider.validate_candidate(original)
    assert enriched.species_id == original.species_id
    assert enriched.scientific_name == original.scientific_name


def test_the_sources_are_independently_constructible():
    assert MockGBIFProvider().source == "GBIF"
    assert MockNCBIProvider().source == "NCBI"
    assert MockGBIFProvider(unavailable=True).lookup("panthera_leo", "P").available is False


def test_a_taxonomy_outage_degrades_the_whole_request_safely():
    result = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")],
        taxonomy=MockTaxonomyProvider(simulate_unavailable=True),
    ).run(request_with())

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition_provenance"]["taxonomy_degraded"] is True
    assert result.output["gbif_id"] is None


# ===========================================================================
# Mock BioCLIP-2 and Mock Qdrant remain mocked
# ===========================================================================

def test_mock_bioclip_is_deterministic_through_the_graph():
    agent = build_agent([reference("panthera_leo", 0.96, "p1", "Panthera leo")])
    assert agent.run(request_with()).output == agent.run(request_with()).output


def test_mock_qdrant_error_branches_reach_a_controlled_failure():
    from ..domain.errors import ErrorCode

    for code in (ErrorCode.RETRIEVAL_UNAVAILABLE, ErrorCode.RETRIEVAL_TIMEOUT):
        result = build_agent(retriever=StubRetriever([], raises=code)).run(request_with())
        assert result.status is AgentStatus.FAILED
        assert result.output["error_code"] == code.value


def test_no_real_component_is_imported_at_module_level():
    """No real model, vector-store or HTTP client is loaded by importing this
    package.

    `qdrant_real.py` does import `qdrant_client` - but lazily, inside the
    constructor, which refuses to run until Chahd's contract is configured. So
    the check is on MODULE-LEVEL imports (column 0): those execute on import,
    and none of them may pull in a real component.
    """
    import pathlib
    import re

    package = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(
        r"^(?:import|from)\s+(torch|open_clip|bioclip|qdrant_client|requests|openai)\b"
    )
    offenders = []
    for path in package.rglob("*.py"):
        if ".venv" in path.parts or path.parent.name == "tests":
            continue
        if path.name == "smoke_test_azure.py":  # a standalone script, never imported
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.match(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, offenders


def test_importing_the_agent_loads_no_real_component():
    """The runtime proof behind the check above."""
    import sys

    for forbidden in ("torch", "open_clip", "bioclip", "qdrant_client"):
        assert forbidden not in sys.modules, forbidden


def test_the_deterministic_plan_covers_every_allowed_step():
    plan = deterministic_plan(intent="recognition", top_k=30)
    assert plan.steps == ALLOWED_PLAN_STEPS
    assert plan.source == "deterministic"
