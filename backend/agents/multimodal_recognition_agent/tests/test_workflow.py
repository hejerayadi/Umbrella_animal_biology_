"""End-to-end workflow scenarios.

These are the deterministic paired scenarios the sprint asks for: a clear match,
close candidates, no match, a text/image conflict, invalid input, delegation,
resume, a retrieval outage and a degraded taxonomy. Each one pins a behaviour
that would otherwise be easy to break silently.
"""
from __future__ import annotations

import pytest

from ..adapters.qdrant_mock import MockQdrantRetriever
from ..adapters.taxonomy import MockTaxonomyProvider
from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
from ..domain.errors import ErrorCode
from ..schema import AgentRequest, AgentStatus
from ..workflows.state import HELPER_OUTPUT_KEYS
from .conftest import StubRetriever, image_entry, make_config, png_bytes, reference

DIMENSION = 32


def build_agent(references=None, *, retriever=None, taxonomy=None, config=None):
    return RecognitionAgent(
        config or make_config(),
        retriever=retriever or StubRetriever(references or []),
        taxonomy_provider=taxonomy or MockTaxonomyProvider(),
    )


def request_with(instruction, extra_context=None):
    context = {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())}
    context.update(extra_context or {})
    return AgentRequest(instruction=instruction, context=context)


# --- S1: a clear match -----------------------------------------------------

def test_clear_match_is_identified_with_the_full_output_contract():
    agent = build_agent([
        reference("panthera_leo", 0.95, "p1", "Panthera leo"),
        reference("panthera_tigris", 0.40, "p2", "Panthera tigris"),
    ])
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert set(result.output) == {
        "recognition", "species", "species_id", "gbif_id", "ncbi_taxid",
        "recognition_candidates", "recognition_provenance",
    }
    assert result.output["recognition"]["decision"] == "identified"
    assert result.output["species"] == "Panthera leo"
    assert result.output["species_id"] == "panthera_leo"
    assert result.output["recognition"]["similarity_is_probability"] is False


def test_completed_output_is_always_a_dict():
    """The orchestrator only merges dict output; anything else vanishes."""
    agent = build_agent([reference("panthera_leo", 0.95)])
    result = agent.run(request_with("Identify this animal."))
    assert isinstance(result.output, dict)


# --- S2: close candidates --------------------------------------------------

def test_close_candidates_are_uncertain():
    agent = build_agent([
        reference("panthera_leo", 0.88, "p1"),
        reference("panthera_tigris", 0.87, "p2"),
    ])
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "uncertain"
    assert result.output["recognition"]["clarification_question"]


# --- S3: no match ----------------------------------------------------------

def test_no_hits_is_completed_not_identified():
    """An honest 'I don't know' is a finished workflow, not a failure."""
    agent = build_agent([])
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "not_identified"
    assert result.output["species"] is None
    assert result.output["species_id"] is None
    assert result.output["recognition_candidates"] == []


def test_weak_hits_are_not_identified():
    agent = build_agent([reference("panthera_leo", 0.20)])
    result = agent.run(request_with("Identify this animal."))
    assert result.output["recognition"]["decision"] == "not_identified"


# --- S4: text / image conflict ---------------------------------------------

def test_conflicting_text_prevents_identification():
    agent = build_agent([reference("panthera_leo", 0.98, "p1", "Panthera leo")])
    result = agent.run(request_with("This is a polar bear, confirm it."))

    assert result.output["recognition"]["text_alignment"] == "conflict"
    assert result.output["recognition"]["decision"] != "identified"


def test_text_cannot_introduce_an_unretrieved_species():
    agent = build_agent([reference("panthera_leo", 0.98, "p1", "Panthera leo")])
    result = agent.run(request_with("This is a polar bear."))

    species_ids = {c["species_id"] for c in result.output["recognition_candidates"]}
    assert "ursus_maritimus" not in species_ids
    assert result.output["species_id"] in (None, "panthera_leo")


def test_agreeing_text_is_reported_and_score_unchanged():
    agent = build_agent([
        reference("panthera_leo", 0.95, "p1", "Panthera leo"),
        reference("panthera_tigris", 0.30, "p2"),
    ])
    neutral = build_agent([
        reference("panthera_leo", 0.95, "p1", "Panthera leo"),
        reference("panthera_tigris", 0.30, "p2"),
    ]).run(request_with("Identify this animal."))
    agreeing = agent.run(request_with("Is this a lion?"))

    assert agreeing.output["recognition"]["text_alignment"] == "agree"
    assert (
        agreeing.output["recognition_candidates"][0]["similarity_score"]
        == neutral.output["recognition_candidates"][0]["similarity_score"]
    )


def test_generic_text_is_neutral():
    agent = build_agent([reference("panthera_leo", 0.95)])
    result = agent.run(request_with("What animal is this?"))
    assert result.output["recognition"]["text_alignment"] == "neutral"


# --- S5: invalid input -----------------------------------------------------

@pytest.mark.parametrize(
    "instruction, context, code",
    [
        # The instruction is optional per the validated decisions; a NON-TEXT
        # instruction is still malformed input.
        (42, None, ErrorCode.EMPTY_INSTRUCTION),
        ("Identify this animal.", {}, ErrorCode.MISSING_IMAGE),
    ],
)
def test_invalid_input_fails_in_a_controlled_way(instruction, context, code):
    agent = build_agent([reference("panthera_leo", 0.9)])
    request = (
        AgentRequest(instruction=instruction, context=context)
        if context is not None
        else request_with(instruction)
    )
    result = agent.run(request)

    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == code.value


def test_no_exception_escapes_the_agent():
    agent = build_agent([reference("panthera_leo", 0.9)])
    for instruction, context in [
        ("", {}), ("x", {"recognition_image": "nonsense"}), ("x", {}),
    ]:
        result = agent.run(AgentRequest(instruction=instruction, context=context))
        assert result.status is AgentStatus.FAILED


# --- S6: delegation and resume --------------------------------------------

def test_scientific_follow_up_after_identification_delegates():
    agent = build_agent([reference("panthera_leo", 0.96, "p1", "Panthera leo")])
    result = agent.run(request_with("What is the evolutionary history of this animal?"))

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Evolution"
    assert "Panthera leo" in result.prompt_to_target_agent
    assert result.output is None


def test_follow_up_after_an_uncertain_result_does_not_delegate():
    """There is no subject to ask about yet, so ask the user instead."""
    agent = build_agent([
        reference("panthera_leo", 0.88, "p1"),
        reference("panthera_tigris", 0.87, "p2"),
    ])
    result = agent.run(request_with("What is the evolutionary history of this animal?"))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "uncertain"
    assert result.output["recognition"]["clarification_question"]


def test_resume_completes_once_the_helper_output_is_in_context():
    agent = build_agent([reference("panthera_leo", 0.96, "p1", "Panthera leo")])
    resumed = agent.run(
        request_with(
            "What is the evolutionary history of this animal?",
            {HELPER_OUTPUT_KEYS["Evolution"]: "Evolutionary relationship completed."},
        )
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.output["species"] == "Panthera leo"


def test_resume_keys_match_the_agent_cards():
    """These are the live cross-agent contract, not names invented here."""
    assert HELPER_OUTPUT_KEYS["Evolution"] == "evolution_analysis"
    assert HELPER_OUTPUT_KEYS["Genome"] == "genome"
    assert HELPER_OUTPUT_KEYS["Biodiversity"] == "biodiversity_report"
    assert HELPER_OUTPUT_KEYS["Trait"] == "traits"
    assert HELPER_OUTPUT_KEYS["Literature"] == "papers"
    assert HELPER_OUTPUT_KEYS["Protein"] == "protein_structure"


def test_agent_never_references_a_peer_endpoint():
    """No import of another agent, no peer URL, no HTTP client."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent
    forbidden = ("localhost:800", "AGENT_URL", "backend.agents.genome_agent",
                 "backend.agents.evolution_agent", "backend.orchestrator",
                 "import httpx", "import requests")
    offenders = []
    for path in package.rglob("*.py"):
        # Skip this agent's own installed dependencies and its test suite.
        if ".venv" in path.parts or path.parent.name == "tests":
            continue
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {n}" for n in forbidden if n in text]
    assert not offenders, offenders


# --- S7: retrieval outage --------------------------------------------------

@pytest.mark.parametrize(
    "code", [ErrorCode.RETRIEVAL_UNAVAILABLE, ErrorCode.RETRIEVAL_TIMEOUT]
)
def test_retrieval_outage_is_a_controlled_failure(code):
    agent = build_agent(retriever=StubRetriever([], raises=code))
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == code.value


def test_unfrozen_qdrant_contract_fails_controlled_not_silently():
    config = make_config(retrieval_mode="real")
    agent = RecognitionAgent.__new__(RecognitionAgent)  # skip provider construction
    from ..agent import _build_retriever
    from ..domain.errors import RecognitionError

    with pytest.raises(RecognitionError) as caught:
        _build_retriever(config)
    assert caught.value.code is ErrorCode.QDRANT_CONTRACT_NOT_FROZEN
    assert agent is not None


# --- S8: taxonomy ----------------------------------------------------------

def test_mock_taxonomy_is_attached_and_labelled():
    agent = build_agent([reference("panthera_leo", 0.96, "p1", "Panthera leo")])
    result = agent.run(request_with("Identify this animal."))

    top = result.output["recognition_candidates"][0]
    assert top["taxonomy_status"] == "mock_verified"
    assert result.output["recognition_provenance"]["taxonomy_mode"] == "mock"


def test_partial_taxonomy_is_reported_as_partial():
    agent = build_agent([reference("ursus_maritimus", 0.96, "p1", "Ursus maritimus")])
    result = agent.run(request_with("Identify this animal."))

    top = result.output["recognition_candidates"][0]
    assert top["taxonomy_status"] == "partial"
    # The fixture has no GBIF id, so none is reported. Never invented.
    assert top["gbif_id"] is None
    assert top["ncbi_taxid"] is not None


def test_missing_identifiers_stay_null():
    agent = build_agent([reference("vulpes_lagopus", 0.96, "p1", "Vulpes lagopus")])
    result = agent.run(request_with("Identify this animal."))

    assert result.output["gbif_id"] is None
    assert result.output["ncbi_taxid"] is None
    assert result.output["recognition_candidates"][0]["taxonomy_status"] == "unverified"


def test_unknown_species_gets_no_invented_taxonomy():
    agent = build_agent([reference("unknown_species_xyz", 0.96, "p1", "Unknown species")])
    result = agent.run(request_with("Identify this animal."))

    top = result.output["recognition_candidates"][0]
    assert top["gbif_id"] is None and top["ncbi_taxid"] is None
    assert top["taxonomy_status"] == "unverified"


def test_taxonomy_outage_degrades_without_failing():
    agent = build_agent(
        [reference("panthera_leo", 0.96, "p1", "Panthera leo")],
        taxonomy=MockTaxonomyProvider(simulate_unavailable=True),
    )
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition_provenance"]["taxonomy_degraded"] is True
    assert result.output["recognition_candidates"][0]["taxonomy_status"] == "unverified"


# --- S9: provenance --------------------------------------------------------

def test_provenance_is_truthful_about_every_mock():
    agent = build_agent([reference("panthera_leo", 0.96)])
    provenance = agent.run(request_with("Identify this animal."))\
        .output["recognition_provenance"]

    assert provenance["embedding_mode"] == "mock"
    assert provenance["embedding_provider"] == "MockBioCLIP2Provider"
    assert provenance["taxonomy_mode"] == "mock"
    assert provenance["reasoning_llm_used"] is False
    assert provenance["similarity_is_probability"] is False
    assert provenance["qdrant_contract_frozen"] is False


def test_known_demo_image_reaches_identified_through_the_fixture_retriever():
    """The full path with no stubs: a demo image mapped to a fixture species
    seed retrieves that species' cluster and clears the confidence gate."""
    from .conftest import sha256_of

    raw = png_bytes()
    config = make_config(image_seed_overrides={sha256_of(raw): "panthera_leo"})
    agent = RecognitionAgent(
        config,
        retriever=MockQdrantRetriever(DIMENSION),
        taxonomy_provider=MockTaxonomyProvider(),
    )

    result = agent.run(
        AgentRequest(
            instruction="Identify this animal.",
            context={RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(raw)},
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "identified"
    assert result.output["species"] == "Panthera leo"
    assert result.output["recognition_candidates"][0]["reference_count"] == 2
    # Still honest about where it came from.
    assert result.output["recognition_provenance"]["retrieval_mode"] == "mock_local_development"


def test_an_unmapped_image_is_not_identified_through_the_fixture_retriever():
    agent = RecognitionAgent(
        make_config(),
        retriever=MockQdrantRetriever(DIMENSION),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    result = agent.run(request_with("Identify this animal."))

    assert result.output["recognition"]["decision"] == "not_identified"
    assert result.output["species"] is None


def test_local_retrieval_never_claims_to_be_the_real_collection():
    agent = RecognitionAgent(
        make_config(), retriever=MockQdrantRetriever(DIMENSION),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    provenance = agent.run(request_with("Identify this animal."))\
        .output["recognition_provenance"]

    assert provenance["retrieval_mode"] == "mock_local_development"
    assert provenance["retrieval_mode"] != "real_minimal"
    assert provenance["collection"] is None


def test_explanation_states_the_mock_provenance():
    agent = build_agent([reference("panthera_leo", 0.96)])
    explanation = agent.run(request_with("Identify this animal."))\
        .output["recognition"]["explanation"]

    lowered = explanation.lower()
    assert "mock" in lowered
    assert "not a probability" in lowered
    assert "not the sprint 2 qdrant collection" in lowered


def test_dimension_source_is_flagged_as_a_local_default():
    agent = build_agent([reference("panthera_leo", 0.96)])
    provenance = agent.run(request_with("Identify this animal."))\
        .output["recognition_provenance"]
    assert provenance["embedding_dimension_source"] == "local_default_pending_qdrant_manifest"


# --- S10: determinism ------------------------------------------------------

def test_the_same_request_produces_the_same_result():
    def run_once():
        return build_agent([reference("panthera_leo", 0.96, "p1", "Panthera leo")])\
            .run(request_with("Identify this animal.")).output

    assert run_once() == run_once()


def test_dropped_payloads_are_surfaced_as_a_warning():
    points = [
        {"vector_seed": "a", "payload": {
            "point_id": "good", "species_id": "panthera_leo", "scientific_name": "Panthera leo",
            "dataset_version": "v0", "embedding_mode": "mock_bioclip2"}},
        {"vector_seed": "b", "payload": {"point_id": "bad", "scientific_name": "No species id"}},
    ]
    agent = RecognitionAgent(
        make_config(), retriever=MockQdrantRetriever(DIMENSION, points=points),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    result = agent.run(request_with("Identify this animal."))

    warnings = result.output["recognition"].get("warnings", [])
    assert any("dropped" in warning for warning in warnings)
