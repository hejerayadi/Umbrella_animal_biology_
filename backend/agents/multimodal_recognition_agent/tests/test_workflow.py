"""End-to-end workflow scenarios.

These are the deterministic paired scenarios the sprint asks for: a clear
classification, close candidates, no candidate, a text/image conflict, invalid
input, delegation, resume, a classifier outage and a degraded taxonomy. Each one
pins a behaviour that would otherwise be easy to break silently.
"""
from __future__ import annotations

import pytest

from ..adapters.bioclip import MockBioCLIP2Provider
from ..adapters.taxonomy import MockTaxonomyProvider
from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
from ..domain.errors import ErrorCode
from ..fixtures.make_demo_images import DEMO_IMAGES, render
from ..schema import AgentRequest, AgentStatus
from ..workflows.state import HELPER_OUTPUT_KEYS
from .conftest import (
    StubClassifier,
    image_entry,
    make_config,
    png_bytes,
    prediction,
)

DEMO = {name: render(size, colour, fmt) for name, size, colour, fmt in DEMO_IMAGES}


def build_agent(predictions=None, *, classifier=None, taxonomy=None, config=None):
    return RecognitionAgent(
        config or make_config(),
        classifier=classifier or StubClassifier(predictions or []),
        taxonomy_provider=taxonomy or MockTaxonomyProvider(),
    )


def request_with(instruction, extra_context=None, raw=None):
    context = {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(raw or png_bytes())}
    context.update(extra_context or {})
    return AgentRequest(instruction=instruction, context=context)


# --- S1: a clear classification --------------------------------------------

def test_clear_match_is_identified_with_the_full_output_contract():
    agent = build_agent([
        prediction("panthera_leo", 0.95, "Panthera leo"),
        prediction("panthera_tigris", 0.40, "Panthera tigris"),
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
    assert result.output["recognition"]["score_is_probability"] is False


def test_completed_output_is_always_a_dict():
    """The orchestrator only merges dict output; anything else vanishes."""
    agent = build_agent([prediction("panthera_leo", 0.95)])
    result = agent.run(request_with("Identify this animal."))
    assert isinstance(result.output, dict)


# --- S2: close candidates --------------------------------------------------

def test_close_candidates_are_uncertain():
    agent = build_agent([
        prediction("panthera_leo", 0.88),
        prediction("panthera_tigris", 0.87),
    ])
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "uncertain"
    assert result.output["recognition"]["clarification_question"]


# --- S3: no candidate ------------------------------------------------------

def test_no_candidate_is_completed_not_identified():
    """An honest 'I don't know' is a finished workflow, not a failure."""
    agent = build_agent([])
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "not_identified"
    assert result.output["species"] is None
    assert result.output["species_id"] is None
    assert result.output["recognition_candidates"] == []


def test_weak_labels_are_not_identified():
    agent = build_agent([prediction("panthera_leo", 0.20)])
    result = agent.run(request_with("Identify this animal."))
    assert result.output["recognition"]["decision"] == "not_identified"


# --- S4: text / image conflict ---------------------------------------------

def test_conflicting_text_prevents_identification():
    agent = build_agent([prediction("panthera_leo", 0.98, "Panthera leo")])
    result = agent.run(request_with("This is a polar bear, confirm it."))

    assert result.output["recognition"]["text_alignment"] == "conflict"
    assert result.output["recognition"]["decision"] != "identified"


def test_text_cannot_introduce_an_unclassified_species():
    agent = build_agent([prediction("panthera_leo", 0.98, "Panthera leo")])
    result = agent.run(request_with("This is a polar bear."))

    species_ids = {c["species_id"] for c in result.output["recognition_candidates"]}
    assert "ursus_maritimus" not in species_ids
    assert result.output["species_id"] in (None, "panthera_leo")


def test_agreeing_text_is_reported_and_score_unchanged():
    predictions = [
        prediction("panthera_leo", 0.95, "Panthera leo"),
        prediction("panthera_tigris", 0.30),
    ]
    neutral = build_agent(predictions).run(request_with("Identify this animal."))
    agreeing = build_agent(predictions).run(request_with("Is this a lion?"))

    assert agreeing.output["recognition"]["text_alignment"] == "agree"
    assert (
        agreeing.output["recognition_candidates"][0]["classification_score"]
        == neutral.output["recognition_candidates"][0]["classification_score"]
        == 0.95
    )


def test_generic_text_is_neutral():
    agent = build_agent([prediction("panthera_leo", 0.95)])
    result = agent.run(request_with("What animal is this?"))
    assert result.output["recognition"]["text_alignment"] == "neutral"


# --- S5: invalid input -----------------------------------------------------

@pytest.mark.parametrize(
    "instruction, context, code",
    [
        (42, None, ErrorCode.EMPTY_INSTRUCTION),
        ("", None, ErrorCode.EMPTY_INSTRUCTION),
        ("   ", None, ErrorCode.EMPTY_INSTRUCTION),
        ("Identify this animal.", {}, ErrorCode.MISSING_IMAGE),
    ],
)
def test_invalid_input_fails_in_a_controlled_way(instruction, context, code):
    agent = build_agent([prediction("panthera_leo", 0.9)])
    request = (
        AgentRequest(instruction=instruction, context=context)
        if context is not None
        else request_with(instruction)
    )
    result = agent.run(request)

    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == code.value


def test_no_exception_escapes_the_agent():
    agent = build_agent([prediction("panthera_leo", 0.9)])
    for instruction, context in [
        ("", {}), ("x", {"recognition_image": "nonsense"}), ("x", {}),
    ]:
        result = agent.run(AgentRequest(instruction=instruction, context=context))
        assert result.status is AgentStatus.FAILED


# --- S6: delegation and resume --------------------------------------------

def test_scientific_follow_up_after_identification_delegates():
    agent = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")])
    result = agent.run(request_with("What is the evolutionary history of this animal?"))

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Evolution"
    assert "Panthera leo" in result.prompt_to_target_agent
    assert result.output is None


def test_follow_up_after_an_uncertain_result_does_not_delegate():
    """There is no subject to ask about yet, so ask the user instead."""
    agent = build_agent([
        prediction("panthera_leo", 0.88),
        prediction("panthera_tigris", 0.87),
    ])
    result = agent.run(request_with("What is the evolutionary history of this animal?"))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "uncertain"
    assert result.output["recognition"]["clarification_question"]


def test_follow_up_after_a_not_identified_result_does_not_delegate():
    agent = build_agent([])
    result = agent.run(request_with("What is the genome of this animal?"))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "not_identified"


def test_resume_completes_once_the_helper_output_is_in_context():
    agent = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")])
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


# --- S7: classifier outage -------------------------------------------------

@pytest.mark.parametrize(
    "code",
    [ErrorCode.CLASSIFICATION_UNAVAILABLE, ErrorCode.CLASSIFICATION_FIXTURE_INVALID,
     ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION],
)
def test_classifier_outage_is_a_controlled_failure(code):
    agent = build_agent(classifier=StubClassifier([], raises=code))
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == code.value


def test_a_classifier_that_returns_unranked_labels_fails_the_request():
    """Defence in depth: the ranking contract is enforced on any provider, not
    only on the shipped fixture."""
    agent = build_agent([prediction("a", 0.2), prediction("b", 0.9)])
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == "CLASSIFICATION_CONTRACT_VIOLATION"


# --- S8: taxonomy ----------------------------------------------------------

def test_mock_taxonomy_is_attached_and_labelled():
    agent = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")])
    result = agent.run(request_with("Identify this animal."))

    top = result.output["recognition_candidates"][0]
    assert top["taxonomy_status"] == "mock_verified"
    provenance = result.output["recognition_provenance"]
    assert provenance["gbif_mode"] == "mock"
    assert provenance["ncbi_mode"] == "mock"


def test_partial_taxonomy_is_reported_as_partial():
    agent = build_agent([prediction("ursus_maritimus", 0.96, "Ursus maritimus")])
    result = agent.run(request_with("Identify this animal."))

    top = result.output["recognition_candidates"][0]
    assert top["taxonomy_status"] == "partial"
    # The fixture has no GBIF id, so none is reported. Never invented.
    assert top["gbif_id"] is None
    assert top["ncbi_taxid"] is not None


def test_missing_identifiers_stay_null():
    agent = build_agent([prediction("vulpes_lagopus", 0.96, "Vulpes lagopus")])
    result = agent.run(request_with("Identify this animal."))

    assert result.output["gbif_id"] is None
    assert result.output["ncbi_taxid"] is None
    assert result.output["recognition_candidates"][0]["taxonomy_status"] == "unverified"


def test_unknown_species_gets_no_invented_taxonomy():
    agent = build_agent([prediction("unknown_species_xyz", 0.96, "Unknown species")])
    result = agent.run(request_with("Identify this animal."))

    top = result.output["recognition_candidates"][0]
    assert top["gbif_id"] is None and top["ncbi_taxid"] is None
    assert top["taxonomy_status"] == "unverified"


def test_taxonomy_outage_degrades_without_failing():
    agent = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")],
        taxonomy=MockTaxonomyProvider(simulate_unavailable=True),
    )
    result = agent.run(request_with("Identify this animal."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition_provenance"]["taxonomy_degraded"] is True
    assert result.output["recognition_candidates"][0]["taxonomy_status"] == "unverified"
    assert any("unverified" in w for w in result.output["recognition"]["warnings"])


# --- S9: provenance --------------------------------------------------------

def test_provenance_is_truthful_about_every_mock():
    agent = build_agent([prediction("panthera_leo", 0.96)])
    provenance = agent.run(request_with("Identify this animal."))\
        .output["recognition_provenance"]

    assert provenance["model_target"] == "BioCLIP-2"
    assert provenance["recognition_mode"] == "mock_classification"
    assert provenance["gbif_mode"] == "mock"
    assert provenance["ncbi_mode"] == "mock"
    assert provenance["reasoning_llm_used"] is False
    assert provenance["score_is_probability"] is False


def test_explanation_states_the_mock_provenance():
    agent = build_agent([prediction("panthera_leo", 0.96)])
    explanation = agent.run(request_with("Identify this animal."))\
        .output["recognition"]["explanation"].lower()

    assert "mock" in explanation
    assert "not a probability" in explanation
    assert "not by real bioclip-2 inference" in explanation
    assert "gbif and ncbi validation are mocked" in explanation


# --- the shipped oracle, end to end, with no stubs at all -------------------

def test_a_known_demo_image_is_identified_through_the_shipped_oracle():
    agent = RecognitionAgent(
        make_config(),
        classifier=MockBioCLIP2Provider(),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    result = agent.run(request_with("Identify this animal.",
                                    raw=DEMO["demo_identified.png"]))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "identified"
    assert result.output["species"] == "Panthera leo"
    assert result.output["gbif_id"] == 5219404
    assert result.output["ncbi_taxid"] == 9689
    assert result.output["recognition_provenance"]["recognition_provider"] \
        == "MockBioCLIP2Provider"


def test_a_close_demo_image_is_uncertain_through_the_shipped_oracle():
    agent = RecognitionAgent(
        make_config(), classifier=MockBioCLIP2Provider(),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    result = agent.run(request_with("Identify this animal.",
                                    raw=DEMO["demo_uncertain.png"]))

    assert result.output["recognition"]["decision"] == "uncertain"
    assert result.output["recognition"]["clarification_question"]


def test_a_partial_taxonomy_demo_image_reports_a_null_identifier():
    agent = RecognitionAgent(
        make_config(), classifier=MockBioCLIP2Provider(),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    result = agent.run(request_with("Identify this animal.",
                                    raw=DEMO["demo_partial_taxonomy.png"]))

    assert result.output["species"] == "Ursus maritimus"
    assert result.output["gbif_id"] is None
    assert result.output["ncbi_taxid"] == 29073
    assert result.output["recognition_candidates"][0]["taxonomy_status"] == "partial"


def test_an_unknown_demo_image_is_not_identified_and_names_nothing():
    agent = RecognitionAgent(
        make_config(), classifier=MockBioCLIP2Provider(),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    result = agent.run(request_with("Identify this animal.",
                                    raw=DEMO["demo_unknown.png"]))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "not_identified"
    assert result.output["species"] is None
    assert result.output["recognition_candidates"] == []
    assert result.output["recognition"]["request_better_image"] is True


def test_the_shipped_oracle_is_deterministic_across_repeated_requests():
    agent = RecognitionAgent(
        make_config(), classifier=MockBioCLIP2Provider(),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    request = request_with("Identify this animal.", raw=DEMO["demo_identified.png"])
    assert agent.run(request).output == agent.run(request).output


# --- S10: determinism ------------------------------------------------------

def test_the_same_request_produces_the_same_result():
    def run_once():
        return build_agent([prediction("panthera_leo", 0.96, "Panthera leo")])\
            .run(request_with("Identify this animal.")).output

    assert run_once() == run_once()


def test_configured_top_k_caps_the_candidates_end_to_end():
    predictions = [prediction(f"species_{i}", 0.9 - i / 100) for i in range(10)]
    agent = build_agent(predictions, config=make_config(top_k_species=3))
    result = agent.run(request_with("Identify this animal."))

    assert len(result.output["recognition_candidates"]) == 3
    assert result.output["recognition_provenance"]["top_k_requested"] == 3
