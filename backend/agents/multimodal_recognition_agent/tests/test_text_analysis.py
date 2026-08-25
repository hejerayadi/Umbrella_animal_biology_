"""Deterministic intent rules and text/image fusion."""
from __future__ import annotations

import pytest

from ..adapters.taxonomy import MockTaxonomyProvider
from ..text_analysis import RuleBasedTextAnalyzer, align_text_with_candidates
from .conftest import candidate


@pytest.fixture
def analyzer():
    return RuleBasedTextAnalyzer(known_names=MockTaxonomyProvider().known_names())


# --- intent ----------------------------------------------------------------

@pytest.mark.parametrize(
    "instruction",
    [
        "What animal is this?",
        "Identify this animal.",
        "Please tell me what species appears in this photograph.",
    ],
)
def test_plain_identification(analyzer, instruction):
    assert analyzer.analyze(instruction).intent == "recognition"


@pytest.mark.parametrize(
    "instruction",
    [
        "Which species look similar to this one?",
        "Show me animals that resemble this.",
        "What does this look like?",
    ],
)
def test_a_similarity_request_is_declined_not_given_its_own_intent(analyzer, instruction):
    """This agent has no similarity feature, so there is no similarity intent to
    fall into. The request is answered by classification and the declined
    capability is named."""
    evidence = analyzer.analyze(instruction)

    assert evidence.intent == "recognition"
    assert evidence.unsupported_capability == "visual_similarity_search"


def test_a_plain_identification_declines_nothing(analyzer):
    assert analyzer.analyze("Identify this animal.").unsupported_capability is None


def test_similarity_is_not_a_member_of_the_intent_enum():
    import typing

    from ..domain.models import Intent

    assert set(typing.get_args(Intent)) == {"recognition", "scientific_follow_up"}


@pytest.mark.parametrize(
    "instruction, capability",
    [
        ("What is the evolutionary history of this animal?", "Evolution"),
        ("Show me the genome of this species.", "Genome"),
        ("What is its conservation status and habitat?", "Biodiversity"),
        ("Which traits define this animal?", "Trait"),
        ("Find published papers about this species.", "Literature"),
        ("Show the protein structure for this animal.", "Protein"),
    ],
)
def test_follow_up_intent_and_capability_hint(analyzer, instruction, capability):
    evidence = analyzer.analyze(instruction)
    assert evidence.intent == "scientific_follow_up"
    assert evidence.requested_capability == capability


def test_clear_intent_uses_no_llm(analyzer):
    """There is no LLM to call - the mode is fixed and rule-based."""
    assert analyzer.mode == "rules"


# --- hints -----------------------------------------------------------------

def test_known_common_name_is_extracted(analyzer):
    assert analyzer.analyze("Is this a lion?").taxon_hint == "lion"


def test_scientific_binomial_is_extracted(analyzer):
    assert analyzer.analyze("Is this Panthera leo?").taxon_hint == "panthera leo"


def test_unknown_binomial_still_recognised_as_a_hint(analyzer):
    hint = analyzer.analyze("Could this be Canis lupus?").taxon_hint
    assert hint == "Canis lupus"
    # ...but it resolves to nothing, so it cannot create a conflict.
    assert analyzer.resolve_hint(hint) is None


def test_no_hint_when_no_species_is_named(analyzer):
    assert analyzer.analyze("what is this?").taxon_hint is None


def test_location_and_habitat_hints(analyzer):
    evidence = analyzer.analyze("I photographed this in Kenya, in the savanna.")
    assert evidence.location_hint == "Kenya"
    assert evidence.habitat_hint == "savanna"


# --- alignment -------------------------------------------------------------

def test_agreement_when_the_named_species_is_the_top_candidate(analyzer):
    evidence = analyzer.analyze("Is this a lion?")
    candidates = [candidate("panthera_leo", 0.9), candidate("panthera_tigris", 0.4)]
    resolved = analyzer.resolve_hint(evidence.taxon_hint)
    assert align_text_with_candidates(evidence, candidates, resolved) == "agree"


def test_conflict_when_a_known_species_was_not_retrieved(analyzer):
    evidence = analyzer.analyze("Is this a polar bear?")
    candidates = [candidate("panthera_leo", 0.9)]
    resolved = analyzer.resolve_hint(evidence.taxon_hint)
    assert align_text_with_candidates(evidence, candidates, resolved) == "conflict"


def test_conflict_when_the_named_species_is_only_a_lower_candidate(analyzer):
    evidence = analyzer.analyze("Is this a tiger?")
    candidates = [candidate("panthera_leo", 0.9), candidate("panthera_tigris", 0.5)]
    resolved = analyzer.resolve_hint(evidence.taxon_hint)
    assert align_text_with_candidates(evidence, candidates, resolved) == "conflict"


def test_neutral_without_a_hint(analyzer):
    evidence = analyzer.analyze("What animal is this?")
    candidates = [candidate("panthera_leo", 0.9)]
    assert align_text_with_candidates(evidence, candidates, None) == "neutral"


def test_unresolvable_name_is_neutral_not_conflict(analyzer):
    """We do not know what the user meant. Guessing would manufacture a conflict."""
    evidence = analyzer.analyze("Could this be Canis lupus?")
    candidates = [candidate("panthera_leo", 0.9)]
    resolved = analyzer.resolve_hint(evidence.taxon_hint)
    assert align_text_with_candidates(evidence, candidates, resolved) == "neutral"


def test_text_never_adds_a_candidate(analyzer):
    """The core rule: naming a species does not put it in the candidate list."""
    evidence = analyzer.analyze("This is definitely a polar bear.")
    candidates = [candidate("panthera_leo", 0.9)]
    before = [c.species_id for c in candidates]

    align_text_with_candidates(evidence, candidates, analyzer.resolve_hint(evidence.taxon_hint))

    assert [c.species_id for c in candidates] == before
    assert "ursus_maritimus" not in before


def test_alignment_is_neutral_when_nothing_was_retrieved(analyzer):
    evidence = analyzer.analyze("Is this a lion?")
    assert align_text_with_candidates(evidence, [], "panthera_leo") == "neutral"
