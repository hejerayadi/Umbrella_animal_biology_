"""Sprint 4 Phase 2: the evaluation runner and the deterministic evaluators.

Most of this file drives the evaluators with **synthetic observations** - hand
written dicts shaped like an `AgentResult` - rather than by running the agent.
That is deliberate: an evaluator test that has to run the whole workflow to
reach one branch tests the workflow, not the evaluator, and cannot easily reach
the branches that only fire on malformed or contradictory output.

The last section does run the real thing: a complete 36-case offline dry run,
with sockets and `time.sleep` both booby-trapped, asserting that every case
produced exactly one result and that nothing reached a network.
"""
from __future__ import annotations

import json
import socket
import time

import pytest

from ..evaluation import consistency, evaluators, report, rubric, runner
from ..evaluation import manifest_schema as ms
from ..evaluation.results_schema import (
    FORBIDDEN_SUBSTRINGS,
    CaseResult,
    MetricOutcome,
    Outcome,
    RunSummary,
    safe_candidates,
    safe_provenance,
    sanitize_answer,
    sanitize_error,
)

MANIFEST = ms.load_manifest()
CASES = ms.cases(MANIFEST)
BY_ID = {c["case_id"]: c for c in CASES}

SEVEN = {
    "species": "Panthera leo", "species_id": "panthera_leo",
    "gbif_id": 5219404, "ncbi_taxid": 9689,
    "recognition": {"decision": "identified", "text_alignment": "neutral",
                    "score_is_probability": False, "explanation": "x",
                    "clarification_question": None},
    "recognition_candidates": [], "recognition_provenance": {},
}


def observation(**overrides):
    """A well-formed observation of a completed, identified request."""
    base = {
        "status": "completed",
        "output": dict(SEVEN),
        "target_agent": None,
        "prompt_to_target_agent": None,
        "decision": "identified",
        "primary_species": "Panthera leo",
        "species_id": "panthera_leo",
        "gbif_id": 5219404,
        "ncbi_taxid": 9689,
        "error_code": None,
        "unsupported_capability": None,
        "candidates": [
            {"species_id": "panthera_leo", "scientific_name": "Panthera leo",
             "classification_score": 0.9, "taxonomy_status": "mock_verified"},
        ],
        "provenance": {
            "score_is_probability": False, "workflow_engine": "langgraph",
            "recognition_mode": "mock_classification",
            "mock_provider_version": "v1", "taxonomy_executed": True,
            "taxonomy_degraded": False, "reasoning_llm_calls": 2,
            "reasoning_llm_used": True,
        },
        "latency_ms": 4.0,
        "classifier_invocations": 1,
    }
    base.update(overrides)
    return base


REAL_CASE = BY_ID["REC-PLEO-02"]
AMBIG_CASE = BY_ID["AMB-01"]
INVALID_CASE = BY_ID["INV-01"]
DELEG_CASE = BY_ID["DEL-01"]


# --- 1. evaluator unit tests: output schema, status --------------------------

def test_output_schema_accepts_exactly_the_seven_keys():
    assert evaluators.evaluate_output_schema(REAL_CASE, observation()).outcome is Outcome.PASS


def test_output_schema_rejects_an_added_key():
    bad = dict(SEVEN, extra="nope")
    result = evaluators.evaluate_output_schema(REAL_CASE, observation(output=bad))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "seven_keys_violated"


def test_output_schema_rejects_a_missing_key():
    bad = {k: v for k, v in SEVEN.items() if k != "gbif_id"}
    assert evaluators.evaluate_output_schema(
        REAL_CASE, observation(output=bad)).outcome is Outcome.FAIL


def test_output_schema_requires_the_two_key_failure_shape():
    ok = observation(status="failed", output={"error_code": "MISSING_IMAGE", "error": "x"})
    assert evaluators.evaluate_output_schema(INVALID_CASE, ok).outcome is Outcome.PASS
    bad = observation(status="failed", output={"error_code": "MISSING_IMAGE"})
    assert evaluators.evaluate_output_schema(INVALID_CASE, bad).outcome is Outcome.FAIL


def test_output_schema_requires_needs_agent_to_carry_no_output():
    ok = observation(status="needs_agent", output=None, target_agent="Genome")
    assert evaluators.evaluate_output_schema(DELEG_CASE, ok).outcome is Outcome.PASS
    bad = observation(status="needs_agent", output=dict(SEVEN), target_agent="Genome")
    assert evaluators.evaluate_output_schema(DELEG_CASE, bad).outcome is Outcome.FAIL


def test_status_compares_against_the_expected_status():
    assert evaluators.evaluate_status(REAL_CASE, observation()).outcome is Outcome.PASS
    assert evaluators.evaluate_status(
        REAL_CASE, observation(status="failed")).outcome is Outcome.FAIL


# --- 2/3. Top-1 and Top-5 ---------------------------------------------------

def test_top1_matches_on_exact_normalised_name():
    assert evaluators.evaluate_top1(
        REAL_CASE, observation(primary_species="  panthera   LEO ")).outcome is Outcome.PASS


def test_top1_does_not_match_a_different_species():
    result = evaluators.evaluate_top1(REAL_CASE, observation(primary_species="Panthera tigris"))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "top1_mismatch"


def test_top1_never_uses_similarity():
    """A near-miss is a miss. `Panthera leon` is not `Panthera leo`."""
    assert evaluators.evaluate_top1(
        REAL_CASE, observation(primary_species="Panthera leon")).outcome is Outcome.FAIL


def test_top1_fails_when_no_species_was_named():
    result = evaluators.evaluate_top1(REAL_CASE, observation(primary_species=None))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "no_primary_species"


def test_top5_finds_the_species_below_rank_one():
    candidates = [
        {"scientific_name": "Canis lupus", "classification_score": 0.6},
        {"scientific_name": "Vulpes vulpes", "classification_score": 0.5},
        {"scientific_name": "Panthera leo", "classification_score": 0.4},
    ]
    result = evaluators.evaluate_top5(
        REAL_CASE, observation(primary_species="Canis lupus", candidates=candidates))
    assert result.outcome is Outcome.PASS
    assert "rank=3" in result.detail


def test_top5_misses_when_the_species_is_absent():
    candidates = [{"scientific_name": "Canis lupus", "classification_score": 0.6}]
    assert evaluators.evaluate_top5(
        REAL_CASE, observation(candidates=candidates)).outcome is Outcome.FAIL


def test_top5_only_considers_the_first_five_labels():
    candidates = [
        {"scientific_name": f"Genus species{i}", "classification_score": 0.9 - i / 100}
        for i in range(5)
    ] + [{"scientific_name": "Panthera leo", "classification_score": 0.1}]
    assert evaluators.evaluate_top5(
        REAL_CASE, observation(candidates=candidates)).outcome is Outcome.FAIL


def test_top5_fails_cleanly_when_there_are_no_candidates():
    result = evaluators.evaluate_top5(REAL_CASE, observation(candidates=[]))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "no_candidates"


# --- 4/5/6. decision handling ----------------------------------------------

def test_a_decision_inside_the_accepted_set_passes():
    challenging = BY_ID["REC-PLEO-01"]
    assert "uncertain" in challenging["accepted_decisions"]
    assert evaluators.evaluate_decision(
        challenging, observation(decision="uncertain")).outcome is Outcome.PASS
    assert evaluators.evaluate_decision(
        challenging, observation(decision="identified")).outcome is Outcome.PASS


def test_a_decision_outside_the_accepted_set_fails():
    clear = BY_ID["REC-PLEO-02"]
    assert clear["accepted_decisions"] == ["identified"]
    assert evaluators.evaluate_decision(
        clear, observation(decision="not_identified")).outcome is Outcome.FAIL


def test_uncertain_is_a_completed_outcome_for_an_ambiguous_case():
    result = evaluators.evaluate_decision(AMBIG_CASE, observation(decision="uncertain"))
    assert result.outcome is Outcome.PASS


def test_not_identified_is_a_completed_outcome_for_an_ambiguous_case():
    result = evaluators.evaluate_decision(AMBIG_CASE, observation(decision="not_identified"))
    assert result.outcome is Outcome.PASS


def test_a_confident_identification_of_a_non_animal_image_fails():
    result = evaluators.evaluate_decision(AMBIG_CASE, observation(decision="identified"))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "decision_not_accepted"


def test_a_failure_case_has_no_decision_to_score():
    assert evaluators.evaluate_decision(
        INVALID_CASE, observation(decision=None)).outcome is Outcome.NOT_APPLICABLE


def test_an_unknown_decision_value_fails_rather_than_passing_quietly():
    assert evaluators.evaluate_decision(
        REAL_CASE, observation(decision="probably")).outcome is Outcome.FAIL


# --- 7. ambiguous metric N/A handling --------------------------------------

@pytest.mark.parametrize("metric", ["top1_species", "top5_species",
                                    "gbif_identifier", "ncbi_taxid"])
def test_species_metrics_are_not_applicable_on_ambiguous_cases(metric):
    assert not evaluators.is_applicable(AMBIG_CASE, metric)


def test_ambiguous_cases_are_never_scored_for_species():
    for evaluate in (evaluators.evaluate_top1, evaluators.evaluate_top5,
                     evaluators.evaluate_gbif, evaluators.evaluate_ncbi):
        result = evaluate(AMBIG_CASE, observation())
        assert result.outcome is Outcome.NOT_APPLICABLE, evaluate.__name__


def test_a_not_applicable_metric_is_neither_a_pass_nor_a_fail():
    result = evaluators.evaluate_top1(AMBIG_CASE, observation())
    assert result.outcome not in (Outcome.PASS, Outcome.FAIL)


# --- 8. taxonomy-unavailable handling ---------------------------------------

def test_a_null_identifier_is_not_penalised_when_taxonomy_never_executed():
    provenance = dict(observation()["provenance"], taxonomy_executed=False)
    obs = observation(gbif_id=None, ncbi_taxid=None, provenance=provenance)
    for evaluate in (evaluators.evaluate_gbif, evaluators.evaluate_ncbi):
        result = evaluate(REAL_CASE, obs)
        assert result.outcome is Outcome.NOT_APPLICABLE
        assert result.reason_code == "taxonomy_unavailable"


def test_a_null_identifier_is_not_penalised_when_taxonomy_degraded():
    provenance = dict(observation()["provenance"], taxonomy_degraded=True)
    obs = observation(gbif_id=None, provenance=provenance)
    assert evaluators.evaluate_gbif(REAL_CASE, obs).reason_code == "taxonomy_unavailable"


def test_a_wrong_identifier_still_fails_when_taxonomy_was_available():
    obs = observation(gbif_id=999999)
    result = evaluators.evaluate_gbif(REAL_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "identifier_mismatch"


# --- 9. GBIF and NCBI comparisons ------------------------------------------

def test_gbif_and_ncbi_match_the_verified_identifiers():
    assert evaluators.evaluate_gbif(REAL_CASE, observation()).outcome is Outcome.PASS
    assert evaluators.evaluate_ncbi(REAL_CASE, observation()).outcome is Outcome.PASS


def test_an_identifier_mismatch_following_a_species_mismatch_is_labelled_as_such():
    """So weakness analysis can separate a broken taxonomy lookup from a wrong
    species that the lookup then faithfully resolved."""
    obs = observation(primary_species="Panthera tigris", gbif_id=5219416)
    result = evaluators.evaluate_gbif(REAL_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "identifier_mismatch_after_species_mismatch"


def test_identifiers_are_not_applicable_when_the_case_records_none():
    assert evaluators.evaluate_gbif(AMBIG_CASE, observation()).outcome is Outcome.NOT_APPLICABLE


# --- 10. delegation ---------------------------------------------------------

def test_delegation_matches_the_expected_capability():
    obs = observation(status="needs_agent", output=None, target_agent="Genome")
    assert evaluators.evaluate_delegation(DELEG_CASE, obs).outcome is Outcome.PASS


def test_delegation_to_the_wrong_capability_fails():
    obs = observation(status="needs_agent", output=None, target_agent="Protein")
    result = evaluators.evaluate_delegation(DELEG_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "delegation_mismatch"


def test_a_missing_escalation_fails():
    result = evaluators.evaluate_delegation(DELEG_CASE, observation(target_agent=None))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "delegation_missing"


def test_a_target_that_is_not_a_registry_capability_fails():
    obs = observation(status="needs_agent", output=None, target_agent="Astrology")
    assert evaluators.evaluate_delegation(
        DELEG_CASE, obs).reason_code == "delegation_not_a_capability"


def test_the_resume_case_expects_no_delegation():
    resume = BY_ID["DEL-03"]
    assert evaluators.evaluate_delegation(
        resume, observation(target_agent=None)).outcome is Outcome.NOT_APPLICABLE
    unexpected = evaluators.evaluate_delegation(resume, observation(target_agent="Genome"))
    assert unexpected.outcome is Outcome.FAIL
    assert unexpected.reason_code == "unexpected_delegation"


# --- 11. task completion ----------------------------------------------------

def test_task_completion_passes_on_a_usable_completed_answer():
    assert evaluators.evaluate_task_completion(REAL_CASE, observation()).outcome is Outcome.PASS


def test_task_completion_fails_when_completed_but_unusable():
    broken = dict(SEVEN, recognition={"decision": None})
    result = evaluators.evaluate_task_completion(REAL_CASE, observation(output=broken))
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "completed_without_usable_answer"


def test_task_completion_requires_a_prompt_on_escalation():
    obs = observation(status="needs_agent", output=None, target_agent="Genome",
                      prompt_to_target_agent=None)
    result = evaluators.evaluate_task_completion(DELEG_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "escalated_without_prompt"


def test_task_completion_requires_an_error_code_on_failure():
    obs = observation(status="failed", output={"error_code": None, "error": "x"},
                      error_code=None)
    assert evaluators.evaluate_task_completion(
        INVALID_CASE, obs).reason_code == "failed_without_error_code"


# --- 12. provenance ---------------------------------------------------------

def test_provenance_passes_on_a_self_consistent_report():
    assert evaluators.evaluate_provenance(REAL_CASE, observation()).outcome is Outcome.PASS


def test_provenance_fails_if_a_score_is_claimed_to_be_a_probability():
    provenance = dict(observation()["provenance"], score_is_probability=True)
    result = evaluators.evaluate_provenance(REAL_CASE, observation(provenance=provenance))
    assert result.outcome is Outcome.FAIL
    assert "score_is_probability" in result.detail


def test_provenance_fails_if_a_real_version_is_filed_under_the_mock_key():
    provenance = dict(observation()["provenance"],
                      recognition_mode="remote_bioclip2_open_domain_species",
                      mock_provider_version="imageomics/bioclip-2")
    result = evaluators.evaluate_provenance(REAL_CASE, observation(provenance=provenance))
    assert result.outcome is Outcome.FAIL


def test_provenance_fails_if_taxonomy_claims_to_have_run_with_no_candidates():
    obs = observation(candidates=[])
    result = evaluators.evaluate_provenance(REAL_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert "taxonomy_executed=True with no candidate" in result.detail


def test_a_spent_but_rejected_llm_call_is_consistent_provenance():
    """`reasoning_llm_used` means "a result was genuinely accepted", not "a call
    was made" (workflows/state.py:65-66). A planner call that was spent and then
    failed or was rejected leaves calls=1 with used=False and the deterministic
    path in charge - which is correct behaviour, not a provenance contradiction.

    Pinned because the live run exposed this on 25 of 30 cases; the offline dry
    run never could, since the fake provider always succeeds."""
    provenance = dict(
        observation()["provenance"],
        reasoning_llm_calls=1, reasoning_llm_used=False,
        plan_source="deterministic", explanation_source="deterministic",
        plan_rejected=False,
    )
    result = evaluators.evaluate_provenance(REAL_CASE, observation(provenance=provenance))
    assert result.outcome is Outcome.PASS, result.detail


@pytest.mark.parametrize("field", ["plan_source", "explanation_source"])
def test_claiming_an_llm_source_without_accepting_a_result_is_inconsistent(field):
    provenance = dict(observation()["provenance"], reasoning_llm_used=False, **{field: "llm"})
    result = evaluators.evaluate_provenance(REAL_CASE, observation(provenance=provenance))
    assert result.outcome is Outcome.FAIL
    assert field in result.detail


def test_claiming_the_model_was_used_without_spending_a_call_is_inconsistent():
    provenance = dict(observation()["provenance"],
                      reasoning_llm_calls=0, reasoning_llm_used=True)
    result = evaluators.evaluate_provenance(REAL_CASE, observation(provenance=provenance))
    assert result.outcome is Outcome.FAIL
    assert "no call was made" in result.detail


def test_provenance_is_not_applicable_on_a_controlled_failure():
    obs = observation(status="failed", output={"error_code": "MISSING_IMAGE", "error": "x"})
    assert evaluators.evaluate_provenance(
        INVALID_CASE, obs).outcome is Outcome.NOT_APPLICABLE


# --- 13. tool selection -----------------------------------------------------

def test_tool_selection_flags_candidates_that_are_not_ranked():
    candidates = [
        {"scientific_name": "Panthera leo", "classification_score": 0.4},
        {"scientific_name": "Canis lupus", "classification_score": 0.9},
    ]
    obs = observation(candidates=candidates)
    result = evaluators.evaluate_tool_selection(DELEG_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert "non-increasing" in result.detail


def test_tool_selection_flags_a_primary_species_that_is_not_the_top_candidate():
    candidates = [
        {"scientific_name": "Canis lupus", "classification_score": 0.9},
        {"scientific_name": "Panthera leo", "classification_score": 0.4},
    ]
    obs = observation(candidates=candidates, primary_species="Panthera leo")
    result = evaluators.evaluate_tool_selection(DELEG_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert "not the top-ranked candidate" in result.detail


def test_tool_selection_flags_a_needs_agent_reply_that_carried_output():
    obs = observation(status="needs_agent", output=dict(SEVEN), target_agent="Genome")
    result = evaluators.evaluate_tool_selection(DELEG_CASE, obs)
    assert result.outcome is Outcome.FAIL


def test_tool_selection_flags_routing_an_unsupported_capability():
    obs = observation(status="needs_agent", output=None, target_agent="Genome",
                      unsupported_capability="visual_similarity_search")
    result = evaluators.evaluate_tool_selection(DELEG_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert "declined" in result.detail


# --- 14. the two-call ceiling ----------------------------------------------

@pytest.mark.parametrize("calls,expected", [(0, Outcome.PASS), (1, Outcome.PASS),
                                            (2, Outcome.PASS), (3, Outcome.FAIL)])
def test_the_agent_call_budget_is_judged_against_the_ceiling_of_two(calls, expected):
    provenance = dict(observation()["provenance"], reasoning_llm_calls=calls)
    result = evaluators.evaluate_agent_llm_budget(REAL_CASE, observation(provenance=provenance))
    assert result.outcome is expected


def test_a_missing_call_count_is_no_evidence_rather_than_a_free_pass():
    result = evaluators.evaluate_agent_llm_budget(REAL_CASE, observation(provenance={}))
    assert result.outcome is Outcome.NO_EVIDENCE
    assert result.reason_code == "no_call_count_available"


# --- 15. evaluator-call accounting -----------------------------------------

def test_evaluator_calls_are_counted_separately_and_are_zero_in_phase_2():
    summary = RunSummary(mode="offline", benchmark_id="b", started="s", finished="f",
                         results=[CaseResult(case_id="X", category="c", mode="offline",
                                             executed=True, agent_llm_calls=2)])
    accounting = summary.llm_call_accounting()
    assert accounting["agent_calls_total"] == 2
    assert accounting["evaluator_calls_total"] == 0
    assert accounting["evaluator_judge_used"] is False
    assert "never added together" in accounting["note"]


def test_no_llm_judge_exists_anywhere_in_the_evaluation_package():
    """Phase 2 uses the documented human rubric. A judge model would have to be
    counted against, and kept out of, the agent's own two-call ceiling."""
    assert rubric.describe()["llm_judge_used"] is False
    package = ms.MANIFEST_PATH.parent
    for module in package.glob("*.py"):
        source = module.read_text(encoding="utf-8").lower()
        for forbidden in ("openai(", "azurechatopenai", "chatcompletion", "judge_model"):
            assert forbidden not in source, f"{module.name} contains {forbidden!r}"


def test_the_human_rubric_cannot_override_a_deterministic_verdict():
    assert rubric.describe()["overrides_deterministic_metrics"] is False
    for criterion in rubric.CRITERIA:
        assert set(criterion["levels"]) == {1, 2, 3, 4, 5}
    assert rubric.validate_score(5) and not rubric.validate_score(6)
    assert not rubric.validate_score("5")


# --- 16. zero retry and controlled errors ----------------------------------

@pytest.mark.parametrize("invocations,expected", [(0, Outcome.PASS), (1, Outcome.PASS),
                                                  (2, Outcome.FAIL)])
def test_zero_retry_is_judged_from_the_invocation_count(invocations, expected):
    assert evaluators.evaluate_zero_retry(
        REAL_CASE, observation(classifier_invocations=invocations)).outcome is expected


def test_zero_retry_without_instrumentation_is_no_evidence():
    result = evaluators.evaluate_zero_retry(
        REAL_CASE, observation(classifier_invocations=None))
    assert result.outcome is Outcome.NO_EVIDENCE


def test_the_expected_controlled_error_code_passes():
    obs = observation(status="failed", error_code="EMPTY_INSTRUCTION")
    assert evaluators.evaluate_controlled_error(INVALID_CASE, obs).outcome is Outcome.PASS


def test_a_different_error_code_fails():
    obs = observation(status="failed", error_code="MISSING_IMAGE")
    result = evaluators.evaluate_controlled_error(INVALID_CASE, obs)
    assert result.outcome is Outcome.FAIL
    assert result.reason_code == "error_code_mismatch"


def test_an_uncontrolled_error_code_fails():
    obs = observation(status="failed", error_code="KABOOM")
    assert evaluators.evaluate_controlled_error(
        INVALID_CASE, obs).reason_code == "error_not_controlled"


def test_no_failure_at_all_on_a_failure_case_fails():
    assert evaluators.evaluate_controlled_error(
        INVALID_CASE, observation(error_code=None)).reason_code == "error_missing"


# --- latency ----------------------------------------------------------------

def test_latency_is_captured_not_judged_against_a_target():
    assert evaluators.evaluate_latency(REAL_CASE, observation()).outcome is Outcome.PASS
    assert evaluators.evaluate_latency(
        REAL_CASE, observation(latency_ms=None)).outcome is Outcome.NO_EVIDENCE
    assert evaluators.evaluate_latency(
        REAL_CASE, observation(latency_ms=-1)).outcome is Outcome.FAIL


# --- 17. every case gets every evaluator ------------------------------------

def test_evaluate_case_returns_one_outcome_per_metric_for_every_case():
    expected = len(evaluators.EVALUATORS) + len(evaluators.DEFERRED_METRICS)
    for case in CASES:
        outcomes = evaluators.evaluate_case(case, observation())
        assert len(outcomes) == expected, case["case_id"]
        names = [o.metric for o in outcomes]
        assert len(set(names)) == len(names), f"{case['case_id']} has a duplicate metric"


def test_every_outcome_carries_a_reason_code():
    for outcome in evaluators.evaluate_case(REAL_CASE, observation()):
        assert outcome.reason_code, outcome.metric


# --- 18. redaction ----------------------------------------------------------

CANARY_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg" + "A" * 240


def test_safe_provenance_drops_anything_not_on_the_allow_list():
    dirty = {"score_is_probability": False, "api_key": "SECRET",
             "raw_image": CANARY_DATA_URL, "Authorization": "Bearer x"}
    safe = safe_provenance(dirty)
    assert safe == {"score_is_probability": False}


def test_safe_candidates_drops_anything_not_on_the_allow_list():
    safe = safe_candidates([{"scientific_name": "Panthera leo", "image": CANARY_DATA_URL}])
    assert safe == [{"scientific_name": "Panthera leo"}]


def test_sanitize_error_records_only_the_exception_class_name():
    assert sanitize_error(ValueError("secret-endpoint https://x.invalid/key")) == "ValueError"


# --- sanitized answer capture (Phase 3 addition) ----------------------------

def test_a_plain_answer_survives_sanitisation_unchanged():
    text = "The image shows Panthera leo. The score is a ranking value, not a probability."
    assert sanitize_answer(text) == text


@pytest.mark.parametrize("blank", [None, "", "   ", 42, [], {}])
def test_a_missing_or_non_string_answer_becomes_none(blank):
    assert sanitize_answer(blank) is None


@pytest.mark.parametrize("canary,marker", [
    (CANARY_DATA_URL, "[redacted-data-url]"),
    ("A" * 300, "[redacted-blob]"),
    ("Authorization: Bearer abc123def456", "[redacted-credential]"),
    ("api_key=sk-live-0123456789abcdef", "[redacted-credential]"),
    ("-----BEGIN PRIVATE KEY-----zzz-----END PRIVATE KEY-----", "[redacted-key-block]"),
    (r"C:\Users\someone\secret\photo.jpg", "[redacted-path]"),
    ("/home/someone/.env", "[redacted-path]"),
    ("/Users/someone/Desktop/private.png", "[redacted-path]"),
    ("AZURE_OPENAI_API_KEY=abcdef123456", "[redacted-env]"),
])
def test_every_canary_is_redacted_out_of_a_captured_answer(canary, marker):
    cleaned = sanitize_answer(f"The animal is Panthera leo. {canary} End.")
    assert marker in cleaned
    assert canary not in cleaned
    assert "Panthera leo" in cleaned, "redaction must not destroy the reviewable answer"


@pytest.mark.parametrize("forbidden", FORBIDDEN_SUBSTRINGS)
def test_a_captured_answer_never_retains_a_forbidden_substring(forbidden):
    cleaned = sanitize_answer(f"prefix {CANARY_DATA_URL} Authorization: Bearer x suffix")
    assert forbidden not in cleaned


def test_a_long_answer_is_truncated_at_the_documented_bound():
    from ..evaluation.results_schema import MAX_ANSWER_CHARS, TRUNCATION_MARKER

    cleaned = sanitize_answer("word " * 2000)
    assert len(cleaned) <= MAX_ANSWER_CHARS + len(TRUNCATION_MARKER)
    assert cleaned.endswith(TRUNCATION_MARKER)


def test_an_answer_at_the_bound_is_not_truncated():
    from ..evaluation.results_schema import MAX_ANSWER_CHARS, TRUNCATION_MARKER

    # Deliberately prose-shaped rather than one long run of characters: an
    # unbroken 2000-character alphanumeric string is exactly what the blob
    # redaction is meant to catch, and it would fire before truncation.
    sentence = "Panthera leo is shown in the photograph. "
    text = (sentence * (MAX_ANSWER_CHARS // len(sentence) + 2))[:MAX_ANSWER_CHARS]
    text = text[:-1] + "x" if text.endswith(" ") else text

    cleaned = sanitize_answer(text)
    assert len(cleaned) == MAX_ANSWER_CHARS
    assert TRUNCATION_MARKER not in cleaned
    assert "[redacted" not in cleaned


def test_whitespace_in_a_captured_answer_is_collapsed():
    assert sanitize_answer("a\n\n  b\tc") == "a b c"


def test_the_offline_run_captures_a_sanitized_answer_for_every_executed_case():
    for result in runner.run().results:
        assert result.final_answer_sanitized, result.case_id
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert forbidden not in result.final_answer_sanitized


def test_a_captured_answer_is_never_the_complete_internal_state():
    """Only the user-facing answer is kept - never the state object, the context
    or the candidate payload."""
    for result in runner.run().results:
        answer = result.final_answer_sanitized or ""
        for leak in ("recognition_provenance", "image_sha256", "normalized",
                     "RecognitionState", "image_bytes"):
            assert leak not in answer, result.case_id


@pytest.mark.parametrize("forbidden", FORBIDDEN_SUBSTRINGS)
def test_a_serialized_run_never_contains_a_payload_or_credential(forbidden):
    summary = runner.run(mode=runner.MODE_OFFLINE)
    assert forbidden not in summary.to_json()


def test_a_serialized_run_contains_no_long_base64_blob():
    import re
    blob = re.findall(r"[A-Za-z0-9+/]{200,}={0,2}", runner.run().to_json())
    assert not blob


def test_the_image_bytes_sent_to_the_agent_never_reach_the_results():
    """The runner builds a real data URL to feed the agent. None of it may come
    back out through the result file."""
    case = BY_ID["REC-PLEO-02"]
    request = runner.build_request(case, mode=runner.MODE_OFFLINE)
    payload = request.context["recognition_image"]["data_url"].split(",", 1)[1]
    assert len(payload) > 100
    assert payload[:80] not in runner.run().to_json()


# --- 19. serialization and aggregation --------------------------------------

def test_the_run_serializes_to_valid_json():
    parsed = json.loads(runner.run().to_json())
    assert parsed["case_count"] == len(CASES)
    assert parsed["dry_run"] is True
    assert parsed["scores_are_infrastructure_validation_only"] is True


def test_metric_totals_account_for_every_case_exactly_once():
    summary = runner.run()
    for metric, counts in summary.metric_totals().items():
        assert sum(counts.values()) == len(summary.results), metric


def test_metric_totals_by_category_sum_to_the_overall_totals():
    summary = runner.run()
    overall = summary.metric_totals()
    per_category = summary.metric_totals_by_category()
    for metric, counts in overall.items():
        for outcome, total in counts.items():
            summed = sum(c.get(metric, {}).get(outcome, 0) for c in per_category.values())
            assert summed == total, f"{metric}/{outcome}"


def test_the_latency_summary_reports_every_executed_case():
    summary = runner.run()
    assert summary.latency_summary()["count"] == len(
        [r for r in summary.results if r.latency_ms is not None])


def test_provider_modes_are_reported_from_what_actually_ran():
    modes = runner.run().provider_modes()
    assert modes["recognition_mode"] == ["mock_classification"]
    assert modes["recognition_provider"] == ["ScriptedDryRunClassifier"]


# --- 20. failed cases are retained ------------------------------------------

def test_failing_cases_appear_in_the_failed_case_list():
    summary = runner.run()
    listed = {f["case_id"] for f in summary.failed_cases()}
    for result in summary.results:
        if any(m.outcome is Outcome.FAIL for m in result.metrics):
            assert result.case_id in listed


def test_a_case_that_raises_still_produces_a_result(monkeypatch):
    """A blow-up must not remove a case from the aggregation."""
    def boom(case, mode):
        raise RuntimeError("secret endpoint https://x.invalid")

    monkeypatch.setattr(runner, "build_request", boom)
    summary = runner.run()
    assert len(summary.results) == len(CASES)
    assert all(r.execution_error == "RuntimeError" for r in summary.results)
    assert all(not r.executed for r in summary.results)
    assert "x.invalid" not in summary.to_json()
    assert len(summary.failed_cases()) == len(CASES)


# --- 21. ordering and identity ----------------------------------------------

def test_results_preserve_the_committed_manifest_order_and_ids():
    summary = runner.run()
    assert [r.case_id for r in summary.results] == [c["case_id"] for c in CASES]


def test_exactly_one_result_exists_for_every_case():
    summary = runner.run()
    assert len(summary.results) == len(CASES) == 36
    ids = [r.case_id for r in summary.results]
    assert len(set(ids)) == len(ids)


def test_two_runs_produce_the_same_verdicts():
    """The dry run is deterministic: same scenarios, same labels, same outcomes.
    Latency is excluded because it is a measurement, not a decision."""
    def verdicts(summary):
        return [(r.case_id, r.status_observed, r.decision_observed,
                 r.primary_species_observed, tuple(
                     (m.metric, m.outcome.value) for m in r.metrics))
                for r in summary.results]

    assert verdicts(runner.run()) == verdicts(runner.run())


# --- 22. offline mode touches no network and never sleeps -------------------

def test_the_offline_dry_run_opens_no_socket(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the offline dry run attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    summary = runner.run(mode=runner.MODE_OFFLINE)
    assert len(summary.results) == len(CASES)


def test_the_offline_dry_run_never_sleeps(monkeypatch):
    def refuse(seconds):
        raise AssertionError(f"the offline dry run slept for {seconds}s")

    monkeypatch.setattr(time, "sleep", refuse)
    assert len(runner.run(mode=runner.MODE_OFFLINE).results) == len(CASES)


def test_the_offline_dry_run_needs_no_environment(monkeypatch):
    for var in list(runner.os.environ):
        if var.startswith(("AZURE_", "NCBI_", "RECOGNITION_", "BIOCLIP_", "TAXONOMY_")):
            monkeypatch.delenv(var, raising=False)
    assert len(runner.run().results) == len(CASES)


# --- 23. the live guard -----------------------------------------------------

def test_live_mode_refuses_without_the_explicit_opt_in():
    with pytest.raises(runner.LiveModeRefused) as excinfo:
        runner.run(mode=runner.MODE_LIVE, environ={})
    assert runner.LIVE_OPT_IN_VAR in str(excinfo.value)
    assert "Nothing was executed" in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE", "1 "])
def test_only_the_exact_opt_in_value_is_accepted(value):
    with pytest.raises(runner.LiveModeRefused):
        runner.require_live_opt_in({runner.LIVE_OPT_IN_VAR: value})


def test_the_opt_in_check_passes_on_exactly_one():
    runner.require_live_opt_in({runner.LIVE_OPT_IN_VAR: "1"})


def test_the_cli_refuses_live_without_opt_in_and_returns_a_nonzero_code(monkeypatch, capsys):
    monkeypatch.delenv(runner.LIVE_OPT_IN_VAR, raising=False)
    assert runner.main(["--mode", "live"]) == 2
    assert runner.LIVE_OPT_IN_VAR in capsys.readouterr().err


def test_offline_is_the_default_mode():
    import inspect
    assert inspect.signature(runner.run).parameters["mode"].default == runner.MODE_OFFLINE
    assert runner.run().mode == runner.MODE_OFFLINE


def test_live_mode_builds_the_agent_through_the_real_factories():
    """No provider is injected in live mode: `build_live_agent` passes nothing,
    so the agent's own factories choose - and refuse a mode they cannot honour."""
    import inspect
    source = inspect.getsource(runner.build_live_agent)
    for injected in ("classifier=", "taxonomy_provider=", "reasoning_llm="):
        assert injected not in source


# --- 24. the markdown report ------------------------------------------------

def test_the_report_renders_and_carries_the_dry_run_warning():
    text = report.render(runner.run())
    assert text.startswith("# Recognition Sprint 4")
    assert "Infrastructure validation only" in text
    assert "not an accuracy measurement" in text.lower()


def test_the_report_lists_every_case_and_every_metric():
    summary = runner.run()
    text = report.render(summary)
    for result in summary.results:
        assert f"`{result.case_id}`" in text
    for metric in summary.metric_names():
        assert f"`{metric}`" in text


def test_the_report_separates_agent_and_evaluator_call_counts():
    text = report.render(runner.run())
    assert "**Evaluator** LLM calls (total)" in text
    assert "never summed" in text


def test_the_report_contains_no_payload_or_credential():
    text = report.render(runner.run())
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden not in text


def test_a_live_summary_carries_no_dry_run_banner():
    summary = runner.run()
    summary.dry_run = False
    summary.mode = "live"
    assert "Infrastructure validation only" not in report.render(summary)


# --- 25. consistency preparation --------------------------------------------

def test_the_consistency_subset_has_at_least_five_cases():
    selected = consistency.select(CASES)
    assert len(selected) >= consistency.MINIMUM_CASES
    assert len(set(selected)) == len(selected)


def test_the_consistency_subset_is_deterministic_and_in_manifest_order():
    first = consistency.select(CASES)
    assert first == consistency.select(CASES)
    order = [c["case_id"] for c in CASES]
    assert first == sorted(first, key=order.index)


def test_the_consistency_subset_excludes_controlled_failures_and_offline_only_cases():
    for case_id in consistency.select(CASES):
        case = BY_ID[case_id]
        assert case["category"] != "invalid_input_or_dependency_failure"
        assert case["applicability"]["live"] is True


def test_consistency_compares_the_required_fields_and_not_the_wording():
    assert set(consistency.COMPARED_FIELDS) == {
        "status", "decision", "primary_species", "candidate_order", "target_agent"}
    assert "explanation" in consistency.EXCLUDED_FIELDS


def test_consistency_detects_an_unstable_repetition():
    stable = [{"status": "completed", "decision": "identified", "primary_species": "A",
               "candidate_order": ["A"], "target_agent": None}] * 3
    assert consistency.compare(stable)["stable"] is True
    unstable = [dict(stable[0]), dict(stable[0], decision="uncertain"), dict(stable[0])]
    verdict = consistency.compare(unstable)
    assert verdict["stable"] is False
    assert verdict["differing_fields"] == ["decision"]


def test_a_single_observation_is_not_reported_as_stable():
    verdict = consistency.compare([{"status": "completed"}])
    assert verdict["stable"] is None
    assert verdict["reason"] == "insufficient_repetitions"


def test_no_consistency_repetitions_were_executed_in_phase_2():
    assert consistency.describe()["executed_in_phase_2"] is False
    for result in runner.run().results:
        assert result.consistency.observations == []
        assert result.consistency.stable is None


def test_selected_cases_request_three_repetitions():
    for result in runner.run().results:
        expected = consistency.REPETITIONS if result.consistency.selected else 0
        assert result.consistency.repetitions_requested == expected


# --- 26. human-review fields ------------------------------------------------

def test_every_result_carries_an_unscored_human_review_slot():
    for result in runner.run().results:
        assert result.human_review.scored is False
        assert result.human_review.relevance is None


def test_relevance_is_reported_as_unscored_rather_than_as_a_pass():
    for result in runner.run().results:
        outcome = result.metric("relevance")
        assert outcome is not None
        assert outcome.outcome in (Outcome.NO_EVIDENCE, Outcome.NOT_APPLICABLE)


# --- 27. the complete 36-case offline dry run -------------------------------

def test_the_complete_offline_dry_run_executes_every_case(monkeypatch):
    """The end-to-end check: all 36 cases, real agent, injected providers, no
    network, no sleeping, one result each."""
    def refuse_socket(*args, **kwargs):
        raise AssertionError("network access during the offline dry run")

    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    monkeypatch.setattr(socket, "create_connection", refuse_socket)
    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(
        AssertionError("real sleep during the offline dry run")))

    summary = runner.run(mode=runner.MODE_OFFLINE)

    assert len(summary.results) == 36
    assert all(r.executed for r in summary.results)
    assert all(r.execution_error is None for r in summary.results)
    assert all(r.skipped_reason is None for r in summary.results)
    assert all(r.latency_ms is not None for r in summary.results)


def test_the_dry_run_reaches_every_terminal_status():
    observed = {r.status_observed for r in runner.run().results}
    assert observed == {"completed", "needs_agent", "failed"}


def test_the_dry_run_never_exceeds_the_agents_two_call_ceiling():
    summary = runner.run()
    assert summary.llm_call_accounting()["agent_calls_max_observed"] == 2
    for result in summary.results:
        if result.agent_llm_calls is not None:
            assert result.agent_llm_calls <= 2


def test_the_dry_run_observes_exactly_one_classification_per_request():
    for result in runner.run().results:
        if result.classifier_invocations is not None:
            assert result.classifier_invocations <= 1


def test_the_scripted_scenarios_are_fixed_and_position_based():
    """The scenario a case receives cannot depend on anything the agent said, or
    the dry run would be scoring itself."""
    for index in range(len(CASES)):
        case = CASES[index % len(CASES)]
        first = runner.scenario_for(case, index)
        assert first == runner.scenario_for(case, index)


def test_the_dry_run_deliberately_includes_wrong_answers():
    """If every scripted label were correct, the mismatch branches of the
    evaluators would never execute and the run would prove much less."""
    summary = runner.run()
    top1 = summary.metric_totals()["top1_species"]
    assert top1["pass"] > 0 and top1["fail"] > 0


def test_results_are_written_into_the_ignored_results_directory(tmp_path):
    summary = runner.run()
    path = runner.write_results(summary, tmp_path)
    assert path.parent == tmp_path
    assert json.loads(path.read_text(encoding="utf-8"))["case_count"] == 36
    gitignore = (ms.MANIFEST_PATH.parent / ".gitignore").read_text(encoding="utf-8")
    assert "results/" in gitignore


def test_the_runner_never_writes_outside_the_results_directory():
    import inspect
    source = inspect.getsource(runner.write_results)
    assert "mkdir" in source and "target_dir" in source
    assert "open(" not in source  # writes go through Path.write_text only
