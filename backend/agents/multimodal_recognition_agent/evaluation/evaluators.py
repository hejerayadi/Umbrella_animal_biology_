"""Deterministic evaluators for the Sprint 4 benchmark.

Every function here takes one manifest case plus one observation of what the
agent actually returned, and produces exactly one `MetricOutcome`. There is no
model in this module, no similarity measure, and no threshold that could be
tuned to improve a score.

Three rules run through all of it.

**Exact names only.** Species comparison is a normalised string equality -
trimmed, whitespace-collapsed, case-folded. Nothing fuzzy. A near-miss is a
miss, because "close enough" is precisely the judgement a benchmark exists to
avoid making silently.

**Applicability is declared by the case, not decided here.** A metric the
manifest lists as inapplicable returns `NOT_APPLICABLE`. It never returns
`PASS` (which would inflate the score) and never returns `FAIL` (which would
punish the agent for a question nobody asked).

**Absent evidence is its own answer.** When a metric applies but the run gave
nothing to judge it by - no call counter, no human score yet - the result is
`NO_EVIDENCE`, not a guess in either direction.
"""
from __future__ import annotations

import re
from typing import Any

from .manifest_schema import (
    VALID_CAPABILITIES,
    VALID_DECISIONS,
    VALID_ERROR_CODES,
    VALID_STATUSES,
)
from .results_schema import MetricOutcome, Outcome

SEVEN_OUTPUT_KEYS = frozenset({
    "species", "species_id", "gbif_id", "ncbi_taxid",
    "recognition", "recognition_candidates", "recognition_provenance",
})

FAILURE_OUTPUT_KEYS = frozenset({"error_code", "error"})

#: Contract guarantees that hold for every case whatever the manifest says, so
#: they are evaluated even when a case does not list them. A case cannot opt out
#: of "the agent must return a valid schema" or "the agent must not retry".
UNIVERSAL_METRICS = frozenset({
    "output_schema", "status", "agent_llm_call_budget", "zero_retry", "latency",
})

#: Metrics whose verdict Phase 2 cannot produce: they need a human reviewer or
#: the Phase 3 repetition protocol. Recorded as NO_EVIDENCE so they stay visible
#: in aggregation instead of quietly disappearing.
DEFERRED_METRICS = {
    "relevance": "human_rubric_not_scored",
    "response_consistency": "requires_phase3_repetitions",
}

TOP_K = 5


def normalise_name(name: Any) -> str:
    """Trim, collapse internal whitespace, case-fold. Nothing else."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name).strip().casefold()


# --- applicability ----------------------------------------------------------

def is_applicable(case: dict, metric: str) -> bool:
    """Whether `metric` should be judged for `case`."""
    applicability = case.get("applicability") or {}
    if metric in (applicability.get("metrics_not_applicable") or ()):
        return False
    if metric in (applicability.get("metrics_applicable") or ()):
        return True
    return metric in UNIVERSAL_METRICS


def _na(metric: str, reason: str = "not_applicable_for_case", detail: str = "") -> MetricOutcome:
    return MetricOutcome(metric, Outcome.NOT_APPLICABLE, reason, detail)


def _verdict(metric: str, ok: bool, pass_code: str, fail_code: str, detail: str = "") -> MetricOutcome:
    return MetricOutcome(
        metric, Outcome.PASS if ok else Outcome.FAIL,
        pass_code if ok else fail_code, detail,
    )


# --- 1. output schema -------------------------------------------------------

def evaluate_output_schema(case: dict, obs: dict) -> MetricOutcome:
    """The agent's public contract, judged against what it actually returned."""
    metric = "output_schema"
    status = obs.get("status")
    output = obs.get("output")

    if status is None:
        return MetricOutcome(metric, Outcome.FAIL, "no_status_returned",
                             "the agent returned no status at all")

    if status not in VALID_STATUSES:
        return MetricOutcome(metric, Outcome.FAIL, "unknown_status",
                             f"status {status!r} is not one Recognition can return")

    if status == "failed":
        ok = isinstance(output, dict) and set(output) == FAILURE_OUTPUT_KEYS
        return _verdict(metric, ok, "failure_shape_exact", "failure_shape_wrong",
                        f"keys={sorted(output) if isinstance(output, dict) else output!r}")

    if status == "needs_agent":
        ok = output is None and bool(obs.get("target_agent"))
        return _verdict(metric, ok, "delegation_shape_exact", "delegation_shape_wrong",
                        "needs_agent must carry output=None and a target_agent")

    ok = isinstance(output, dict) and set(output) == SEVEN_OUTPUT_KEYS
    detail = f"keys={sorted(output) if isinstance(output, dict) else output!r}"
    return _verdict(metric, ok, "seven_keys_exact", "seven_keys_violated", detail)


# --- 2. status --------------------------------------------------------------

def evaluate_status(case: dict, obs: dict) -> MetricOutcome:
    metric = "status"
    expected, observed = case["expected_status"], obs.get("status")
    return _verdict(metric, expected == observed, "status_as_expected", "status_mismatch",
                    f"expected={expected!r} observed={observed!r}")


# --- 3. decision ------------------------------------------------------------

def evaluate_decision(case: dict, obs: dict) -> MetricOutcome:
    """Scored against the case's *accepted* set, not one forced answer.

    A challenging photograph that the agent honestly calls `uncertain` has not
    got it wrong; a non-animal image that it confidently identifies has.
    """
    metric = "decision"
    if not is_applicable(case, metric):
        return _na(metric)
    accepted = case.get("accepted_decisions") or []
    if not accepted:
        return _na(metric, "no_decision_expected", "this case expects a controlled failure")

    observed = obs.get("decision")
    if observed is None:
        return MetricOutcome(metric, Outcome.NO_EVIDENCE, "no_decision_returned",
                             f"accepted={accepted}")
    if observed not in VALID_DECISIONS:
        return MetricOutcome(metric, Outcome.FAIL, "unknown_decision",
                             f"observed={observed!r}")
    return _verdict(metric, observed in accepted, "decision_accepted", "decision_not_accepted",
                    f"observed={observed!r} accepted={accepted}")


# --- 4/5. Top-1 and Top-5 ---------------------------------------------------

def evaluate_top1(case: dict, obs: dict) -> MetricOutcome:
    metric = "top1_species"
    if not is_applicable(case, metric):
        return _na(metric, "no_species_ground_truth")
    expected = case.get("expected_species")
    if not expected:
        return _na(metric, "no_species_ground_truth")

    observed = obs.get("primary_species")
    if observed is None:
        return _verdict(metric, False, "top1_match", "no_primary_species",
                        f"expected={expected!r} but the agent named no species")
    return _verdict(
        metric, normalise_name(observed) == normalise_name(expected),
        "top1_match", "top1_mismatch", f"expected={expected!r} observed={observed!r}",
    )


def evaluate_top5(case: dict, obs: dict) -> MetricOutcome:
    """Is the expected species anywhere in the ranked labels the agent returned?

    Accepted labels come from the case when it declares them, so a species with
    a scientifically justified alternative name is not marked wrong on spelling.
    """
    metric = "top5_species"
    if not is_applicable(case, metric):
        return _na(metric, "no_species_ground_truth")
    expected = case.get("expected_species")
    if not expected:
        return _na(metric, "no_species_ground_truth")

    accepted = {normalise_name(n) for n in (case.get("accepted_top_k_labels") or [expected])}
    accepted.add(normalise_name(expected))

    candidates = obs.get("candidates") or []
    names = [normalise_name(c.get("scientific_name")) for c in candidates[:TOP_K]]
    if not names:
        return _verdict(metric, False, "top5_hit", "no_candidates",
                        f"expected={expected!r} but no candidate was returned")
    hit = any(name in accepted for name in names)
    rank = names.index(next(n for n in names if n in accepted)) + 1 if hit else None
    return _verdict(metric, hit, "top5_hit", "top5_miss",
                    f"expected={expected!r} rank={rank} of {len(names)}")


# --- 6/7. taxonomy identifiers ---------------------------------------------

def _taxonomy_unavailable(obs: dict) -> bool:
    provenance = obs.get("provenance") or {}
    return (provenance.get("taxonomy_executed") is False
            or provenance.get("taxonomy_degraded") is True)


def _evaluate_identifier(case: dict, obs: dict, metric: str,
                         expected_key: str, observed_key: str) -> MetricOutcome:
    if not is_applicable(case, metric):
        return _na(metric, "no_identifier_ground_truth")
    expected = case.get(expected_key)
    if expected is None:
        return _na(metric, "no_identifier_ground_truth")

    observed = obs.get(observed_key)
    if observed is None and _taxonomy_unavailable(obs):
        # The agent is required to leave a missing identifier null rather than
        # infer one. Scoring that as a failure would reward inventing a number.
        return _na(metric, "taxonomy_unavailable",
                   "taxonomy did not execute or degraded; a null identifier is correct here")

    if observed == expected:
        return MetricOutcome(metric, Outcome.PASS, "identifier_match",
                             f"expected={expected} observed={observed}")

    species_mismatch = (
        case.get("expected_species")
        and normalise_name(obs.get("primary_species")) != normalise_name(case["expected_species"])
    )
    reason = (
        "identifier_mismatch_after_species_mismatch" if species_mismatch
        else "identifier_mismatch"
    )
    detail = f"expected={expected} observed={observed}"
    if species_mismatch:
        detail += " (the primary species was already wrong, so this follows from that)"
    return MetricOutcome(metric, Outcome.FAIL, reason, detail)


def evaluate_gbif(case: dict, obs: dict) -> MetricOutcome:
    return _evaluate_identifier(case, obs, "gbif_identifier", "expected_gbif_id", "gbif_id")


def evaluate_ncbi(case: dict, obs: dict) -> MetricOutcome:
    return _evaluate_identifier(case, obs, "ncbi_taxid", "expected_ncbi_taxid", "ncbi_taxid")


# --- 8. delegation ----------------------------------------------------------

def evaluate_delegation(case: dict, obs: dict) -> MetricOutcome:
    metric = "delegation_capability"
    expected = case.get("expected_delegation_capability")
    observed = obs.get("target_agent")

    # Checked BEFORE applicability, deliberately. Escalating when the case
    # expects no escalation is a contract violation however the case declares
    # its metrics - and the resume case is exactly the one that declares this
    # metric inapplicable while being the case where a second escalation would
    # matter most. Short-circuiting on applicability first would let the agent
    # loop back to the Capability Resolver unnoticed.
    if expected is None and observed:
        return MetricOutcome(metric, Outcome.FAIL, "unexpected_delegation",
                             f"escalated to {observed!r} when none was expected")

    if not is_applicable(case, metric):
        return _na(metric)
    if expected is None:
        return _na(metric, "no_delegation_expected")

    if observed is None:
        return MetricOutcome(metric, Outcome.FAIL, "delegation_missing",
                             f"expected capability {expected!r}, agent escalated to nobody")
    if observed not in VALID_CAPABILITIES:
        return MetricOutcome(metric, Outcome.FAIL, "delegation_not_a_capability",
                             f"observed={observed!r}")
    return _verdict(metric, observed == expected, "delegation_match", "delegation_mismatch",
                    f"expected={expected!r} observed={observed!r}")


# --- 9. task completion -----------------------------------------------------

def evaluate_task_completion(case: dict, obs: dict) -> MetricOutcome:
    """Did the agent reach the terminal state this case requires, carrying the
    payload that state is obliged to carry?

    Broader than `status`: a `completed` reply missing its decision, or a
    `needs_agent` reply with no prompt for the receiving agent, has produced the
    right label on an unusable answer.
    """
    metric = "task_completion"
    if not is_applicable(case, metric):
        return _na(metric)

    expected, status = case["expected_status"], obs.get("status")
    if status != expected:
        return MetricOutcome(metric, Outcome.FAIL, "wrong_terminal_state",
                             f"expected={expected!r} observed={status!r}")

    if status == "completed":
        output = obs.get("output")
        ok = (isinstance(output, dict) and set(output) == SEVEN_OUTPUT_KEYS
              and isinstance(output.get("recognition"), dict)
              and output["recognition"].get("decision") in VALID_DECISIONS)
        return _verdict(metric, ok, "completed_with_usable_answer",
                        "completed_without_usable_answer", "")
    if status == "needs_agent":
        ok = bool(obs.get("target_agent")) and bool(obs.get("prompt_to_target_agent"))
        return _verdict(metric, ok, "escalated_with_prompt", "escalated_without_prompt", "")

    ok = bool(obs.get("error_code"))
    return _verdict(metric, ok, "failed_with_error_code", "failed_without_error_code", "")


# --- 10. provenance truthfulness -------------------------------------------

def evaluate_provenance(case: dict, obs: dict) -> MetricOutcome:
    """Does provenance describe what actually ran?

    Checks the invariants the agent itself claims: a score is never a
    probability, a real model version is never filed under a mock key, and
    taxonomy cannot report that it answered when there was nothing to look up.
    """
    metric = "provenance_truthfulness"
    if not is_applicable(case, metric):
        return _na(metric)

    provenance = obs.get("provenance") or {}
    if obs.get("status") == "failed":
        # A controlled refusal carries no provenance by contract.
        return _na(metric, "no_provenance_on_controlled_failure")
    if not provenance:
        return MetricOutcome(metric, Outcome.NO_EVIDENCE, "no_provenance_returned", "")

    problems: list[str] = []
    if provenance.get("score_is_probability") is not False:
        problems.append("score_is_probability is not False")
    if provenance.get("workflow_engine") != "langgraph":
        problems.append(f"workflow_engine={provenance.get('workflow_engine')!r}")

    mode = provenance.get("recognition_mode")
    if not isinstance(mode, str) or not mode:
        problems.append("recognition_mode missing")
    else:
        is_mock = mode == "mock_classification"
        version = provenance.get("mock_provider_version")
        if is_mock and not version:
            problems.append("mock mode reported no mock_provider_version")
        if not is_mock and version is not None:
            problems.append("a non-mock run filed a version under mock_provider_version")

    executed = provenance.get("taxonomy_executed")
    has_candidates = bool(obs.get("candidates"))
    if executed is True and not has_candidates:
        problems.append("taxonomy_executed=True with no candidate to validate")
    if executed is False and has_candidates:
        problems.append("taxonomy_executed=False despite candidates being present")

    # `reasoning_llm_used` means "a result was genuinely accepted", NOT "a call
    # was made" - see workflows/state.py:65-66. So calls>0 with used=False is
    # perfectly consistent: the call was spent and its result was rejected or
    # never arrived, and the deterministic path took over. An earlier version of
    # this check treated that as an inconsistency, which was wrong; the offline
    # dry run never exposed it because the fake provider always succeeds.
    #
    # The genuinely impossible states are the reverse ones.
    calls = provenance.get("reasoning_llm_calls")
    used = provenance.get("reasoning_llm_used")
    if isinstance(calls, int) and calls == 0 and used is True:
        problems.append("reasoning_llm_used is True but no call was made")
    for field in ("plan_source", "explanation_source"):
        if provenance.get(field) == "llm" and used is not True:
            problems.append(f"{field} is 'llm' but reasoning_llm_used is not True")

    return _verdict(metric, not problems, "provenance_consistent", "provenance_inconsistent",
                    "; ".join(problems))


# --- 11. agent/tool selection ----------------------------------------------

def evaluate_tool_selection(case: dict, obs: dict) -> MetricOutcome:
    """The routing and sourcing rules the agent must not break.

    Recognition may name a capability and stop; it may never call a peer, never
    let taxonomy reorder the classifier's ranking, and never quietly substitute
    classification for a capability it does not have.
    """
    metric = "agent_tool_selection"
    if not is_applicable(case, metric):
        return _na(metric)

    problems: list[str] = []
    target = obs.get("target_agent")
    if target is not None:
        if target not in VALID_CAPABILITIES:
            problems.append(f"target_agent {target!r} is not a known capability")
        if obs.get("status") != "needs_agent":
            problems.append("a target_agent was named without a needs_agent status")
        if obs.get("output") is not None:
            problems.append("needs_agent carried an output payload")

    candidates = obs.get("candidates") or []
    scores = [c.get("classification_score") for c in candidates]
    if any(s is None for s in scores):
        if candidates:
            problems.append("a candidate carried no classification score")
    else:
        if scores != sorted(scores, reverse=True):
            problems.append("candidates are not in non-increasing score order")
        if len(set(scores)) != len(scores):
            problems.append("candidate scores are not distinct")

    primary = obs.get("primary_species")
    if primary and candidates:
        if normalise_name(primary) != normalise_name(candidates[0].get("scientific_name")):
            problems.append("the primary species is not the top-ranked candidate")

    unsupported = obs.get("unsupported_capability")
    if unsupported and target is not None:
        problems.append("an unsupported capability was routed instead of declined")

    return _verdict(metric, not problems, "tool_selection_rules_held",
                    "tool_selection_rules_broken", "; ".join(problems))


# --- 12. agent LLM call budget ---------------------------------------------

def evaluate_agent_llm_budget(case: dict, obs: dict) -> MetricOutcome:
    """At most two agent LLM calls per request - when the run supplies a count.

    The count comes from the agent's own provenance, so a case that never
    reached the workflow yields no evidence rather than a free pass.
    """
    metric = "agent_llm_call_budget"
    calls = (obs.get("provenance") or {}).get("reasoning_llm_calls")
    if not isinstance(calls, int):
        return MetricOutcome(metric, Outcome.NO_EVIDENCE, "no_call_count_available",
                             "the run reported no reasoning_llm_calls")
    return _verdict(metric, calls <= 2, "within_two_call_ceiling", "two_call_ceiling_exceeded",
                    f"reasoning_llm_calls={calls} ceiling=2")


# --- 13. zero retry ---------------------------------------------------------

def evaluate_zero_retry(case: dict, obs: dict) -> MetricOutcome:
    """One logical classification per request, or no evidence.

    Offline runs instrument the injected classifier and can count invocations
    exactly. A live run has no such counter, which is reported as absent rather
    than assumed to be one.
    """
    metric = "zero_retry"
    invocations = obs.get("classifier_invocations")
    if not isinstance(invocations, int):
        return MetricOutcome(metric, Outcome.NO_EVIDENCE, "no_invocation_count_available",
                             "no instrumented provider was in place for this run")
    return _verdict(metric, invocations <= 1, "single_attempt", "retry_detected",
                    f"classifier invocations={invocations}")


# --- 14. controlled error code ---------------------------------------------

def evaluate_controlled_error(case: dict, obs: dict) -> MetricOutcome:
    metric = "controlled_error_code"
    expected = case.get("expected_error_code")
    observed = obs.get("error_code")

    if expected is None:
        if observed:
            if not is_applicable(case, metric):
                return _na(metric, "no_error_expected")
            return MetricOutcome(metric, Outcome.FAIL, "unexpected_error",
                                 f"observed={observed!r} when none was expected")
        return _na(metric, "no_error_expected")

    if observed is None:
        return MetricOutcome(metric, Outcome.FAIL, "error_missing",
                             f"expected={expected!r}, the agent did not fail")
    if observed not in VALID_ERROR_CODES:
        return MetricOutcome(metric, Outcome.FAIL, "error_not_controlled",
                             f"observed={observed!r} is not a known ErrorCode")
    return _verdict(metric, observed == expected, "error_code_match", "error_code_mismatch",
                    f"expected={expected!r} observed={observed!r}")


# --- 15. latency ------------------------------------------------------------

def evaluate_latency(case: dict, obs: dict) -> MetricOutcome:
    """Latency is captured, not judged. There is no target to pass or miss."""
    metric = "latency"
    latency = obs.get("latency_ms")
    if not isinstance(latency, (int, float)):
        return MetricOutcome(metric, Outcome.NO_EVIDENCE, "no_latency_captured", "")
    if latency < 0:
        return MetricOutcome(metric, Outcome.FAIL, "negative_latency", f"{latency}")
    return MetricOutcome(metric, Outcome.PASS, "latency_captured", f"{round(latency, 3)} ms")


# --- deferred metrics -------------------------------------------------------

def deferred_outcomes(case: dict) -> list[MetricOutcome]:
    """Metrics a case declares but Phase 2 structurally cannot decide."""
    outcomes = []
    for metric, reason in DEFERRED_METRICS.items():
        if not is_applicable(case, metric):
            outcomes.append(_na(metric))
            continue
        outcomes.append(MetricOutcome(
            metric, Outcome.NO_EVIDENCE, reason,
            "human rubric scores are recorded separately and never override a "
            "deterministic biological verdict" if metric == "relevance"
            else "filled by the Phase 3 repetition protocol",
        ))
    return outcomes


#: Order matters only for readability of the report.
EVALUATORS = (
    evaluate_output_schema,
    evaluate_status,
    evaluate_decision,
    evaluate_top1,
    evaluate_top5,
    evaluate_gbif,
    evaluate_ncbi,
    evaluate_delegation,
    evaluate_task_completion,
    evaluate_provenance,
    evaluate_tool_selection,
    evaluate_agent_llm_budget,
    evaluate_zero_retry,
    evaluate_controlled_error,
    evaluate_latency,
)


def evaluate_case(case: dict, obs: dict) -> list[MetricOutcome]:
    """Run every evaluator over one case. Always returns one outcome each."""
    return [evaluator(case, obs) for evaluator in EVALUATORS] + deferred_outcomes(case)
