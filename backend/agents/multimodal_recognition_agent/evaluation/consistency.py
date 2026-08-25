"""Consistency-protocol support: selection now, repetitions in Phase 3.

Phase 3 must run at least five representative valid cases three times each in the
same live configuration and compare what came back. Phase 2's job is to make
that reproducible: fix *which* cases, fix *what* is compared, and fix what an
unstable run means - before anyone has seen a result and can be tempted to pick
the five that happened to agree.

Explanation wording is deliberately excluded from the comparison. The explainer
is a language model when one is enabled; identical wording across three runs is
not required and demanding it would report normal phrasing variation as
instability.
"""
from __future__ import annotations

from typing import Any

REPETITIONS = 3
MINIMUM_CASES = 5

#: What must match across repetitions of the same case.
COMPARED_FIELDS = (
    "status",
    "decision",
    "primary_species",
    "candidate_order",
    "target_agent",
)

#: What must NOT be compared, and why.
EXCLUDED_FIELDS = {
    "explanation": "phrasing varies by design when a language model writes it",
    "latency_ms": "a timing measurement, not a decision",
    "clarification_question": "derived from the decision, which is already compared",
}


def select(cases: list[dict]) -> list[str]:
    """Choose the representative subset, deterministically and in advance.

    Representative means: valid, live-runnable cases that between them cover the
    outcomes the agent can reach - a clear identification, a challenging one, an
    honestly inconclusive one, and both halves of the delegation contract. One
    case per category-and-difficulty bucket, taken in committed manifest order so
    the choice cannot drift between runs.
    """
    buckets: dict[tuple[str, Any], str] = {}
    for case in cases:
        applicability = case.get("applicability") or {}
        if not applicability.get("live"):
            continue
        if case["category"] == "invalid_input_or_dependency_failure":
            continue  # a controlled refusal is deterministic by construction
        key = (case["category"], case.get("difficulty"))
        buckets.setdefault(key, case["case_id"])

    selected = list(buckets.values())

    # Top up in manifest order if the buckets alone did not reach the minimum.
    if len(selected) < MINIMUM_CASES:
        for case in cases:
            if len(selected) >= MINIMUM_CASES:
                break
            applicability = case.get("applicability") or {}
            if (applicability.get("live")
                    and case["category"] != "invalid_input_or_dependency_failure"
                    and case["case_id"] not in selected):
                selected.append(case["case_id"])

    order = {case["case_id"]: i for i, case in enumerate(cases)}
    return sorted(selected, key=lambda cid: order[cid])


def observation(result_dict: dict[str, Any]) -> dict[str, Any]:
    """Reduce one run of one case to just the fields that are compared."""
    candidates = result_dict.get("candidates_observed") or []
    return {
        "status": result_dict.get("status_observed"),
        "decision": result_dict.get("decision_observed"),
        "primary_species": result_dict.get("primary_species_observed"),
        "candidate_order": [c.get("scientific_name") for c in candidates],
        "target_agent": result_dict.get("target_agent_observed"),
    }


def compare(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Are these repetitions of one case mutually consistent?

    Fewer than two observations is not stability - it is no evidence, and it is
    reported as such rather than as a pass.
    """
    if len(observations) < 2:
        return {
            "stable": None,
            "reason": "insufficient_repetitions",
            "observed_repetitions": len(observations),
            "differing_fields": [],
        }

    differing = [
        field for field in COMPARED_FIELDS
        if len({_freeze(obs.get(field)) for obs in observations}) > 1
    ]
    return {
        "stable": not differing,
        "reason": "all_compared_fields_identical" if not differing else "fields_differed",
        "observed_repetitions": len(observations),
        "differing_fields": differing,
    }


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def describe() -> dict[str, Any]:
    return {
        "repetitions_per_case": REPETITIONS,
        "minimum_cases": MINIMUM_CASES,
        "compared_fields": list(COMPARED_FIELDS),
        "excluded_fields": EXCLUDED_FIELDS,
        "executed_in_phase_2": False,
        "note": "Selection is fixed in Phase 2; the repetitions themselves are a "
                "Phase 3 live activity and were not run here.",
    }
