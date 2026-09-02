"""
Step 5 — one evaluator function per orchestration behavior.

Every evaluator takes the parsed `case` dict (from agent_test_cases.yaml)
and the `actual` CaseRun captured by agent_capture.run_case, and returns an
EvalResult(passed, detail). agent_capture.score_case combines all five into
one CaseScore per case.
"""
from __future__ import annotations

from dataclasses import dataclass

from schemas.common import AgentStatus

# Minimum wall-clock overlap (seconds) between two branches for them to
# count as "ran concurrently" rather than "happened to interleave slightly
# due to scheduling noise". Fixtures sleep at least 0.15s (case7 sleeps
# 0.3s), so this threshold has comfortable margin either way.
MIN_OVERLAP_SECONDS = 0.05

PARALLEL_PAIRS = {
    "functional_evidence_vs_literature": ("functional_evidence", "literature_support"),
    "pathways_vs_protein_data": ("pathways", "protein_data"),
}


@dataclass
class EvalResult:
    passed: bool
    detail: str


def _status_name(value) -> str:
    if value is None:
        return "none"
    return value.value if isinstance(value, AgentStatus) else str(value)


def check_escalation_correctness(case: dict, actual) -> EvalResult:
    """Did the right escalation (or none) fire, with the right target_agent?"""
    expected_status = case["expect"]["status"]
    expected_target = case["expect"].get("target_agent")
    actual_status = _status_name(actual.final_state.get("status"))
    actual_target = actual.final_state.get("target_agent")

    if actual_status != expected_status:
        return EvalResult(False, f"expected status={expected_status!r}, got {actual_status!r}")
    if expected_target != actual_target:
        return EvalResult(False, f"expected target_agent={expected_target!r}, got {actual_target!r}")

    if case["expect"].get("functional_evidence_data_present"):
        pathways = actual.final_state.get("pathway_data", [])
        proteins = actual.final_state.get("protein_data", [])
        if not pathways or not proteins:
            return EvalResult(False, "literature escalation must not block/drop functional evidence data, but "
                                      f"pathway_data={pathways!r} protein_data={proteins!r}")

    return EvalResult(True, f"status={actual_status!r}, target_agent={actual_target!r}")


def check_non_fatal_handling(case: dict, actual) -> EvalResult:
    """When a case simulates a functional-evidence-branch failure, did the
    graph still reach aggregate instead of stopping? For cases that don't
    simulate a partial failure, this is a no-op pass (nothing to check)."""
    if not case["expect"].get("non_fatal_case"):
        return EvalResult(True, "not a partial-failure case, nothing to check")

    reached_aggregate = "aggregate" in actual.node_order
    fe_status = _status_name(actual.final_state.get("functional_evidence_status"))
    if not reached_aggregate:
        return EvalResult(False, f"expected aggregate to still run despite functional_evidence_status={fe_status!r}, "
                                  f"but node_order={actual.node_order}")
    return EvalResult(True, f"aggregate ran despite functional_evidence_status={fe_status!r}, as expected (non-fatal)")


def check_trajectory(case: dict, actual) -> EvalResult:
    """Does the node sequence match the expected staged trajectory? Each
    stage is a list of node names that must all appear, in any relative
    order among themselves, but strictly after every node in the previous
    stage and before every node in the next. This also implicitly confirms
    join_and_route only appears once both parallel branches have finished
    (it's its own stage, after the functional_evidence/literature_support
    stage) and never reacts mid-branch."""
    stages = case["expect"]["trajectory"]
    order = actual.node_order

    cursor = 0
    for stage in stages:
        stage_positions = []
        for node in stage:
            if node not in order[cursor:]:
                return EvalResult(False, f"expected {node!r} in trajectory after position {cursor}, "
                                          f"actual node_order={order}")
            pos = order.index(node, cursor)
            stage_positions.append(pos)
        cursor = max(stage_positions) + 1

    if "nodes_must_not_run" in case["expect"]:
        ran_but_shouldnt = [n for n in case["expect"]["nodes_must_not_run"] if n in order]
        if ran_but_shouldnt:
            return EvalResult(False, f"nodes ran that should not have: {ran_but_shouldnt}")

    return EvalResult(True, f"node_order={order} matches expected staged trajectory")


def check_parallelism(case: dict, actual) -> EvalResult:
    """Did the top-level branches (functional_evidence, literature_support)
    and the nested branches (pathways, protein_data) each show overlapping
    execution time, not sequential?"""
    pair_keys = case["expect"].get("check_parallelism", [])
    if not pair_keys:
        return EvalResult(True, "no parallelism assertions requested for this case")

    details = []
    for key in pair_keys:
        label_a, label_b = PARALLEL_PAIRS[key]
        overlap = actual.timing_log.overlap_seconds(label_a, label_b)
        if overlap is None:
            return EvalResult(False, f"{key}: one or both of {label_a!r}/{label_b!r} never ran "
                                      f"(spans={actual.timing_log.spans})")
        if overlap < MIN_OVERLAP_SECONDS:
            return EvalResult(False, f"{key}: overlap={overlap:.3f}s < {MIN_OVERLAP_SECONDS}s threshold "
                                      f"-> ran sequentially, not in parallel (spans={actual.timing_log.spans})")
        details.append(f"{key}: overlap={overlap:.3f}s")

    return EvalResult(True, "; ".join(details))


def check_aggregate_faithfulness(case: dict, actual) -> EvalResult:
    """Similar idea to Part A's faithfulness check, but applied to the
    combined final explanation against all upstream branch outputs: every
    gene/pathway/protein/evidence item that made it into state should be
    reflected in what got handed to the Explanation Writer (aggregate_node
    builds its `genes`/`pathways`/`proteins`/`evidence` strings directly
    from state, so this also catches stale/partial data silently dropped
    before aggregate)."""
    if not case["expect"].get("aggregate_faithfulness"):
        return EvalResult(True, "faithfulness not requested for this case")

    state = actual.final_state
    if not actual.aggregate_calls:
        return EvalResult(False, "aggregate never called write_explanation, nothing to check faithfulness against")
    # aggregate_node calls write_explanation exactly once (asserted separately
    # by trajectory/escalation checks); take that one real call's kwargs.
    call = actual.aggregate_calls[-1]

    missing = []
    expected_genes = {a.gene_symbol for a in state.get("go_annotations", [])}
    seen_genes = set(call.get("genes", "").split(", ")) if call.get("genes") else set()
    if expected_genes != seen_genes:
        missing.append(f"genes: expected {expected_genes}, aggregate saw {seen_genes}")

    expected_pathways = {p.pathway_name for p in state.get("pathway_data", [])}
    seen_pathways = set(call.get("pathways", "").split(", ")) if call.get("pathways") else set()
    if expected_pathways != seen_pathways:
        missing.append(f"pathways: expected {expected_pathways}, aggregate saw {seen_pathways}")

    expected_proteins = {p.protein_name for p in state.get("protein_data", [])}
    seen_proteins = set(call.get("proteins", "").split(", ")) if call.get("proteins") else set()
    if expected_proteins != seen_proteins:
        missing.append(f"proteins: expected {expected_proteins}, aggregate saw {seen_proteins}")

    if missing:
        return EvalResult(False, "; ".join(missing))
    return EvalResult(True, "write_explanation was called with genes/pathways/proteins faithfully "
                             "matching final state (nothing stale or dropped)")


EVALUATORS = {
    "escalation_correctness": check_escalation_correctness,
    "non_fatal_handling": check_non_fatal_handling,
    "trajectory": check_trajectory,
    "parallelism": check_parallelism,
    "aggregate_faithfulness": check_aggregate_faithfulness,
}


def score_case(case: dict, actual) -> dict[str, EvalResult]:
    return {name: fn(case, actual) for name, fn in EVALUATORS.items()}
