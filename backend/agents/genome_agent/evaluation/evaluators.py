"""
evaluators.py — Sprint 4, Part A Step 5.

Exactly the 6 evaluator functions the guide specifies, plus the shared
EvalScore / EvalResult types used by run_eval.py.

Every function is pure (no I/O, no side effects) so each can be tested
independently of the runner.

Guide spec (verbatim):
    check_task_success    did actual.status == "COMPLETED" and do expected output
                          fields exist with plausible values?
    check_trajectory      does actual.node_sequence match case["expected_path"]?
    check_tool_selection  does the set of tool names called match
                          case["expected_tool_calls"]?
    check_tool_arguments  for each matched tool call, are the arguments sane /
                          non-empty / matching the input?
    check_escalation      does actual trigger reconstruction escalation exactly
                          when case["expected_escalation"] says it should?
    check_efficiency      count tool calls and LLM calls, flag any case that is
                          an outlier vs the median for its category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ── shared types ──────────────────────────────────────────────────────────────

@dataclass
class EvalScore:
    passed: bool
    score: float      # 0.0 – 1.0
    reason: str


@dataclass
class EvalResult:
    """Everything run_case() captures about one execution."""
    status: str                   # "completed" | "error_end" | "failed" | "error"
    node_sequence: list[str]      # nodes visited in order
    tool_calls_log: list[dict]    # [{"tool": name, "args": {...}}, ...]
    output_fields: dict[str, Any] # keys the agent produced (assembly_id, genome_size, …)
    escalation_triggered: bool    # reconstruction_need status == NEEDS_AGENT
    error_end_reached: bool       # "error_end" in node_sequence
    explanation: str | None       # explanation_writer output text
    upstream_data: dict[str, Any] # raw metadata + annotation for hallucination check
    wall_time_s: float
    errors: list[str]


# ── 1. task success ───────────────────────────────────────────────────────────

def check_task_success(case: dict, actual: EvalResult) -> EvalScore:
    """
    Did the agent complete and do the expected output fields exist with
    plausible (non-empty) values?

    For nonexistent-species cases (expected_error_end: true) we accept
    error_end as the correct terminal state instead of completed.
    """
    expects_error_end: bool = case.get("expected_error_end", False)
    expected_fields: list[str] = case.get("expected_output_fields", [])

    if expects_error_end:
        if actual.error_end_reached:
            return EvalScore(True, 1.0, "error_end reached as expected")
        return EvalScore(
            False, 0.0,
            f"expected error_end but got status={actual.status!r}, "
            f"nodes={actual.node_sequence}"
        )

    if actual.status not in ("completed", "error_end"):
        return EvalScore(False, 0.0, f"status={actual.status!r} (expected 'completed')")

    if actual.status == "error_end" and not expects_error_end:
        return EvalScore(False, 0.0, "hit error_end unexpectedly — species not resolved")

    missing = [f for f in expected_fields if not actual.output_fields.get(f)]
    if missing:
        score = round(1 - len(missing) / max(len(expected_fields), 1), 2)
        return EvalScore(False, score, f"missing output fields: {missing}")

    return EvalScore(True, 1.0, f"completed with all {len(expected_fields)} expected fields")


# ── 2. trajectory ─────────────────────────────────────────────────────────────

def check_trajectory(case: dict, actual: EvalResult) -> EvalScore:
    """
    Does actual.node_sequence match case["expected_path"]?

    Uses relative-order matching rather than exact equality to avoid
    brittleness from LangGraph's non-deterministic parallel branch order:
    - get_genome_metadata and get_gene_annotation may execute in either order
    - generate_visualization is optional (query router may add it)

    Every node in expected_path must appear in actual.node_sequence in the
    same relative order. Parallel pair treated as unordered set.
    """
    expected: list[str] = case.get("expected_path", [])
    actual_seq: list[str] = actual.node_sequence

    if not expected:
        return EvalScore(True, 1.0, "no expected_path defined")

    # Nodes that run in parallel — either order is correct
    _PARALLEL = {"get_genome_metadata", "get_gene_annotation"}
    # Nodes the query router may add on any question — not a failure if present
    _OPTIONAL = {"generate_visualization"}

    # Build a collapsed required list: treat the parallel pair as one checkpoint
    required: list[str] = []
    parallel_placeholder_added = False
    for node in expected:
        if node in _OPTIONAL:
            continue
        if node in _PARALLEL:
            if not parallel_placeholder_added:
                required.append("__parallel__")
                parallel_placeholder_added = True
        else:
            required.append(node)

    missing: list[str] = []
    pos = 0
    for req in required:
        if req == "__parallel__":
            expected_parallel = [n for n in expected if n in _PARALLEL]
            if any(n not in actual_seq for n in expected_parallel):
                missing.extend(
                    n for n in expected_parallel if n not in actual_seq
                )
        else:
            found = False
            while pos < len(actual_seq):
                if actual_seq[pos] == req:
                    pos += 1
                    found = True
                    break
                pos += 1
            if not found:
                missing.append(req)

    if missing:
        extra = sorted(set(actual_seq) - set(expected) - _OPTIONAL)
        return EvalScore(
            False,
            round(1 - len(missing) / max(len(required), 1), 2),
            f"missing nodes: {missing}; extra non-optional: {extra}; "
            f"actual: {actual_seq}"
        )

    return EvalScore(True, 1.0, f"trajectory matches expected_path")


# ── 3. tool selection ─────────────────────────────────────────────────────────

def check_tool_selection(case: dict, actual: EvalResult) -> EvalScore:
    """
    Does the set of tool names called match case["expected_tool_calls"]?
    """
    expected: list[str] = case.get("expected_tool_calls", [])
    if not expected:
        return EvalScore(True, 1.0, "no expected_tool_calls defined")

    actual_tools = {e["tool"] for e in actual.tool_calls_log}
    missing = [t for t in expected if t not in actual_tools]

    if missing:
        return EvalScore(
            False,
            round(1 - len(missing) / len(expected), 2),
            f"tools not called: {missing}; called: {sorted(actual_tools)}"
        )
    return EvalScore(True, 1.0, f"all {len(expected)} expected tools called")


# ── 4. tool arguments ─────────────────────────────────────────────────────────

def check_tool_arguments(case: dict, actual: EvalResult) -> EvalScore:
    """
    For each matched tool call, are the arguments sane / non-empty /
    matching the input?

    Checks (not exact string match — checks the right value is present):
    - ncbi_taxonomy_search : query is non-empty, not a placeholder string
    - ncbi_assembly_lookup / ncbi_assembly_stats / ncbi_gene_list :
      assembly_id matches GCF_ or GCA_ pattern
    """
    _ASSEMBLY_RE = re.compile(r"^GC[FA]_\d+\.\d+$")
    issues: list[str] = []

    for entry in actual.tool_calls_log:
        tool = entry.get("tool", "")
        args = entry.get("args", {})

        if tool == "ncbi_taxonomy_search":
            q = str(args.get("query", "")).strip()
            if not q or q.lower() in {"none", "unknown", "null", ""}:
                issues.append(
                    f"ncbi_taxonomy_search called with empty/invalid query: {q!r}"
                )

        elif tool in ("ncbi_assembly_lookup", "ncbi_assembly_stats", "ncbi_gene_list"):
            aid = str(args.get("assembly_id", "")).strip()
            if not aid:
                issues.append(f"{tool} called with empty assembly_id")
            elif not _ASSEMBLY_RE.match(aid):
                issues.append(
                    f"{tool} called with non-standard assembly_id: {aid!r}"
                )

    if issues:
        return EvalScore(False, 0.0, "; ".join(issues))
    return EvalScore(True, 1.0, "all tool arguments are sane")


# ── 5. escalation ─────────────────────────────────────────────────────────────

def check_escalation(case: dict, actual: EvalResult) -> EvalScore:
    """
    Does actual trigger reconstruction escalation exactly when
    case["expected_escalation"] says it should?

    True  → reconstruction_resolver must be in node_sequence AND
             escalation_triggered must be True
    False → reconstruction_resolver must NOT be in node_sequence
    """
    expected: bool = case.get("expected_escalation", False)
    node_hit = "reconstruction_resolver" in actual.node_sequence
    flag_set = actual.escalation_triggered

    if expected:
        if flag_set and node_hit:
            return EvalScore(True, 1.0, "reconstruction escalation fired correctly")
        problems = []
        if not flag_set:
            problems.append("escalation_triggered=False")
        if not node_hit:
            problems.append("reconstruction_resolver not visited")
        return EvalScore(False, 0.0, f"expected escalation but: {'; '.join(problems)}")
    else:
        if not flag_set and not node_hit:
            return EvalScore(True, 1.0, "no escalation — correct for complete assembly")
        false_positives = []
        if flag_set:
            false_positives.append("escalation_triggered=True (false positive)")
        if node_hit:
            false_positives.append("reconstruction_resolver visited (false positive)")
        return EvalScore(False, 0.0, "; ".join(false_positives))


# ── 6. efficiency ─────────────────────────────────────────────────────────────

def check_efficiency(
    case: dict,
    actual: EvalResult,
    category_medians: dict[str, float] | None = None,
) -> EvalScore:
    """
    Count tool calls, flag any case that is an outlier vs the median for
    its category (relative check — not a hard threshold until we have enough
    runs to know what "normal" looks like).

    Also flags duplicate tool calls (same tool + same args twice in one run).

    On first run there are no medians yet — returns informational pass so the
    scorecard still shows raw counts.
    """
    tool_count = len(actual.tool_calls_log)
    issues: list[str] = []

    # Duplicate calls — same tool + same assembly_id called twice
    seen: set[str] = set()
    for entry in actual.tool_calls_log:
        key = f"{entry['tool']}::{entry.get('args', {})}"
        if key in seen:
            issues.append(f"duplicate call: {entry['tool']}({entry.get('args', {})})")
        seen.add(key)

    # Outlier check vs category median
    category = case.get("category", "")
    if category_medians and category in category_medians:
        median = category_medians[category]
        if median > 0 and tool_count > median * 2:
            issues.append(
                f"tool_count={tool_count} is >2x median "
                f"({median:.0f}) for category '{category}'"
            )

    if issues:
        return EvalScore(
            False, 0.0,
            f"{'; '.join(issues)} "
            f"(wall_time={actual.wall_time_s:.1f}s, tool_calls={tool_count})"
        )

    return EvalScore(
        True, 1.0,
        f"tool_calls={tool_count}, wall_time={actual.wall_time_s:.1f}s"
    )


# ── aggregate ─────────────────────────────────────────────────────────────────

def score_all(
    case: dict,
    actual: EvalResult,
    category_medians: dict | None = None,
) -> dict[str, EvalScore]:
    """Run all 6 evaluators and return {metric: EvalScore}."""
    return {
        "task_success":   check_task_success(case, actual),
        "trajectory":     check_trajectory(case, actual),
        "tool_selection": check_tool_selection(case, actual),
        "tool_arguments": check_tool_arguments(case, actual),
        "escalation":     check_escalation(case, actual),
        "efficiency":     check_efficiency(case, actual, category_medians),
    }
