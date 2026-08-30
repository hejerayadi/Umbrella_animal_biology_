"""
run_eval.py — Sprint 4, Part A Steps 4-7.

Exactly 4 functions as the guide specifies:

    load_cases(path)         reads test_cases.yaml into a list of dicts
    run_case(case)           calls the orchestrator, captures node_sequence,
                             tool_calls_log, and wall-clock time
    score_case(case, actual) delegates to evaluators.py, returns per-metric scores
    main()                   loops all cases, writes results/scorecard_<ts>.json

Usage (from repo root):
    python -m backend.agents.genome_agent.evaluation.run_eval
    python -m backend.agents.genome_agent.evaluation.run_eval --case happy_human_genome_size
    python -m backend.agents.genome_agent.evaluation.run_eval --category scaffold_escalation
    python -m backend.agents.genome_agent.evaluation.run_eval --fast
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import unittest.mock as mock
from pathlib import Path
from typing import Any

import yaml

from .evaluators import EvalResult, EvalScore, score_all

_HERE    = Path(__file__).parent
_CASES   = _HERE / "test_cases.yaml"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval")


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_cases
# ─────────────────────────────────────────────────────────────────────────────

def load_cases(path: Path = _CASES) -> list[dict]:
    """Read test_cases.yaml into a list of dicts."""
    with path.open(encoding="utf-8") as fh:
        cases = yaml.safe_load(fh)
    if not isinstance(cases, list):
        raise ValueError(f"test_cases.yaml must be a YAML list, got {type(cases)}")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# 2. run_case
# ─────────────────────────────────────────────────────────────────────────────

def _make_failure_patch(tool_name: str, error_cls: type) -> contextlib.AbstractContextManager:
    """Patch one NCBI helper to raise the given error class."""
    _PATHS = {
        "ncbi_taxonomy_search":
            "backend.agents.genome_agent.subagents.species_resolver._search_taxonomy_core",
        "ncbi_assembly_lookup":
            "backend.agents.genome_agent.subagents.species_resolver._search_assembly_by_taxid_core",
        "ncbi_assembly_stats":
            "backend.agents.genome_agent.subagents.genome_metadata.get_genome_metadata",
        "ncbi_gene_list":
            "backend.agents.genome_agent.subagents.gene_annotation.search_genes",
    }
    path = _PATHS.get(tool_name)
    if not path:
        return contextlib.nullcontext()

    async def _raise(*a, **kw):
        raise error_cls(f"Simulated {error_cls.__name__} for {tool_name}")

    return mock.patch(path, side_effect=_raise)


async def _run_case_async(case: dict) -> EvalResult:
    from backend.agents.genome_agent.orchestrator import GenomeAgentLangGraphOrchestrator

    species_hint = case.get("species_hint") or case.get("input", "")
    simulate     = case.get("simulate_failure") or {}
    fail_tool    = simulate.get("tool")
    fail_error   = {"TimeoutError": TimeoutError, "ConnectionError": ConnectionError}.get(
        simulate.get("error", ""), Exception
    )

    patch_ctx = _make_failure_patch(fail_tool, fail_error) if fail_tool else contextlib.nullcontext()

    # Tag this run in LangSmith if tracing is enabled
    ls_ctx = _langsmith_tag(case)

    t0 = time.perf_counter()
    try:
        with patch_ctx, ls_ctx:
            orch  = GenomeAgentLangGraphOrchestrator()
            state = await orch.run(
                user_question=case["input"],
                species_name=species_hint,
            )
    except Exception as exc:
        wall = time.perf_counter() - t0
        logger.warning("[run_case] %s raised %s: %s", case["id"], type(exc).__name__, exc)
        return EvalResult(
            status="error",
            node_sequence=[],
            tool_calls_log=[],
            output_fields={},
            escalation_triggered=False,
            error_end_reached=False,
            explanation=None,
            upstream_data={},
            wall_time_s=wall,
            errors=[str(exc)],
        )

    wall = time.perf_counter() - t0

    # Collect output fields
    output: dict[str, Any] = {}
    species = state.species or {}
    if state.assembly_id:
        output["assembly_id"] = state.assembly_id
    if species.get("scientific_name"):
        output["scientific_name"] = species["scientific_name"]
    if state.metadata and state.metadata.get("genome_size_bp"):
        output["genome_size"] = state.metadata["genome_size_bp"]
    if state.annotation and state.annotation.get("gene_list"):
        output["gene_list"] = state.annotation["gene_list"]
    if state.explanation:
        output["explanation"] = state.explanation

    need        = state.reconstruction_need or {}
    escalation  = need.get("status") == "NEEDS_AGENT"
    error_end   = "error_end" in state.node_sequence

    if error_end:
        status = "error_end"
    elif state.assembly_id:
        status = "completed"
    else:
        status = "failed"

    upstream: dict[str, Any] = {}
    if state.metadata:
        upstream["metadata"] = state.metadata
    if state.annotation:
        upstream["annotation"] = state.annotation

    return EvalResult(
        status=status,
        node_sequence=list(state.node_sequence),
        tool_calls_log=list(state.tool_calls_log),
        output_fields=output,
        escalation_triggered=escalation,
        error_end_reached=error_end,
        explanation=state.explanation,
        upstream_data=upstream,
        wall_time_s=wall,
        errors=list(state.errors),
    )


def run_case(case: dict) -> EvalResult:
    """
    Calls GenomeAgentLangGraphOrchestrator.run() and captures:
    - the node_sequence actually visited
    - the tool_calls_log actually made, with their arguments
    - wall-clock time for the run
    """
    return asyncio.run(_run_case_async(case))


# ─────────────────────────────────────────────────────────────────────────────
# 3. score_case
# ─────────────────────────────────────────────────────────────────────────────

def score_case(
    case: dict,
    actual: EvalResult,
    category_medians: dict | None = None,
) -> dict[str, EvalScore]:
    """
    Hands case (expected) and actual (what really happened) to the
    functions in evaluators.py and returns a dict of per-metric pass/fail.
    """
    return score_all(case, actual, category_medians)


# ─────────────────────────────────────────────────────────────────────────────
# 4. main
# ─────────────────────────────────────────────────────────────────────────────

def _category_medians(all_results: list[tuple[dict, EvalResult]]) -> dict[str, float]:
    from statistics import median
    by_cat: dict[str, list[int]] = {}
    for case, actual in all_results:
        cat = case.get("category", "unknown")
        by_cat.setdefault(cat, []).append(len(actual.tool_calls_log))
    return {cat: median(counts) for cat, counts in by_cat.items()}


def _build_scorecard(
    all_scores: list[tuple[dict, EvalResult, dict[str, EvalScore]]]
) -> dict:
    by_category: dict[str, dict] = {}
    total_pass = total_cases = 0

    for case, actual, scores in all_scores:
        cat = case.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"pass": 0, "total": 0, "failures": []}

        case_passed = all(s.passed for s in scores.values())
        by_category[cat]["total"] += 1
        total_cases += 1
        if case_passed:
            by_category[cat]["pass"] += 1
            total_pass += 1
        else:
            by_category[cat]["failures"].append({
                "id":             case["id"],
                "input":          case["input"],
                "failed_metrics": {
                    m: {"passed": s.passed, "score": s.score, "reason": s.reason}
                    for m, s in scores.items() if not s.passed
                },
                "node_sequence": actual.node_sequence,
                "tool_calls":    [e["tool"] for e in actual.tool_calls_log],
                "wall_time_s":   round(actual.wall_time_s, 2),
                "errors":        actual.errors,
            })

    category_rates = {
        cat: f"{d['pass']}/{d['total']}"
        for cat, d in by_category.items()
    }
    failure_ranking = sorted(
        [(cat, d["total"] - d["pass"]) for cat, d in by_category.items()],
        key=lambda x: -x[1],
    )
    return {
        "overall":         f"{total_pass}/{total_cases}",
        "pass_rate":       round(total_pass / max(total_cases, 1), 3),
        "by_category":     category_rates,
        "failure_ranking": failure_ranking,
        "details":         by_category,
    }


def main(filter_id: str | None = None, filter_cat: str | None = None) -> None:
    """
    Loop over all cases, collect scores, write a summary to results/.
    Results are grouped by category field — not just an overall pass rate.
    """
    cases = load_cases()
    if filter_id:
        cases = [c for c in cases if c["id"] == filter_id]
    if filter_cat:
        cases = [c for c in cases if c.get("category") == filter_cat]
    if not cases:
        print("No cases matched the filter.")
        sys.exit(1)

    print(f"\nRunning {len(cases)} case(s)...\n")

    all_results: list[tuple[dict, EvalResult]] = []
    for case in cases:
        print(f"  [{case['id']}] ", end="", flush=True)
        actual = run_case(case)
        all_results.append((case, actual))
        print(
            f"status={actual.status}  "
            f"nodes={actual.node_sequence}  "
            f"tools={[e['tool'] for e in actual.tool_calls_log]}  "
            f"t={actual.wall_time_s:.1f}s"
        )

    medians = _category_medians(all_results)

    all_scores: list[tuple[dict, EvalResult, dict[str, EvalScore]]] = []
    for case, actual in all_results:
        scores = score_case(case, actual, medians)
        all_scores.append((case, actual, scores))

    scorecard = _build_scorecard(all_scores)

    # Write JSON scorecard
    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_path = _RESULTS / f"scorecard_{ts}.json"
    out_path.write_text(json.dumps(scorecard, indent=2, default=str), encoding="utf-8")

    # Print summary — grouped by category
    print(f"\n{'='*60}")
    print(f"  Overall: {scorecard['overall']}  "
          f"({scorecard['pass_rate']*100:.0f}%)")
    print(f"\n  By category:")
    for cat, rate in scorecard["by_category"].items():
        print(f"    {cat:<30} {rate}")

    print(f"\n  Failure ranking (most to least failures):")
    for cat, count in scorecard["failure_ranking"]:
        if count:
            print(f"    {cat:<30} {count} failing case(s)")

    # For every failing case: which specific evaluator failed it
    for cat, data in scorecard["details"].items():
        for fail in data["failures"]:
            print(f"\n  FAIL [{fail['id']}]")
            for metric, info in fail["failed_metrics"].items():
                print(f"      {metric}: {info['reason']}")

    print(f"\n  Full scorecard -> {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LangSmith tag helper (Part B — zero-code-change tracing)
# ─────────────────────────────────────────────────────────────────────────────

def _langsmith_tag(case: dict) -> contextlib.AbstractContextManager:
    """Tag this run in LangSmith with case id and category when tracing is on."""
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() not in ("true", "1"):
        return contextlib.nullcontext()
    try:
        from langsmith import trace as ls_trace
        return ls_trace(
            name=case["id"],
            metadata={
                "case_id":  case["id"],
                "category": case.get("category", ""),
                "input":    case.get("input", ""),
            },
            tags=[case.get("category", "eval"), "sprint4"],
        )
    except Exception:
        return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genome Agent evaluator — Sprint 4"
    )
    parser.add_argument("--case",     help="Run a single case by id")
    parser.add_argument("--category", help="Run all cases in one category")
    parser.add_argument(
        "--fast", action="store_true",
        help="List cases only — no live NCBI calls"
    )
    args = parser.parse_args()

    if args.fast:
        cases = load_cases()
        if args.category:
            cases = [c for c in cases if c.get("category") == args.category]
        print(f"Loaded {len(cases)} case(s):")
        for c in cases:
            print(f"  {c['id']:<45} [{c.get('category', '')}]")
        sys.exit(0)

    main(filter_id=args.case, filter_cat=args.category)
