"""
trace_review.py — Part B Step 5

Pulls recent traces from LangSmith and checks the three things the
Sprint 4 guide calls out:

  1. Orchestration decisions — does join_parallel's child match the input?
  2. Hallucination signals   — do all numbers in explanation appear upstream?
  3. Latency / duplicate calls — slowest node, any duplicate NCBI spans?

Usage (from repo root):
    python -m backend.agents.genome_agent.evaluation.trace_review

Requires:
    LANGCHAIN_API_KEY, LANGCHAIN_PROJECT set (see langsmith_setup.md)
    pip install langsmith

Output:
    Prints a review table to stdout.
    Writes evaluation/results/trace_review_<timestamp>.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("trace_review")

_RESULTS = Path(__file__).parent / "results"
_RESULTS.mkdir(exist_ok=True)

# Expected tail nodes after join_parallel for each assembly level.
# Scaffold/Contig → reconstruction_resolver, everything else → explanation_writer
_SCAFFOLD_LEVELS = {"scaffold", "contig"}


def _require_langsmith():
    """Import langsmith or print a clear error and exit."""
    try:
        from langsmith import Client
        return Client
    except ImportError:
        print(
            "langsmith is not installed.\n"
            "Run:  pip install langsmith\n"
            "Then set LANGCHAIN_API_KEY and LANGCHAIN_PROJECT in your .env"
        )
        sys.exit(1)


# ── 1. fetch traces ───────────────────────────────────────────────────────────

def fetch_traces(limit: int = 20) -> list[dict]:
    """Pull the most recent `limit` top-level runs from LangSmith."""
    Client = _require_langsmith()
    api_key  = os.getenv("LANGCHAIN_API_KEY")
    project  = os.getenv("LANGCHAIN_PROJECT", "genome-agent-sprint4")

    if not api_key:
        print("LANGCHAIN_API_KEY is not set — see langsmith_setup.md")
        sys.exit(1)

    client = Client(api_key=api_key)
    print(f"Fetching up to {limit} traces from project '{project}' …")

    # Fetch more than needed then filter to top-level only client-side
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        all_runs = list(client.list_runs(
            project_name=project,
            limit=limit * 8,
        ))

    # Top-level = no parent_run_id
    top_level = [r for r in all_runs if getattr(r, "parent_run_id", None) is None][:limit]
    print(f"  fetched {len(top_level)} top-level trace(s) "
          f"(from {len(all_runs)} total spans)\n")

    # Build full tree: group children by parent id
    children_by_parent: dict[str, list] = {}
    for r in all_runs:
        pid = str(getattr(r, "parent_run_id", None) or "")
        if pid:
            children_by_parent.setdefault(pid, []).append(r)

    def _to_dict_with_children(run) -> dict:
        d = _run_to_dict(run)
        kids = children_by_parent.get(str(run.id), [])
        d["child_runs"] = [_to_dict_with_children(c) for c in kids]
        return d

    return [_to_dict_with_children(r) for r in top_level]


def _run_to_dict(run: Any) -> dict:
    """Flatten a LangSmith Run object into a plain dict we can analyse."""
    return {
        "id":         str(run.id),
        "name":       run.name or "",
        "status":     run.status or "unknown",
        "start_time": run.start_time.isoformat() if run.start_time else None,
        "end_time":   run.end_time.isoformat()   if run.end_time   else None,
        "latency_s":  (
            (run.end_time - run.start_time).total_seconds()
            if run.start_time and run.end_time else None
        ),
        "inputs":     run.inputs  or {},
        "outputs":    run.outputs or {},
        "error":      run.error,
        "child_runs": [_run_to_dict(c) for c in (run.child_runs or [])],
    }


# ── 2. check orchestration decisions ─────────────────────────────────────────

def check_orchestration(trace: dict) -> dict:
    """
    Check 1: Does the tail after join_parallel match what we'd expect?

    Returns:
      {passed, expected_tail, actual_tail, note}
    """
    all_nodes = _collect_node_names(trace)

    # Determine expected tail
    has_scaffold = _has_scaffold_assembly(trace)
    expected_tail = "reconstruction_resolver" if has_scaffold else "explanation_writer"

    # Check what actually followed join_parallel
    actual_tail = _find_tail_after_join(trace)

    passed = (actual_tail == expected_tail) if actual_tail else None

    # Check for the parallel-serialisation bug:
    # get_genome_metadata and get_gene_annotation should overlap in time.
    parallel_serial = _check_parallel_serialized(trace)

    return {
        "passed":          passed,
        "expected_tail":   expected_tail,
        "actual_tail":     actual_tail or "not_found",
        "parallel_serial_bug": parallel_serial,
        "all_nodes":       all_nodes,
    }


def _collect_node_names(run: dict) -> list[str]:
    names = []
    if run.get("name"):
        names.append(run["name"])
    for child in run.get("child_runs", []):
        names.extend(_collect_node_names(child))
    return names


def _has_scaffold_assembly(trace: dict) -> bool:
    """Heuristic: look for 'Scaffold' or 'Contig' in outputs anywhere."""
    text = json.dumps(trace.get("outputs", {})).lower()
    return "scaffold" in text or "contig" in text


def _find_tail_after_join(trace: dict) -> str | None:
    """Find the first child run that appears after join_parallel in the tree."""
    children = trace.get("child_runs", [])
    join_idx = next(
        (i for i, c in enumerate(children) if "join" in (c.get("name") or "").lower()),
        None,
    )
    if join_idx is None or join_idx + 1 >= len(children):
        return None
    return children[join_idx + 1].get("name")


def _check_parallel_serialized(trace: dict) -> bool:
    """
    True if get_genome_metadata and get_gene_annotation ran sequentially
    instead of in parallel (i.e. one finished before the other started).
    This is the bug the guide specifically calls out.
    """
    children = trace.get("child_runs", [])
    meta_run = next(
        (c for c in children if "genome_metadata" in (c.get("name") or "")), None
    )
    annot_run = next(
        (c for c in children if "gene_annotation" in (c.get("name") or "")), None
    )
    if not meta_run or not annot_run:
        return False  # can't tell

    # Both need start/end times
    def _parse_time(t: str | None) -> float | None:
        if not t:
            return None
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return None

    m_start = _parse_time(meta_run.get("start_time"))
    m_end   = _parse_time(meta_run.get("end_time"))
    a_start = _parse_time(annot_run.get("start_time"))
    a_end   = _parse_time(annot_run.get("end_time"))

    if None in (m_start, m_end, a_start, a_end):
        return False

    # Serialized = one fully finishes before the other starts
    serialized = (m_end <= a_start) or (a_end <= m_start)
    return serialized


# ── 3. check hallucination ────────────────────────────────────────────────────

def check_hallucination(trace: dict) -> dict:
    """
    Check 2: Are all numbers in explanation_writer's output traceable to
    upstream tool outputs?

    Extracts numbers from explanation, checks against genome_metadata and
    gene_annotation outputs in the same trace.
    """
    explanation = _find_explanation(trace)
    if not explanation:
        return {"passed": None, "note": "no explanation found in this trace"}

    upstream = _collect_upstream_numbers(trace)
    numbers_in_exp = set(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", explanation))

    def _norm(n: str) -> str:
        return n.replace(",", "")

    ungrounded = [
        n for n in numbers_in_exp
        if _norm(n) not in {_norm(u) for u in upstream}
        and float(_norm(n)) > 10  # ignore small counts/indices
    ]

    return {
        "passed":        len(ungrounded) == 0,
        "ungrounded":    ungrounded,
        "upstream_nums": sorted(upstream)[:20],  # first 20 for display
        "note": (
            f"{len(ungrounded)} number(s) in explanation not found upstream"
            if ungrounded else "all numbers traceable to upstream data"
        ),
    }


def _find_explanation(run: dict) -> str | None:
    if "explanation_writer" in (run.get("name") or ""):
        outputs = run.get("outputs") or {}
        return outputs.get("explanation") or str(outputs)[:500]
    for child in run.get("child_runs", []):
        result = _find_explanation(child)
        if result:
            return result
    return None


def _collect_upstream_numbers(run: dict) -> set[str]:
    """Recursively collect all numbers from genome_metadata and gene_annotation outputs."""
    nums: set[str] = set()
    name = run.get("name") or ""
    if "metadata" in name or "annotation" in name or "ncbi" in name.lower():
        text = json.dumps(run.get("outputs") or {})
        nums.update(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text))
    for child in run.get("child_runs", []):
        nums.update(_collect_upstream_numbers(child))
    return nums


# ── 4. check latency / duplicate calls ───────────────────────────────────────

def check_latency_and_duplicates(trace: dict) -> dict:
    """
    Check 3: Find the slowest node, flag any duplicate NCBI calls.
    """
    all_spans = _flatten_runs(trace)

    # Slowest span
    timed = [(s["name"], s["latency_s"]) for s in all_spans if s.get("latency_s")]
    timed.sort(key=lambda x: -x[1])
    slowest = timed[:3] if timed else []

    # Duplicate NCBI calls: same name + same inputs, twice in same trace
    ncbi_calls: list[tuple[str, str]] = []
    for span in all_spans:
        if "ncbi" in (span.get("name") or "").lower():
            key = json.dumps(span.get("inputs") or {}, sort_keys=True)
            ncbi_calls.append((span["name"], key))

    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for name, key in ncbi_calls:
        if (name, key) in seen:
            duplicates.append(f"{name}({key[:80]})")
        seen.add((name, key))

    total_latency = trace.get("latency_s")

    return {
        "total_latency_s": total_latency,
        "slowest_spans":   slowest,
        "duplicate_ncbi_calls": duplicates,
        "passed": len(duplicates) == 0,
        "note": (
            f"{len(duplicates)} duplicate NCBI call(s) detected"
            if duplicates else "no duplicate NCBI calls"
        ),
    }


def _flatten_runs(run: dict) -> list[dict]:
    result = [run]
    for child in run.get("child_runs", []):
        result.extend(_flatten_runs(child))
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def review_traces(limit: int = 20) -> None:
    traces = fetch_traces(limit)

    if not traces:
        print("No traces found. Run some eval cases first with LANGCHAIN_TRACING_V2=true.")
        return

    rows = []
    for trace in traces:
        orch  = check_orchestration(trace)
        hall  = check_hallucination(trace)
        lat   = check_latency_and_duplicates(trace)

        row = {
            "id":            trace["id"][:8],
            "name":          trace.get("name", "")[:40],
            "status":        trace.get("status"),
            "latency_s":     round(trace.get("latency_s") or 0, 1),
            "orchestration": orch,
            "hallucination": hall,
            "latency_dupes": lat,
        }
        rows.append(row)

        # Print one-line summary
        orch_ok   = "✓" if orch["passed"] else ("?" if orch["passed"] is None else "✗")
        hall_ok   = "✓" if hall["passed"] else ("?" if hall["passed"] is None else "✗")
        lat_ok    = "✓" if lat["passed"]  else "✗"
        serial    = " ⚠ SERIAL_PARALLEL" if orch.get("parallel_serial_bug") else ""
        print(
            f"  {trace['id'][:8]}  orch={orch_ok}  hall={hall_ok}  eff={lat_ok}"
            f"  {row['latency_s']}s  nodes={orch['all_nodes'][:6]}{serial}"
        )
        if orch["passed"] is False:
            print(f"    orchestration: expected={orch['expected_tail']} actual={orch['actual_tail']}")
        if hall["passed"] is False:
            print(f"    hallucination: {hall['note']}  ungrounded={hall.get('ungrounded', [])}")
        if lat["passed"] is False:
            print(f"    duplicates: {lat['note']}")

    # Summary
    orch_pass  = sum(1 for r in rows if r["orchestration"]["passed"] is True)
    hall_pass  = sum(1 for r in rows if r["hallucination"]["passed"] is True)
    lat_pass   = sum(1 for r in rows if r["latency_dupes"]["passed"])
    serial_cnt = sum(1 for r in rows if r["orchestration"].get("parallel_serial_bug"))
    n = len(rows)

    print(f"\n{'='*60}")
    print(f"  Traces reviewed: {n}")
    print(f"  Orchestration correct:      {orch_pass}/{n}")
    print(f"  Hallucination-free:         {hall_pass}/{n}")
    print(f"  No duplicate NCBI calls:    {lat_pass}/{n}")
    if serial_cnt:
        print(f"  ⚠  PARALLEL-SERIALISED BUG: {serial_cnt}/{n} traces")
    print(f"{'='*60}\n")

    # Write JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = _RESULTS / f"trace_review_{ts}.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"  Full review → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LangSmith trace reviewer — Sprint 4 Part B")
    p.add_argument("--limit", type=int, default=20, help="Number of recent traces to fetch")
    args = p.parse_args()
    review_traces(limit=args.limit)
