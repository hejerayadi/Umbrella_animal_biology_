"""Real Evaluation Runner for the Evolution Agent — Sprint 4.

Implements the full evaluation framework shown in the workshop slides,
mirroring the recognition agent's approach:

  Technique          Purpose
  ─────────────────  ─────────────────────────────────────────
  Dataset-based      Compare outputs with ground truth
  Rule-based         Calculate deterministic metrics
  Human evaluation   Assess communication quality (LLM-as-judge)

Metrics scored
──────────────
BIOLOGICAL
  • Top-1 correctness      closest pair matches published % identity ranking
  • Pair ordering          relative score ranking matches published identity order
  • Tree topology          published clade constraints satisfied in Newick
  • Bootstrap present      UFBoot values present when >=4 species used
  • Confidence decision    numeric confidence present when expected

AGENT BEHAVIOR
  • Task completion        right terminal status, right species resolved
  • Tool/agent selection   Planner routed to the expected feature
  • Provenance             source_agents field populated; NCBI accessions noted
  • Consistency            same feature + species across N repeated runs
  • Latency                wall-clock time within per-case budget
  • Controlled errors      invalid/out-of-catalogue inputs fail cleanly

COMMUNICATION QUALITY (LLM-as-judge, 1–5)
  • Relevance              does interpretation answer what was asked?
  • Explanation quality    is the explanation grounded and clear?

Usage (server must be running on port 8002):
    python backend\\agents\\evolution_agent\\evals\\real_eval.py

Writes real_eval_report.json and real_eval_report.md next to this file.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.agents.evolution_agent.evals.real_dataset import (  # noqa: E402
    CASES,
    REAL_TOPOLOGY_CONSTRAINTS,
    RealEvalCase,
)

API_URL = "http://127.0.0.1:8002/execute"

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    name: str
    passed: bool | None      # None = not applicable / not scored
    score: float | None      # 1–5 for LLM-judge metrics; else None
    detail: str


@dataclass
class CaseResult:
    case_id:   str
    category:  str
    prompt:    str
    latency_s: float
    status:    str            # completed | continue | failed | error

    # biological
    top1_correctness:    Metric
    pair_ordering:       Metric
    tree_topology:       Metric
    bootstrap_present:   Metric
    confidence_decision: Metric

    # agent behavior
    task_completion:  Metric
    tool_selection:   Metric
    provenance:       Metric
    latency_ok:       Metric
    controlled_error: Metric

    # consistency (populated only for consistency_repeats > 1)
    consistency: Metric | None

    # communication quality (LLM-as-judge)
    relevance:           Metric
    explanation_quality: Metric

    raw_response: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _call(prompt: str, timeout: float) -> tuple[dict[str, Any], float]:
    """Return (response_dict, latency_s). Raises on HTTP error."""
    t0 = time.perf_counter()
    resp = requests.post(
        API_URL,
        json={"instruction": prompt, "context": {}},
        timeout=timeout,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json(), latency


def _call_safe(prompt: str, timeout: float) -> tuple[dict[str, Any], float]:
    """Like _call but returns an error sentinel instead of raising."""
    try:
        return _call(prompt, timeout)
    except Exception as exc:
        return {"status": "error", "_error": str(exc)}, 0.0


# ---------------------------------------------------------------------------
# Feature inference (black-box, same as run_eval.py)
# ---------------------------------------------------------------------------

def _infer_feature(response: dict[str, Any]) -> str:
    status = response.get("status")
    if status == "continue":
        return "clarification_required"
    if status in ("failed", "error"):
        return status
    out = response.get("output") or {}
    has_tree   = bool(out.get("newick_tree"))
    has_scores = bool(out.get("similarity_scores"))
    if has_tree and has_scores:
        return "full_analysis"
    if has_tree:
        return "phylogenetic_tree"
    if has_scores:
        return "molecular_comparison"
    return "unknown"


# ---------------------------------------------------------------------------
# BIOLOGICAL METRICS
# ---------------------------------------------------------------------------

def _score_top1(case: RealEvalCase, out: dict) -> Metric:
    """Top-1 correctness: is the highest-scoring pair the expected one?"""
    if not case.expect_closest_pair:
        return Metric("top1_correctness", None, None, "not defined for this case")

    scores = out.get("similarity_scores") or []
    if not scores:
        return Metric("top1_correctness", False, None, "no similarity_scores in response")

    top = max(scores, key=lambda e: e.get("score", 0))
    actual_pair  = frozenset({top["species_a"].lower(), top["species_b"].lower()})
    expected_pair = frozenset({s.lower() for s in case.expect_closest_pair})
    passed = actual_pair == expected_pair
    return Metric(
        "top1_correctness", passed, None,
        f"expected={set(case.expect_closest_pair)} "
        f"actual={{{top['species_a']}, {top['species_b']}}} "
        f"score={top.get('score')}"
    )


def _score_pair_ordering(case: RealEvalCase, out: dict) -> Metric:
    """Pair ordering: relative ranking matches published cytochrome-b identity."""
    if not case.expect_pair_ordering:
        return Metric("pair_ordering", None, None, "not defined for this case")

    scores = out.get("similarity_scores") or []
    if not scores:
        return Metric("pair_ordering", False, None, "no similarity_scores in response")

    score_map: dict[frozenset, float] = {
        frozenset({e["species_a"].lower(), e["species_b"].lower()}): e.get("score", 0)
        for e in scores
    }

    violations: list[str] = []
    for (ha, hb, la, lb) in case.expect_pair_ordering:
        higher_key = frozenset({ha.lower(), hb.lower()})
        lower_key  = frozenset({la.lower(), lb.lower()})
        s_higher = score_map.get(higher_key)
        s_lower  = score_map.get(lower_key)
        if s_higher is None or s_lower is None:
            violations.append(f"missing pair: {ha}/{hb} or {la}/{lb}")
        elif s_higher <= s_lower:
            violations.append(
                f"score({ha},{hb})={s_higher:.4f} should be > "
                f"score({la},{lb})={s_lower:.4f}"
            )

    passed = len(violations) == 0
    detail = "all orderings correct" if passed else "; ".join(violations)
    return Metric("pair_ordering", passed, None, detail)


def _score_topology(case: RealEvalCase, out: dict) -> Metric:
    """Tree topology: published clade constraints satisfied in the Newick string."""
    if not case.expect_topology_clades:
        return Metric("tree_topology", None, None, "no topology constraints defined")

    newick = out.get("newick_tree") or ""
    if not newick:
        return Metric("tree_topology", False, None, "no newick_tree in response")

    # Build a map of clade_name → constraint dict
    constraint_map = {c["clade"]: c for c in REAL_TOPOLOGY_CONSTRAINTS}

    violations: list[str] = []
    for clade_name in case.expect_topology_clades:
        constraint = constraint_map.get(clade_name)
        if constraint is None:
            violations.append(f"unknown clade constraint: {clade_name!r}")
            continue

        ingroup  = constraint["ingroup"]
        outgroup = constraint.get("outgroup_examples", [])

        # Check: every ingroup species appears as a leaf
        for sp in ingroup:
            # match quoted or unquoted binomials
            pattern = re.escape(sp) + r"[:\),]"
            if not re.search(pattern, newick):
                violations.append(f"{clade_name}: {sp!r} not found as leaf in tree")

        # Check: in a valid clade the ingroup species must share a subtree
        # to the exclusion of outgroup species.
        # Heuristic: find the smallest parenthesised subtree that contains
        # all ingroup members — none of the outgroup members should be in it.
        if outgroup:
            subtree = _smallest_subtree_containing(newick, ingroup)
            if subtree:
                for og in outgroup:
                    if og in subtree:
                        violations.append(
                            f"{clade_name}: outgroup {og!r} found inside "
                            f"ingroup subtree"
                        )
            # If we can't find a subtree it might be a star tree (3 species,
            # no bootstrap) — don't penalise, just note.

    passed = len(violations) == 0
    detail = "all topology constraints satisfied" if passed else "; ".join(violations)
    return Metric("tree_topology", passed, None, detail)


def _smallest_subtree_containing(newick: str, species: list[str]) -> str | None:
    """
    Return the smallest parenthesised substring of `newick` that contains
    all `species` as substrings. Returns None if no such substring exists.
    """
    # Find all parenthesised substrings
    results: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(newick):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = newick[start:i + 1]
                if all(sp in candidate for sp in species):
                    results.append(candidate)
                start = -1

    if not results:
        return None
    # Return the shortest (most specific) matching subtree
    return min(results, key=len)


def _score_bootstrap(case: RealEvalCase, out: dict) -> Metric:
    """Bootstrap present when expected (>=4 species)."""
    if not case.expect_bootstrap_present:
        return Metric("bootstrap_present", None, None, "not expected for this case")

    bootstrap = out.get("bootstrap_support") or {}
    passed = len(bootstrap) > 0
    return Metric(
        "bootstrap_present", passed, None,
        f"bootstrap_support has {len(bootstrap)} entries"
        + (" (expected >0)" if not passed else "")
    )


def _score_confidence_decision(case: RealEvalCase, out: dict) -> Metric:
    """Confidence present/absent as expected."""
    conf = out.get("overall_confidence")
    if case.expect_confidence_present:
        passed = conf is not None
        return Metric(
            "confidence_decision", passed, None,
            f"overall_confidence={conf}" + (" (expected a value)" if not passed else "")
        )
    else:
        # None is acceptable; a value is also acceptable (not penalised)
        return Metric("confidence_decision", True, None, f"overall_confidence={conf} (not required)")


# ---------------------------------------------------------------------------
# AGENT BEHAVIOR METRICS
# ---------------------------------------------------------------------------

def _score_task_completion(case: RealEvalCase, response: dict) -> Metric:
    status = response.get("status")
    status_ok = status == case.expected_status
    parts = [f"status expected={case.expected_status!r} actual={status!r}"]

    species_ok = True
    if case.expected_species and status == "completed":
        out = response.get("output") or {}
        actual = set(out.get("species_list") or [])
        expected = set(case.expected_species)
        species_ok = actual == expected
        parts.append(f"species expected={sorted(expected)} actual={sorted(actual)}")

    passed = status_ok and species_ok
    return Metric("task_completion", passed, None, "; ".join(parts))


def _score_tool_selection(case: RealEvalCase, response: dict) -> Metric:
    status = response.get("status")
    if status in ("failed", "error"):
        # Tool selection not inferrable for failed requests — task_completion
        # already covers whether the failure was expected.
        return Metric(
            "tool_selection", None, None,
            "not applicable — request did not complete"
        )
    actual  = _infer_feature(response)
    passed  = actual == case.expected_feature
    return Metric(
        "tool_selection", passed, None,
        f"expected={case.expected_feature!r} actual={actual!r}"
    )


def _score_provenance(case: RealEvalCase, response: dict) -> Metric:
    """
    Provenance check:
      1. source_agents must be non-empty.
      2. 'Evolution Agent Orchestrator' must always be present.
      3. For completed cases, at least one worker agent must be listed.
      4. NCBI accessions are noted (Sprint 2 mock cannot populate them,
         so we record expected accessions without failing on their absence).
    """
    source = response.get("source_agents") or []
    parts: list[str] = []
    ok = True

    if not source:
        return Metric("provenance", False, None, "source_agents is empty")

    if "Evolution Agent Orchestrator" not in source:
        ok = False
        parts.append("'Evolution Agent Orchestrator' missing from source_agents")

    status = response.get("status")
    if status == "completed":
        worker_agents = [a for a in source if a != "Evolution Agent Orchestrator"]
        if not worker_agents:
            ok = False
            parts.append("no worker agent listed in source_agents for completed result")
        else:
            parts.append(f"workers={worker_agents}")

    if case.expected_ncbi_accessions:
        parts.append(
            f"expected_ncbi={case.expected_ncbi_accessions} "
            f"(not yet populated by mock — verified in Sprint 3+)"
        )

    detail = "; ".join(parts) if parts else f"source_agents={source}"
    return Metric("provenance", ok, None, detail)


def _score_latency(case: RealEvalCase, latency_s: float) -> Metric:
    passed = latency_s <= case.max_latency_s
    return Metric(
        "latency", passed, None,
        f"{latency_s:.2f}s (budget={case.max_latency_s}s)"
        + (" OVER BUDGET" if not passed else "")
    )


def _score_controlled_error(case: RealEvalCase, response: dict) -> Metric:
    """
    For invalid_input and delegation categories: the agent must not crash or
    hallucinate. It must return a clear failure message.
    """
    if case.category not in ("invalid_input", "delegation"):
        return Metric("controlled_error", None, None, "not applicable for this category")

    status = response.get("status")
    out = response.get("output") or {}

    # Must not be a server-side crash (status='error')
    if status == "error":
        return Metric(
            "controlled_error", False, None,
            f"server error: {response.get('_error', 'unknown')}"
        )

    # For expected failures: output must be a non-empty string message
    if case.expected_status == "failed":
        msg = out if isinstance(out, str) else str(out)
        has_message = len(msg.strip()) > 10
        return Metric(
            "controlled_error", has_message and status == "failed", None,
            f"status={status} message={msg[:120]!r}"
        )

    # For expected 'continue' (clarification): output must contain a question
    if case.expected_status == "continue":
        question = ""
        if isinstance(out, dict):
            question = out.get("clarification_question", "")
        has_question = len(question.strip()) > 5
        return Metric(
            "controlled_error", has_question and status == "continue", None,
            f"status={status} clarification_question={question!r}"
        )

    return Metric("controlled_error", True, None, f"status={status}")


# ---------------------------------------------------------------------------
# CONSISTENCY METRIC
# ---------------------------------------------------------------------------

def _score_consistency(
    case: RealEvalCase, responses: list[dict]
) -> Metric | None:
    if len(responses) < 2:
        return None

    features = [_infer_feature(r) for r in responses]
    species_sets = [
        frozenset((r.get("output") or {}).get("species_list") or [])
        for r in responses
        if r.get("status") == "completed"
    ]

    feature_stable = len(set(features)) == 1
    species_stable = len(set(species_sets)) <= 1

    # For consistency cases we also check closest-pair stability
    pair_stable = True
    if case.expect_closest_pair:
        pairs = []
        for r in responses:
            out = r.get("output") or {}
            scores = out.get("similarity_scores") or []
            if scores:
                top = max(scores, key=lambda e: e.get("score", 0))
                pairs.append(frozenset({top["species_a"].lower(), top["species_b"].lower()}))
        pair_stable = len(set(pairs)) <= 1

    passed = feature_stable and species_stable and pair_stable
    return Metric(
        "consistency", passed, None,
        f"features={features}; "
        f"species_stable={species_stable}; "
        f"pair_stable={pair_stable}"
    )


# ---------------------------------------------------------------------------
# LLM-AS-JUDGE METRICS
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating a bioinformatics agent's response. You will see the
user's original question and the agent's plain-English interpretation.

Score TWO criteria, 1-5 each:

relevance (1-5)
  5 = directly and completely answers what the user asked
  3 = partially answers but misses key points
  1 = does not address the question at all

explanation_quality (1-5)
  5 = explanation is grounded in the data, clear, no fabricated claims
  3 = mostly grounded but includes vague or unsupported statements
  1 = explanation fabricates numbers or claims not supported by the data

Reply with STRICT JSON only — no prose, no code fences:
{"relevance": <1-5>, "explanation_quality": <1-5>, "reasoning": "<one sentence>"}
"""


def _llm_judge(prompt: str, interpretation: str, llm) -> tuple[Metric, Metric]:
    """Returns (relevance_metric, explanation_quality_metric)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    msg = f"User question:\n{prompt}\n\nAgent interpretation:\n{interpretation}"
    try:
        resp = llm.invoke([
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(content=msg),
        ])
        text  = getattr(resp, "content", str(resp))
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            raise ValueError("no JSON in judge response")
        payload = json.loads(match.group(0))
        rel   = float(payload.get("relevance", 0))
        eq    = float(payload.get("explanation_quality", 0))
        reason = payload.get("reasoning", "")
        return (
            Metric("relevance",           None, rel, reason),
            Metric("explanation_quality", None, eq,  reason),
        )
    except Exception as exc:
        skip = Metric("judge_skipped", None, None, f"judge call failed: {exc}")
        return skip, skip


def _skip_judge(reason: str) -> tuple[Metric, Metric]:
    m = Metric("judge_skipped", None, None, reason)
    return m, m


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def _run_single(
    case: RealEvalCase, llm
) -> tuple[CaseResult, list[dict]]:
    """Run one case (possibly repeated for consistency). Returns (result, all_responses)."""

    n_runs = max(1, case.consistency_repeats)
    responses: list[dict] = []
    latencies: list[float] = []

    for i in range(n_runs):
        label = f"[{case.id}] run {i+1}/{n_runs}"
        print(f"  {label}: sending prompt…")
        resp, lat = _call_safe(case.prompt, timeout=case.max_latency_s + 30)
        latencies.append(lat)
        responses.append(resp)
        print(f"  {label}: status={resp.get('status')} latency={lat:.1f}s")

    primary  = responses[0]
    latency  = latencies[0]
    out      = (primary.get("output") or {}) if isinstance(primary.get("output"), dict) else {}
    status   = primary.get("status", "error")

    # --- biological metrics (only for completed responses) ---
    if status == "completed":
        top1   = _score_top1(case, out)
        pairs  = _score_pair_ordering(case, out)
        topo   = _score_topology(case, out)
        boot   = _score_bootstrap(case, out)
        conf   = _score_confidence_decision(case, out)
    else:
        na = Metric("n/a", None, None, f"not applicable (status={status})")
        top1 = pairs = topo = boot = conf = na

    # --- agent behavior ---
    task   = _score_task_completion(case, primary)
    tool   = _score_tool_selection(case, primary)
    prov   = _score_provenance(case, primary)
    lat_m  = _score_latency(case, latency)
    cerr   = _score_controlled_error(case, primary)

    # --- consistency ---
    cons = _score_consistency(case, responses) if n_runs > 1 else None

    # --- LLM-as-judge ---
    interpretation = out.get("interpretation") or out.get("explanation") or ""
    if llm is not None and status == "completed" and interpretation:
        rel_m, eq_m = _llm_judge(case.prompt, interpretation, llm)
    else:
        reason = (
            "no LLM configured" if llm is None
            else f"status={status}" if status != "completed"
            else "no interpretation in response"
        )
        rel_m, eq_m = _skip_judge(reason)

    result = CaseResult(
        case_id=case.id, category=case.category,
        prompt=case.prompt, latency_s=latency, status=status,
        top1_correctness=top1, pair_ordering=pairs,
        tree_topology=topo, bootstrap_present=boot,
        confidence_decision=conf,
        task_completion=task, tool_selection=tool,
        provenance=prov, latency_ok=lat_m,
        controlled_error=cerr, consistency=cons,
        relevance=rel_m, explanation_quality=eq_m,
        raw_response=primary, notes=case.notes,
    )
    return result, responses


def run() -> list[CaseResult]:
    try:
        from backend.agents.evolution_agent.framework.llm_client import (
            LLMUnavailable, get_llm,
        )
        llm = get_llm()
        print("[real_eval] LLM configured — judge scoring enabled")
    except Exception:
        llm = None
        print("[real_eval] no LLM configured — judge scoring skipped", file=sys.stderr)

    results: list[CaseResult] = []
    for i, case in enumerate(CASES):
        print(f"\n[{i+1}/{len(CASES)}] {case.id} ({case.category})")
        try:
            result, _ = _run_single(case, llm)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            # build a skeleton error result so the report is complete
            na = Metric("error", False, None, str(exc))
            result = CaseResult(
                case_id=case.id, category=case.category,
                prompt=case.prompt, latency_s=0.0, status="error",
                top1_correctness=na, pair_ordering=na, tree_topology=na,
                bootstrap_present=na, confidence_decision=na,
                task_completion=na, tool_selection=na,
                provenance=na, latency_ok=na, controlled_error=na,
                consistency=None, relevance=na, explanation_quality=na,
            )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def _pass_rate(results: list[CaseResult], attr: str) -> str:
    metrics = [getattr(r, attr) for r in results]
    scored  = [m for m in metrics if m is not None and m.passed is not None]
    if not scored:
        return "n/a"
    p = sum(1 for m in scored if m.passed)
    return f"{p}/{len(scored)}"


def _mean_score(results: list[CaseResult], attr: str) -> str:
    scores = [
        getattr(r, attr).score
        for r in results
        if getattr(r, attr) is not None and getattr(r, attr).score is not None
    ]
    if not scores:
        return "n/a"
    return f"{sum(scores)/len(scores):.2f}/5"


def _fmt(m: Metric | None) -> str:
    if m is None:
        return "n/a"
    if m.score is not None:
        return f"{m.score}/5 — {m.detail}"
    if m.passed is None:
        return f"— {m.detail}"
    return f"{'PASS' if m.passed else 'FAIL'} — {m.detail}"


def write_reports(results: list[CaseResult], out_dir: Path) -> None:
    json_path = out_dir / "real_eval_report.json"
    md_path   = out_dir / "real_eval_report.md"

    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    lines: list[str] = [
        "# Evolution Agent — Real Evaluation Report",
        "",
        "Benchmark: 36 cases, 12 species, real NCBI cytochrome-b ground truth.",
        "",
        "## Main Results",
        "",
        "### Biological Metrics",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Top-1 closest pair correctness | {_pass_rate(results, 'top1_correctness')} |",
        f"| Pair ordering (relative ranking) | {_pass_rate(results, 'pair_ordering')} |",
        f"| Tree topology (clade constraints) | {_pass_rate(results, 'tree_topology')} |",
        f"| Bootstrap support present | {_pass_rate(results, 'bootstrap_present')} |",
        f"| Confidence decision correct | {_pass_rate(results, 'confidence_decision')} |",
        "",
        "### Agent Behavior Metrics",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Task completion | {_pass_rate(results, 'task_completion')} |",
        f"| Tool / agent selection | {_pass_rate(results, 'tool_selection')} |",
        f"| Provenance | {_pass_rate(results, 'provenance')} |",
        f"| Latency within budget | {_pass_rate(results, 'latency_ok')} |",
        f"| Controlled error handling | {_pass_rate(results, 'controlled_error')} |",
    ]

    cons_scored = [r.consistency for r in results if r.consistency is not None]
    if cons_scored:
        cp = sum(1 for m in cons_scored if m.passed)
        lines.append(f"| Consistency (×3 repeats) | {cp}/{len(cons_scored)} |")

    lines += [
        "",
        "### Communication Quality (LLM-as-judge)",
        "",
        "| Criterion | Score |",
        "|---|---|",
        f"| Relevance | {_mean_score(results, 'relevance')} |",
        f"| Explanation quality | {_mean_score(results, 'explanation_quality')} |",
        "",
        "### Latency",
        "",
    ]

    lats = [r.latency_s for r in results if r.latency_s > 0]
    if lats:
        lats_s = sorted(lats)
        median = lats_s[len(lats_s) // 2]
        lines += [
            f"- Median latency: {median:.1f}s",
            f"- Min: {min(lats):.1f}s  Max: {max(lats):.1f}s",
            f"- Cases over budget: "
            f"{sum(1 for r in results if r.latency_ok.passed is False)}",
            "",
        ]

    # Failures
    lines += ["## Failures and Weaknesses", ""]
    failures = [
        (r, attr, getattr(r, attr))
        for r in results
        for attr in [
            "top1_correctness", "pair_ordering", "tree_topology",
            "bootstrap_present", "confidence_decision",
            "task_completion", "tool_selection", "provenance",
            "latency_ok", "controlled_error",
        ]
        if getattr(r, attr) is not None and getattr(r, attr).passed is False
    ]
    judge_failures = [
        r for r in results
        if r.relevance.score is not None and r.relevance.score < 3
    ]

    if not failures and not judge_failures:
        lines.append("No failures — all scored criteria passed.")
    else:
        for r, attr, m in failures:
            lines.append(f"- **{r.case_id}** ({attr}): {m.detail}")
        for r in judge_failures:
            lines.append(
                f"- **{r.case_id}** (relevance {r.relevance.score}/5): "
                f"{r.relevance.detail}"
            )
    lines.append("")

    # Per-case detail
    lines += ["## Per-Case Detail", ""]
    for r in results:
        lines.append(f"### `{r.case_id}` ({r.category}) — {r.status} in {r.latency_s:.1f}s")
        lines.append(f"- Prompt: {r.prompt!r}")
        if r.notes:
            lines.append(f"- Notes: {r.notes}")
        lines.append(f"- Task completion:       {_fmt(r.task_completion)}")
        lines.append(f"- Tool selection:        {_fmt(r.tool_selection)}")
        lines.append(f"- Top-1 correctness:     {_fmt(r.top1_correctness)}")
        lines.append(f"- Pair ordering:         {_fmt(r.pair_ordering)}")
        lines.append(f"- Tree topology:         {_fmt(r.tree_topology)}")
        lines.append(f"- Bootstrap present:     {_fmt(r.bootstrap_present)}")
        lines.append(f"- Confidence decision:   {_fmt(r.confidence_decision)}")
        lines.append(f"- Provenance:            {_fmt(r.provenance)}")
        lines.append(f"- Latency:               {_fmt(r.latency_ok)}")
        lines.append(f"- Controlled error:      {_fmt(r.controlled_error)}")
        if r.consistency is not None:
            lines.append(f"- Consistency:           {_fmt(r.consistency)}")
        lines.append(f"- Relevance (judge):     {_fmt(r.relevance)}")
        lines.append(f"- Explanation quality:   {_fmt(r.explanation_quality)}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[real_eval] wrote {json_path}")
    print(f"[real_eval] wrote {md_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[real_eval] starting — {len(CASES)} cases")
    results = run()

    passed_all = sum(
        1 for r in results
        if r.task_completion.passed is True
    )
    print(f"\n[real_eval] task completion: {passed_all}/{len(results)}")

    write_reports(results, Path(__file__).resolve().parent)
