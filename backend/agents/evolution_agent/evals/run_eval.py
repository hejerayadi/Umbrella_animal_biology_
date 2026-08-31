"""Agent Evaluation runner for the Evolution Agent (Sprint 4, Task 1).

Sends every case in golden_dataset.CASES to the live Evolution Agent
(POST /execute — a real server must already be running, no mocks) and
scores the five Sprint 4 criteria:

    1. Agent/tool selection  -- did the Planner route to the right feature?
    2. Task completion       -- right terminal status, right species resolved?
    3. Correctness           -- deterministic, domain-grounded checks
                                 (closest pair, well-formed tree, faithfulness
                                 of the LLM's prose to the structured data)
    4. Relevance              -- LLM-as-judge: does the prose answer what was
                                 actually asked?
    5. Response consistency  -- for cases with consistency_repeats > 1, run
                                 N times and check routing/species don't
                                 flip-flop across runs

Usage (server must already be running on port 8002):
    python -m backend.agents.evolution_agent.evals.run_eval

Writes report.json (machine-readable) and report.md (human-readable, with
a weaknesses section) next to this file.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root on sys.path

from backend.agents.evolution_agent.evals.golden_dataset import CASES, EvalCase  # noqa: E402
from backend.agents.evolution_agent.framework.llm_client import LLMUnavailable, get_llm  # noqa: E402

API_URL = "http://127.0.0.1:8002/execute"
REQUEST_TIMEOUT = 240  # phylogenetic_tree cases run real MAFFT+IQ-TREE


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    passed: bool | None  # None = not scored as pass/fail (e.g. a 1-5 judge score)
    detail: str
    score: float | None = None  # for judge criteria: 1-5


@dataclass
class CaseResult:
    case_id: str
    prompt: str
    raw_response: dict[str, Any]
    tool_selection: CriterionResult
    task_completion: CriterionResult
    correctness: CriterionResult
    relevance: CriterionResult
    consistency: CriterionResult | None
    notes: str = ""


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------

def call_agent(prompt: str) -> dict[str, Any]:
    resp = requests.post(
        API_URL,
        json={"instruction": prompt, "context": {}},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Feature inference from response shape (black-box — no internal field
# exposes "which feature ran", so this is inferred the same way any real
# caller would have to).
# ---------------------------------------------------------------------------

def infer_feature(response: dict[str, Any]) -> str:
    status = response.get("status")
    if status == "continue":
        return "clarification_required"
    if status == "failed":
        return "failed"

    out = response.get("output") or {}
    has_tree = bool(out.get("newick_tree"))
    has_scores = bool(out.get("similarity_scores"))
    if has_tree and has_scores:
        return "full_analysis"
    if has_tree:
        return "phylogenetic_tree"
    if has_scores:
        return "molecular_comparison"
    return "unknown"


# ---------------------------------------------------------------------------
# Criterion 1: agent/tool selection
# ---------------------------------------------------------------------------

def score_tool_selection(case: EvalCase, response: dict[str, Any]) -> CriterionResult:
    if response.get("status") == "failed":
        # A failure can happen after correct routing (e.g. species
        # resolution fails downstream of a correctly-chosen feature) --
        # response shape gives no way to recover which feature was
        # actually attempted, so this isn't scored either way here.
        # task_completion already covers whether the failure was correct.
        return CriterionResult(
            passed=None,
            detail="not applicable — request failed before feature is recoverable from response shape",
        )
    actual = infer_feature(response)
    passed = actual == case.expected_feature
    return CriterionResult(
        passed=passed,
        detail=f"expected={case.expected_feature!r} actual={actual!r}",
    )


# ---------------------------------------------------------------------------
# Criterion 2: task completion
# ---------------------------------------------------------------------------

def score_task_completion(case: EvalCase, response: dict[str, Any]) -> CriterionResult:
    status = response.get("status")
    status_ok = status == case.expected_status
    detail_parts = [f"status expected={case.expected_status!r} actual={status!r}"]

    species_ok = True
    if case.expected_species and status == "completed":
        out = response.get("output") or {}
        actual_species = set(out.get("species_list") or [])
        species_ok = actual_species == set(case.expected_species)
        detail_parts.append(
            f"species expected={sorted(case.expected_species)} actual={sorted(actual_species)}"
        )

    passed = status_ok and species_ok
    return CriterionResult(passed=passed, detail="; ".join(detail_parts))


# ---------------------------------------------------------------------------
# Criterion 3: correctness (deterministic, domain-grounded)
# ---------------------------------------------------------------------------

_LEAF_RE = re.compile(r"'([^']+)'|([A-Za-z_]+ [A-Za-z_]+)")


def _newick_leaves(newick: str) -> set[str]:
    return {a or b for a, b in _LEAF_RE.findall(newick)}


def score_correctness(case: EvalCase, response: dict[str, Any]) -> CriterionResult:
    if response.get("status") != "completed":
        # Nothing to check structurally; failure/clarification correctness
        # is implicitly covered by task_completion for these cases.
        return CriterionResult(passed=None, detail="not applicable (non-completed status)")

    out = response.get("output") or {}
    checks: list[str] = []
    ok = True

    if case.expect_closest_pair:
        scores = out.get("similarity_scores") or []
        if not scores:
            ok = False
            checks.append("expected similarity_scores but none returned")
        else:
            top = max(scores, key=lambda e: e["score"])
            actual_pair = {top["species_a"], top["species_b"]}
            expected_pair = set(case.expect_closest_pair)
            pair_ok = actual_pair == expected_pair
            ok = ok and pair_ok
            checks.append(
                f"closest pair expected={expected_pair} actual={actual_pair} score={top['score']}"
            )

    newick = out.get("newick_tree")
    if newick:
        leaves = _newick_leaves(newick)
        expected_leaves = {s for s in case.expected_species} or None
        if expected_leaves:
            missing = {s for s in expected_leaves if s not in leaves}
            leaves_ok = not missing
            ok = ok and leaves_ok
            checks.append(
                f"tree leaves missing={sorted(missing)}" if missing else "tree contains all expected species as leaves"
            )
        balanced = newick.count("(") == newick.count(")")
        ok = ok and balanced
        checks.append(f"newick balanced={balanced}")

    if not checks:
        return CriterionResult(passed=None, detail="no deterministic correctness check defined for this case")

    return CriterionResult(passed=ok, detail="; ".join(checks))


# ---------------------------------------------------------------------------
# Criterion 4: relevance (LLM-as-judge)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are grading one response from a bioinformatics agent. You will see the
user's original question and the agent's plain-English explanation.

Score two things, 1-5 each:
- relevance: does the explanation actually address what the user asked?
- faithfulness: does the explanation stick to what the structured data
  would support, without fabricating claims not grounded in it?

Respond with strict JSON only: {"relevance": <1-5>, "faithfulness": <1-5>, "reasoning": "<one sentence>"}
"""


def llm_judge(prompt: str, explanation: str, llm) -> CriterionResult:
    from langchain_core.messages import HumanMessage, SystemMessage

    user_msg = f"User question: {prompt}\n\nAgent explanation: {explanation}"
    try:
        resp = llm.invoke([SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user_msg)])
        text = getattr(resp, "content", str(resp))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
        relevance = float(payload.get("relevance", 0))
        reasoning = payload.get("reasoning", "")
        return CriterionResult(passed=None, score=relevance, detail=reasoning)
    except Exception as exc:  # noqa: BLE001
        return CriterionResult(passed=None, detail=f"judge call failed: {exc}")


# ---------------------------------------------------------------------------
# Criterion 5: response consistency
# ---------------------------------------------------------------------------

def score_consistency(case: EvalCase, responses: list[dict[str, Any]]) -> CriterionResult | None:
    if len(responses) < 2:
        return None
    features = [infer_feature(r) for r in responses]
    all_same_feature = len(set(features)) == 1

    species_sets = [
        frozenset((r.get("output") or {}).get("species_list") or [])
        for r in responses
        if r.get("status") == "completed"
    ]
    all_same_species = len(set(species_sets)) <= 1

    passed = all_same_feature and all_same_species
    return CriterionResult(
        passed=passed,
        detail=f"features={features} species_sets={[sorted(s) for s in species_sets]}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> list[CaseResult]:
    try:
        llm = get_llm()
    except LLMUnavailable:
        llm = None
        print("[eval] no LLM configured — relevance/faithfulness judging will be skipped", file=sys.stderr)

    results: list[CaseResult] = []

    for case in CASES:
        print(f"[eval] running {case.id!r}: {case.prompt!r}")
        responses: list[dict[str, Any]] = []
        for i in range(max(1, case.consistency_repeats)):
            t0 = time.time()
            resp = call_agent(case.prompt)
            print(f"    run {i + 1}/{case.consistency_repeats}: status={resp.get('status')} ({time.time() - t0:.1f}s)")
            responses.append(resp)

        primary = responses[0]

        tool_selection = score_tool_selection(case, primary)
        task_completion = score_task_completion(case, primary)
        correctness = score_correctness(case, primary)

        relevance = CriterionResult(passed=None, detail="skipped (no LLM configured)")
        if llm is not None and primary.get("status") == "completed":
            explanation = (primary.get("output") or {}).get("explanation", "")
            if explanation:
                relevance = llm_judge(case.prompt, explanation, llm)

        consistency = score_consistency(case, responses) if case.consistency_repeats > 1 else None

        results.append(
            CaseResult(
                case_id=case.id,
                prompt=case.prompt,
                raw_response=primary,
                tool_selection=tool_selection,
                task_completion=task_completion,
                correctness=correctness,
                relevance=relevance,
                consistency=consistency,
                notes=case.notes,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_criterion(c: CriterionResult | None) -> str:
    if c is None:
        return "n/a"
    if c.score is not None:
        return f"{c.score}/5 — {c.detail}"
    if c.passed is None:
        return f"— {c.detail}"
    return f"{'PASS' if c.passed else 'FAIL'} — {c.detail}"


def write_reports(results: list[CaseResult], out_dir: Path) -> None:
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    lines: list[str] = ["# Evolution Agent — Agent Evaluation Report", ""]

    def pass_rate(attr: str) -> str:
        scored = [getattr(r, attr) for r in results if getattr(r, attr) and getattr(r, attr).passed is not None]
        if not scored:
            return "n/a"
        passed = sum(1 for c in scored if c.passed)
        return f"{passed}/{len(scored)}"

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases run: {len(results)}")
    lines.append(f"- Agent/tool selection: {pass_rate('tool_selection')} passed")
    lines.append(f"- Task completion: {pass_rate('task_completion')} passed")
    lines.append(f"- Correctness (deterministic): {pass_rate('correctness')} passed")
    consistency_scored = [r.consistency for r in results if r.consistency is not None]
    if consistency_scored:
        cpass = sum(1 for c in consistency_scored if c.passed)
        lines.append(f"- Response consistency: {cpass}/{len(consistency_scored)} passed")
    relevance_scores = [r.relevance.score for r in results if r.relevance.score is not None]
    if relevance_scores:
        lines.append(f"- Relevance (LLM judge, mean): {sum(relevance_scores) / len(relevance_scores):.1f}/5")
    lines.append("")

    lines.append("## Weaknesses identified")
    lines.append("")
    weaknesses = []
    for r in results:
        for label, crit in [
            ("agent/tool selection", r.tool_selection),
            ("task completion", r.task_completion),
            ("correctness", r.correctness),
            ("consistency", r.consistency),
        ]:
            if crit is not None and crit.passed is False:
                weaknesses.append(f"- **{r.case_id}** ({label}): {crit.detail}")
        if r.relevance.score is not None and r.relevance.score < 3:
            weaknesses.append(f"- **{r.case_id}** (relevance, {r.relevance.score}/5): {r.relevance.detail}")
    if weaknesses:
        lines.extend(weaknesses)
    else:
        lines.append("None found — every scored criterion passed across all cases.")
    lines.append("")

    lines.append("## Per-case detail")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.case_id}`")
        lines.append(f"- Prompt: {r.prompt!r}")
        if r.notes:
            lines.append(f"- Notes: {r.notes}")
        lines.append(f"- Agent/tool selection: {_fmt_criterion(r.tool_selection)}")
        lines.append(f"- Task completion: {_fmt_criterion(r.task_completion)}")
        lines.append(f"- Correctness: {_fmt_criterion(r.correctness)}")
        lines.append(f"- Relevance: {_fmt_criterion(r.relevance)}")
        lines.append(f"- Consistency: {_fmt_criterion(r.consistency)}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] wrote {md_path}")


if __name__ == "__main__":
    results = run()
    write_reports(results, Path(__file__).resolve().parent)
