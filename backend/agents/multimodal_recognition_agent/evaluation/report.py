"""Render a `RunSummary` as Markdown.

The generator is deliberately dumb: it formats what the run recorded and adds
nothing. The only editorialising it does is the banner at the top of a dry run,
which exists because a table of green numbers is exactly the artefact somebody
will screenshot without reading the surrounding prose.
"""
from __future__ import annotations

from . import consistency, rubric
from .results_schema import Outcome, RunSummary

DRY_RUN_BANNER = """\
> ## ⚠ Infrastructure validation only — not an accuracy measurement
>
> This run used **injected, scripted providers**. Every species label below came
> from a fixed table in `runner.py`, not from BioCLIP-2, and the scenario each
> case received was decided in advance by position, not by any model looking at
> any image. A third of the real-recognition cases were scripted to be *wrong on
> purpose*, to drive the evaluator branches that only fire on a mismatch.
>
> **Nothing here measures recognition quality.** It measures whether the runner
> executes every case, whether the fifteen evaluators reach the right verdicts,
> and whether results serialize safely. Live measurement is Phase 3.
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def _metric_rows(totals: dict[str, dict[str, int]]) -> list[list[str]]:
    rows = []
    for metric, counts in totals.items():
        judged = counts["pass"] + counts["fail"]
        rate = f"{counts['pass']}/{judged}" if judged else "—"
        rows.append([
            f"`{metric}`", counts["pass"], counts["fail"],
            counts["not_applicable"], counts["no_evidence"], rate,
        ])
    return rows


def render(summary: RunSummary) -> str:
    parts: list[str] = []
    parts.append(f"# Recognition Sprint 4 — evaluation results ({summary.mode} run)\n")
    parts.append(
        f"**Benchmark:** `{summary.benchmark_id}`  \n"
        f"**Mode:** `{summary.mode}`  \n"
        f"**Started:** {summary.started}  \n"
        f"**Finished:** {summary.finished}  \n"
        f"**Cases evaluated:** {len(summary.results)}\n"
    )
    if summary.dry_run:
        parts.append(DRY_RUN_BANNER)
    parts.append(f"\n{summary.notes}\n")

    # -- metrics overall ----------------------------------------------------
    parts.append("\n## Metric outcomes — overall\n")
    parts.append(
        "`N/A` means the case never claimed the metric. `no evidence` means it "
        "applied but the run produced nothing to judge it by. Neither is counted "
        "as a pass.\n\n"
    )
    parts.append(_table(
        ["Metric", "Pass", "Fail", "N/A", "No evidence", "Pass rate (judged)"],
        _metric_rows(summary.metric_totals()),
    ))

    # -- metrics by category ------------------------------------------------
    parts.append("\n## Metric outcomes — by category\n")
    for category, totals in summary.metric_totals_by_category().items():
        count = len([r for r in summary.results if r.category == category])
        parts.append(f"\n### `{category}` — {count} case(s)\n\n")
        parts.append(_table(
            ["Metric", "Pass", "Fail", "N/A", "No evidence", "Pass rate (judged)"],
            _metric_rows(totals),
        ))

    # -- per-case -----------------------------------------------------------
    parts.append("\n## Per-case results\n\n")
    rows = []
    for result in summary.results:
        failing = [m.metric for m in result.metrics if m.outcome is Outcome.FAIL]
        rows.append([
            f"`{result.case_id}`",
            result.status_observed or "—",
            result.decision_observed or "—",
            result.primary_species_observed or "—",
            result.target_agent_observed or "—",
            result.error_code_observed or "—",
            f"{result.latency_ms:.1f}" if result.latency_ms is not None else "—",
            ", ".join(f"`{m}`" for m in failing) if failing else "—",
        ])
    parts.append(_table(
        ["Case", "Status", "Decision", "Primary species", "Delegated to",
         "Error code", "Latency (ms)", "Failing metrics"],
        rows,
    ))

    # -- failures -----------------------------------------------------------
    failures = summary.failed_cases()
    parts.append(f"\n## Cases with at least one failing metric — {len(failures)}\n\n")
    if not failures:
        parts.append("_None._\n")
    else:
        rows = []
        for failure in failures:
            for metric in failure["failed_metrics"]:
                rows.append([
                    f"`{failure['case_id']}`", failure["category"],
                    f"`{metric['metric']}`", f"`{metric['reason_code']}`",
                    metric["detail"] or "—",
                ])
            if not failure["failed_metrics"]:
                rows.append([
                    f"`{failure['case_id']}`", failure["category"],
                    "`execution`", f"`{failure['execution_error']}`", "—",
                ])
        parts.append(_table(["Case", "Category", "Metric", "Reason", "Detail"], rows))

    # -- latency ------------------------------------------------------------
    latency = summary.latency_summary()
    parts.append("\n## Latency\n\n")
    parts.append(_table(
        ["Samples", "Min (ms)", "Median (ms)", "Max (ms)", "Total (ms)"],
        [[latency["count"], latency["min_ms"], latency["median_ms"],
          latency["max_ms"], latency["total_ms"]]],
    ))

    # -- providers ----------------------------------------------------------
    parts.append("\n## Provider modes and versions observed\n\n")
    modes = summary.provider_modes()
    parts.append(_table(
        ["Field", "Observed values"],
        [[f"`{k}`", ", ".join(f"`{x}`" for x in v)] for k, v in modes.items()],
    ) if modes else "_(none recorded)_\n")

    # -- LLM accounting -----------------------------------------------------
    calls = summary.llm_call_accounting()
    parts.append("\n## LLM call accounting\n\n")
    parts.append(
        "Agent calls and evaluator calls are counted separately and never summed: "
        "the agent's two-call ceiling is only verifiable if evaluation calls stay "
        "outside it.\n\n"
    )
    parts.append(_table(
        ["Counter", "Value"],
        [
            ["Agent LLM calls (total)", calls["agent_calls_total"]],
            ["Agent LLM calls (max observed in one request)", calls["agent_calls_max_observed"]],
            ["Agent call ceiling", calls["agent_call_ceiling"]],
            ["Cases with call evidence", calls["agent_calls_cases_with_evidence"]],
            ["**Evaluator** LLM calls (total)", calls["evaluator_calls_total"]],
            ["Evaluator judge model used", calls["evaluator_judge_used"]],
        ],
    ))

    # -- human rubric -------------------------------------------------------
    described = rubric.describe()
    scored = [r for r in summary.results if r.human_review.scored]
    parts.append("\n## Human rubric\n\n")
    parts.append(
        f"Method: **{described['method']}** (no LLM judge). Scale "
        f"{described['scale']['min']}–{described['scale']['max']}. "
        f"Scored so far: **{len(scored)} / {len(summary.results)}**. "
        "A human score never overrides a deterministic biological verdict.\n\n"
    )
    parts.append(_table(
        ["Criterion", "Question"],
        [[f"`{c['criterion']}`", c["question"]] for c in described["criteria"]],
    ))

    # -- consistency --------------------------------------------------------
    selected = [r.case_id for r in summary.results if r.consistency.selected]
    described_consistency = consistency.describe()
    parts.append("\n## Consistency protocol\n\n")
    parts.append(
        f"Selected subset ({len(selected)} cases, chosen deterministically before "
        f"any result was seen): {', '.join('`' + c + '`' for c in selected)}.\n\n"
        f"Each will be run **{described_consistency['repetitions_per_case']}×** in the "
        "same live configuration in Phase 3, comparing "
        f"{', '.join('`' + f + '`' for f in described_consistency['compared_fields'])}. "
        "Explanation wording is deliberately excluded — phrasing varies by design "
        "when a language model writes it.\n\n"
        f"**Executed in this run: {described_consistency['executed_in_phase_2']}.**\n"
    )

    return "".join(parts)
