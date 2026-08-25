"""Result models for the Sprint 4 evaluation runner, and the redaction boundary.

Everything the runner records passes through this module, which is what makes
the privacy guarantee checkable in one place rather than scattered across the
runner and the report generator.

Two decisions are load-bearing.

**Four outcomes, not two.** A metric is `pass`, `fail`, `not_applicable` or
`no_evidence`. Collapsing the last two into either of the first two is how an
evaluation quietly lies: an ambiguous image scored `fail` on Top-1 would punish
the agent for refusing to guess, and a two-call check scored `pass` when no call
counter was available would claim a guarantee nobody measured.

**Provenance is allow-listed, never copied wholesale.** `_PROVENANCE_KEYS` names
exactly the fields that may be recorded. A future provenance key - remote or
otherwise - is dropped until somebody adds it here deliberately, so a provider
cannot widen the result file on its own.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Outcome(str, Enum):
    """What an evaluator concluded."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    #: The metric applies, but the run produced nothing to judge it by. Distinct
    #: from `not_applicable`, which means it never applied to this case.
    NO_EVIDENCE = "no_evidence"


#: Provenance fields that may be written into a result file. Chosen because each
#: is a mode, a count, a boolean or a fixed label - never user content.
_PROVENANCE_KEYS = (
    "model_target",
    "recognition_provider",
    "recognition_mode",
    "mock_provider_version",
    "model_version",
    "top_k_requested",
    "gbif_mode",
    "ncbi_mode",
    "taxonomy_executed",
    "taxonomy_degraded",
    "score_is_probability",
    "score_kind",
    "text_analysis_mode",
    "workflow_engine",
    "reasoning_llm_enabled",
    "reasoning_llm_provider",
    "plan_source",
    "plan_rejected",
    "explanation_source",
    "reasoning_llm_calls",
    "reasoning_llm_used",
    "remote_space_id",
    "remote_space_revision",
)

#: Candidate fields that may be recorded. Labels, scores and identifiers only.
_CANDIDATE_KEYS = (
    "species_id",
    "scientific_name",
    "common_name",
    "rank",
    "classification_score",
    "gbif_id",
    "ncbi_taxid",
    "taxonomy_status",
)

#: Substrings that must never appear in a serialized result. Checked by test.
FORBIDDEN_SUBSTRINGS = ("data:image", ";base64,", "-----BEGIN", "Authorization")


def safe_provenance(provenance: Any) -> dict[str, Any]:
    """The allow-listed view of `recognition_provenance`.

    The nested per-species `taxonomy_report` is deliberately reduced to its
    mode/availability shape: it is safe, but it is also long, and the metrics
    that read it only need which source answered.
    """
    if not isinstance(provenance, dict):
        return {}
    safe = {k: provenance[k] for k in _PROVENANCE_KEYS if k in provenance}
    report = provenance.get("taxonomy_report")
    if isinstance(report, dict):
        reduced: dict[str, Any] = {}
        for species_id, entry in report.items():
            if not isinstance(entry, dict):
                continue
            reduced[str(species_id)] = {
                source: {
                    "mode": value.get("mode"),
                    "available": value.get("available"),
                    "matched": value.get("matched"),
                    "inconsistent_record": value.get("inconsistent_record"),
                }
                for source, value in entry.items()
                if isinstance(value, dict)
            }
            if isinstance(entry.get("status"), str):
                reduced[str(species_id)]["status"] = entry["status"]
        safe["taxonomy_report"] = reduced
    return safe


def safe_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Ranked labels, reduced to the allow-listed fields."""
    if not isinstance(candidates, list):
        return []
    out = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            out.append({k: candidate.get(k) for k in _CANDIDATE_KEYS if k in candidate})
    return out


def sanitize_error(exc: BaseException) -> str:
    """The only thing an exception is allowed to contribute to a result.

    Just the class name. The agent's own adapters follow the same rule for the
    same reason: a remote error body can echo the request that caused it.
    """
    return type(exc).__name__


@dataclass(frozen=True)
class MetricOutcome:
    """One evaluator's verdict on one case."""

    metric: str
    outcome: Outcome
    #: A short, stable, machine-comparable reason. Never free prose alone.
    reason_code: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


@dataclass
class HumanReview:
    """Recorded human rubric scores. Empty until a reviewer fills it in.

    Deliberately separate from the deterministic metrics, and deliberately never
    consulted by them: a human score may add information about relevance and
    explanation quality, but it may not overturn whether the right species was
    named.
    """

    relevance: int | None = None
    explanation_quality: int | None = None
    task_completion: int | None = None
    reviewer: str | None = None
    notes: str = ""
    scored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencySlot:
    """Prepared in Phase 2, filled by the Phase 3 repetition protocol."""

    selected: bool = False
    repetitions_requested: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    stable: bool | None = None
    compared_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    """Exactly one of these exists for every manifest case, always."""

    case_id: str
    category: str
    mode: str
    executed: bool
    #: What the agent actually returned.
    status_observed: str | None = None
    decision_observed: str | None = None
    primary_species_observed: str | None = None
    species_id_observed: str | None = None
    gbif_id_observed: int | None = None
    ncbi_taxid_observed: int | None = None
    target_agent_observed: str | None = None
    error_code_observed: str | None = None
    unsupported_capability_observed: str | None = None
    candidates_observed: list[dict[str, Any]] = field(default_factory=list)
    output_keys_observed: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    #: LLM calls made by the agent, as reported by its own provenance.
    agent_llm_calls: int | None = None
    #: LLM calls made by evaluation itself. Structurally zero in Phase 2: the
    #: rubric is human and no judge model exists. Kept as a field so Phase 3
    #: cannot report the two together by accident.
    evaluator_llm_calls: int = 0
    #: Classifier invocations observed for this request. >1 would be a retry.
    classifier_invocations: int | None = None
    metrics: list[MetricOutcome] = field(default_factory=list)
    human_review: HumanReview = field(default_factory=HumanReview)
    consistency: ConsistencySlot = field(default_factory=ConsistencySlot)
    #: Sanitized class name if the runner itself could not complete the case.
    execution_error: str | None = None
    skipped_reason: str | None = None

    def metric(self, name: str) -> MetricOutcome | None:
        for outcome in self.metrics:
            if outcome.metric == name:
                return outcome
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = [m.to_dict() for m in self.metrics]
        payload["human_review"] = self.human_review.to_dict()
        payload["consistency"] = self.consistency.to_dict()
        return payload


@dataclass
class RunSummary:
    """One evaluation run: its results, its aggregates and its disclaimers."""

    mode: str
    benchmark_id: str
    started: str
    finished: str
    results: list[CaseResult] = field(default_factory=list)
    #: True whenever providers were injected rather than real. Carried into the
    #: report so a dry run can never be presented as a live measurement.
    dry_run: bool = True
    notes: str = ""

    # -- aggregation --------------------------------------------------------

    def metric_names(self) -> list[str]:
        seen: list[str] = []
        for result in self.results:
            for outcome in result.metrics:
                if outcome.metric not in seen:
                    seen.append(outcome.metric)
        return seen

    def metric_totals(self, results: list[CaseResult] | None = None) -> dict[str, dict[str, int]]:
        """pass / fail / not_applicable / no_evidence per metric.

        Every case contributes to exactly one bucket per metric it carries, so a
        failing case cannot vanish from the totals.
        """
        pool = self.results if results is None else results
        totals: dict[str, dict[str, int]] = {}
        for name in self.metric_names():
            totals[name] = {outcome.value: 0 for outcome in Outcome}
        for result in pool:
            for outcome in result.metrics:
                totals.setdefault(
                    outcome.metric, {o.value: 0 for o in Outcome}
                )[outcome.outcome.value] += 1
        return totals

    def metric_totals_by_category(self) -> dict[str, dict[str, dict[str, int]]]:
        categories = []
        for result in self.results:
            if result.category not in categories:
                categories.append(result.category)
        return {
            category: self.metric_totals(
                [r for r in self.results if r.category == category]
            )
            for category in categories
        }

    def latency_summary(self) -> dict[str, float | int | None]:
        samples = sorted(r.latency_ms for r in self.results if r.latency_ms is not None)
        if not samples:
            return {"count": 0, "min_ms": None, "median_ms": None, "max_ms": None,
                    "total_ms": None}
        middle = len(samples) // 2
        median = (
            samples[middle]
            if len(samples) % 2
            else (samples[middle - 1] + samples[middle]) / 2
        )
        return {
            "count": len(samples),
            "min_ms": round(samples[0], 3),
            "median_ms": round(median, 3),
            "max_ms": round(samples[-1], 3),
            "total_ms": round(sum(samples), 3),
        }

    def failed_cases(self) -> list[dict[str, Any]]:
        """Every case with at least one failing metric, or no result at all."""
        failures = []
        for result in self.results:
            failing = [m for m in result.metrics if m.outcome is Outcome.FAIL]
            if failing or result.execution_error:
                failures.append({
                    "case_id": result.case_id,
                    "category": result.category,
                    "execution_error": result.execution_error,
                    "failed_metrics": [m.to_dict() for m in failing],
                })
        return failures

    def llm_call_accounting(self) -> dict[str, Any]:
        """Agent calls and evaluator calls, counted separately and never summed.

        The agent's own two-call ceiling is a property of the agent. Evaluation
        calls, if any ever exist, are outside it. Reporting one number for both
        would make the ceiling unverifiable.
        """
        agent = [r.agent_llm_calls for r in self.results if r.agent_llm_calls is not None]
        return {
            "agent_calls_total": sum(agent),
            "agent_calls_max_observed": max(agent) if agent else None,
            "agent_calls_cases_with_evidence": len(agent),
            "agent_call_ceiling": 2,
            "evaluator_calls_total": sum(r.evaluator_llm_calls for r in self.results),
            "evaluator_judge_used": False,
            "note": (
                "Agent and evaluator calls are counted separately and never added "
                "together. Phase 2 uses a documented human rubric, so the evaluator "
                "total is structurally zero."
            ),
        }

    def provider_modes(self) -> dict[str, Any]:
        modes: dict[str, set[str]] = {}
        for result in self.results:
            for key in ("recognition_mode", "recognition_provider", "gbif_mode",
                        "ncbi_mode", "reasoning_llm_provider", "model_version",
                        "remote_space_revision", "mock_provider_version"):
                value = result.provenance.get(key)
                if isinstance(value, str) and value:
                    modes.setdefault(key, set()).add(value)
        return {key: sorted(values) for key, values in modes.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "scores_are_infrastructure_validation_only": self.dry_run,
            "started": self.started,
            "finished": self.finished,
            "notes": self.notes,
            "case_count": len(self.results),
            "metric_totals": self.metric_totals(),
            "metric_totals_by_category": self.metric_totals_by_category(),
            "latency": self.latency_summary(),
            "llm_calls": self.llm_call_accounting(),
            "provider_modes": self.provider_modes(),
            "failed_cases": self.failed_cases(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)
