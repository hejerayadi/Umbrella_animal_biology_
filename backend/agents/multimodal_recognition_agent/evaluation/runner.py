"""The bounded Sprint 4 evaluation runner.

Two modes, and the difference between them is the whole point of this module.

**Offline dry run (the default).** Every provider is injected: a scripted
classifier, the agent's own fixture-backed taxonomy provider, and the offline
deterministic GPT-5 mini stand-in. No credential is read, no socket is opened,
no second passes. It exists to prove that the runner and the fifteen evaluators
work end to end over all 36 cases - and **its scores say nothing whatsoever
about BioCLIP-2's accuracy**, because the labels come from a table in this file
rather than from a model. That is stated in the results payload itself
(`dry_run: true`, `scores_are_infrastructure_validation_only: true`) so a
downstream reader cannot lose the caveat.

**Live (prepared, not executed in Phase 2).** Requires both an explicit
`--mode live` and the `RECOGNITION_EVAL_LIVE=1` opt-in; refuses clearly when
either is missing. It builds the agent through its ordinary constructor so the
real factories choose the providers - which means a mode with no implementation
behind it refuses to start rather than degrading to a mock. Cases run
sequentially, under the agent's own bounded deadlines, with its own zero-retry
behaviour untouched.

The runner never modifies the agent. It imports `RecognitionAgent`, builds an
`AgentRequest`, and reads the `AgentResult` - the same public contract `api.py`
uses.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..adapters.reasoning_llm import FakeGPT5MiniProvider
from ..adapters.taxonomy import MockTaxonomyProvider
from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY, RecognitionConfig
from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import BioCLIPTaxonPrediction
from ..schema import AgentRequest, AgentStatus
from . import consistency
from .evaluators import evaluate_case
from .manifest_schema import cases as manifest_cases
from .manifest_schema import load_manifest
from .results_schema import (
    CaseResult,
    ConsistencySlot,
    RunSummary,
    safe_candidates,
    safe_provenance,
    sanitize_error,
)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
RESULTS = HERE / "results"

LIVE_OPT_IN_VAR = "RECOGNITION_EVAL_LIVE"

MODE_OFFLINE = "offline"
MODE_LIVE = "live"


# --- the scripted offline classifier ---------------------------------------

#: Deterministic scenarios the dry run cycles through. These are *evaluator
#: exercises*, not predictions: the point is to drive every branch of every
#: evaluator, which is why a third of them are deliberately wrong.
TOP1_CORRECT = "top1_correct"
TOP5_ONLY = "top5_only"
WRONG_SPECIES = "wrong_species"
NO_CANDIDATES = "no_candidates"
CLASSIFIER_ERROR = "classifier_error"

_REAL_CYCLE = (TOP1_CORRECT, TOP5_ONLY, WRONG_SPECIES, NO_CANDIDATES)

#: Stand-in labels used when a scenario needs a species that is *not* the
#: expected one. Real binomials, so ranking and taxonomy behave normally; the
#: choice is arbitrary and carries no biological meaning.
_DECOYS = ("Canis lupus", "Vulpes vulpes", "Lynx lynx", "Ursus arctos")


def species_id_for(scientific_name: str) -> str:
    """The id convention the agent's own taxonomy fixture uses."""
    return scientific_name.strip().lower().replace(" ", "_")


def scenario_for(case: dict, index_in_category: int) -> str:
    """Which scripted behaviour this case gets. Fixed, and independent of any
    observed result, so a rerun cannot drift."""
    category = case["category"]
    if category == "real_recognition":
        return _REAL_CYCLE[index_in_category % len(_REAL_CYCLE)]
    if category == "ambiguous_or_non_animal":
        # Matches what the case accepts: no identifiable animal, so no label.
        return NO_CANDIDATES
    if category == "delegation_resume":
        # Delegation cannot fire unless the decision is `identified`.
        return TOP1_CORRECT
    if case.get("failure_kind") == "classifier_unavailable":
        return CLASSIFIER_ERROR
    return NO_CANDIDATES


def _predictions(scenario: str, expected: str | None) -> list[BioCLIPTaxonPrediction]:
    def make(name: str, score: float) -> BioCLIPTaxonPrediction:
        return BioCLIPTaxonPrediction(
            species_id=species_id_for(name), scientific_name=name,
            rank="species", classification_score=score,
        )

    decoys = [d for d in _DECOYS if d != expected]
    if scenario == NO_CANDIDATES or expected is None:
        return []
    if scenario == TOP1_CORRECT:
        # Above the identified floor with a wide margin: a decisive top label.
        return [make(expected, 0.92), make(decoys[0], 0.30), make(decoys[1], 0.21)]
    if scenario == TOP5_ONLY:
        # Below the identified floor and tightly packed: the expected species is
        # present but not on top, which is exactly the Top-5-but-not-Top-1 case.
        return [make(decoys[0], 0.62), make(decoys[1], 0.55),
                make(expected, 0.44), make(decoys[2], 0.30)]
    if scenario == WRONG_SPECIES:
        return [make(decoys[0], 0.90), make(decoys[1], 0.28), make(decoys[2], 0.19)]
    return []


class ScriptedClassifier:
    """A deterministic stand-in for BioCLIP-2, and an invocation counter.

    It implements the same `classify(image, top_k)` signature the real provider
    does, so the workflow runs its ordinary path. `invocations` is what gives the
    zero-retry evaluator real evidence: the agent makes one logical
    classification call per request, so anything above one is a retry.
    """

    provider_name = "ScriptedDryRunClassifier"
    #: Deliberately the mock mode string: provenance must describe what actually
    #: ran, and what ran here is emphatically not remote BioCLIP-2.
    recognition_mode = "mock_classification"
    version = "sprint4-dry-run-scripted-v1"

    def __init__(self, scenario: str, expected_species: str | None) -> None:
        self._scenario = scenario
        self._expected = expected_species
        self.invocations = 0

    def classify(self, image: Any, top_k: int) -> list[BioCLIPTaxonPrediction]:
        self.invocations += 1
        if self._scenario == CLASSIFIER_ERROR:
            # The same controlled code every remote failure maps onto.
            raise RecognitionError(ErrorCode.CLASSIFICATION_UNAVAILABLE)
        return _predictions(self._scenario, self._expected)[:top_k]


# --- request construction ---------------------------------------------------

_MEDIA_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


def synthetic_image(case: dict) -> bytes:
    """A small, valid, deterministic image of the case's declared media type.

    The dry run must not depend on `assets/`, which is git-ignored and usually
    empty. The bytes are derived from the case id, so the same case always
    produces the same image and the same SHA-256.
    """
    from PIL import Image

    media_type = (case.get("asset") or {}).get("media_type", "image/png")
    digest = hashlib.sha256(case["case_id"].encode("utf-8")).digest()
    colour = (digest[0], digest[1], digest[2])
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), colour).save(
        buffer, format=_MEDIA_FORMAT.get(media_type, "PNG")
    )
    return buffer.getvalue()


def _data_url(raw: bytes, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.b64encode(raw).decode("ascii")


def build_request(case: dict, *, mode: str) -> AgentRequest:
    """One `AgentRequest`, honouring the case's declared asset mode."""
    context: dict[str, Any] = {}
    asset = case.get("asset")
    asset_mode = case.get("asset_mode", "image_required" if asset else "none")

    if asset_mode == "none":
        pass
    elif asset_mode == "synthetic_context":
        # A remote URL where an inline data URL is required. The agent must
        # refuse it; nothing is ever fetched.
        context[RECOGNITION_IMAGE_CONTEXT_KEY] = {
            "data_url": "https://example.invalid/not-fetched.jpg",
            "filename": "not-fetched.jpg",
        }
    elif asset:
        media_type = asset["media_type"]
        if mode == MODE_LIVE:
            path = HERE / asset["local_path"]
            if not path.exists():
                raise FileNotFoundError(asset["local_path"])
            raw = path.read_bytes()
            filename = Path(asset["local_path"]).name
        else:
            raw = synthetic_image(case)
            filename = f"{case['case_id']}.{_MEDIA_FORMAT.get(media_type, 'PNG').lower()}"
        context[RECOGNITION_IMAGE_CONTEXT_KEY] = {
            "data_url": _data_url(raw, media_type),
            "filename": filename,
        }

    if case.get("resume_context_present") and case.get("resume_context_key"):
        # The resume half of the delegation contract: the helper's output key is
        # already present, so the agent must complete instead of escalating.
        context[case["resume_context_key"]] = {
            "summary": "helper output placeholder recorded by the evaluation runner",
        }

    return AgentRequest(instruction=case["instruction"], context=context)


# --- observation ------------------------------------------------------------

def observe(result: Any, *, latency_ms: float, classifier_invocations: int | None) -> dict:
    """Reduce an `AgentResult` to the sanitized facts the evaluators read.

    Nothing that entered the agent comes back out here: no image, no data URL,
    no context. Only labels, identifiers, modes, counts and the decision.
    """
    status = result.status.value if isinstance(result.status, AgentStatus) else result.status
    output = result.output
    obs: dict[str, Any] = {
        "status": status,
        "output": output,
        "target_agent": result.target_agent,
        "prompt_to_target_agent": result.prompt_to_target_agent,
        "latency_ms": latency_ms,
        "classifier_invocations": classifier_invocations,
        "candidates": [],
        "provenance": {},
    }

    if isinstance(output, dict):
        if "error_code" in output:
            obs["error_code"] = output.get("error_code")
        recognition = output.get("recognition")
        if isinstance(recognition, dict):
            obs["decision"] = recognition.get("decision")
            obs["unsupported_capability"] = recognition.get("unsupported_capability")
        obs["primary_species"] = output.get("species")
        obs["species_id"] = output.get("species_id")
        obs["gbif_id"] = output.get("gbif_id")
        obs["ncbi_taxid"] = output.get("ncbi_taxid")
        obs["candidates"] = safe_candidates(output.get("recognition_candidates"))
        obs["provenance"] = safe_provenance(output.get("recognition_provenance"))
    return obs


# --- agent construction -----------------------------------------------------

def build_offline_agent(case: dict, scenario: str) -> tuple[RecognitionAgent, ScriptedClassifier]:
    """The agent with every provider injected. No env, no network, no credential."""
    classifier = ScriptedClassifier(scenario, case.get("expected_species"))
    agent = RecognitionAgent(
        RecognitionConfig.from_env(),
        classifier=classifier,
        taxonomy_provider=MockTaxonomyProvider(),
        # The offline deterministic stand-in: it exercises the real two-call
        # contract, which is what gives the call-budget evaluator evidence.
        reasoning_llm=FakeGPT5MiniProvider(),
    )
    return agent, classifier


def build_live_agent() -> RecognitionAgent:
    """The ordinary constructor, so the real factories choose the providers.

    Nothing is injected. A mode with no implementation behind it raises rather
    than degrading to a mock, which is the agent's own guarantee and the reason
    this function is three lines rather than thirty.
    """
    return RecognitionAgent(RecognitionConfig.from_env())


class LiveModeRefused(RuntimeError):
    """Raised when live execution is requested without the explicit opt-in."""


def require_live_opt_in(environ: dict[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    if env.get(LIVE_OPT_IN_VAR) != "1":
        raise LiveModeRefused(
            f"Live evaluation refused: set {LIVE_OPT_IN_VAR}=1 to contact real "
            "providers. Nothing was executed and no request was sent."
        )


# --- the run ----------------------------------------------------------------

def run(
    *,
    mode: str = MODE_OFFLINE,
    manifest: dict | None = None,
    clock: Callable[[], float] = time.perf_counter,
    environ: dict[str, str] | None = None,
) -> RunSummary:
    """Evaluate every case in committed order. One result each, always."""
    if mode not in (MODE_OFFLINE, MODE_LIVE):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == MODE_LIVE:
        require_live_opt_in(environ)

    manifest = manifest or load_manifest()
    cases = manifest_cases(manifest)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    consistency_subset = set(consistency.select(cases))
    live_agent = build_live_agent() if mode == MODE_LIVE else None

    counters: dict[str, int] = {}
    results: list[CaseResult] = []

    for case in cases:
        category = case["category"]
        index = counters.get(category, 0)
        counters[category] = index + 1

        result = CaseResult(case_id=case["case_id"], category=category, mode=mode,
                            executed=False)
        result.consistency = ConsistencySlot(
            selected=case["case_id"] in consistency_subset,
            repetitions_requested=(
                consistency.REPETITIONS if case["case_id"] in consistency_subset else 0
            ),
            compared_fields=list(consistency.COMPARED_FIELDS),
        )

        classifier: ScriptedClassifier | None = None
        obs: dict[str, Any]
        try:
            if mode == MODE_OFFLINE:
                agent, classifier = build_offline_agent(case, scenario_for(case, index))
            else:
                agent = live_agent  # type: ignore[assignment]
            request = build_request(case, mode=mode)

            start = clock()
            agent_result = agent.run(request)
            latency_ms = (clock() - start) * 1000.0

            obs = observe(
                agent_result, latency_ms=latency_ms,
                classifier_invocations=classifier.invocations if classifier else None,
            )
            result.executed = True
        except Exception as exc:  # noqa: BLE001 - recorded, never propagated
            # A case that blew up still gets a result. Dropping it here is how a
            # failure would silently leave the aggregation.
            result.execution_error = sanitize_error(exc)
            obs = {
                "status": None, "output": None, "target_agent": None,
                "latency_ms": None, "candidates": [], "provenance": {},
                "classifier_invocations": classifier.invocations if classifier else None,
            }

        result.status_observed = obs.get("status")
        result.decision_observed = obs.get("decision")
        result.primary_species_observed = obs.get("primary_species")
        result.species_id_observed = obs.get("species_id")
        result.gbif_id_observed = obs.get("gbif_id")
        result.ncbi_taxid_observed = obs.get("ncbi_taxid")
        result.target_agent_observed = obs.get("target_agent")
        result.error_code_observed = obs.get("error_code")
        result.unsupported_capability_observed = obs.get("unsupported_capability")
        result.candidates_observed = obs.get("candidates") or []
        result.provenance = obs.get("provenance") or {}
        result.latency_ms = obs.get("latency_ms")
        result.classifier_invocations = obs.get("classifier_invocations")
        result.agent_llm_calls = result.provenance.get("reasoning_llm_calls")
        result.evaluator_llm_calls = 0  # no judge model exists in Phase 2
        output = obs.get("output")
        result.output_keys_observed = sorted(output) if isinstance(output, dict) else []
        result.metrics = evaluate_case(case, obs)
        results.append(result)

    return RunSummary(
        mode=mode,
        benchmark_id=manifest.get("benchmark_id", "unknown"),
        started=started,
        finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results=results,
        dry_run=(mode == MODE_OFFLINE),
        notes=(
            "OFFLINE DRY RUN - INFRASTRUCTURE VALIDATION ONLY. Every label below "
            "came from a scripted table in runner.py, not from BioCLIP-2. These "
            "numbers validate the runner and the evaluators; they are not a "
            "measurement of recognition accuracy and must never be reported as one."
            if mode == MODE_OFFLINE else
            "Live run against the configured real providers."
        ),
    )


def write_results(summary: RunSummary, directory: Path | None = None) -> Path:
    """Write the machine-readable results into the git-ignored results directory."""
    target_dir = directory or RESULTS
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = summary.finished.replace(":", "").replace("-", "")
    path = target_dir / f"results_{summary.mode}_{stamp}.json"
    path.write_text(summary.to_json(), encoding="utf-8")
    return path


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sprint 4 Recognition evaluation runner (bounded benchmark).",
    )
    parser.add_argument("--mode", choices=[MODE_OFFLINE, MODE_LIVE], default=MODE_OFFLINE,
                        help="offline (default, injected providers) or live (guarded)")
    parser.add_argument("--out", default=None, help="results directory (default: evaluation/results)")
    parser.add_argument("--report", action="store_true", help="also write the Markdown report")
    args = parser.parse_args(argv)

    try:
        summary = run(mode=args.mode)
    except LiveModeRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    directory = Path(args.out) if args.out else None
    path = write_results(summary, directory)
    print(f"{len(summary.results)} cases evaluated -> {path.name}")
    if summary.dry_run:
        print("DRY RUN: infrastructure validation only. Not a measure of recognition accuracy.")

    if args.report:
        from .report import render

        report_path = path.with_suffix(".md")
        report_path.write_text(render(summary), encoding="utf-8")
        print(f"report -> {report_path.name}")

    totals = summary.metric_totals()
    failures = sum(t["fail"] for t in totals.values())
    print(f"metric outcomes: {json.dumps({k: v for k, v in list(totals.items())[:1]})} ...")
    print(f"{len(summary.failed_cases())} case(s) with at least one failing metric; "
          f"{failures} failing metric outcome(s) in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
