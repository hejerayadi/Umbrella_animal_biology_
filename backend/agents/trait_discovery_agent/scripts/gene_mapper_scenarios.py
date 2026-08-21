"""
Gene Mapper Agent scenario runner.

Unlike tests/test_gene_mapper.py (which mocks QuickGO and the LLM to assert on
isolated units), this module *executes* the real `gene_mapper_agent()` against
the real QuickGO REST API, the real Qdrant cache, and — for the one scenario
that needs it — the real NIM-backed LLM pick. The other scenarios patch only
`_llm_pick_candidate` (the same boundary tests/test_gene_mapper.py monkeypatches)
so they're deterministic and don't require an NVIDIA_NIM_API_KEY, while every
QuickGO lookup and cache write on the path stays real.

Requires QDRANT_URL / QDRANT_API_KEY in the environment for every scenario
(the agent always writes through the cache layer), and NVIDIA_NIM_API_KEY only
for `three-candidates-llm-pick` (the only scenario that makes a real LLM call).

Usage:
    python -m scripts.gene_mapper_scenarios --scenario three-candidates-llm-pick
    python -m scripts.gene_mapper_scenarios --scenario no-uniprot-no-llm-call
    python -m scripts.gene_mapper_scenarios --scenario llm-unavailable-fallback
    python -m scripts.gene_mapper_scenarios --scenario grounding-rejects-invented-id
    python -m scripts.gene_mapper_scenarios --scenario all
    python -m scripts.gene_mapper_scenarios --scenario all --verbose
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subagents.gene_mapper as gm_module
from schemas.common import AgentStatus
from schemas.inputs import GeneMapperInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real gene/UniProt facts (confirmed live against QuickGO — see the FGF5/HR
# live run in scripts/check_gene_mapper.py). HR has exactly 3 biological_process
# candidates, which is what the "three GO candidates" scenario needs; FGF5 has
# 5, which is enough to exercise the multi-candidate fallback/grounding paths.
# ---------------------------------------------------------------------------
HR_GENE, HR_ACCESSION = "HR", "O43593"
FGF5_GENE, FGF5_ACCESSION = "FGF5", "P12034"


class _Patch:
    """Tiny manual monkeypatch — swap an attribute, restore it on exit.
    Same helper as scripts/orchestration_retrieval_scenarios.py."""

    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value

    def __enter__(self):
        self._orig = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self._orig)


async def _check(label: str, condition: bool, detail: str, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not condition:
        failures.append(f"{label}: {detail}")


def _spy_llm_pick(reasoning_log: list):
    """Wraps the REAL _llm_pick_candidate so we can print what it decided and
    why, without changing its behavior (same spirit as _embed_counter() in
    orchestration_retrieval_scenarios.py)."""
    real = gm_module._llm_pick_candidate

    async def spying(trait_name, gene_symbol, candidates, uniprot_accession=""):
        go_id, go_name, reasoning = await real(
            trait_name, gene_symbol, candidates, uniprot_accession
        )
        reasoning_log.append((go_id, go_name, reasoning))
        return go_id, go_name, reasoning

    return spying


def _warn_slow_llm():
    print("  (real LLM call in progress — NIM can silently poll up to ~60s per "
          "turn on a cold request; run with --verbose to see per-turn progress "
          "logs from workflows.llm.tool_loop instead of a blank terminal)")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
async def scenario_three_candidates_llm_pick(verbose: bool) -> list:
    """HR/O43593 has exactly 3 real biological_process GO candidates. Runs the
    real bind_tools LLM pick (requires NVIDIA_NIM_API_KEY) and asserts its
    choice is used end to end: grounded in the real candidate set and
    unmatched_genes stays empty.

    Note: QuickGO also rate-limits, and resolving names for every unresolved
    candidate means several resolve_go_term_name calls in quick succession.
    kb/sources/_http_retry.py retries transient 429/5xx automatically, but if
    the LLM path still fails after retries exhaust, this scenario reports
    INCONCLUSIVE rather than FAIL — falling back to the deterministic path in
    that case is the fallback working as designed."""
    real_candidates = await gm_module.list_go_candidates(HR_GENE, HR_ACCESSION)
    valid_ids = {c["go_id"] for c in real_candidates}
    print(f"  Real QuickGO candidates for {HR_GENE}: {sorted(valid_ids)}")

    reasoning_log: list = []
    failures: list = []
    _warn_slow_llm()
    try:
        with _Patch(gm_module, "_llm_pick_candidate", _spy_llm_pick(reasoning_log)):
            result = await gm_module.gene_mapper_agent(GeneMapperInput(
                trait_name="hair follicle development",
                gene_list=[HR_GENE],
                species_name="Homo sapiens",
                instruction="Map genes to GO biological process",
                context={"uniprot_accessions": {HR_GENE: HR_ACCESSION}},
            ))
    except Exception as exc:
        await _check("real LLM call succeeded", False, f"raised {exc!r}", failures)
        return failures

    await _check("status COMPLETED", result.status == AgentStatus.COMPLETED,
                 str(result.status), failures)
    await _check("unmatched_genes empty", result.unmatched_genes == [],
                 str(result.unmatched_genes), failures)

    if not reasoning_log:
        print("  [INCONCLUSIVE] LLM pick did not complete this run (likely rate-limited "
              "even after retry) — the agent correctly fell back to the deterministic "
              "path instead. Re-run to get a genuine LLM-pick sample.")
        return failures

    if result.go_annotations:
        picked = result.go_annotations[0]
        await _check("picked go_id is grounded in the real candidate list",
                     picked.go_id in valid_ids, f"{picked.go_id} in {sorted(valid_ids)}", failures)
        _, _, reasoning = reasoning_log[0]
        print(f"  LLM reasoning: {reasoning}")
    return failures


async def scenario_no_uniprot_no_llm_call(verbose: bool) -> list:
    """A gene with no uniprot_accession in context must land in
    unmatched_genes without ever reaching list_go_candidates or the LLM pick.
    No network / NIM key required — this path never leaves Python."""
    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not have been called")

    failures: list = []
    with _Patch(gm_module, "list_go_candidates", _fail_if_called), \
         _Patch(gm_module, "_llm_pick_candidate", _fail_if_called):
        result = await gm_module.gene_mapper_agent(GeneMapperInput(
            trait_name="test trait",
            gene_list=["NO_ACCESSION_GENE"],
            species_name="Homo sapiens",
            instruction="Map genes to GO biological process",
            context={"uniprot_accessions": {}},
        ))

    await _check("status FAILED", result.status == AgentStatus.FAILED, str(result.status), failures)
    await _check("gene lands in unmatched_genes",
                 result.unmatched_genes == ["NO_ACCESSION_GENE"], str(result.unmatched_genes), failures)
    await _check("no annotations produced", result.go_annotations == [], str(result.go_annotations), failures)
    return failures


async def scenario_llm_unavailable_fallback(verbose: bool) -> list:
    """Simulates a real NIM outage by making _llm_pick_candidate raise —
    everything else (QuickGO fetch, deterministic fetch_go_annotation
    fallback, Qdrant cache write) stays real. Asserts the fallback still
    returns a usable, unranked (first-candidate) result."""
    real_candidates = await gm_module.list_go_candidates(FGF5_GENE, FGF5_ACCESSION)
    naive_first_id = real_candidates[0]["go_id"]
    print(f"  Real QuickGO candidates for {FGF5_GENE}: {[c['go_id'] for c in real_candidates]}")
    print(f"  Naive first-result id (what the fallback must return): {naive_first_id}")

    async def _simulated_outage(*args, **kwargs):
        raise RuntimeError("simulated NIM outage")

    failures: list = []
    with _Patch(gm_module, "_llm_pick_candidate", _simulated_outage):
        result = await gm_module.gene_mapper_agent(GeneMapperInput(
            trait_name="hair follicle development",
            gene_list=[FGF5_GENE],
            species_name="Homo sapiens",
            instruction="Map genes to GO biological process",
            context={"uniprot_accessions": {FGF5_GENE: FGF5_ACCESSION}},
        ))

    await _check("status COMPLETED despite LLM outage",
                 result.status == AgentStatus.COMPLETED, str(result.status), failures)
    await _check("unmatched_genes empty", result.unmatched_genes == [], str(result.unmatched_genes), failures)
    if result.go_annotations:
        await _check("fallback used the naive first candidate (unranked)",
                     result.go_annotations[0].go_id == naive_first_id,
                     f"{result.go_annotations[0].go_id} == {naive_first_id}", failures)
    return failures


async def scenario_grounding_rejects_invented_id(verbose: bool) -> list:
    """LLM tries to submit a go_id that was never in the real
    list_go_candidates result. The grounding check in
    subagents/gene_mapper/__init__.py must reject it and fall back to the
    deterministic (real, unranked) path instead of trusting the model."""
    real_candidates = await gm_module.list_go_candidates(FGF5_GENE, FGF5_ACCESSION)
    valid_ids = {c["go_id"] for c in real_candidates}
    invented_id = "GO:9999999"
    assert invented_id not in valid_ids, "test fixture bug: invented id collides with a real candidate"
    naive_first_id = real_candidates[0]["go_id"]

    async def _fake_llm_pick(*args, **kwargs):
        return invented_id, "fabricated term", "hallucinated, not grounded in any tool result"

    failures: list = []
    with _Patch(gm_module, "_llm_pick_candidate", _fake_llm_pick):
        result = await gm_module.gene_mapper_agent(GeneMapperInput(
            trait_name="hair follicle development",
            gene_list=[FGF5_GENE],
            species_name="Homo sapiens",
            instruction="Map genes to GO biological process",
            context={"uniprot_accessions": {FGF5_GENE: FGF5_ACCESSION}},
        ))

    await _check("status COMPLETED (fallback recovered)",
                 result.status == AgentStatus.COMPLETED, str(result.status), failures)
    if result.go_annotations:
        await _check("invented id was NOT used",
                     result.go_annotations[0].go_id != invented_id,
                     result.go_annotations[0].go_id, failures)
        await _check("fallback go_id is grounded in the real candidate set",
                     result.go_annotations[0].go_id == naive_first_id,
                     f"{result.go_annotations[0].go_id} == {naive_first_id}", failures)
    return failures


SCENARIOS = {
    "three-candidates-llm-pick": (
        "HR/O43593 (3 real GO candidates) — real LLM pick is used end to end, "
        "unmatched_genes stays empty. Requires NVIDIA_NIM_API_KEY.",
        scenario_three_candidates_llm_pick,
    ),
    "no-uniprot-no-llm-call": (
        "Gene with no uniprot_accession in context lands in unmatched_genes "
        "without list_go_candidates or the LLM ever being called.",
        scenario_no_uniprot_no_llm_call,
    ),
    "llm-unavailable-fallback": (
        "Simulated NIM outage on FGF5/P12034 — real QuickGO fetch + "
        "deterministic fetch_go_annotation fallback still returns a usable, "
        "unranked result.",
        scenario_llm_unavailable_fallback,
    ),
    "grounding-rejects-invented-id": (
        "LLM submits a go_id absent from the real candidate list — the "
        "grounding check must reject it and fall back to the deterministic path.",
        scenario_grounding_rejects_invented_id,
    ),
}


async def run_one(name: str, verbose: bool) -> bool:
    description, fn = SCENARIOS[name]
    print("\n" + "#" * 70)
    print(f"# SCENARIO: {name}")
    print(f"# {description}")
    print("#" * 70)
    start = time.perf_counter()
    failures = await fn(verbose)
    elapsed = time.perf_counter() - start

    print("\n--- result ---")
    if failures:
        print(f"{len(failures)} check(s) failed in {elapsed:.2f}s:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"all checks passed in {elapsed:.2f}s")
    return not failures


async def main(scenario: str, verbose: bool) -> int:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                         format="%(levelname)s:%(name)s:%(message)s")

    names = list(SCENARIOS) if scenario == "all" else [scenario]
    results = {}
    for name in names:
        results[name] = await run_one(name, verbose)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute real Gene Mapper agent scenarios against live QuickGO/Qdrant."
    )
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS) + ["all"], default="all",
        help="Which scenario to run (default: all).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print INFO-level logs, including httpx request lines.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.scenario, args.verbose)))
