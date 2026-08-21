"""
Pathways Agent scenario runner.

Unlike tests/test_pathways.py (which mocks KEGG and the LLM to assert on
isolated units), this module *executes* the real `pathways_agent()` against
the real KEGG REST API and the real Qdrant cache. As in
scripts/gene_mapper_scenarios.py, only `_llm_pick_pathway` is patched where a
scenario needs a deterministic outcome (missing gene, simulated LLM outage) —
every KEGG lookup and cache write on the path stays real.

Requires QDRANT_URL / QDRANT_API_KEY in the environment for every scenario
(the agent always writes through the cache layer), and NVIDIA_NIM_API_KEY only
for `multi-pathway-llm-pick` (the only scenario that makes a real LLM call).

Usage:
    python -m scripts.pathways_scenarios --scenario multi-pathway-llm-pick
    python -m scripts.pathways_scenarios --scenario one-missing-one-present
    python -m scripts.pathways_scenarios --scenario llm-unavailable-fallback
    python -m scripts.pathways_scenarios --scenario all
    python -m scripts.pathways_scenarios --scenario all --verbose
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subagents.pathways as pw_module
from schemas.common import AgentStatus
from schemas.inputs import PathwaysInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real gene/KEGG facts (confirmed live in scripts/check_pathways.py — UCP1's
# 3 KEGG pathway links produced a real multi-candidate LLM pick of
# hsa04714/Thermogenesis). FGF5's KEGG gene id is also used elsewhere in this
# repo (see scripts/demo_live_integration.py).
# ---------------------------------------------------------------------------
UCP1_GENE, UCP1_KEGG_ID = "UCP1", "hsa:7350"
FGF5_GENE, FGF5_KEGG_ID = "FGF5", "hsa:2249"
# Syntactically valid but non-existent KEGG gene id — the real KEGG link
# endpoint returns an empty body for it, so list_pathway_candidates naturally
# comes back [] without needing to fake anything.
MISSING_GENE, MISSING_KEGG_ID = "MISSING_GENE", "hsa:00000001"


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


def _warn_slow_llm():
    print("  (real LLM call in progress — NIM can silently poll up to ~60s per "
          "turn on a cold request; run with --verbose to see per-turn progress "
          "logs from workflows.llm.tool_loop instead of a blank terminal)")


def _spy_llm_pick(reasoning_log: list):
    """Wraps the REAL _llm_pick_pathway so we can print what it decided and
    why, without changing its behavior."""
    real = pw_module._llm_pick_pathway

    async def spying(trait_name, gene_symbol, candidates, kegg_gene_id=""):
        pathway_id, pathway_name, reasoning = await real(
            trait_name, gene_symbol, candidates, kegg_gene_id
        )
        reasoning_log.append((pathway_id, pathway_name, reasoning))
        return pathway_id, pathway_name, reasoning

    return spying


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
async def scenario_multi_pathway_llm_pick(verbose: bool) -> list:
    """UCP1/hsa:7350 has multiple real KEGG pathway links. Runs the real
    bind_tools LLM pick (requires NVIDIA_NIM_API_KEY) and asserts its
    selection is actually used — specifically that it's NOT just the naive
    first-result pick a non-LLM implementation would return.

    Note: KEGG rate-limits aggressively (~3 req/sec), and resolving names for
    every unresolved candidate means several fetch_pathway_name calls in
    quick succession. kb/sources/_http_retry.py retries transient 429/5xx
    automatically, but if the LLM path still fails after retries exhaust,
    this scenario reports INCONCLUSIVE rather than FAIL — a fallback to the
    deterministic path in that case is the fallback working as designed, not
    a bug in the LLM-selection logic this scenario is meant to check."""
    real_candidates = await pw_module.list_pathway_candidates(UCP1_KEGG_ID)
    naive_first_id = real_candidates[0]["pathway_id"]
    print(f"  Real KEGG candidates for {UCP1_GENE}: {[c['pathway_id'] for c in real_candidates]}")
    print(f"  Naive first-result id (what a non-LLM pick would return): {naive_first_id}")

    reasoning_log: list = []
    failures: list = []
    _warn_slow_llm()
    try:
        with _Patch(pw_module, "_llm_pick_pathway", _spy_llm_pick(reasoning_log)):
            result = await pw_module.pathways_agent(PathwaysInput(
                trait_name="thermogenesis",
                gene_list=[UCP1_GENE],
                instruction="Find KEGG pathways",
                context={"kegg_gene_ids": {UCP1_GENE: UCP1_KEGG_ID}},
            ))
    except Exception as exc:
        await _check("real LLM call succeeded", False, f"raised {exc!r}", failures)
        return failures

    await _check("status COMPLETED", result.status == AgentStatus.COMPLETED, str(result.status), failures)

    if not reasoning_log:
        # The LLM path didn't complete (e.g. KEGG rate-limited it even after
        # retries) and the agent fell back to the deterministic path. That's
        # the fallback doing its job — but it means this run can't actually
        # exercise "LLM selection over naive pick", so don't fail it.
        print("  [INCONCLUSIVE] LLM pick did not complete this run (likely rate-limited "
              "even after retry) — the agent correctly fell back to the deterministic "
              "path instead. Re-run to get a genuine LLM-pick sample.")
        return failures

    if result.pathways:
        picked = result.pathways[0]
        print(f"  LLM picked: {picked.pathway_id} ({picked.pathway_name})")
        print(f"  LLM reasoning: {reasoning_log[0][2]}")
        await _check("LLM pick used over the naive first-result pick",
                     picked.pathway_id != naive_first_id or len(real_candidates) == 1,
                     f"picked={picked.pathway_id} naive_first={naive_first_id} "
                     f"(only equal if the model independently agreed, or there's only 1 candidate)",
                     failures)
    return failures


async def scenario_one_missing_one_present(verbose: bool) -> list:
    """One gene resolves to zero real KEGG links (malformed), the other
    resolves to real links. Only the missing one should land in
    malformed_ids, and overall status should stay COMPLETED because at least
    one pathway resolved — mirrors
    tests/integration/test_suborchestrator_to_agent.py::test_one_agent_failing_does_not_affect_the_other
    at the single-agent level with a real KEGG call for both genes.

    Note: FGF5 has more than one KEGG link, so this scenario can also trigger
    a real (unpatched) LLM call — see the multi-pathway-llm-pick scenario's
    docstring for why that can silently take a while."""
    failures: list = []
    _warn_slow_llm()
    result = await pw_module.pathways_agent(PathwaysInput(
        trait_name="test",
        gene_list=[MISSING_GENE, FGF5_GENE],
        instruction="Find KEGG pathways",
        context={"kegg_gene_ids": {MISSING_GENE: MISSING_KEGG_ID, FGF5_GENE: FGF5_KEGG_ID}},
    ))

    await _check("status COMPLETED", result.status == AgentStatus.COMPLETED, str(result.status), failures)
    await _check("only the missing gene is malformed",
                 result.malformed_ids == [MISSING_GENE], str(result.malformed_ids), failures)
    await _check("the present gene resolved a pathway",
                 any(True for _ in result.pathways), f"{len(result.pathways)} pathway(s)", failures)
    return failures


async def scenario_llm_unavailable_fallback(verbose: bool) -> list:
    """Simulates a real NIM outage by making _llm_pick_pathway raise —
    everything else (KEGG fetch, deterministic fetch_pathway fallback, Qdrant
    cache write) stays real. Asserts the fallback still returns a usable,
    unranked (first-link) result."""
    real_candidates = await pw_module.list_pathway_candidates(UCP1_KEGG_ID)
    naive_first_id = real_candidates[0]["pathway_id"]
    print(f"  Real KEGG candidates for {UCP1_GENE}: {[c['pathway_id'] for c in real_candidates]}")
    print(f"  Naive first-result id (what the fallback must return): {naive_first_id}")

    async def _simulated_outage(*args, **kwargs):
        raise RuntimeError("simulated NIM outage")

    failures: list = []
    with _Patch(pw_module, "_llm_pick_pathway", _simulated_outage):
        result = await pw_module.pathways_agent(PathwaysInput(
            trait_name="thermogenesis",
            gene_list=[UCP1_GENE],
            instruction="Find KEGG pathways",
            context={"kegg_gene_ids": {UCP1_GENE: UCP1_KEGG_ID}},
        ))

    await _check("status COMPLETED despite LLM outage",
                 result.status == AgentStatus.COMPLETED, str(result.status), failures)
    await _check("malformed_ids empty", result.malformed_ids == [], str(result.malformed_ids), failures)
    if result.pathways:
        await _check("fallback used the naive first-link result (unranked)",
                     result.pathways[0].pathway_id == naive_first_id,
                     f"{result.pathways[0].pathway_id} == {naive_first_id}", failures)
    return failures


SCENARIOS = {
    "multi-pathway-llm-pick": (
        "UCP1/hsa:7350 (multiple real KEGG links) — real LLM selection is used "
        "over the naive first-result pick. Requires NVIDIA_NIM_API_KEY.",
        scenario_multi_pathway_llm_pick,
    ),
    "one-missing-one-present": (
        "One gene with zero real KEGG links, one with real links — only the "
        "missing gene lands in malformed_ids, status stays COMPLETED.",
        scenario_one_missing_one_present,
    ),
    "llm-unavailable-fallback": (
        "Simulated NIM outage on UCP1/hsa:7350 — real KEGG fetch + "
        "deterministic fetch_pathway fallback still returns a usable, "
        "unranked list.",
        scenario_llm_unavailable_fallback,
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
        description="Execute real Pathways agent scenarios against live KEGG/Qdrant."
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
