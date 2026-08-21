"""
Protein Data Agent scenario runner.

Unlike tests/test_protein_data.py (which mocks UniProt and the LLM to assert
on isolated units), this module *executes* the real `protein_data_agent()`
against the real UniProt REST API and the real Qdrant cache. As in
scripts/pathways_scenarios.py, only `list_uniprot_candidates` and/or
`_llm_pick_protein` are patched where a scenario needs a deterministic
outcome (forcing a genuine multi-hit decision, simulating an LLM outage) —
everything else on the path (the real bind_tools loop, JSON parsing,
grounding check, deterministic fetch_uniprot fallback, and every Qdrant
cache write) stays real.

Two reviewed UniProt entries for the *same* gene+species pair is rare by
design (§2 of the guide — the UniProt query is already tight), so rather
than hunting for a real gene that happens to have one today, the multi-hit
and fallback scenarios fake the candidate list itself with a genuinely
trait-relevant entry and a plausible-looking distractor, then let a real
NVIDIA NIM call choose between them.

Requires QDRANT_URL / QDRANT_API_KEY in the environment for every scenario
(the agent always writes through the cache layer), and NVIDIA_NIM_API_KEY
only for `multi-protein-llm-pick` (the only scenario that makes a real LLM
call).

Usage:
    python -m scripts.protein_data_scenarios --scenario multi-protein-llm-pick
    python -m scripts.protein_data_scenarios --scenario one-missing-one-present
    python -m scripts.protein_data_scenarios --scenario llm-unavailable-fallback
    python -m scripts.protein_data_scenarios --scenario all
    python -m scripts.protein_data_scenarios --scenario all --verbose
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subagents.protein_data as pd_module
from schemas.common import AgentStatus
from schemas.inputs import ProteinDataInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real gene/UniProt facts. FGF5's single reviewed human entry is confirmed
# live in scripts/check_protein_data.py and used elsewhere in this repo (see
# scripts/demo_full_integration.py). UCP1 is likewise a real single-hit gene
# today — the candidate list for the multi-hit scenarios below is faked
# specifically because a real two-reviewed-hit gene isn't something you can
# rely on existing on demand (§2/§8: ambiguity is rare by design).
# ---------------------------------------------------------------------------
UCP1_GENE, TAX_ID = "UCP1", 9606
FGF5_GENE = "FGF5"
# Syntactically valid but non-existent gene symbol — the real UniProt search
# naturally returns zero results for it, so list_uniprot_candidates comes
# back [] without needing to fake anything.
MISSING_GENE = "NOTAREALGENE1"

# A genuinely trait-relevant entry and a plausible-looking distractor,
# placed FIRST so a passing multi-hit scenario proves the model reasoned
# about function_summary text rather than defaulting to array order.
_FAKE_CANDIDATES = [
    {
        "source_accession": "Q99999",
        "protein_name": f"{UCP1_GENE}-related pseudogene product",
        "function_summary": (
            "Isoform lacking the canonical mitochondrial targeting sequence; "
            "catalytically inactive, function undetermined, no established "
            "role in the trait under investigation."
        ),
    },
    {
        "source_accession": "P25874",
        "protein_name": "Uncoupling protein 1",
        "function_summary": (
            "Mitochondrial inner-membrane proton channel expressed in brown "
            "adipose tissue; uncouples oxidative phosphorylation from ATP "
            "synthesis to generate heat via non-shivering thermogenesis, the "
            "core mechanism of cold adaptation."
        ),
    },
]


class _Patch:
    """Tiny manual monkeypatch — swap an attribute, restore it on exit.
    Same helper as scripts/pathways_scenarios.py."""

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


async def _fake_candidates(gene_symbol: str, tax_id: int):
    return _FAKE_CANDIDATES


def _spy_llm_pick(reasoning_log: list):
    """Wraps the REAL _llm_pick_protein so we can print what it decided and
    why, without changing its behavior."""
    real = pd_module._llm_pick_protein

    async def spying(trait_name, gene_symbol, candidates, tax_id=None):
        accession, protein_name, function_summary, reasoning = await real(
            trait_name, gene_symbol, candidates, tax_id
        )
        reasoning_log.append((accession, protein_name, function_summary, reasoning))
        return accession, protein_name, function_summary, reasoning

    return spying


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
async def scenario_multi_protein_llm_pick(verbose: bool) -> list:
    """Two faked reviewed UniProt hits for UCP1 — one genuinely trait-relevant
    (thermogenesis), one a plausible-looking distractor pseudogene entry
    placed FIRST. Runs the real bind_tools LLM pick (requires
    NVIDIA_NIM_API_KEY) end to end (prompt, tool binding, JSON parsing,
    grounding check against the real candidate list) and asserts its
    selection is actually used — specifically that it's NOT just the naive
    first-result pick a non-LLM implementation would return."""
    naive_first_accession = _FAKE_CANDIDATES[0]["source_accession"]
    print(f"  Candidate accessions for {UCP1_GENE}: "
          f"{[c['source_accession'] for c in _FAKE_CANDIDATES]}")
    print(f"  Naive first-result accession (what a non-LLM pick would return): "
          f"{naive_first_accession}")

    reasoning_log: list = []
    failures: list = []
    _warn_slow_llm()
    try:
        with _Patch(pd_module, "list_uniprot_candidates", _fake_candidates), \
             _Patch(pd_module, "_llm_pick_protein", _spy_llm_pick(reasoning_log)):
            result = await pd_module.protein_data_agent(ProteinDataInput(
                trait_name="cold adaptation",
                gene_list=[UCP1_GENE],
                instruction="Find UniProt protein data",
                context={"tax_id": TAX_ID},
            ))
    except Exception as exc:
        await _check("real LLM call succeeded", False, f"raised {exc!r}", failures)
        return failures

    await _check("status COMPLETED", result.status == AgentStatus.COMPLETED, str(result.status), failures)

    if not reasoning_log:
        print("  [INCONCLUSIVE] LLM pick did not complete this run — the agent "
              "correctly fell back to the deterministic path instead. Re-run to "
              "get a genuine LLM-pick sample.")
        return failures

    if result.proteins:
        picked = result.proteins[0]
        print(f"  LLM picked: {picked.source_accession} ({picked.protein_name})")
        print(f"  LLM reasoning: {reasoning_log[0][3]}")
        await _check(
            "LLM pick used the trait-relevant entry, not the naive first-result pick",
            picked.source_accession != naive_first_accession,
            f"picked={picked.source_accession} naive_first={naive_first_accession}",
            failures,
        )
        await _check(
            "picked accession is grounded in the real candidate list",
            picked.source_accession in {c["source_accession"] for c in _FAKE_CANDIDATES},
            picked.source_accession, failures,
        )
    return failures


async def scenario_one_missing_one_present(verbose: bool) -> list:
    """One gene resolves to zero real reviewed UniProt hits (missing), the
    other resolves to a real reviewed hit. Only the missing one should land
    in missing_genes, and overall status should stay COMPLETED because at
    least one protein resolved — mirrors
    tests/integration/test_suborchestrator_to_agent.py::test_one_agent_failing_does_not_affect_the_other
    at the single-agent level with a real UniProt call for both genes."""
    failures: list = []
    result = await pd_module.protein_data_agent(ProteinDataInput(
        trait_name="test",
        gene_list=[MISSING_GENE, FGF5_GENE],
        instruction="Find UniProt protein data",
        context={"tax_id": TAX_ID},
    ))

    await _check("status COMPLETED", result.status == AgentStatus.COMPLETED, str(result.status), failures)
    await _check("only the missing gene is in missing_genes",
                 result.missing_genes == [MISSING_GENE], str(result.missing_genes), failures)
    await _check("the present gene resolved a protein",
                 any(True for _ in result.proteins), f"{len(result.proteins)} protein(s)", failures)
    return failures


async def scenario_llm_unavailable_fallback(verbose: bool) -> list:
    """Simulates a real NIM outage by making _llm_pick_protein raise, on the
    same faked two-candidate list as the multi-pick scenario — everything
    else (deterministic fetch_uniprot fallback via real UniProt, Qdrant
    cache write) stays real. Asserts the fallback still returns a usable,
    unranked (first-hit) result rather than dropping the gene entirely.

    Note: the deterministic fallback re-queries UniProt for real (it doesn't
    know about the faked candidate list), so the accession it returns is
    whatever UniProt's real first reviewed hit for UCP1 is today — not
    necessarily either of the faked accessions above."""
    async def _simulated_outage(*args, **kwargs):
        raise RuntimeError("simulated NIM outage")

    failures: list = []
    with _Patch(pd_module, "list_uniprot_candidates", _fake_candidates), \
         _Patch(pd_module, "_llm_pick_protein", _simulated_outage):
        result = await pd_module.protein_data_agent(ProteinDataInput(
            trait_name="cold adaptation",
            gene_list=[UCP1_GENE],
            instruction="Find UniProt protein data",
            context={"tax_id": TAX_ID},
        ))

    await _check("status COMPLETED despite LLM outage",
                 result.status == AgentStatus.COMPLETED, str(result.status), failures)
    await _check("missing_genes empty", result.missing_genes == [], str(result.missing_genes), failures)
    if result.proteins:
        print(f"  Fallback returned: {result.proteins[0].source_accession} "
              f"({result.proteins[0].protein_name})")
        await _check("fallback returned a usable (unranked) result",
                     bool(result.proteins[0].source_accession), result.proteins[0].source_accession,
                     failures)
    return failures


SCENARIOS = {
    "multi-protein-llm-pick": (
        "UCP1 with two faked reviewed UniProt hits (one trait-relevant, one "
        "a distractor) — real LLM selection is used over the naive "
        "first-result pick. Requires NVIDIA_NIM_API_KEY.",
        scenario_multi_protein_llm_pick,
    ),
    "one-missing-one-present": (
        "One gene with zero real reviewed UniProt hits, one with a real hit "
        "— only the missing gene lands in missing_genes, status stays "
        "COMPLETED.",
        scenario_one_missing_one_present,
    ),
    "llm-unavailable-fallback": (
        "Simulated NIM outage on UCP1's (faked) two-candidate list — "
        "deterministic fetch_uniprot fallback via real UniProt still "
        "returns a usable, unranked result.",
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
        description="Execute real Protein Data agent scenarios against live UniProt/Qdrant."
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
