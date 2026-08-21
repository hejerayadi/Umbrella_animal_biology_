"""
Functional Evidence sub-orchestrator scenario runner (§4).

Unlike tests/test_functional_evidence_merge.py (which calls merge_node in
isolation with hand-built statuses) and
tests/integration/test_suborchestrator_to_agent.py (which mocks the KEGG/
UniProt clients), this module *executes* the real, compiled
functional_evidence_graph end to end: real Pathways agent, real Protein Data
agent, real KEGG/UniProt REST calls, real Qdrant cache writes, real merge_node
— for every scenario except the dedicated LLM-purity one, which calls
merge_node directly since that's the only node under test there and it has no
business touching KEGG/UniProt/Qdrant at all.

Only the gene→id lookups are faked where a scenario needs a deterministic
zero-hit gene (no real gene reliably has zero KEGG links AND zero reviewed
UniProt hits on demand) — every fetch and cache write on the path stays real.

Requires QDRANT_URL / QDRANT_API_KEY in the environment for every scenario
that runs the real graph (both agents always write through the cache layer).
No NVIDIA_NIM_API_KEY is required by any scenario here: FGF5/UCP1 are used in
their single-KEGG-link / single-reviewed-hit form, so neither child agent's
own LLM disambiguation path is exercised — that's already covered by
pathways_scenarios.py / protein_data_scenarios.py. This module is about
merge_node's coordination logic, not either child's decision-making.

Usage:
    python -m scripts.functional_evidence_scenarios --scenario both-succeed
    python -m scripts.functional_evidence_scenarios --scenario one-fails-one-succeeds
    python -m scripts.functional_evidence_scenarios --scenario both-fail
    python -m scripts.functional_evidence_scenarios --scenario no-llm-in-merge
    python -m scripts.functional_evidence_scenarios --scenario all
    python -m scripts.functional_evidence_scenarios --scenario all --verbose
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workflows.llm as llm_module
from schemas.common import AgentStatus
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.nodes.functional_evidence_nodes import merge_node
from workflows.state import FunctionalEvidenceState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real gene facts (same fixtures as pathways_scenarios.py / protein_data_
# scenarios.py — FGF5 and UCP1 both resolve to exactly one KEGG link and one
# reviewed UniProt hit today, so no child agent makes a real LLM call here).
# ---------------------------------------------------------------------------
FGF5_GENE, FGF5_KEGG_ID = "FGF5", "hsa:2249"
UCP1_GENE, UCP1_KEGG_ID = "UCP1", "hsa:7350"
TAX_ID = 9606
# Syntactically valid but non-existent — the real KEGG link endpoint returns
# an empty body and the real UniProt search returns zero hits for these, so
# both child agents naturally land in FAILED without faking anything.
MISSING_GENE, MISSING_KEGG_ID = "NOTAREALGENE1", "hsa:00000001"


async def _check(label: str, condition: bool, detail: str, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not condition:
        failures.append(f"{label}: {detail}")


def _state(gene_list: list[str], kegg_gene_ids: dict, trait_name: str = "test") -> FunctionalEvidenceState:
    return FunctionalEvidenceState(
        gene_list=gene_list,
        trait_name=trait_name,
        instruction="find functional evidence",
        context={"kegg_gene_ids": kegg_gene_ids, "tax_id": TAX_ID},
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
async def scenario_both_succeed(verbose: bool) -> list:
    """FGF5 resolves real data on both sides (KEGG link + reviewed UniProt
    hit) — merge_node's AND-of-failure rule (§0.2/§8) has nothing to fail on,
    so the sub-orchestrator status is COMPLETED. §10 inverse case of the
    existing one-child-fails test."""
    failures: list = []
    app = build_functional_evidence_graph()
    result = await app.ainvoke(_state([FGF5_GENE], {FGF5_GENE: FGF5_KEGG_ID}, trait_name="hair growth"))

    await _check("pathways_status COMPLETED",
                 result["pathways_status"] == AgentStatus.COMPLETED, str(result["pathways_status"]), failures)
    await _check("protein_data_status COMPLETED",
                 result["protein_data_status"] == AgentStatus.COMPLETED, str(result["protein_data_status"]), failures)
    await _check("subgraph status COMPLETED",
                 result["status"] == AgentStatus.COMPLETED, str(result["status"]), failures)
    return failures


async def scenario_one_fails_one_succeeds(verbose: bool) -> list:
    """Pathways gets a gene with no real KEGG link (FAILED); Protein Data
    gets a real gene with a reviewed UniProt hit (COMPLETED). Mirrors
    test_one_agent_failing_does_not_affect_the_other at the live-graph
    level: one child failing on its own is non-critical (§0.2), so the
    sub-orchestrator still completes."""
    failures: list = []
    app = build_functional_evidence_graph()
    result = await app.ainvoke(_state([MISSING_GENE], {MISSING_GENE: MISSING_KEGG_ID}))

    await _check("pathways_status FAILED (no real KEGG link for this id)",
                 result["pathways_status"] == AgentStatus.FAILED, str(result["pathways_status"]), failures)
    # Protein Data uses gene_list for its own UniProt lookup, independent of
    # kegg_gene_ids — swap in a gene with a real reviewed hit so this
    # scenario genuinely exercises "one failed, one succeeded" rather than
    # "both happened to fail for the same reason".
    result2 = await app.ainvoke(_state([FGF5_GENE], {FGF5_GENE: MISSING_KEGG_ID}))
    await _check("protein_data_status COMPLETED (real UniProt hit for FGF5)",
                 result2["protein_data_status"] == AgentStatus.COMPLETED, str(result2["protein_data_status"]), failures)
    await _check("pathways_status FAILED (KEGG id points nowhere)",
                 result2["pathways_status"] == AgentStatus.FAILED, str(result2["pathways_status"]), failures)
    await _check("subgraph status still COMPLETED (§0.2: single-child failure is non-critical)",
                 result2["status"] == AgentStatus.COMPLETED, str(result2["status"]), failures)
    return failures


async def scenario_both_fail(verbose: bool) -> list:
    """Neither child resolves anything real — no KEGG link, no reviewed
    UniProt hit — so both land in FAILED and merge_node's AND-of-failure
    rule fails the whole sub-orchestrator."""
    failures: list = []
    app = build_functional_evidence_graph()
    result = await app.ainvoke(_state([MISSING_GENE], {MISSING_GENE: MISSING_KEGG_ID}))

    await _check("pathways_status FAILED",
                 result["pathways_status"] == AgentStatus.FAILED, str(result["pathways_status"]), failures)
    await _check("protein_data_status FAILED",
                 result["protein_data_status"] == AgentStatus.FAILED, str(result["protein_data_status"]), failures)
    await _check("subgraph status FAILED (both children failed)",
                 result["status"] == AgentStatus.FAILED, str(result["status"]), failures)
    return failures


async def scenario_no_llm_in_merge(verbose: bool) -> list:
    """§4.2/§10 negative test, protein_structure-style: merge_node itself is
    called directly (not via the child agents, which are free to use the LLM
    for their own disambiguation) with every status combination, while
    workflows.llm.get_llm — the single choke point every real LLM
    invocation in this codebase funnels through — is poisoned to raise.
    A pass proves there is no code path in merge_node that could ever
    construct an LLM client, regardless of what its inputs are."""
    failures: list = []

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("merge_node must never construct an LLM client")

    combos = [
        (AgentStatus.COMPLETED, AgentStatus.COMPLETED, AgentStatus.COMPLETED),
        (AgentStatus.FAILED, AgentStatus.COMPLETED, AgentStatus.COMPLETED),
        (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.COMPLETED),
        (AgentStatus.FAILED, AgentStatus.FAILED, AgentStatus.FAILED),
    ]

    original_get_llm = llm_module.get_llm
    llm_module.get_llm = _fail_if_called
    try:
        for pathways_status, protein_data_status, expected in combos:
            state = FunctionalEvidenceState(
                gene_list=[FGF5_GENE],
                instruction="find functional evidence",
                pathways_status=pathways_status,
                protein_data_status=protein_data_status,
            )
            try:
                result = await merge_node(state)
                await _check(
                    f"merge_node({pathways_status.name}, {protein_data_status.name}) "
                    f"-> {expected.name}, no LLM call",
                    result["status"] == expected,
                    str(result["status"]),
                    failures,
                )
            except AssertionError as exc:
                await _check(
                    f"merge_node({pathways_status.name}, {protein_data_status.name}) "
                    "never touches the LLM",
                    False, f"raised {exc!r}", failures,
                )
    finally:
        llm_module.get_llm = original_get_llm

    return failures


SCENARIOS = {
    "both-succeed": (
        "FGF5 resolves real data on both sides — merge_node's AND-of-failure "
        "rule has nothing to fail on, subgraph status is COMPLETED.",
        scenario_both_succeed,
    ),
    "one-fails-one-succeeds": (
        "One child fails (no real KEGG link / bad gene), the other succeeds "
        "— subgraph still completes (§0.2: single-child failure is "
        "non-critical).",
        scenario_one_fails_one_succeeds,
    ),
    "both-fail": (
        "Neither child resolves anything real — subgraph status is FAILED.",
        scenario_both_fail,
    ),
    "no-llm-in-merge": (
        "Negative test: merge_node is called directly across every status "
        "combination with workflows.llm.get_llm poisoned — asserts no code "
        "path in this node can ever construct an LLM client.",
        scenario_no_llm_in_merge,
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
        description="Execute real Functional Evidence sub-orchestrator scenarios "
                     "against the live graph (KEGG/UniProt/Qdrant) plus a merge_node LLM-purity check."
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
