"""
Step 7 — run the RAG eval over rag_test_cases.yaml and produce a retrieval
scorecard broken down by collection (GO / KEGG / UniProt / literature),
not one blended number.

Usage:
    python evaluation/run_eval.py                 # all genes
    python evaluation/run_eval.py --gene TP53      # single gene, verbose
    python evaluation/run_eval.py --json results/latest.json

Requires QDRANT_URL / QDRANT_API_KEY (and whatever the real gene_mapper
LLM/QuickGO path needs) to be configured for a live-KB run -- pathways and
protein_data use the mock subagents by default so this stays runnable
offline (see capture.py docstring for how to swap in the real ones).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# RAG_EVAL_DEBUG_RELEVANCY=1 also enables per-claim raw-vs-sigmoid logging in
# rag_evaluators.py -- without a handler configured here those logger.info()
# calls are silently dropped by Python's default logging setup (root logger
# defaults to WARNING with no handler), so this is required, not cosmetic.
if os.environ.get("RAG_EVAL_DEBUG_RELEVANCY") == "1":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_capture import load_cases as load_agent_cases, run_case  # noqa: E402
from agent_evaluators import score_case  # noqa: E402
from capture import CAPTURE_FNS, NodeCapture  # noqa: E402
from rag_evaluators import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    payload_contract,
)

TEST_CASES_PATH = Path(__file__).parent / "rag_test_cases.yaml"
NODE_TO_COLLECTION = {
    "gene_mapper": "go_annotations",
    "pathways": "kegg_pathways",
    "protein_data": "uniprot_proteins",
    "literature_support": "literature (no Qdrant collection)",
}
NODE_TO_EXPECTED_KEY = {
    "gene_mapper": "expected_go_terms",
    "pathways": "expected_pathways",
    "protein_data": "expected_protein_summary_contains",
}


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH) as f:
        return yaml.safe_load(f)


async def run_one_node(node: str, case: dict) -> dict[str, Any]:
    gene = case["gene"]
    trait_name = case["trait_name"]
    species_name = case.get("species_name", "")

    capture_fn = CAPTURE_FNS[node]
    if node == "literature_support":
        cap: NodeCapture = await capture_fn(gene, trait_name)
    else:
        # gene_mapper, pathways, protein_data all need species_name now to
        # resolve a tax_id / KEGG org code for live UniProt+KEGG lookups.
        cap = await capture_fn(gene, trait_name, species_name)

    result: dict[str, Any] = {
        "gene": gene, "node": node, "collection": cap.collection,
        "error": cap.error, "num_chunks": len(cap.context_chunks),
        "unresolved": cap.unresolved,
    }

    if cap.error:
        return result

    result["faithfulness"] = asdict(faithfulness(cap.answer_claims, cap.context_chunks))
    # unresolved (no UniProt/KEGG id to key off of, zero reviewed hits, etc.)
    # is a coverage gap, not a relevancy judgment -- scoring it into
    # answer_relevancy is what was flattening kegg_pathways to 0.00 even
    # though the pathways that *did* resolve were genuinely on-topic. It's
    # still recorded (see "resolution rate" in the scorecard) just not
    # blended into this average.
    if not cap.unresolved:
        # relevancy_claims falls back to answer_claims for nodes that don't
        # set it explicitly (protein_data, literature_support already
        # produce sentence-shaped text with nothing extra to add -- see
        # NodeCapture.relevancy_claims).
        relevancy_input = cap.relevancy_claims or cap.answer_claims
        result["answer_relevancy"] = asdict(await answer_relevancy(relevancy_input, gene, trait_name))
    if node in NODE_TO_EXPECTED_KEY:
        expected_terms = case.get(NODE_TO_EXPECTED_KEY[node]) or []
        result["context_recall"] = asdict(context_recall(expected_terms, cap.context_chunks))
    if cap.collection:
        result["context_precision"] = asdict(context_precision(cap.context_chunks, gene))
        contract = payload_contract(cap.collection, cap.context_chunks)
        result["payload_contract"] = {
            "total_chunks": contract.total_chunks,
            "valid_chunks": contract.valid_chunks,
            "failures": contract.failures,
        }
    return result


async def run_eval(genes: list[str] | None = None) -> list[dict[str, Any]]:
    cases = load_test_cases()
    if genes:
        wanted = {g.upper() for g in genes}
        cases = [c for c in cases if c["gene"].upper() in wanted]

    results = []
    for case in cases:
        for node in ("gene_mapper", "pathways", "protein_data", "literature_support"):
            results.append(await run_one_node(node, case))
    return results


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_scorecard(results: list[dict[str, Any]]) -> None:
    by_collection: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_collection[r["collection"] or "literature (no Qdrant collection)"].append(r)

    print("\n=== Retrieval scorecard, by collection ===\n")
    for collection, rows in sorted(by_collection.items(), key=lambda kv: str(kv[0])):
        errored = [r for r in rows if r["error"]]
        ok = [r for r in rows if not r["error"]]
        unresolved = [r for r in ok if r.get("unresolved")]
        print(f"[{collection}] — {len(rows)} gene(s) evaluated, {len(errored)} errored")
        if ok:
            resolved_n = len(ok) - len(unresolved)
            print(f"  {'resolution_rate':<18} {resolved_n}/{len(ok)}  (an answer was actually produced)")
            if unresolved:
                genes = ", ".join(r["gene"] for r in unresolved)
                print(f"  {'':<18} unresolved genes: {genes}")
            for metric in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
                scores = [r[metric]["score"] for r in ok if metric in r]
                if scores:
                    print(f"  {metric:<18} avg={_avg(scores):.2f}  (n={len(scores)})")
            contract_rows = [r for r in ok if "payload_contract" in r]
            if contract_rows:
                total = sum(r["payload_contract"]["total_chunks"] for r in contract_rows)
                valid = sum(r["payload_contract"]["valid_chunks"] for r in contract_rows)
                print(f"  {'payload_contract':<18} {valid}/{total} chunks valid")
        for r in errored:
            print(f"  ERROR gene={r['gene']} node={r['node']}: {r['error']}")
        print()


async def run_agent_eval(case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Part B — orchestration/escalation/parallel fan-out. Every case builds
    and drives the real compiled trait_discovery graph, patching only the
    subagent/resolver/writer boundary each scenario needs (see
    agent_fixtures.py). Returns one result dict per case with the five
    evaluator verdicts from agent_evaluators.score_case."""
    cases = load_agent_cases()
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]

    results = []
    for case in cases:
        run = await run_case(case)
        result: dict[str, Any] = {"id": case["id"], "description": case["description"].strip(),
                                   "node_order": run.node_order, "error": run.error}
        if not run.error:
            scores = score_case(case, run)
            result["checks"] = {name: {"passed": r.passed, "detail": r.detail} for name, r in scores.items()}
            result["passed"] = all(r.passed for r in scores.values())
        else:
            result["passed"] = False
        results.append(result)
    return results


def print_agent_scorecard(results: list[dict[str, Any]]) -> None:
    print("\n=== Orchestration scorecard, by scenario ===\n")
    passed = sum(1 for r in results if r["passed"])
    print(f"{passed}/{len(results)} scenario(s) fully passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']} — {r['description']}")
        if r["error"]:
            print(f"    RUNNER ERROR: {r['error']}")
            continue
        for name, check in r["checks"].items():
            mark = "ok" if check["passed"] else "XX"
            print(f"    [{mark}] {name}: {check['detail']}")
        print()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trait_discovery_agent eval (Sprint 4 guide, Parts A & B).")
    parser.add_argument("--gene", action="append", help="Restrict Part A (RAG) to one gene (repeatable).")
    parser.add_argument("--case", action="append", help="Restrict Part B (orchestration) to one case id (repeatable).")
    parser.add_argument("--skip-rag", action="store_true", help="Skip Part A (retrieval quality).")
    parser.add_argument("--skip-agent", action="store_true", help="Skip Part B (orchestration/escalation).")
    parser.add_argument("--json", type=Path, default=None, help="Write raw results (both parts) to this path.")
    args = parser.parse_args()

    rag_results: list[dict[str, Any]] = []
    agent_results: list[dict[str, Any]] = []
    had_error = False

    # Step 6: two separate scorecards, never blended into one number — a low
    # score in one tells a different team something different than the other.
    if not args.skip_rag:
        rag_results = await run_eval(genes=args.gene)
        print_scorecard(rag_results)
        had_error = had_error or any(r["error"] for r in rag_results)

    if not args.skip_agent:
        agent_results = await run_agent_eval(case_ids=args.case)
        print_agent_scorecard(agent_results)
        had_error = had_error or any(not r["passed"] for r in agent_results)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rag": rag_results, "agent": agent_results}, indent=2, default=str))
        print(f"raw results written to {args.json}")

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))