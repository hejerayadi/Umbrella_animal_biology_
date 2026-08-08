"""
Orchestration + Retrieval scenario runner for the Functional Evidence sub-orchestrator.

Unlike the pytest suite (which asserts on isolated units), this module actually
*executes* the real compiled LangGraph sub-orchestrator
(`build_functional_evidence_graph()`, unmodified) with `astream(..., stream_mode="updates")`
so you see every node fire in real time, and it drives the *real* Qdrant KB layer
(`kb.qdrant_store` / `kb.retrieval`) — no monkeypatching of the caching, embedding, or
retrieval code paths. The only thing faked is the outbound HTTP call to UniProt/KEGG
(`fetch_uniprot` / `fetch_pathway`), the same boundary the integration tests fake, so
scenarios are fast and deterministic without needing network access to those APIs.

Each scenario:
  1. purges the relevant Qdrant collections (clean slate, no cross-scenario pollution)
  2. streams the real graph execution, printing each node as it completes
  3. runs real retrieval checks against the KB the graph just wrote to
     (semantic_search, get_cached, validate_document) and prints PASS/FAIL per check

Requires QDRANT_URL / QDRANT_API_KEY in the environment (see .env.example) and
qdrant-client / sentence-transformers installed — same requirements as the app itself.

Usage:
    python -m scripts.orchestration_retrieval_scenarios --scenario cache-dataflow
    python -m scripts.orchestration_retrieval_scenarios --scenario retrieval-round-trip
    python -m scripts.orchestration_retrieval_scenarios --scenario metadata-isolation
    python -m scripts.orchestration_retrieval_scenarios --scenario unresolved-gene
    python -m scripts.orchestration_retrieval_scenarios --scenario all
    python -m scripts.orchestration_retrieval_scenarios --scenario all --verbose
"""
import argparse
import asyncio
import logging
import sys
import time
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kb.qdrant_store as qdrant_store
import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
from kb.qdrant_store import COLLECTIONS, ensure_collections, get_cached, get_client
from kb.retrieval import semantic_search, validate_document
from schemas.common import AgentStatus
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.state import FunctionalEvidenceState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Faked external boundary only. Everything past this point — caching, dedup,
# embedding, upsert, semantic search, schema validation — is the real code.
# ---------------------------------------------------------------------------
UNIPROT_DB = {
    ("FGF5", 9606): dict(gene_symbol="FGF5", protein_name="Fibroblast growth factor 5",
                          function_summary="Regulates hair follicle growth cycle.", source_accession="P12034"),
    ("UCP1", 9606): dict(gene_symbol="UCP1", protein_name="Uncoupling protein 1",
                          function_summary="Mitochondrial proton channel, generates heat.", source_accession="P25874"),
}
KEGG_DB = {
    "hsa:FGF5": dict(pathway_id="hsa04010", pathway_name="MAPK signaling pathway"),
    "hsa:UCP1": dict(pathway_id="hsa00071", pathway_name="Fatty acid degradation"),
}


def make_fake_uniprot(log: list):
    async def fake_fetch_uniprot(gene_symbol: str, tax_id: int):
        log.append((gene_symbol, tax_id))
        row = UNIPROT_DB.get((gene_symbol, tax_id))
        if row is None:
            return None
        from schemas.outputs import ProteinEntry
        return ProteinEntry(**row)
    return fake_fetch_uniprot


def make_fake_kegg(log: list):
    async def fake_fetch_pathway(kegg_gene_id: str):
        log.append(kegg_gene_id)
        row = KEGG_DB.get(kegg_gene_id)
        if row is None:
            return None
        from schemas.outputs import PathwayEntry
        return PathwayEntry(**row)
    return fake_fetch_pathway


class _Patch:
    """Tiny manual monkeypatch — swap an attribute, restore it on exit."""

    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value

    def __enter__(self):
        self._orig = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self._orig)


def _embed_counter():
    """Wraps the REAL embed_text so we can count calls without changing behavior."""
    calls: list[str] = []
    real_embed_text = qdrant_store.embed_text

    async def counting_embed_text(text: str):
        calls.append(text)
        return await real_embed_text(text)

    return calls, counting_embed_text


async def _purge(collections=COLLECTIONS) -> None:
    from qdrant_client.models import Filter, FilterSelector
    client = get_client()
    for name in collections:
        try:
            await client.delete(collection_name=name, points_selector=FilterSelector(filter=Filter()))
        except Exception as exc:  # pragma: no cover - collection may not exist yet
            logger.debug("purge skipped for %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Live orchestration trace (identical spirit to workflows/scenario_runner.py,
# scoped to the functional-evidence subgraph)
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "pathways": "Pathways Agent (KEGG)",
    "fetch_protein_data": "Protein Data Agent (UniProt)",
    "merge": "Merge & Status",
}


def _status_str(v):
    return v.value if isinstance(v, AgentStatus) else str(v)


def _print_node(step: int, node_name: str, payload: dict) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    header = f"STEP {step} — {label}"
    print(f"\n{header}\n" + "-" * len(header))
    if node_name == "pathways":
        print(f"  status: {_status_str(payload['pathways_status'])}")
        for p in payload.get("pathway_data", []):
            print(f"    pathway: {p.pathway_name} ({p.pathway_id})")
    elif node_name == "fetch_protein_data":
        print(f"  status: {_status_str(payload['protein_data_status'])}")
        for p in payload.get("protein_data", []):
            print(f"    protein: {p.protein_name} [{p.source_accession}]")
    elif node_name == "merge":
        print(f"  final status: {_status_str(payload['status'])}")
    else:  # pragma: no cover
        print(f"  {payload}")


async def run_orchestration(gene_list, context, verbose: bool) -> tuple[dict, list, list, list]:
    """Streams the real sub-orchestrator graph and returns
    (final_state, node_order, uniprot_log, kegg_log, embed_calls)."""
    uniprot_log: list = []
    kegg_log: list = []
    embed_calls, counting_embed = _embed_counter()

    with _Patch(protein_data_module, "fetch_uniprot", make_fake_uniprot(uniprot_log)), \
         _Patch(pathways_module, "fetch_pathway", make_fake_kegg(kegg_log)), \
         _Patch(qdrant_store, "embed_text", counting_embed):

        app = build_functional_evidence_graph()
        state = FunctionalEvidenceState(
            gene_list=gene_list, instruction="scenario run", context=context,
        )
        accumulated = {f.name: getattr(state, f.name) for f in fields(state)}
        node_order: list[str] = []
        step = 0
        async for update in app.astream(state, stream_mode="updates"):
            (node_name, payload), = update.items()
            step += 1
            node_order.append(node_name)
            if verbose:
                _print_node(step, node_name, payload)
            accumulated.update(payload)

    return accumulated, node_order, uniprot_log, kegg_log, embed_calls


# ---------------------------------------------------------------------------
# Retrieval / dataflow checks, run against the REAL KB after orchestration
# ---------------------------------------------------------------------------
async def _check(label: str, condition: bool, detail: str, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not condition:
        failures.append(f"{label}: {detail}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
async def scenario_cache_dataflow(verbose: bool) -> list:
    """Run the same gene twice through the real graph. Second run must hit the
    real Qdrant cache and must NOT call the real embedding model again."""
    print("\nPurging uniprot_proteins / kegg_pathways ...")
    await _purge(["uniprot_proteins", "kegg_pathways"])

    gene_list = ["FGF5"]
    context = {"tax_id": 9606, "kegg_gene_ids": {"FGF5": "hsa:FGF5"}}

    print("\n=== Run 1 (cold — expect embed) ===")
    r1, _, u1, k1, e1 = await run_orchestration(gene_list, context, verbose)
    print(f"  external calls -> uniprot: {len(u1)}, kegg: {len(k1)}, embeds so far: {len(e1)}")

    print("\n=== Run 2 (repeat — expect cache hit, no re-embed) ===")
    r2, _, u2, k2, e2 = await run_orchestration(gene_list, context, verbose)
    print(f"  external calls -> uniprot: {len(u2)}, kegg: {len(k2)}, embeds this run: {len(e2)}")

    failures: list = []
    await _check("uniprot re-fetched on repeat", len(u2) == 1,
                 f"fetch_uniprot has no request-level cache, expected 1 call, got {len(u2)}", failures)
    await _check("kegg re-fetched on repeat", len(k2) == 1,
                 f"fetch_pathway has no request-level cache, expected 1 call, got {len(k2)}", failures)
    await _check("no re-embedding on repeat", len(e2) == 0,
                 f"payload unchanged, upsert_point should skip embed_text; embed_calls={len(e2)}", failures)
    await _check("status COMPLETED on both runs",
                 r1["status"] == r2["status"] == AgentStatus.COMPLETED,
                 f"run1={_status_str(r1['status'])} run2={_status_str(r2['status'])}", failures)
    return failures


async def scenario_retrieval_round_trip(verbose: bool) -> list:
    """Run the graph once, then confirm the documents it wrote are actually
    retrievable via semantic_search and pass schema validation."""
    print("\nPurging kegg_pathways ...")
    await _purge(["kegg_pathways"])

    gene_list = ["FGF5", "UCP1"]
    context = {"tax_id": 9606, "kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"}}
    result, _, _, _, _ = await run_orchestration(gene_list, context, verbose)

    print("\nSemantic search: 'MAPK signaling pathway' against kegg_pathways ...")
    hits = await semantic_search("kegg_pathways", "MAPK signaling pathway", top_k=3)

    failures: list = []
    await _check("workflow output completed", result["status"] == AgentStatus.COMPLETED,
                 _status_str(result["status"]), failures)
    await _check("hits returned", bool(hits), f"{len(hits)} hit(s)", failures)
    await _check("FGF5 pathway document present",
                 any(h.payload.get("gene_symbol") == "FGF5" for h in hits),
                 f"gene_symbols seen: {[h.payload.get('gene_symbol') for h in hits]}", failures)
    for hit in hits:
        v = validate_document("kegg_pathways", hit.payload)
        await _check(f"schema valid for {hit.payload.get('dedup_key')}", v.is_valid,
                     f"missing={v.missing_fields} warnings={v.warnings}", failures)
    return failures


async def scenario_metadata_isolation(verbose: bool) -> list:
    """Run the graph for two different genes, then confirm a filtered search
    for one gene never returns the other gene's document."""
    print("\nPurging uniprot_proteins ...")
    await _purge(["uniprot_proteins"])

    print("\n=== Run for FGF5 ===")
    await run_orchestration(["FGF5"], {"tax_id": 9606, "kegg_gene_ids": {}}, verbose)
    print("\n=== Run for UCP1 ===")
    await run_orchestration(["UCP1"], {"tax_id": 9606, "kegg_gene_ids": {}}, verbose)

    print("\nFiltered semantic search: gene_symbol=UCP1 on uniprot_proteins ...")
    hits = await semantic_search("uniprot_proteins", "protein function", top_k=5,
                                  filters={"gene_symbol": "UCP1"})

    failures: list = []
    await _check("hits returned for UCP1 filter", bool(hits), f"{len(hits)} hit(s)", failures)
    await _check("no cross-gene leakage",
                 all(h.payload.get("gene_symbol") == "UCP1" for h in hits),
                 f"gene_symbols seen: {[h.payload.get('gene_symbol') for h in hits]}", failures)
    return failures


async def scenario_unresolved_gene(verbose: bool) -> list:
    """A gene the fake KEGG client can't resolve must write nothing to the KB."""
    print("\nPurging kegg_pathways ...")
    await _purge(["kegg_pathways"])

    gene_list = ["NOT_A_REAL_GENE"]
    context = {"kegg_gene_ids": {"NOT_A_REAL_GENE": "hsa:doesnotexist"}}
    result, _, _, _, _ = await run_orchestration(gene_list, context, verbose)

    hits = await semantic_search("kegg_pathways", "NOT_A_REAL_GENE", top_k=5)

    failures: list = []
    await _check("pathways status FAILED", result["pathways_status"] == AgentStatus.FAILED,
                 _status_str(result["pathways_status"]), failures)
    await _check("nothing written to the KB for the unresolved gene",
                 not any(h.payload.get("gene_symbol") == "NOT_A_REAL_GENE" for h in hits),
                 f"{len(hits)} hit(s) returned", failures)
    return failures


SCENARIOS = {
    "cache-dataflow": (
        "Real graph run twice with identical inputs — verifies dedup_key/text_hash "
        "caching skips re-embedding on the second run without skipping the external fetch.",
        scenario_cache_dataflow,
    ),
    "retrieval-round-trip": (
        "Real graph writes to the KB, then a real semantic_search retrieves and "
        "schema-validates what was just written.",
        scenario_retrieval_round_trip,
    ),
    "metadata-isolation": (
        "Real graph run for two different genes — a filtered semantic_search for one "
        "gene must never leak the other gene's documents.",
        scenario_metadata_isolation,
    ),
    "unresolved-gene": (
        "A gene the (faked) external API can't resolve must reach FAILED status and "
        "write nothing to the KB.",
        scenario_unresolved_gene,
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
    await ensure_collections()

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
        description="Execute real orchestration + real Qdrant retrieval scenarios."
    )
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS) + ["all"], default="all",
        help="Which scenario to run (default: all).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print a live, step-by-step node trace of the real graph execution.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.scenario, args.verbose)))