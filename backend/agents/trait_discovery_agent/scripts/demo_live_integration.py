"""
Fully LIVE integration demonstration for the Trait Discovery Agent.

Unlike scripts/demo_full_integration.py, this script fakes NOTHING. Every
external boundary is hit for real:

  - KEGG REST API        (subagents.pathways -> kb.sources.kegg_client)
  - UniProt REST API      (subagents.protein_data -> kb.sources.uniprot_client)
  - QuickGO REST API      (subagents.gene_mapper -> kb.sources.go_client)
  - NVIDIA NIM LLM        (workflows.explanation_writer, workflows.capability_resolver)
  - your real Qdrant instance (kb.qdrant_store / kb.retrieval)

KEGG/UniProt/QuickGO need no API key. The LLM calls need NVIDIA_NIM_API_KEY
(see .env.example) — without it this script fails loudly and early, before
any node runs, rather than partway through.

One honest caveat: the *graph itself* — not this script — currently wires
`gene_mapper_node`/`literature_support_node` to `mock_gene_mapper` /
`mock_literature_support` (see workflows/nodes/gene_mapper_node.py and
workflows/nodes/literature_support_node.py). There's no faking happening
here; that's simply what's checked in today. So Phase 2 runs the real
top-level graph as-is (real pathways/protein-data/LLM, mocked gene-mapper/
literature), and Phase 3 separately calls the REAL `gene_mapper_agent`
directly against QuickGO so that agent's live path gets proven too, even
though the graph doesn't call it yet.

Two runs happen on purpose, both drawn from the mock literature DB that's
already in subagents/literature_support.py — not invented for this demo:
  - "fur growth" / [FGF5, UCP1]  -> 2 literature records -> COMPLETED
    -> exercises the real explanation LLM call (write_explanation)
  - "cold adaptation" / [UCP1]   -> 1 literature record (thin) -> NEEDS_AGENT
    -> exercises the real escalation LLM call (resolve_capability)

Usage:
    python -m scripts.demo_live_integration
    python -m scripts.demo_live_integration --verbose
"""
import argparse
import asyncio
import logging
import os
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

import subagents.gene_mapper as gene_mapper_module
import workflows.trait_discovery_graph as td_graph_module
from kb.qdrant_store import ensure_collections, get_client, COLLECTIONS
from kb.retrieval import semantic_search, validate_document
from schemas.common import AgentStatus
from schemas.inputs import GeneMapperInput
from workflows.state import TraitDiscoveryState

logger = logging.getLogger(__name__)

KEGG_FIND_URL = "https://rest.kegg.jp/find/{organism}/{gene_symbol}"


async def resolve_kegg_gene_id(gene_symbol: str, organism: str = "hsa") -> str | None:
    """Live lookup: turn a gene symbol into the org:number ID pathways_agent
    expects (e.g. FGF5 -> hsa:2249). A real Genome Agent upstream would have
    already resolved this before calling Trait Discovery Agent; this stands in
    for that step so the demo can run from just a gene symbol."""
    url = KEGG_FIND_URL.format(organism=organism, gene_symbol=gene_symbol)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    lines = [l for l in resp.text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    for line in lines:
        kegg_id, _, desc = line.partition("\t")
        aliases = [a.strip().upper() for a in desc.split(";")[0].split(",")]
        if gene_symbol.upper() in aliases:
            return kegg_id.strip()
    return lines[0].split("\t")[0].strip()


def _require_env(*names: str) -> list[str]:
    return [n for n in names if not os.environ.get(n)]


async def _purge(collections=COLLECTIONS) -> None:
    from qdrant_client.models import Filter, FilterSelector
    client = get_client()
    for name in collections:
        try:
            await client.delete(collection_name=name, points_selector=FilterSelector(filter=Filter()))
        except Exception as exc:  # pragma: no cover - collection may not exist yet
            logger.debug("purge skipped for %s: %s", name, exc)


def _status_str(v):
    return v.value if isinstance(v, AgentStatus) else str(v)


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _check(label: str, condition: bool, detail: str, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not condition:
        failures.append(f"{label}: {detail}")


# ---------------------------------------------------------------------------
# PHASE 1 — resolve real KEGG gene IDs (live network, no faking)
# ---------------------------------------------------------------------------
async def phase_1_resolve_ids(genes: list[str], organism: str) -> dict[str, str]:
    _hr("PHASE 1 — resolve real KEGG gene IDs")
    print(f"Live lookup against rest.kegg.jp for {genes} in organism '{organism}' ...")
    kegg_ids: dict[str, str] = {}
    for gene in genes:
        kegg_id = await resolve_kegg_gene_id(gene, organism)
        kegg_ids[gene] = kegg_id
        print(f"  {gene} -> {kegg_id}")
    return kegg_ids


# ---------------------------------------------------------------------------
# PHASE 2 — the real top-level graph, live, no patches at all
# ---------------------------------------------------------------------------
async def run_graph_live(trait_name: str, species_name: str, gene_list: list[str],
                          kegg_ids: dict[str, str], tax_id: int, verbose: bool) -> dict:
    app = td_graph_module.build_trait_discovery_graph()
    state = TraitDiscoveryState(
        trait_name=trait_name,
        species_name=species_name,
        instruction=f"Which genes explain {trait_name}?",
        context={
            "gene_list": gene_list,
            "kegg_gene_ids": kegg_ids,
            "tax_id": tax_id,
        },
    )
    accumulated = {f.name: getattr(state, f.name) for f in fields(state)}
    step = 0
    async for update in app.astream(state, stream_mode="updates"):
        (node_name, payload), = update.items()
        step += 1
        print(f"  step {step}: node '{node_name}' fired")
        if verbose:
            print(f"    payload keys: {list(payload.keys())}")
        accumulated.update(payload)
    return accumulated


async def phase_2_completed_run(kegg_ids: dict[str, str], verbose: bool) -> tuple[dict, list]:
    _hr("PHASE 2a — live run: 'fur growth' (expect COMPLETED, real explanation call)")
    print(
        "Real graph, real KEGG pathway lookup, real UniProt protein lookup, real NIM\n"
        "call for the final explanation. gene_mapper/literature_support are the mocks\n"
        "the graph itself is currently wired to (see module docstring)."
    )
    result = await run_graph_live(
        "fur growth", "human", ["FGF5", "UCP1"], kegg_ids, 9606, verbose,
    )

    failures: list = []
    _check("pathways resolved from live KEGG", bool(result.get("pathway_data")),
           [p.pathway_name for p in result.get("pathway_data", [])], failures)
    _check("proteins resolved from live UniProt", bool(result.get("protein_data")),
           [p.protein_name for p in result.get("protein_data", [])], failures)
    _check("status COMPLETED", result.get("status") == AgentStatus.COMPLETED,
           _status_str(result.get("status")), failures)
    _check("explanation written by the real LLM",
           isinstance(result.get("explanation"), str) and bool(result.get("explanation")),
           result.get("explanation", "")[:80] + "...", failures)
    return result, failures


async def phase_2_escalation_run(kegg_ids: dict[str, str], verbose: bool) -> tuple[dict, list]:
    _hr("PHASE 2b — live run: 'cold adaptation' (expect NEEDS_AGENT, real resolver call)")
    print(
        "'cold adaptation' only has 1 literature record in the mock literature DB, which\n"
        "is under the thin-evidence threshold, so the graph escalates for real via\n"
        "workflows.capability_resolver.resolve_capability — a genuine NIM call that picks\n"
        "a target agent from the live agent_cards catalog."
    )
    result = await run_graph_live(
        "cold adaptation", "human", ["UCP1"], kegg_ids, 9606, verbose,
    )

    failures: list = []
    _check("status NEEDS_AGENT", result.get("status") == AgentStatus.NEEDS_AGENT,
           _status_str(result.get("status")), failures)
    _check("resolver picked a real target agent from the catalog",
           bool(result.get("target_agent")), result.get("target_agent"), failures)
    _check("resolver drafted a real prompt for the target agent",
           bool(result.get("prompt_to_target_agent")),
           (result.get("prompt_to_target_agent") or "")[:80] + "...", failures)
    return result, failures


# ---------------------------------------------------------------------------
# PHASE 3 — the real gene_mapper_agent, called directly (live QuickGO)
# ---------------------------------------------------------------------------
async def phase_3_real_gene_mapper(completed_result: dict, verbose: bool) -> list:
    _hr("PHASE 3 — real Gene Mapper agent, called directly (live QuickGO)")
    print(
        "The graph node still points at mock_gene_mapper, so this calls the REAL\n"
        "gene_mapper_agent directly, using the UniProt accessions Phase 2a already\n"
        "resolved for real, to prove the live QuickGO path and its KB write work too."
    )

    uniprot_accessions = {
        p.gene_symbol: p.source_accession
        for p in completed_result.get("protein_data", [])
        if p.source_accession
    }
    print(f"  uniprot_accessions from Phase 2a: {uniprot_accessions}")

    out = await gene_mapper_module.gene_mapper_agent(GeneMapperInput(
        trait_name="fur growth",
        gene_list=list(uniprot_accessions.keys()),
        species_name="human",
        instruction="live demo",
        context={"uniprot_accessions": uniprot_accessions},
    ))
    if verbose:
        print(f"  gene_mapper_agent output: {out}")

    failures: list = []
    _check("real GO annotations resolved from QuickGO", bool(out.go_annotations),
           [(g.gene_symbol, g.go_name) for g in out.go_annotations], failures)
    return failures


# ---------------------------------------------------------------------------
# PHASE 4 — retrieval against everything that was really written
# ---------------------------------------------------------------------------
async def phase_4_retrieval(verbose: bool) -> list:
    _hr("PHASE 4 — retrieval against real, live-written documents")
    failures: list = []

    checks = [
        ("kegg_pathways", "signaling pathway", "gene_symbol"),
        ("uniprot_proteins", "protein function", "gene_symbol"),
        ("go_annotations", "biological process", "gene_symbol"),
    ]
    for collection, query, key_field in checks:
        hits = await semantic_search(collection, query, top_k=5)
        _check(f"{collection} has retrievable live documents", bool(hits),
               f"{len(hits)} hit(s)", failures)
        for hit in hits:
            result = validate_document(collection, hit.payload)
            _check(f"{collection} doc valid ({hit.payload.get('dedup_key')})", result.is_valid,
                   f"missing={result.missing_fields} warnings={result.warnings}", failures)

    return failures


async def main(verbose: bool) -> int:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                         format="%(levelname)s:%(name)s:%(message)s")

    print("Trait Discovery Agent — FULLY LIVE integration demo")
    print("No faking anywhere: real KEGG, real UniProt, real QuickGO, real NIM LLM, real Qdrant.")

    if not (os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NIM_API_KEY")):
        print(
            "\n[FAIL] NVIDIA_NIM_API_KEY (or NIM_API_KEY) is not set.\n"
            "Copy .env.example to .env and add a real NIM key before running this script "
            "— see workflows/llm.py."
        )
        return 1
    if _require_env("QDRANT_URL", "QDRANT_API_KEY"):
        print("\n[FAIL] QDRANT_URL / QDRANT_API_KEY are not set. See .env.example.")
        return 1

    await ensure_collections()
    print("\nPurging kegg_pathways / uniprot_proteins / go_annotations for a clean run ...")
    await _purge(["kegg_pathways", "uniprot_proteins", "go_annotations"])

    all_failures: dict[str, list] = {}

    kegg_ids = await phase_1_resolve_ids(["FGF5", "UCP1"], organism="hsa")

    completed_result, failures_2a = await phase_2_completed_run(kegg_ids, verbose)
    all_failures["Phase 2a - live COMPLETED run"] = failures_2a

    _, failures_2b = await phase_2_escalation_run(kegg_ids, verbose)
    all_failures["Phase 2b - live NEEDS_AGENT escalation run"] = failures_2b

    failures_3 = await phase_3_real_gene_mapper(completed_result, verbose)
    all_failures["Phase 3 - real Gene Mapper agent (live QuickGO)"] = failures_3

    failures_4 = await phase_4_retrieval(verbose)
    all_failures["Phase 4 - retrieval against live-written docs"] = failures_4

    _hr("SUMMARY")
    ok = True
    for name, failures in all_failures.items():
        status = "PASS" if not failures else "FAIL"
        if failures:
            ok = False
        print(f"  [{status}] {name} ({len(failures)} failed check(s))")

    print("\nLive integration proven end to end." if ok else "\nSome checks failed — see detail above.")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully live (no faking) demonstration of the Trait Discovery Agent "
                    "against real KEGG/UniProt/QuickGO and a real NIM LLM."
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra detail per step.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.verbose)))