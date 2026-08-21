"""
Full-stack integration demonstration for the Trait Discovery Agent.

This is NOT a pytest suite — it's a runnable, narrated demo that proves the whole
stack actually talks to itself correctly, end to end:

  1. Orchestrator -> sub-orchestrator communication
     (`trait_discovery_graph` invoking `functional_evidence_graph` as one of its
     nodes, and merging the sub-orchestrator's output back into its own state).

  2. Sub-orchestrator -> agent communication
     (`functional_evidence_graph` fanning out to the real `pathways_agent` and
     `protein_data_agent` subagents, and fanning back in at `merge`).

  3. Retrieval within agent workflows
     (the real Qdrant-backed `kb.qdrant_store` / `kb.retrieval` layer: the
     documents the agents just wrote are independently semantic-searched back
     out and schema-validated).

  4. End-to-end data flow
     (a trait/species request goes in one end and a fully-populated
     `TraitDiscoveryOutput`-shaped result — genes, pathways, proteins,
     literature, explanation — comes out the other end).


Usage:
    python -m scripts.demo_full_integration
    python -m scripts.demo_full_integration --verbose
"""
import argparse
import asyncio
import logging
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
import workflows.capability_resolver as resolver_module
import workflows.trait_discovery_graph as td_graph_module
from kb.qdrant_store import ensure_collections, get_client, COLLECTIONS
from kb.retrieval import semantic_search, validate_document
from schemas.common import AgentStatus
from schemas.outputs import GOAnnotation, LiteratureRecord, PathwayEntry, ProteinEntry
from workflows.capability_resolver import CapabilityResolution
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.state import FunctionalEvidenceState, TraitDiscoveryState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The only faked boundaries: outbound HTTP (KEGG/UniProt) and outbound LLM
# (NIM). Everything downstream of these two calls is real production code.
# ---------------------------------------------------------------------------
KEGG_DB = {
    "hsa:FGF5": dict(pathway_id="hsa04010", pathway_name="MAPK signaling pathway"),
    "hsa:UCP1": dict(pathway_id="hsa00071", pathway_name="Fatty acid degradation"),
}
UNIPROT_DB = {
    ("FGF5", 9606): dict(gene_symbol="FGF5", protein_name="Fibroblast growth factor 5",
                          function_summary="Regulates hair follicle growth cycle.",
                          source_accession="P12034"),
    ("UCP1", 9606): dict(gene_symbol="UCP1", protein_name="Uncoupling protein 1",
                          function_summary="Mitochondrial proton channel, generates heat.",
                          source_accession="P25874"),
}


def make_fake_kegg(log: list):
    async def fake_fetch_pathway(kegg_gene_id: str):
        log.append(kegg_gene_id)
        row = KEGG_DB.get(kegg_gene_id)
        return PathwayEntry(**row) if row else None
    return fake_fetch_pathway


def make_fake_uniprot(log: list):
    async def fake_fetch_uniprot(gene_symbol: str, tax_id: int):
        log.append((gene_symbol, tax_id))
        row = UNIPROT_DB.get((gene_symbol, tax_id))
        return ProteinEntry(**row) if row else None
    return fake_fetch_uniprot


def make_fake_explain(log: list):
    async def fake_write_explanation(**kwargs):
        log.append(kwargs)
        return (
            f"Demo explanation: {kwargs['trait_name']} in {kwargs['species_name']} "
            f"is linked to genes [{kwargs['genes']}], pathways [{kwargs['pathways']}], "
            f"proteins [{kwargs['proteins']}], supported by literature [{kwargs['evidence']}]."
        )
    return fake_write_explanation


def make_fake_resolve(log: list):
    async def fake_resolve_capability(*, waiting_agent, need_description, known_context):
        log.append((waiting_agent, need_description))
        target = "Literature Agent" if waiting_agent == "Literature Support" else "Genome Agent"
        return CapabilityResolution(
            target_agent=target,
            prompt_to_target_agent=f"[demo escalation] {need_description}",
            reasoning="stubbed LLM boundary for demo run",
        )
    return fake_resolve_capability


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
# PHASE 1 — orchestrator -> sub-orchestrator communication
# ---------------------------------------------------------------------------
async def phase_1_orchestrator_to_suborchestrator(verbose: bool) -> tuple[dict, list]:
    _hr("PHASE 1 — orchestrator -> sub-orchestrator communication")
    print(
        "Running the REAL top-level graph (build_trait_discovery_graph). When its\n"
        "'functional_evidence' node fires, it hands gene_list/instruction/context to a\n"
        "freshly-invoked, real, compiled sub-orchestrator (build_functional_evidence_graph)\n"
        "and folds that sub-orchestrator's returned state straight back into its own."
    )

    app = td_graph_module.build_trait_discovery_graph()
    state = TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={
            "gene_list": ["FGF5", "UCP1"],
            "kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"},
            "tax_id": 9606,
        },
    )

    accumulated = {f.name: getattr(state, f.name) for f in fields(state)}
    node_order: list[str] = []
    step = 0
    async for update in app.astream(state, stream_mode="updates"):
        (node_name, payload), = update.items()
        step += 1
        node_order.append(node_name)
        marker = " <- sub-orchestrator merged in here" if node_name == "functional_evidence" else ""
        print(f"  step {step}: node '{node_name}' fired{marker}")
        if verbose:
            print(f"    payload keys: {list(payload.keys())}")
        accumulated.update(payload)

    failures: list = []
    _check("top-level graph visited 'functional_evidence' node",
           "functional_evidence" in node_order, f"node_order={node_order}", failures)
    _check("sub-orchestrator's pathway_data merged into parent state",
           bool(accumulated.get("pathway_data")),
           f"{len(accumulated.get('pathway_data', []))} pathway(s)", failures)
    _check("sub-orchestrator's protein_data merged into parent state",
           bool(accumulated.get("protein_data")),
           f"{len(accumulated.get('protein_data', []))} protein(s)", failures)
    _check("sub-orchestrator status surfaced on parent state",
           accumulated.get("functional_evidence_status") == AgentStatus.COMPLETED,
           _status_str(accumulated.get("functional_evidence_status")), failures)

    return accumulated, failures


# ---------------------------------------------------------------------------
# PHASE 2 — sub-orchestrator -> agent communication
# ---------------------------------------------------------------------------
async def phase_2_suborchestrator_to_agent(verbose: bool) -> list:
    _hr("PHASE 2 — sub-orchestrator -> agent communication")
    print(
        "Running the REAL sub-orchestrator standalone (same compiled graph object the\n"
        "top-level node uses internally) and streaming it node-by-node, so the fan-out to\n"
        "the two real agent functions (pathways_agent, protein_data_agent) and the\n"
        "fan-in at 'merge' are both directly visible."
    )

    app = build_functional_evidence_graph()
    state = FunctionalEvidenceState(
        gene_list=["FGF5", "UCP1"],
        instruction="collect functional evidence",
        context={
            "kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"},
            "tax_id": 9606,
        },
    )

    accumulated = {f.name: getattr(state, f.name) for f in fields(state)}
    node_order: list[str] = []
    step = 0
    async for update in app.astream(state, stream_mode="updates"):
        (node_name, payload), = update.items()
        step += 1
        node_order.append(node_name)
        print(f"  step {step}: agent node '{node_name}' fired")
        if verbose:
            print(f"    payload: {payload}")
        accumulated.update(payload)

    failures: list = []
    _check("both agents fanned out from START",
           {"pathways", "fetch_protein_data"} <= set(node_order),
           f"node_order={node_order}", failures)
    _check("merge ran after both agents", node_order[-1] == "merge",
           f"last node={node_order[-1]}", failures)
    _check("pathways_agent returned real data",
           {p.pathway_id for p in accumulated["pathway_data"]} == {"hsa04010", "hsa00071"},
           [p.pathway_id for p in accumulated["pathway_data"]], failures)
    _check("protein_data_agent returned real data",
           {p.gene_symbol for p in accumulated["protein_data"]} == {"FGF5", "UCP1"},
           [p.gene_symbol for p in accumulated["protein_data"]], failures)

    return failures


# ---------------------------------------------------------------------------
# PHASE 3 — retrieval within agent workflows
# ---------------------------------------------------------------------------
async def phase_3_retrieval(verbose: bool) -> list:
    _hr("PHASE 3 — retrieval within agent workflows")
    print(
        "The pathways_agent/protein_data_agent calls in Phases 1-2 already wrote real\n"
        "points into Qdrant (kb.qdrant_store.upsert_point). This phase independently\n"
        "reads them back with kb.retrieval.semantic_search and checks the payload shape\n"
        "with kb.retrieval.validate_document — proving retrieval works against what the\n"
        "agents actually produced, not against seeded fixtures."
    )

    failures: list = []

    print("\n  semantic_search('MAPK signaling pathway') on kegg_pathways ...")
    kegg_hits = await semantic_search("kegg_pathways", "MAPK signaling pathway", top_k=3)
    _check("kegg_pathways hits returned", bool(kegg_hits), f"{len(kegg_hits)} hit(s)", failures)
    _check("FGF5 pathway document retrievable",
           any(h.payload.get("gene_symbol") == "FGF5" for h in kegg_hits),
           [h.payload.get("gene_symbol") for h in kegg_hits], failures)

    print("  semantic_search('mitochondrial proton channel') on uniprot_proteins ...")
    uniprot_hits = await semantic_search("uniprot_proteins", "mitochondrial proton channel", top_k=3)
    _check("uniprot_proteins hits returned", bool(uniprot_hits), f"{len(uniprot_hits)} hit(s)", failures)
    _check("UCP1 protein document retrievable",
           any(h.payload.get("gene_symbol") == "UCP1" for h in uniprot_hits),
           [h.payload.get("gene_symbol") for h in uniprot_hits], failures)

    print("  schema-validating every retrieved document ...")
    for hit in kegg_hits:
        result = validate_document("kegg_pathways", hit.payload)
        _check(f"kegg_pathways doc valid ({hit.payload.get('dedup_key')})", result.is_valid,
               f"missing={result.missing_fields} warnings={result.warnings}", failures)
    for hit in uniprot_hits:
        result = validate_document("uniprot_proteins", hit.payload)
        _check(f"uniprot_proteins doc valid ({hit.payload.get('dedup_key')})", result.is_valid,
               f"missing={result.missing_fields} warnings={result.warnings}", failures)

    print("\n  filtered retrieval: gene_symbol=UCP1 must not leak FGF5 ...")
    filtered = await semantic_search("uniprot_proteins", "protein function", top_k=5,
                                      filters={"gene_symbol": "UCP1"})
    _check("filtered search returns only UCP1",
           bool(filtered) and all(h.payload.get("gene_symbol") == "UCP1" for h in filtered),
           [h.payload.get("gene_symbol") for h in filtered], failures)

    return failures


# ---------------------------------------------------------------------------
# PHASE 4 — end-to-end data flow validation
# ---------------------------------------------------------------------------
async def phase_4_end_to_end_data_flow(explain_log: list, verbose: bool) -> list:
    _hr("PHASE 4 — end-to-end data flow validation")
    print(
        "Running the full pipeline one more time, start to finish, and checking that the\n"
        "final state matches TraitDiscoveryOutput field-for-field: trait/species in one\n"
        "end, genes -> pathways/proteins -> literature -> explanation out the other."
    )

    calls_before = len(explain_log)

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={
            "gene_list": ["FGF5", "UCP1"],
            "kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"},
            "tax_id": 9606,
        },
    ))

    calls_this_phase = len(explain_log) - calls_before

    if verbose:
        print(f"\n  final explanation: {result['explanation']!r}")

    failures: list = []
    _check("gene_list carried through from context", result["gene_list"] == ["FGF5", "UCP1"],
           result["gene_list"], failures)
    _check("go_annotations populated with the right type",
           bool(result["go_annotations"]) and all(isinstance(g, GOAnnotation) for g in result["go_annotations"]),
           len(result["go_annotations"]), failures)
    _check("pathway_data populated with the right type",
           bool(result["pathway_data"]) and all(isinstance(p, PathwayEntry) for p in result["pathway_data"]),
           len(result["pathway_data"]), failures)
    _check("protein_data populated with the right type",
           bool(result["protein_data"]) and all(isinstance(p, ProteinEntry) for p in result["protein_data"]),
           len(result["protein_data"]), failures)
    _check("evidence populated with the right type",
           bool(result["evidence"]) and all(isinstance(e, LiteratureRecord) for e in result["evidence"]),
           len(result["evidence"]), failures)
    _check("explanation is non-empty prose", isinstance(result["explanation"], str) and bool(result["explanation"]),
           result["explanation"][:60] + "...", failures)
    _check("overall status COMPLETED", result["status"] == AgentStatus.COMPLETED,
           _status_str(result["status"]), failures)
    _check("aggregate ran exactly once this phase", calls_this_phase == 1,
           f"{calls_this_phase} call(s)", failures)

    return failures


async def main(verbose: bool) -> int:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                         format="%(levelname)s:%(name)s:%(message)s")

    print("Trait Discovery Agent — full-stack integration demo")
    print("Real orchestrator + real sub-orchestrator + real agents + real Qdrant KB.")
    print("Faked: outbound HTTP to KEGG/UniProt, outbound calls to the NIM LLM.")

    await ensure_collections()
    print("\nPurging kegg_pathways / uniprot_proteins for a clean, deterministic run ...")
    await _purge(["kegg_pathways", "uniprot_proteins"])

    kegg_log: list = []
    uniprot_log: list = []
    explain_log: list = []
    resolve_log: list = []

    all_failures: dict[str, list] = {}

    with _Patch(pathways_module, "fetch_pathway", make_fake_kegg(kegg_log)), \
         _Patch(protein_data_module, "fetch_uniprot", make_fake_uniprot(uniprot_log)), \
         _Patch(td_graph_module, "write_explanation", make_fake_explain(explain_log)), \
         _Patch(resolver_module, "resolve_capability", make_fake_resolve(resolve_log)):

        _, failures_1 = await phase_1_orchestrator_to_suborchestrator(verbose)
        all_failures["Phase 1 - orchestrator -> sub-orchestrator"] = failures_1

        failures_2 = await phase_2_suborchestrator_to_agent(verbose)
        all_failures["Phase 2 - sub-orchestrator -> agent"] = failures_2

        failures_3 = await phase_3_retrieval(verbose)
        all_failures["Phase 3 - retrieval within agent workflows"] = failures_3

        failures_4 = await phase_4_end_to_end_data_flow(explain_log, verbose)
        all_failures["Phase 4 - end-to-end data flow"] = failures_4

    _hr("SUMMARY")
    print(f"external calls made -> kegg: {len(kegg_log)}, uniprot: {len(uniprot_log)}, "
          f"llm explain: {len(explain_log)}, llm resolve: {len(resolve_log)}")
    ok = True
    for name, failures in all_failures.items():
        status = "PASS" if not failures else "FAIL"
        if failures:
            ok = False
        print(f"  [{status}] {name} ({len(failures)} failed check(s))")

    print("\nIntegration proven end to end." if ok else "\nSome checks failed — see detail above.")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Narrated, real-code demonstration of orchestrator -> sub-orchestrator "
                    "-> agent -> retrieval integration for the Trait Discovery Agent."
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra detail per step.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.verbose)))