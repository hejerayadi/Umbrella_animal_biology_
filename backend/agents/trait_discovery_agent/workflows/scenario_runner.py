"""
Scenario runner for the Trait Discovery Agent.

Every scenario below invokes the *real* compiled LangGraph graph
(`build_trait_discovery_graph()`, unmodified) and, wherever the workflow
normally would, makes a *real* call to the NVIDIA NIM model — the Capability
Resolver and Explanation Writer are never monkeypatched or mocked here.
NVIDIA_NIM_API_KEY (or NVIDIA_NIM_API_KEY) must be set in the environment.

This module only *observes* execution via `app.astream(state,
stream_mode="updates")`, which yields the return value of each node the
instant it finishes, in the order the graph actually visits them. Nothing
about routing, nodes, or edges is touched — this is purely a trace printer
built on top of them.

CLI entry point is in `workflows/__main__.py`:
    python -m workflows --scenario success
    python -m workflows --scenario missing-genes
    python -m workflows --scenario literature-escalation
    python -m workflows --scenario unknown-genes
"""
import time
from dataclasses import fields

from workflows.agent_catalog import load_agent_cards
from workflows.state import TraitDiscoveryState
from workflows.trait_discovery_graph import build_trait_discovery_graph
from schemas.common import AgentStatus

# ---------------------------------------------------------------------------
# Scenario definitions — inputs only. All behavior/decisions come from the
# real mocked data layer (subagents/*) and the real LLM calls, not from here.
# ---------------------------------------------------------------------------
SCENARIOS = {
    "success": dict(
        description="Normal successful run — full pipeline, ends in an LLM-written explanation.",
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    ),
    "missing-genes": dict(
        description='No gene_list in context — forces the Capability Resolver to pick an agent.',
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={},
    ),
    "literature-escalation": dict(
        description="Literature evidence for this trait is thin — forces escalation to Literature Agent.",
        trait_name="cold adaptation",
        species_name="mouse",
        instruction="Which genes drive cold adaptation?",
        context={"gene_list": ["UCP1"]},
    ),
    "unknown-genes": dict(
        description="Gene not present in any mock database — Gene Mapper fails deterministically, no LLM call.",
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["ZZZTOP"]},
    ),
}

NODE_LABELS = {
    "check_gene_list": "Check Gene List",
    "gene_mapper": "Gene Mapper",
    "functional_evidence": "Functional Evidence (Pathways + Protein Data)",
    "literature_support": "Literature Support",
    "join_and_route": "Join & Route",
    "escalate_genome_agent": "Capability Resolver → Genome Agent escalation",
    "escalate_literature_agent": "Capability Resolver → Literature Agent escalation",
    "aggregate": "Explanation Writer",
    "failed": "Failed",
}

# Nodes that make a real call to the NVIDIA NIM model.
LLM_NODES = {"escalate_genome_agent", "escalate_literature_agent", "aggregate"}


def _status_str(value) -> str:
    return value.value if isinstance(value, AgentStatus) else str(value)


def _print_step(step_num: int, node_name: str, before: dict, update: dict) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    is_llm = node_name in LLM_NODES
    header = f"STEP {step_num} — {label}" + (" (NVIDIA NIM)" if is_llm else "")
    print(f"\n{header}")
    print("─" * len(header))

    if node_name == "check_gene_list":
        genes = before.get("gene_list") or []
        print("Input:")
        print(f"  gene_list from context: {genes or '(none provided)'}")
        print("Status:")
        print(f"  {_status_str(update.get('status', AgentStatus.COMPLETED))}")

    elif node_name == "gene_mapper":
        print("Input:")
        for g in before.get("gene_list", []):
            print(f"  {g}")
        print("Status:")
        print(f"  {_status_str(update['gene_mapper_status'])}")
        annotations = update.get("go_annotations", [])
        print("Output:")
        print(f"  {len(annotations)} GO annotation(s)")
        for a in annotations:
            print(f"    - {a.gene_symbol}: {a.go_name} ({a.go_id})")

    elif node_name == "functional_evidence":
        print("Input:")
        print(f"  gene_list: {before.get('gene_list')}")
        print("Status:")
        print(f"  {_status_str(update['functional_evidence_status'])}")
        pathways = update.get("pathway_data", [])
        proteins = update.get("protein_data", [])
        print("Output:")
        print(f"  {len(pathways)} pathway(s), {len(proteins)} protein(s)")
        for p in pathways:
            print(f"    - pathway: {p.pathway_name} ({p.pathway_id})")
        for p in proteins:
            print(f"    - protein: {p.protein_name}")

    elif node_name == "literature_support":
        print("Input:")
        print(f"  trait: {before.get('trait_name')}, genes: {before.get('gene_list')}")
        print("Status:")
        print(f"  {_status_str(update['literature_status'])}")
        evidence = update.get("evidence", [])
        print("Output:")
        print(f"  {len(evidence)} record(s)")
        for e in evidence:
            print(f"    - PMID {e.pmid}: {e.title} ({e.year})")
        if update.get("_literature_prompt"):
            print("Flagged as thin evidence — escalation will be needed:")
            print(f"  {update['_literature_prompt']}")

    elif node_name == "join_and_route":
        print(f"Routing decision -> status = {_status_str(update['status'])}")

    elif node_name in ("escalate_genome_agent", "escalate_literature_agent"):
        waiting_agent = "Trait Discovery Agent" if node_name == "escalate_genome_agent" else "Literature Support"
        catalog_names = [c.name for c in load_agent_cards() if c.name != waiting_agent]
        print("Waiting agent:")
        print(f"  {waiting_agent}")
        print("\nLoading agent catalog...")
        print("Available capabilities:")
        for name in catalog_names:
            print(f"  • {name}")
        print("\nLLM Decision")
        print("------------")
        print("Target agent:")
        print(f"  {update['target_agent']}")
        print("Reason:")
        print(f"  {update.get('resolution_reasoning', '(no reasoning returned)')}")
        print("Prompt generated for target agent:")
        print(f"  {update['prompt_to_target_agent']}")
        print("\nFinal status:")
        print(f"  {_status_str(update['status'])}")

    elif node_name == "aggregate":
        genes = ", ".join(a.gene_symbol for a in before.get("go_annotations", []))
        pathways = ", ".join(p.pathway_name for p in before.get("pathway_data", []))
        proteins = ", ".join(p.protein_name for p in before.get("protein_data", []))
        evidence = "; ".join(e.short_summary for e in before.get("evidence", []))
        print("Input handed to Explanation Writer:")
        print(f"  genes:    {genes or 'none'}")
        print(f"  pathways: {pathways or 'none'}")
        print(f"  proteins: {proteins or 'none'}")
        print(f"  evidence: {evidence or 'none'}")
        print("\nLLM-generated explanation:")
        print(f"  {update['explanation']}")
        print("\nFinal status:")
        print(f"  {_status_str(update['status'])}")

    elif node_name == "failed":
        print("Workflow terminated — unrecoverable failure, no LLM call made.")
        print("Final status:")
        print(f"  {_status_str(update['status'])}")

    else:  # pragma: no cover - defensive fallback if the graph gains a node
        print(update)


def _print_summary(node_order: list[str], target_agent, final_status, elapsed: float) -> None:
    llm_used = [n for n in node_order if n in LLM_NODES]
    print("\n" + "=" * 50)
    print("WORKFLOW SUMMARY")
    print("=" * 50)
    print("Path taken:")
    print("  " + " → ".join(NODE_LABELS.get(n, n) for n in node_order))
    print("\nNodes executed:")
    for n in node_order:
        tag = "LLM call — NVIDIA NIM" if n in LLM_NODES else "deterministic"
        print(f"  - {NODE_LABELS.get(n, n)}  [{tag}]")
    print("\nLLM calls made:")
    if llm_used:
        for n in llm_used:
            print(f"  - {NODE_LABELS.get(n, n)}")
    else:
        print("  (none — deterministic routing handled this run end to end)")
    print("\nAgent selected by Capability Resolver:")
    print(f"  {target_agent or '(none — no escalation needed)'}")
    print("\nFinal status:")
    print(f"  {_status_str(final_status)}")
    print("\nExecution time:")
    print(f"  {elapsed:.2f}s")
    print("=" * 50)


async def run_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{name}'. Choices: {', '.join(sorted(SCENARIOS))}")

    spec = SCENARIOS[name]
    print("\n" + "#" * 60)
    print(f"# SCENARIO: {name}")
    print(f"# {spec['description']}")
    print("#" * 60)

    app = build_trait_discovery_graph()
    state = TraitDiscoveryState(
        trait_name=spec["trait_name"],
        species_name=spec["species_name"],
        instruction=spec["instruction"],
        context=spec["context"],
    )

    accumulated = {f.name: getattr(state, f.name) for f in fields(state)}
    node_order: list[str] = []
    step_num = 0

    start = time.perf_counter()
    async for update in app.astream(state, stream_mode="updates"):
        (node_name, payload), = update.items()
        step_num += 1
        node_order.append(node_name)
        _print_step(step_num, node_name, accumulated, payload)
        accumulated.update(payload)
    elapsed = time.perf_counter() - start

    _print_summary(node_order, accumulated.get("target_agent"), accumulated.get("status"), elapsed)
    return accumulated
