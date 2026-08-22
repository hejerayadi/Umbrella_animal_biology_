"""
Genome Agent — Full Integration Demo
(sub-orchestrator <-> agent communication + retrieval + end-to-end data flow)

Unlike the other two scripts in this folder, which each isolate one layer:

    - run_ncbi_live_check.py        -> subagents only, called directly
    - run_orchestrator_scenarios.py -> the full graph, printed as a trace

...this script wires ALL of it together in one run and *asserts* that data
actually flows from one layer to the next, instead of just printing it. It
exists to answer three questions in one go:

  1. RETRIEVAL WITHIN AGENT WORKFLOWS
     Do the real subagents (species_resolver, genome_metadata,
     gene_annotation) actually hit NCBI eutils and come back with usable
     data, when called the way the orchestrator calls them (not in
     isolation)?

  2. SUB-ORCHESTRATOR -> AGENT COMMUNICATION
     The Genome Agent has no nested LangGraph subgraph the way
     trait_discovery_agent's functional_evidence_graph does, but it has two
     real analogues of "one orchestration layer calling another agent", and
     this script exercises both:
       a) subagents/visualization.py's size_comparison scope is itself a
          tiny orchestrator: it calls resolve_species()/get_genome_metadata()
          AGAIN, live, for a set of reference species, in parallel. That's
          agent-calls-agent retrieval happening *inside* a single node.
       b) workflows/nodes/capability_resolver_node.py is the orchestrator
          escalating: when generate_visualization returns NEEDS_AGENT, the
          capability resolver reads agent_cards/*.json (the agent catalog)
          and picks an external agent (Reconstruction Agent / Protein
          Structure Visualization Agent) to hand off to. This is the
          Genome Agent's actual "sub-orchestrator picks an agent" moment.

  3. END-TO-END DATA FLOW
     Does assembly_id resolved in species_resolver actually reach
     genome_metadata and gene_annotation? Does genome_size_bp/gene_table
     actually reach visualization? Does a NEEDS_AGENT visualization
     actually reach capability_resolver and produce a valid target? Does
     all of it actually reach the final explanation? Each scenario below
     asserts on this instead of eyeballing printed output.

This is a plain executable script (asserts + exit code), not a pytest
file — run it by hand or wire it into CI as a smoke check.

Requires network access to eutils.ncbi.nlm.nih.gov. No NVIDIA_API_KEY is
required: query_router / capability_resolver / explanation_writer all fall
back to deterministic logic when the LLM is unreachable, and this script
works correctly either way (it never asserts on the *source* of a routing
decision, only on its shape and on the data flow around it).

Run everything:
    python -m genome_agent.scripts.run_full_integration_demo

Run one part:
    python -m genome_agent.scripts.run_full_integration_demo --part retrieval
    python -m genome_agent.scripts.run_full_integration_demo --part suborchestrator
    python -m genome_agent.scripts.run_full_integration_demo --part e2e

Use a different species (must be NCBI-resolvable):
    python -m genome_agent.scripts.run_full_integration_demo --species "house mouse"

or, from inside the container:
    python scripts/run_full_integration_demo.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any

# Node-level INFO logs are useful in isolation but drown out the trace below.
logging.getLogger().setLevel(logging.WARNING)

# Every check function below uses relative imports (`from ..subagents...`),
# matching the rest of this package. That only works when the module is
# loaded as part of the `genome_agent` package (`python -m
# genome_agent.scripts.run_full_integration_demo`). When run directly
# (`python scripts/run_full_integration_demo.py` from inside genome_agent/),
# fix up sys.path and __package__ ONCE, up front, so every relative import
# below resolves correctly on the first pass instead of failing partway
# through a check.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    __package__ = "genome_agent.scripts"

_WIDTH = 70


# ---------------------------------------------------------------------------
# Small pass/fail harness — plain asserts, collected instead of raised, so
# one failed check doesn't hide the rest of the demo.
# ---------------------------------------------------------------------------


class Checklist:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, description: str, condition: bool, detail: str = "") -> bool:
        self.results.append((description, condition, detail))
        mark = "PASS" if condition else "FAIL"
        line = f"  [{mark}] {description}"
        if detail and not condition:
            line += f"  -- {detail}"
        print(line)
        return condition

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self) -> str:
        passed = sum(1 for _, ok, _ in self.results if ok)
        return f"{passed}/{len(self.results)} checks passed"


def _header(title: str) -> None:
    print()
    print("=" * _WIDTH)
    print(title)
    print("=" * _WIDTH)


def _kv(label: str, value: Any) -> None:
    print(f"  {label:<24}: {value}")


# ---------------------------------------------------------------------------
# Part 1 — Retrieval within agent workflows
#
# Calls the real subagents in the same sequence the orchestrator uses
# (species -> assembly_id -> metadata + annotation, fanned out from that
# one assembly_id) and asserts the data is real and the id actually
# propagated, rather than each call being independently lucky.
# ---------------------------------------------------------------------------


async def run_retrieval_checks(species_name: str, checklist: Checklist) -> dict[str, Any]:
    from ..subagents.gene_annotation import get_gene_annotation
    from ..subagents.genome_metadata import get_genome_metadata
    from ..subagents.species_resolver import resolve_species

    _header(f"PART 1: Retrieval within agent workflows (species={species_name!r})")

    t0 = time.monotonic()
    species = await resolve_species(species_name)
    _kv("species_resolver", f"{time.monotonic() - t0:.2f}s")
    _kv("  assembly_id", species.get("assembly_id"))
    _kv("  scientific_name", species.get("scientific_name"))

    assembly_id = species.get("assembly_id")
    checklist.check(
        "species_resolver returns a usable assembly_id",
        bool(assembly_id),
        f"got {assembly_id!r}",
    )
    if not assembly_id:
        return {"species": species, "assembly_id": None, "metadata": None, "annotation": None}

    t0 = time.monotonic()
    metadata, annotation = await asyncio.gather(
        get_genome_metadata(assembly_id),
        get_gene_annotation(assembly_id),
    )
    _kv("metadata+annotation (parallel)", f"{time.monotonic() - t0:.2f}s")
    _kv("  genome_size_bp", metadata.get("genome_size_bp"))
    _kv("  gene count", len(annotation.get("gene_list") or []))

    checklist.check(
        "genome_metadata returned a real genome_size_bp for that assembly_id",
        metadata.get("genome_size_bp") is not None,
        f"got {metadata.get('genome_size_bp')!r}",
    )
    checklist.check(
        "gene_annotation returned at least one gene for that assembly_id",
        bool(annotation.get("gene_list")),
        f"got {annotation.get('gene_list')!r}",
    )

    return {"species": species, "assembly_id": assembly_id, "metadata": metadata, "annotation": annotation}


# ---------------------------------------------------------------------------
# Part 2 — Sub-orchestrator <-> agent communication
#
# 2a) visualization.py's size_comparison scope calling OTHER subagents
#     (resolve_species/get_genome_metadata) live, for reference species —
#     an agent calling agents, inside one node.
# 2b) capability_resolver_node escalating to an external agent picked from
#     the agent_cards/*.json catalog when a scope can't be handled locally.
# ---------------------------------------------------------------------------


async def run_suborchestrator_checks(retrieved: dict[str, Any], checklist: Checklist) -> None:
    from ..subagents.visualization import generate_visualization
    from ..workflows.agent_catalog import load_agent_catalog
    from ..workflows.capability_resolver import resolve_capability_fallback

    species = retrieved["species"]
    metadata = retrieved["metadata"]
    assembly_id = retrieved["assembly_id"]

    if not assembly_id or not metadata:
        print("\n(skipping PART 2 — Part 1 did not resolve enough data to build on)")
        checklist.check("PART 2 skipped due to missing Part 1 data", False, "no assembly_id/metadata")
        return

    # --- 2a: agent-calls-agent retrieval inside visualization ---------------
    _header("PART 2a: visualization (agent) calling species_resolver + genome_metadata (agents) live")
    t0 = time.monotonic()
    viz = await generate_visualization(
        scope="size_comparison",
        genome_size_bp=metadata.get("genome_size_bp"),
        assembly_id=assembly_id,
        common_name=species.get("common_name"),
        scientific_name=species.get("scientific_name"),
    )
    _kv("elapsed", f"{time.monotonic() - t0:.2f}s")
    _kv("status", viz.get("status"))
    comparisons = viz.get("comparisons") or []
    for c in comparisons:
        marker = " <- queried species (from Part 1)" if c.get("is_queried_species") else ""
        print(f"    - {c['common_name']}: {c['genome_size_bp'] / 1_000_000_000:.2f} Gb{marker}")

    checklist.check(
        "size_comparison completed",
        viz.get("status") == "COMPLETED",
        f"got status={viz.get('status')!r}",
    )
    checklist.check(
        "size_comparison pulled in >=1 reference species via its own internal "
        "resolve_species()/get_genome_metadata() calls (agent-calls-agent)",
        len(comparisons) >= 2,  # queried species + at least 1 live-resolved peer
        f"got {len(comparisons)} rows",
    )
    checklist.check(
        "the queried species from Part 1 is represented in visualization's output",
        any(c.get("is_queried_species") for c in comparisons),
        "no row flagged is_queried_species=True",
    )

    # --- 2b: orchestrator escalating to an external agent via the catalog ---
    _header("PART 2b: capability_resolver — orchestrator escalating to an external agent")
    catalog = load_agent_catalog()
    checklist.check(
        "agent catalog (agent_cards/*.json) loaded and non-empty",
        bool(catalog.strip()),
    )
    print("  Catalog entries found:")
    for line in catalog.splitlines():
        if line.startswith("- "):
            print(f"    {line}")

    # protein_structure is the one local scope that always defers (NEEDS_AGENT)
    protein_viz = await generate_visualization(scope="protein_structure")
    checklist.check(
        "generate_visualization(protein_structure) requests escalation (NEEDS_AGENT)",
        protein_viz.get("status") == "NEEDS_AGENT",
        f"got {protein_viz.get('status')!r}",
    )

    decision = resolve_capability_fallback(
        current_agent="visualization",
        prompt_to_target_agent=protein_viz.get("prompt_to_target_agent", ""),
    )
    _kv("prompt_to_target_agent", protein_viz.get("prompt_to_target_agent"))
    _kv("-> target_agent", decision.target_agent)
    _kv("-> handoff_message", decision.handoff_message)

    checklist.check(
        "capability_resolver picked a real agent from the catalog (not 'none')",
        decision.target_agent.lower() != "none",
        f"got target_agent={decision.target_agent!r}",
    )
    checklist.check(
        "capability_resolver picked the Protein Structure Visualization Agent "
        "specifically (matches the prompt content, not a random catalog entry)",
        decision.target_agent == "Protein Structure Visualization Agent",
        f"got {decision.target_agent!r}",
    )

    # Negative path: a request nothing in the catalog can serve. (Deliberately
    # avoids every keyword/agent-name the fallback matches on — "genome",
    # "protein", "structure", "reconstruct", "literature", "paper", "3d" —
    # so this genuinely exercises the "no agent fits" branch.)
    dead_end = resolve_capability_fallback(
        current_agent="visualization",
        prompt_to_target_agent="Book a hotel room for the field research team.",
    )
    checklist.check(
        "capability_resolver correctly returns 'none' when no catalog agent fits",
        dead_end.target_agent.lower() == "none",
        f"got {dead_end.target_agent!r}",
    )


# ---------------------------------------------------------------------------
# Part 3 — End-to-end data flow through the compiled orchestrator graph
#
# Runs the REAL compiled LangGraph (query_router -> species_resolver ->
# parallel_kickoff -> get_genome_metadata/get_gene_annotation ->
# join_parallel -> generate_visualization -> capability_resolver ->
# explanation_writer -> END) and asserts the same data threaded through
# every hop, plus that the final explanation reflects what happened.
# ---------------------------------------------------------------------------


async def run_end_to_end_checks(species_name: str, checklist: Checklist) -> None:
    from ..orchestrator import GenomeAgentLangGraphOrchestrator

    _header(f"PART 3: End-to-end graph run (species={species_name!r}, scope=protein_structure)")

    orch = GenomeAgentLangGraphOrchestrator()
    t0 = time.monotonic()
    result = await orch.run(
        # Deliberately mentions genome size AND protein structure so the
        # query_router (LLM or fallback) sets needs_metadata=True as well
        # as needs_annotation=True — otherwise a router that (correctly)
        # decides a pure protein-structure question doesn't need genome
        # size will skip get_genome_metadata_node entirely, and the
        # "metadata reached this hop" check below would be asserting on
        # a hop that was never supposed to run. visualization_scope is
        # still passed explicitly, so escalation happens either way.
        user_question=f"Show me the genome size and predict the 3D protein structure for the {species_name}.",
        species_name=species_name,
        visualization_scope="protein_structure",
    )
    elapsed = time.monotonic() - t0
    steps = orch.get_execution_history(result)

    _kv("elapsed", f"{elapsed:.2f}s")
    _kv("execution history", " -> ".join(steps))
    _kv("assembly_id", result.assembly_id)
    _kv("visualization status", (result.visualization or {}).get("status"))
    _kv("visualization target_agent", (result.visualization or {}).get("target_agent"))
    _kv("errors", result.errors)
    print("  explanation:")
    for line in (result.explanation or "").splitlines() or [""]:
        print(f"    {line}")

    checklist.check(
        "graph resolved a real assembly_id for the species",
        bool(result.assembly_id),
        f"got {result.assembly_id!r}",
    )
    checklist.check(
        "IF query_router decided metadata was needed, genome_metadata actually "
        "delivered it (query_router's needs_metadata decision is made by a live "
        "LLM here, so this doesn't assume metadata was requested — only that "
        "it was fetched correctly when it was)",
        (not result.needs_metadata) or bool(result.metadata and result.metadata.get("genome_size_bp")),
        f"needs_metadata={result.needs_metadata}, metadata={result.metadata!r}",
    )
    checklist.check(
        "IF query_router decided annotation was needed, gene_annotation actually delivered it",
        (not result.needs_annotation) or bool(result.annotation and result.annotation.get("gene_list")),
        f"needs_annotation={result.needs_annotation}, annotation={result.annotation!r}",
    )
    checklist.check(
        "generate_visualization escalated (NEEDS_AGENT) and capability_resolver ran",
        (result.visualization or {}).get("target_agent") is not None,
        f"visualization={result.visualization!r}",
    )
    checklist.check(
        "capability_resolver's chosen target_agent is a real catalog entry",
        (result.visualization or {}).get("target_agent")
        == "Protein Structure Visualization Agent",
        f"got {(result.visualization or {}).get('target_agent')!r}",
    )
    checklist.check(
        "the final explanation was produced (explanation_writer ran after escalation)",
        bool(result.explanation),
    )
    checklist.check(
        "execution history shows every hop in order: species -> metadata -> annotation -> visualization",
        steps == ["species_resolver", "get_genome_metadata", "get_gene_annotation", "generate_visualization"],
        f"got {steps}",
    )

    # Second run: unresolvable species -> confirms the graph short-circuits
    # cleanly (error_end) instead of limping through with partial/garbage data.
    _header("PART 3b: End-to-end graph run — unresolvable species (error_end path)")
    result2 = await orch.run(
        user_question="Show me the genome of the dragon.",
        species_name="dragon",
    )
    checklist.check(
        "unresolvable species short-circuits to error_end with no assembly_id",
        result2.assembly_id is None,
    )
    checklist.check(
        "unresolvable species produces no metadata/annotation (nothing ran on garbage data)",
        result2.metadata is None and result2.annotation is None,
    )
    checklist.check(
        "unresolvable species records a clear error message "
        "(either 'not found' from NCBI, or a raised exception if NCBI/network is unreachable)",
        bool(result2.errors),
        f"errors={result2.errors}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", default="tiger", help="Species to use throughout (default: tiger)")
    parser.add_argument(
        "--part",
        choices=["retrieval", "suborchestrator", "e2e", "all"],
        default="all",
        help="Run only one part instead of the full demo.",
    )
    args = parser.parse_args(argv)

    llm_mode = "LIVE (NVIDIA_API_KEY set)" if os.getenv("NVIDIA_API_KEY") else "FALLBACK (no NVIDIA_API_KEY set)"
    print("#" * _WIDTH)
    print("# Genome Agent — Full Integration Demo")
    print("# retrieval + sub-orchestrator<->agent + end-to-end data flow")
    print(f"# LLM mode: {llm_mode}")
    print(f"# Species: {args.species}")
    print("#" * _WIDTH)

    checklist = Checklist()
    retrieved: dict[str, Any] = {}

    if args.part in ("retrieval", "suborchestrator", "all"):
        # suborchestrator checks build on Part 1's data (assembly_id +
        # metadata), so retrieval always runs first when either is selected.
        retrieved = await run_retrieval_checks(args.species, checklist)

    if args.part in ("suborchestrator", "all"):
        await run_suborchestrator_checks(retrieved, checklist)

    if args.part in ("e2e", "all"):
        await run_end_to_end_checks(args.species, checklist)

    _header("SUMMARY")
    print(f"  {checklist.summary()}")
    for description, ok, detail in checklist.results:
        if not ok:
            print(f"  FAILED: {description}" + (f" ({detail})" if detail else ""))
    print("=" * _WIDTH)

    return 0 if checklist.all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))