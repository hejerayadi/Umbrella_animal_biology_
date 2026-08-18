"""
Scenario runner for the Genome Agent LangGraph orchestrator.

Unlike the pytest suite, this is a plain executable script: it builds the
*real* compiled graph (`build_genome_graph`) via `GenomeAgentLangGraphOrchestrator`
and runs it end to end against a handful of scenarios, printing a readable
trace of what happened at each step. Nothing is mocked at the node/routing
level — the only thing that varies run to run is whether NVIDIA_API_KEY is
set:

    - If it IS set, query_router / capability_resolver / explanation_writer
      call the real LLM.
    - If it is NOT set, each of those three modules catches the failure
      internally and falls back to its deterministic keyword-based logic
      (route_query_fallback / resolve_capability_fallback / raw findings
      text) — so the workflow still runs end to end without an API key.

Run every scenario:
    python -m genome_agent.scripts.run_orchestrator_scenarios

Run just one (handy on a free-tier key, or while debugging one branch):
    python -m genome_agent.scripts.run_orchestrator_scenarios --scenario happy-path

Run a few at once:
    python -m genome_agent.scripts.run_orchestrator_scenarios --scenario happy-path reconstruction-scaffold

List available scenario names:
    python -m genome_agent.scripts.run_orchestrator_scenarios --list

or, from inside the container:
    python scripts/run_orchestrator_scenarios.py --scenario reconstruction-scaffold
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

# Node-level INFO logs are useful in isolation but drown out the trace below.
logging.getLogger().setLevel(logging.WARNING)

_WIDTH = 64


@dataclass
class Scenario:
    slug: str
    title: str
    run_kwargs: dict[str, Any]
    workflow_lines: list[str]
    # When set, get_genome_metadata is wrapped to force this assembly_level
    # into the real NCBI response.  Use this for reconstruction scenarios
    # where the live assembly may have been upgraded by NCBI since the
    # scenario was written (e.g. axolotl went Scaffold → Chromosome).
    forced_assembly_level: str | None = None

    @property
    def input_lines(self) -> list[str]:
        lines = [f"Species: {self.run_kwargs['species_name']}"]
        scope = self.run_kwargs.get("visualization_scope")
        if scope is not None:
            lines.append(f"Visualization scope: {scope}")
        if self.forced_assembly_level:
            lines.append(f"[forced assembly_level: {self.forced_assembly_level}]")
        return lines


@contextlib.asynccontextmanager
async def _force_assembly_level(level: str):
    """
    Wrap get_genome_metadata so it always returns the given assembly_level,
    while leaving every other field (genome_size_bp, chromosome_count, …)
    as returned by the real NCBI call.

    This makes reconstruction scenarios deterministic regardless of whether
    NCBI has since upgraded the assembly.
    """
    # Patch the name inside the node module, not the subagent module.
    # The node does `from ...subagents.genome_metadata import get_genome_metadata`
    # which binds the name locally — patching the subagent module has no effect
    # on that already-bound reference.
    import genome_agent.workflows.nodes.genome_data_nodes as _node_mod
    real_fn = _node_mod.get_genome_metadata

    async def _patched(assembly_id: str) -> dict:
        result = await real_fn(assembly_id)
        result["assembly_level"] = level
        return result

    with patch.object(_node_mod, "get_genome_metadata", _patched):
        yield


SCENARIOS: list[Scenario] = [
    # ------------------------------------------------------------------ #
    # Original scenarios (unchanged)                                       #
    # ------------------------------------------------------------------ #
    Scenario(
        slug="happy-path",
        title="Happy path — tiger, chromosome_map",
        run_kwargs=dict(
            user_question="Show me the genome size and genes of the tiger.",
            species_name="tiger",
            visualization_scope="chromosome_map",
        ),
        workflow_lines=[
            "query_router -> species_resolver",
            "species_resolver -> parallel_kickoff (assembly_id resolved)",
            "parallel_kickoff -> get_genome_metadata + get_gene_annotation (parallel)",
            "join_parallel -> generate_visualization -> explanation_writer -> END",
        ],
    ),
    Scenario(
        slug="unknown-species",
        title="Unknown species — dragon (error_end routing)",
        run_kwargs=dict(
            user_question="Show me the genome of the dragon.",
            species_name="dragon",
        ),
        workflow_lines=[
            "query_router -> species_resolver",
            "species_resolver -> error_end (assembly_id is None, no downstream calls)",
            "error_end -> END",
        ],
    ),
    Scenario(
        slug="escalation",
        title="Escalation — house mouse, protein_structure (NEEDS_AGENT)",
        run_kwargs=dict(
            user_question="Predict the 3D protein structure for the house mouse.",
            species_name="house mouse",
            visualization_scope="protein_structure",
        ),
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata + get_gene_annotation -> join_parallel",
            "join_parallel -> generate_visualization (returns NEEDS_AGENT)",
            "generate_visualization -> capability_resolver -> explanation_writer -> END",
        ],
    ),
    Scenario(
        slug="unknown-scope",
        title="Partial failure — tiger, unknown_scope (visualization FAILED)",
        run_kwargs=dict(
            user_question="Show me the genome size and genes of the tiger.",
            species_name="tiger",
            visualization_scope="unknown_scope",
        ),
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata + get_gene_annotation -> join_parallel",
            "join_parallel -> generate_visualization (returns FAILED, not escalated)",
            "generate_visualization -> explanation_writer -> END",
        ],
    ),
    Scenario(
        slug="partial-results",
        title="Partial results — asian elephant (empty gene_list, metadata OK)",
        run_kwargs=dict(
            user_question="Show me the genome size and genes of the asian elephant.",
            species_name="asian elephant",
            visualization_scope="chromosome_map",
        ),
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata (OK) + get_gene_annotation (empty gene_list, non-fatal) -> join_parallel",
            "join_parallel -> generate_visualization -> explanation_writer -> END",
        ],
    ),
    Scenario(
        slug="size-comparison",
        title="Visualization variety — tiger, size_comparison",
        run_kwargs=dict(
            user_question="Compare the genome size of the tiger to other cats.",
            species_name="tiger",
            visualization_scope="size_comparison",
        ),
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata + get_gene_annotation -> join_parallel",
            "join_parallel -> generate_visualization (size_comparison, COMPLETED) -> explanation_writer -> END",
        ],
    ),
    Scenario(
        slug="no-visualization",
        title="No visualization requested — house mouse, scope=none",
        run_kwargs=dict(
            user_question="How many chromosomes does the house mouse have?",
            species_name="house mouse",
            visualization_scope="none",
        ),
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata + get_gene_annotation -> join_parallel",
            "join_parallel -> explanation_writer -> END (generate_visualization skipped)",
        ],
    ),
    Scenario(
        slug="escalation-partial",
        title="Escalation + partial annotation — asian elephant, protein_structure",
        run_kwargs=dict(
            user_question="Predict the 3D protein structure for the asian elephant.",
            species_name="asian elephant",
            visualization_scope="protein_structure",
        ),
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata (OK) + get_gene_annotation (empty gene_list, non-fatal) -> join_parallel",
            "join_parallel -> generate_visualization (returns NEEDS_AGENT)",
            "generate_visualization -> capability_resolver -> explanation_writer -> END",
        ],
    ),

    # ------------------------------------------------------------------ #
    # NEW: Reconstruction escalation scenarios                             #
    # ------------------------------------------------------------------ #
    Scenario(
        slug="reconstruction-scaffold",
        title="Reconstruction — scaffold-level assembly (NEEDS_AGENT → Reconstruction Agent)",
        run_kwargs=dict(
            user_question="Reconstruct the complete genome for the axolotl.",
            species_name="axolotl",
            visualization_scope="none",
        ),
        forced_assembly_level="Scaffold",   # axolotl was upgraded to Chromosome by NCBI
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata: NCBI reports assembly_level='Scaffold'",
            "  → metadata node sets reconstruction_need={status:NEEDS_AGENT, target_agent:None}",
            "get_gene_annotation -> join_parallel",
            "join_parallel: _route_after_join_parallel sees reconstruction_need.status==NEEDS_AGENT",
            "  → routes to reconstruction_resolver (NOT generate_visualization)",
            "reconstruction_resolver: calls resolve_capability / resolve_capability_fallback",
            "  → fills target_agent='Reconstruction Agent', updates handoff prompt",
            "reconstruction_resolver -> explanation_writer -> END",
            "adapter to_result(): reconstruction_need.status==NEEDS_AGENT",
            "  → returns AgentResult(status=NEEDS_AGENT, target_agent='Reconstruction Agent')",
            "Global Orchestrator receives NEEDS_AGENT → invokes Reconstruction Agent",
        ],
    ),
    Scenario(
        slug="reconstruction-contig",
        title="Reconstruction — contig-level assembly (NEEDS_AGENT → Reconstruction Agent)",
        run_kwargs=dict(
            user_question="The coelacanth genome has many gaps, please reconstruct it.",
            species_name="coelacanth",
            visualization_scope="none",
        ),
        forced_assembly_level="Contig",     # force contig level regardless of current NCBI state
        workflow_lines=[
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata: NCBI reports assembly_level='Contig'",
            "  → metadata node sets reconstruction_need={status:NEEDS_AGENT, target_agent:None}",
            "get_gene_annotation -> join_parallel",
            "join_parallel: _route_after_join_parallel sees reconstruction_need.status==NEEDS_AGENT",
            "  → routes to reconstruction_resolver",
            "reconstruction_resolver: resolves target_agent='Reconstruction Agent'",
            "reconstruction_resolver -> explanation_writer -> END",
            "adapter: returns AgentResult(status=NEEDS_AGENT, target_agent='Reconstruction Agent')",
            "Global Orchestrator receives NEEDS_AGENT → invokes Reconstruction Agent",
        ],
    ),
]

_SCENARIOS_BY_SLUG: dict[str, Scenario] = {s.slug: s for s in SCENARIOS}


def _print_case(scenario: Scenario, result) -> None:
    # result is a GenomeAgentState — derive the logical status from state fields.
    need = result.reconstruction_need or {}
    viz  = result.visualization or {}

    if need.get("status") == "NEEDS_AGENT":
        effective_status       = "NEEDS_AGENT"
        effective_target       = need.get("target_agent")
        effective_prompt       = need.get("prompt_to_target_agent")
    elif viz.get("status") == "NEEDS_AGENT":
        effective_status       = "NEEDS_AGENT"
        effective_target       = viz.get("target_agent")
        effective_prompt       = viz.get("prompt_to_target_agent")
    elif result.assembly_id is None:
        effective_status       = "ERROR"
        effective_target       = None
        effective_prompt       = None
    else:
        effective_status       = "COMPLETED"
        effective_target       = None
        effective_prompt       = None

    print("\n" + "=" * _WIDTH)
    print(f"CASE: {scenario.title}")
    print("=" * _WIDTH)
    print("Input:")
    for line in scenario.input_lines:
        print(f"  {line}")
    print("Workflow:")
    for line in scenario.workflow_lines:
        print(f"  {line}")
    print("Result:")
    print(f"  Status               : {effective_status}")
    print(f"  Target agent         : {effective_target}")
    print(f"  Handoff prompt       : {effective_prompt!r}")
    print(f"  reconstruction_need  : {need or None}")
    print(f"  Species              : {result.species}")
    print(f"  Metadata             : {result.metadata}")
    print(f"  Annotation           : {result.annotation}")
    print(f"  Visualization        : {result.visualization}")
    print(f"  Explanation          : {result.explanation}")
    print(f"  Errors               : {result.errors}")

    # Reconstruction-specific assertion — doubles as a live smoke test.
    if scenario.slug.startswith("reconstruction-"):
        ok = (
            effective_status == "NEEDS_AGENT"
            and effective_target == "Reconstruction Agent"
        )
        verdict = "✅ PASS" if ok else "❌ FAIL — expected NEEDS_AGENT / Reconstruction Agent"
        print(f"  [reconstruction assertion] {verdict}")

    print("=" * _WIDTH)


def _print_list() -> None:
    print("Available scenarios:")
    for s in SCENARIOS:
        print(f"  {s.slug:<30} {s.title}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Genome Agent orchestrator scenarios.",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        nargs="+",
        metavar="SLUG",
        choices=[s.slug for s in SCENARIOS],
        help="Run only the named scenario(s) instead of all of them. "
        "Use --list to see available slugs.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Print available scenario slugs and exit.",
    )
    return parser.parse_args(argv)


async def run_all(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.list:
        _print_list()
        return

    try:
        from ..orchestrator import GenomeAgentLangGraphOrchestrator
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from genome_agent.orchestrator import GenomeAgentLangGraphOrchestrator

    scenarios_to_run = (
        [_SCENARIOS_BY_SLUG[slug] for slug in args.scenario] if args.scenario else SCENARIOS
    )

    llm_mode = (
        "LIVE (NVIDIA_API_KEY set)"
        if os.getenv("NVIDIA_API_KEY")
        else "FALLBACK (no NVIDIA_API_KEY — keyword-based routing/resolution)"
    )
    print("\n" + "#" * _WIDTH)
    print(f"# Genome Agent — Orchestrator Scenario Run")
    print(f"# LLM mode: {llm_mode}")
    if args.scenario:
        print(
            f"# Running {len(scenarios_to_run)}/{len(SCENARIOS)} selected "
            f"scenario(s): {', '.join(args.scenario)}"
        )
    print("#" * _WIDTH)

    orch = GenomeAgentLangGraphOrchestrator()

    for scenario in scenarios_to_run:
        if scenario.forced_assembly_level:
            async with _force_assembly_level(scenario.forced_assembly_level):
                result = await orch.run(**scenario.run_kwargs)
        else:
            result = await orch.run(**scenario.run_kwargs)
        _print_case(scenario, result)

    print("\n" + "#" * _WIDTH)
    print(f"# {len(scenarios_to_run)} scenario(s) executed.")
    if not args.scenario:
        print(
            "# Branch coverage:\n"
            "#   happy path, error_end, all 3 visualization statuses\n"
            "#   (COMPLETED ×2 scopes, NEEDS_AGENT, FAILED), no-visualization,\n"
            "#   partial-result degradation, partial + escalation,\n"
            "#   reconstruction-scaffold (NEW), reconstruction-contig (NEW)."
        )
    print("#" * _WIDTH)


if __name__ == "__main__":
    asyncio.run(run_all())