"""
Genome Agent Orchestrator

LangGraph wiring for the Genome Agent: routing decisions (which subagents
are needed, which agent to escalate to, and the final write-up) are made by
the LLM via LangChain (query_router, capability_resolver, explanation_writer
in workflows/), each with a deterministic fallback for when the LLM is
unreachable. Subagent data itself stays hardcoded mock data (see subagents/).

Node bodies live in workflows/nodes/ — this file only wires them into a
graph. Tests live in tests/test_orchestrator.py, not here.

Graph shape::

  START → query_router
  query_router → species_resolver
  species_resolver → (conditional)
    ├─ assembly_id None → error_end → END
    └─ assembly_id present → parallel_kickoff
  parallel_kickoff → get_genome_metadata (if needed)
  parallel_kickoff → get_gene_annotation (if needed)
  get_genome_metadata → join_parallel
  get_gene_annotation → join_parallel
  join_parallel → (conditional)
    ├─ gaps detected      → reconstruction_resolver → explanation_writer → END
    ├─ viz needed         → generate_visualization
    └─ viz not needed     → explanation_writer → END
  generate_visualization → (conditional)
    ├─ NEEDS_AGENT → capability_resolver → explanation_writer → END
    ├─ COMPLETED   → explanation_writer → END
    └─ FAILED      → explanation_writer → END
  explanation_writer → END
  error_end → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .workflows.nodes.capability_resolver_node import capability_resolver_node
from .workflows.nodes.explanation_node import error_end_node, explanation_writer_node
from .workflows.nodes.genome_data_nodes import get_gene_annotation_node, get_genome_metadata_node
from .workflows.nodes.query_router_node import query_router_node
from .workflows.nodes.reconstruction_resolver_node import reconstruction_resolver_node
from .workflows.nodes.species_resolver_node import species_resolver_node
from .workflows.nodes.visualization_node import generate_visualization_node
from .workflows.state import GenomeAgentState


# ------------------------------------------------------------------
# Trivial fan-out/fan-in plumbing nodes (no business logic, so they stay
# here rather than in workflows/nodes/ alongside the real node bodies).
# ------------------------------------------------------------------


def _parallel_kickoff_node(state: GenomeAgentState) -> dict[str, Any]:
    # No-op fan-out node — must still write to at least one channel
    # (langgraph enforces this), so we echo back an unchanged value.
    return {"errors": []}


def _join_parallel_node(state: GenomeAgentState) -> dict[str, Any]:
    # No-op join node — must still write to at least one channel.
    # `errors` uses operator.add as its reducer, so [] is a true no-op merge.
    return {"errors": []}


# ------------------------------------------------------------------
# Conditional edge functions
# ------------------------------------------------------------------


def _route_after_species_resolver(state: GenomeAgentState) -> str:
    if state.assembly_id is None:
        return "error_end"
    return "parallel_kickoff"


def _route_after_join_parallel(state: GenomeAgentState) -> str:
    need = state.reconstruction_need or {}
    if need.get("status") == "NEEDS_AGENT":
        return "reconstruction_resolver"
    if state.visualization_scope == "none":
        return "explanation_writer"
    return "generate_visualization"


def _route_after_visualization(state: GenomeAgentState) -> str:
    viz = state.visualization
    if viz is None:
        return "explanation_writer"

    status = viz.get("status")
    if status == "NEEDS_AGENT":
        return "capability_resolver"

    return "explanation_writer"


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------


def build_genome_graph() -> CompiledStateGraph:
    graph = StateGraph(GenomeAgentState)

    graph.add_node("query_router", query_router_node)
    graph.add_node("species_resolver", species_resolver_node)
    graph.add_node("parallel_kickoff", _parallel_kickoff_node)
    graph.add_node("get_genome_metadata", get_genome_metadata_node)
    graph.add_node("get_gene_annotation", get_gene_annotation_node)
    graph.add_node("join_parallel", _join_parallel_node)
    graph.add_node("generate_visualization", generate_visualization_node)
    graph.add_node("reconstruction_resolver", reconstruction_resolver_node)
    graph.add_node("capability_resolver", capability_resolver_node)
    graph.add_node("explanation_writer", explanation_writer_node)
    graph.add_node("error_end", error_end_node)

    graph.add_edge(START, "query_router")

    graph.add_edge("query_router", "species_resolver")

    graph.add_conditional_edges(
        "species_resolver",
        _route_after_species_resolver,
        {
            "error_end": "error_end",
            "parallel_kickoff": "parallel_kickoff",
        },
    )

    graph.add_edge("parallel_kickoff", "get_genome_metadata")
    graph.add_edge("parallel_kickoff", "get_gene_annotation")

    graph.add_edge("get_genome_metadata", "join_parallel")
    graph.add_edge("get_gene_annotation", "join_parallel")

    # Single conditional edge from join_parallel — covers all three branches.
    graph.add_conditional_edges(
        "join_parallel",
        _route_after_join_parallel,
        {
            "reconstruction_resolver": "reconstruction_resolver",
            "generate_visualization": "generate_visualization",
            "explanation_writer": "explanation_writer",
        },
    )

    graph.add_conditional_edges(
        "generate_visualization",
        _route_after_visualization,
        {
            "capability_resolver": "capability_resolver",
            "explanation_writer": "explanation_writer",
        },
    )

    graph.add_edge("reconstruction_resolver", "explanation_writer")
    graph.add_edge("capability_resolver", "explanation_writer")
    graph.add_edge("explanation_writer", END)
    graph.add_edge("error_end", END)

    return graph.compile()


class GenomeAgentLangGraphOrchestrator:
    """LangGraph-based orchestrator for the Genome Agent."""

    def __init__(self) -> None:
        self._graph = build_genome_graph()

    async def run(
        self,
        user_question: str,
        species_name: str,
        visualization_scope: str = "",
    ) -> GenomeAgentState:
        """Execute the LangGraph workflow for a user question.

        visualization_scope: "chromosome_map", "size_comparison",
            "protein_structure", "none", or "" (default) to let
            query_router infer the scope from user_question. Any
            non-empty value is treated as an explicit request and is
            never overridden by the router.
        """

        initial_state = GenomeAgentState(
            user_question=user_question,
            species_name=species_name,
            visualization_scope=visualization_scope,
        )

        result = await self._graph.ainvoke(initial_state)
        return result if isinstance(result, GenomeAgentState) else GenomeAgentState(**result)

    def get_execution_history(self, state: GenomeAgentState) -> list[str]:
        """Return a human-readable list of steps taken."""
        steps = []
        if state.species is not None:
            steps.append("species_resolver")
        if state._metadata_done or state.metadata is not None:
            steps.append("get_genome_metadata")
        if state._annotation_done or state.annotation is not None:
            steps.append("get_gene_annotation")
        if state.visualization is not None:
            steps.append("generate_visualization")
        return steps