"""
Genome Agent Orchestrator (Task 1 — Sprint 1)
Coordinates the 4 mock subagents in the required order and parallelism.

Call order:
  1. resolve_species()                             — sequential, gating step
  2. get_genome_metadata() + get_gene_annotation() — parallel (asyncio.gather)
  3. generate_visualization()                      — sequential, uses step 2 outputs

Fallback rules:
  - assembly_id is None → stop early, return species_not_found error
  - metadata or annotation fails/empty → include whatever succeeded, keep going
  - visualization status == "NEEDS_AGENT" → bubble up as-is, never retry
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .subagents.species_resolver import resolve_species
from .subagents.genome_metadata import get_genome_metadata
from .subagents.gene_annotation import get_gene_annotation
from .subagents.visualization import generate_visualization
from .workflows.agent_catalog import load_agent_catalog
from .workflows.capability_resolver import resolve_capability, resolve_capability_fallback
from .workflows.explanation_writer import write_explanation
from .workflows.llm import get_llm_client
from .workflows.query_router import route_query, route_query_fallback
from .workflows.state import GenomeAgentState

logger = logging.getLogger(__name__)


class GenomeAgentOrchestrator:
    def __init__(self):
        pass

    async def run(self, species_name: str, visualization_scope: str = "chromosome_map") -> dict:
        """
        Main entry point for the Genome Agent.

        Args:
            species_name:        Common or scientific name of the target species.
            visualization_scope: One of "chromosome_map", "size_comparison",
                                 or "protein_structure". Defaults to "chromosome_map".

        Returns:
            A single merged output dict with keys:
              - species      : resolved species info (always present)
              - metadata     : genome metadata (None if unavailable)
              - annotation   : gene annotation  (None if unavailable)
              - visualization: visualization result, may carry status="NEEDS_AGENT"
              - errors       : list of non-fatal error messages (may be empty)
        """

        output = {
            "species": None,
            "metadata": None,
            "annotation": None,
            "visualization": None,
            "errors": [],
        }

        # ------------------------------------------------------------------
        # Step 1 — Resolve species (sequential, gating)
        # ------------------------------------------------------------------
        try:
            species = await resolve_species(species_name)
        except Exception as exc:
            output["errors"].append(f"species_resolver raised an exception: {exc}")
            return output

        output["species"] = species

        if species.get("assembly_id") is None:
            # Hard stop — everything downstream needs an assembly_id
            output["errors"].append(
                f"Species '{species_name}' could not be resolved to a genome assembly. "
                "No further data can be retrieved."
            )
            return output

        assembly_id = species["assembly_id"]

        # ------------------------------------------------------------------
        # Step 2 — Metadata + Annotation in parallel
        # ------------------------------------------------------------------
        metadata_result, annotation_result = await asyncio.gather(
            get_genome_metadata(assembly_id),
            get_gene_annotation(assembly_id),
            return_exceptions=True,  # prevents one failure from cancelling the other
        )

        # Unpack metadata
        if isinstance(metadata_result, Exception):
            output["errors"].append(f"get_genome_metadata raised an exception: {metadata_result}")
            metadata = None
        elif metadata_result.get("genome_size_bp") is None:
            output["errors"].append(
                f"Genome metadata returned empty for assembly '{assembly_id}'."
            )
            metadata = None
        else:
            metadata = metadata_result

        output["metadata"] = metadata

        # Unpack annotation
        if isinstance(annotation_result, Exception):
            output["errors"].append(f"get_gene_annotation raised an exception: {annotation_result}")
            annotation = None
        elif not annotation_result.get("gene_list"):
            # Empty gene list is not necessarily an error — log it softly
            output["errors"].append(
                f"Gene annotation returned no genes for assembly '{assembly_id}'."
            )
            annotation = annotation_result  # keep it (has empty lists, not None)
        else:
            annotation = annotation_result

        output["annotation"] = annotation

        # ------------------------------------------------------------------
        # Step 3 — Visualization (sequential, uses step 2 outputs)
        # ------------------------------------------------------------------
        genome_size = metadata["genome_size_bp"] if metadata else None
        gene_table = annotation["gene_table"] if annotation else None

        try:
            viz = await generate_visualization(
                scope=visualization_scope,
                genome_size_bp=genome_size,
                gene_table=gene_table,
            )
        except Exception as exc:
            output["errors"].append(f"generate_visualization raised an exception: {exc}")
            viz = None

        # If the visualization needs another agent, bubble it up unchanged — never resolve here
        output["visualization"] = viz

        return output


# --------------------------------------------------------------------------
# Task 3 — LangGraph Orchestrator
# Replaces the hand-rolled asyncio.gather control flow with an actual
# LangGraph graph using StateGraph, nodes, edges, and conditional routing.
#
# Graph shape::
#
#   START → query_router
#   query_router → species_resolver
#   species_resolver → (conditional)
#     ├─ assembly_id None → error_end → END
#     └─ assembly_id present → parallel_kickoff
#   parallel_kickoff → get_genome_metadata (if needed)
#   parallel_kickoff → get_gene_annotation (if needed)
#   get_genome_metadata → join_parallel
#   get_gene_annotation → join_parallel
#   join_parallel → (conditional)
#     ├─ viz needed → generate_visualization
#     └─ viz not needed → explanation_writer → END
#   generate_visualization → (conditional)
#     ├─ NEEDS_AGENT → capability_resolver → generate_visualization
#     ├─ COMPLETED → explanation_writer → END
#     └─ FAILED → explanation_writer → END
#   explanation_writer → END
#   error_end → END
# --------------------------------------------------------------------------

import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .subagents.species_resolver import resolve_species
from .subagents.genome_metadata import get_genome_metadata
from .subagents.gene_annotation import get_gene_annotation
from .subagents.visualization import generate_visualization
from .workflows.agent_catalog import load_agent_catalog
from .workflows.capability_resolver import resolve_capability, resolve_capability_fallback
from .workflows.explanation_writer import write_explanation
from .workflows.llm import get_llm_client
from .workflows.query_router import route_query, route_query_fallback
from .workflows.state import GenomeAgentState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Node functions
# ------------------------------------------------------------------


async def _query_router_node(state: GenomeAgentState) -> dict[str, Any]:
    user_question = state.user_question
    logger.info("[query_router] routing user_question=%r", user_question)

    decision = route_query(user_question)
    if decision is None:
        decision = route_query_fallback(user_question)

    current_scope = state.visualization_scope
    new_scope = current_scope if current_scope not in ("chromosome_map", "") else decision.visualization_scope

    return {
        "needs_metadata": decision.needs_metadata,
        "needs_annotation": decision.needs_annotation,
        "visualization_scope": new_scope,
    }


async def _species_resolver_node(state: GenomeAgentState) -> dict[str, Any]:
    species_name = state.species_name
    logger.info("[species_resolver] resolving species=%r", species_name)

    try:
        species = await resolve_species(species_name)
    except Exception as exc:
        return {
            "errors": [*state.errors, f"species_resolver raised an exception: {exc}"],
            "assembly_id": None,
        }

    assembly_id = species.get("assembly_id")
    if assembly_id is None:
        return {
            "species": species,
            "assembly_id": None,
            "errors": [
                *state.errors,
                f"Species '{species_name}' could not be resolved to a genome assembly. "
                "No further data can be retrieved.",
            ],
        }

    return {
        "species": species,
        "assembly_id": assembly_id,
    }


def _parallel_kickoff_node(state: GenomeAgentState) -> dict[str, Any]:
    return {}


async def _get_genome_metadata_node(state: GenomeAgentState) -> dict[str, Any]:
    if not state.needs_metadata:
        return {"_metadata_done": True}

    assembly_id = state.assembly_id
    logger.info("[get_genome_metadata] fetching metadata for assembly=%r", assembly_id)

    try:
        result = await get_genome_metadata(assembly_id)
    except Exception as exc:
        return {
            "errors": [*state.errors, f"get_genome_metadata raised an exception: {exc}"],
            "metadata": None,
            "_metadata_done": True,
        }

    if result.get("genome_size_bp") is None:
        return {
            "errors": [
                *state.errors,
                f"Genome metadata returned empty for assembly '{assembly_id}'.",
            ],
            "metadata": None,
            "_metadata_done": True,
        }

    return {
        "metadata": result,
        "_metadata_done": True,
    }


async def _get_gene_annotation_node(state: GenomeAgentState) -> dict[str, Any]:
    if not state.needs_annotation:
        return {"_annotation_done": True}

    assembly_id = state.assembly_id
    logger.info("[get_gene_annotation] fetching annotation for assembly=%r", assembly_id)

    try:
        result = await get_gene_annotation(assembly_id)
    except Exception as exc:
        return {
            "errors": [
                *state.errors,
                f"get_gene_annotation raised an exception: {exc}",
            ],
            "annotation": None,
            "_annotation_done": True,
        }

    if not result.get("gene_list"):
        return {
            "errors": [
                *state.errors,
                f"Gene annotation returned no genes for assembly '{assembly_id}'.",
            ],
            "annotation": result,
            "_annotation_done": True,
        }

    return {
        "annotation": result,
        "_annotation_done": True,
    }


def _join_parallel_node(state: GenomeAgentState) -> dict[str, Any]:
    return {}


async def _generate_visualization_node(state: GenomeAgentState) -> dict[str, Any]:
    if state.visualization is not None:
        return {}

    scope = state.visualization_scope
    if scope == "none":
        return {}

    genome_size = state.metadata["genome_size_bp"] if state.metadata else None
    gene_table = state.annotation["gene_table"] if state.annotation else None
    logger.info("[generate_visualization] scope=%r", scope)

    try:
        result = await generate_visualization(
            scope=scope,
            genome_size_bp=genome_size,
            gene_table=gene_table,
        )
    except Exception as exc:
        return {
            "errors": [
                *state.errors,
                f"generate_visualization raised an exception: {exc}",
            ],
            "visualization": None,
        }

    if result.get("status") == "FAILED":
        return {
            "visualization": result,
            "errors": [
                *state.errors,
                f"Visualization failed with status FAILED for scope '{scope}'.",
            ],
        }

    return {"visualization": result}


async def _capability_resolver_node(state: GenomeAgentState) -> dict[str, Any]:
    viz = state.visualization or {}
    prompt_to_target = viz.get("prompt_to_target_agent", "")
    known_context = {
        "species": state.species,
        "metadata": state.metadata,
        "annotation": state.annotation,
    }

    decision = resolve_capability(
        current_agent="visualization",
        prompt_to_target_agent=prompt_to_target,
        known_context=known_context,
    )
    if decision is None:
        decision = resolve_capability_fallback(
            current_agent="visualization",
            prompt_to_target_agent=prompt_to_target,
        )

    if decision.target_agent == "none":
        return {
            "errors": [
                *state.errors,
                f"No agent in the catalog can resolve: {prompt_to_target}",
            ],
            "visualization": {
                **viz,
                "status": "FAILED",
                "target_agent": None,
                "prompt_to_target_agent": None,
            },
        }

    return {
        "visualization": {
            **viz,
            "status": "NEEDS_AGENT",
            "target_agent": decision.target_agent,
            "prompt_to_target_agent": decision.handoff_message,
        },
    }


async def _explanation_writer_node(state: GenomeAgentState) -> dict[str, Any]:
    explanation = write_explanation(
        user_question=state.user_question,
        species=state.species,
        metadata=state.metadata,
        annotation=state.annotation,
        visualization=state.visualization,
    )
    return {"explanation": explanation}


def _error_end_node(state: GenomeAgentState) -> dict[str, Any]:
    logger.info("[error_end] species resolver failed, stopping")
    return {}


# ------------------------------------------------------------------
# Conditional edge functions
# ------------------------------------------------------------------


def _route_after_query_router(state: GenomeAgentState) -> str:
    return "species_resolver"


def _route_after_species_resolver(state: GenomeAgentState) -> str:
    if state.assembly_id is None:
        return "error_end"
    return "parallel_kickoff"


def _route_after_join_parallel(state: GenomeAgentState) -> str:
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

    graph.add_node("query_router", _query_router_node)
    graph.add_node("species_resolver", _species_resolver_node)
    graph.add_node("parallel_kickoff", _parallel_kickoff_node)
    graph.add_node("get_genome_metadata", _get_genome_metadata_node)
    graph.add_node("get_gene_annotation", _get_gene_annotation_node)
    graph.add_node("join_parallel", _join_parallel_node)
    graph.add_node("generate_visualization", _generate_visualization_node)
    graph.add_node("capability_resolver", _capability_resolver_node)
    graph.add_node("explanation_writer", _explanation_writer_node)
    graph.add_node("error_end", _error_end_node)

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

    graph.add_conditional_edges(
        "join_parallel",
        _route_after_join_parallel,
        {
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

    graph.add_edge("capability_resolver", "explanation_writer")
    graph.add_edge("explanation_writer", END)
    graph.add_edge("error_end", END)

    return graph.compile()


class GenomeAgentLangGraphOrchestrator:
    """LangGraph-based orchestrator for the Genome Agent (Task 3).

    Replaces the hand-raised `asyncio.gather` control flow from Task 1
    with a proper LangGraph `StateGraph` that uses nodes, edges, and
    conditional routing for the same sequential → parallel → sequential
    execution order.
    """

    def __init__(self) -> None:
        self._graph = build_genome_graph()

    async def run(
        self,
        user_question: str,
        species_name: str,
        visualization_scope: str = "chromosome_map",
    ) -> GenomeAgentState:
        """Execute the LangGraph workflow for a user question."""

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


# --------------------------------------------------------------------------
# Quick sanity-check — run with: python -m backend.agents.genome_agent.orchestrator
# --------------------------------------------------------------------------
if __name__ == "__main__":

    async def _run_task1_tests():
        orch = GenomeAgentOrchestrator()
        sep = "-" * 60

        # Test 1: Known species — happy path
        print(sep)
        print("TEST 1 — Tiger, chromosome_map (happy path)")
        print(sep)
        result = await orch.run("tiger", visualization_scope="chromosome_map")
        print(f"  Species   : {result['species']}")
        print(f"  Metadata  : {result['metadata']}")
        print(f"  Annotation: {result['annotation']}")
        print(f"  Viz       : {result['visualization']}")
        print(f"  Errors    : {result['errors']}")
        assert result["species"]["assembly_id"] == "GCF_000464555.1"
        assert result["metadata"]["chromosome_count"] == 38
        assert result["annotation"]["gene_list"] == ["Mc1r"]
        assert result["visualization"]["status"] == "COMPLETED"
        assert not result["errors"]
        print("  ✅ PASSED\n")

        # Test 2: Unknown species — early stop
        print(sep)
        print("TEST 2 — Unknown species 'dragon' (early-stop fallback)")
        print(sep)
        result = await orch.run("dragon")
        print(f"  Species   : {result['species']}")
        print(f"  Metadata  : {result['metadata']}")
        print(f"  Annotation: {result['annotation']}")
        print(f"  Viz       : {result['visualization']}")
        print(f"  Errors    : {result['errors']}")
        assert result["species"]["assembly_id"] is None
        assert result["metadata"] is None
        assert result["annotation"] is None
        assert result["visualization"] is None
        assert len(result["errors"]) == 1
        assert "could not be resolved" in result["errors"][0]
        print("  ✅ PASSED\n")

        # Test 3: protein_structure — NEEDS_AGENT must bubble up
        print(sep)
        print("TEST 3 — House mouse, protein_structure scope (NEEDS_AGENT bubble-up)")
        print(sep)
        result = await orch.run("house mouse", visualization_scope="protein_structure")
        print(f"  Species   : {result['species']}")
        print(f"  Metadata  : {result['metadata']}")
        print(f"  Annotation: {result['annotation']}")
        print(f"  Viz       : {result['visualization']}")
        print(f"  Errors    : {result['errors']}")
        assert result["species"]["assembly_id"] == "GCF_000001635.27"
        assert result["visualization"]["status"] == "NEEDS_AGENT"
        assert result["visualization"]["target_agent"] is None
        assert "prompt_to_target_agent" in result["visualization"]
        print("  ✅ PASSED\n")

        print(sep)
        print("All Task 1 orchestrator tests passed ✅")
        print(sep)

    asyncio.run(_run_task1_tests())

    # ------------------------------------------------------------------
    # Task 3 — LangGraph orchestrator integration tests
    # ------------------------------------------------------------------
    sep = "-" * 60

    async def _run_task3_tests():
        lg_orch = GenomeAgentLangGraphOrchestrator()

        # Test 4: Happy path — sequential → parallel → sequential
        print(sep)
        print("TEST 4 — Tiger, chromosome_map (LangGraph happy path)")
        print(sep)
        result = await lg_orch.run(
            user_question="Show me the genome size and genes of the tiger.",
            species_name="tiger",
            visualization_scope="chromosome_map",
        )
        print(f"  Species     : {result.species}")
        print(f"  Metadata    : {result.metadata}")
        print(f"  Annotation  : {result.annotation}")
        print(f"  Viz         : {result.visualization}")
        print(f"  Explanation : {result.explanation}")
        print(f"  Errors      : {result.errors}")
        assert result.species is not None
        assert result.species["assembly_id"] == "GCF_000464555.1"
        assert result.metadata is not None
        assert result.metadata["chromosome_count"] == 38
        assert result.annotation is not None
        assert result.annotation["gene_list"] == ["Mc1r"]
        assert result.visualization is not None
        assert result.visualization["status"] == "COMPLETED"
        assert result.explanation is not None
        assert not result.errors
        print("  ✅ PASSED\n")

        # Test 5: Unknown species — error_end routing
        print(sep)
        print("TEST 5 — Unknown species 'dragon' (error_end routing)")
        print(sep)
        result = await lg_orch.run(
            user_question="Show me the genome of the dragon.",
            species_name="dragon",
        )
        print(f"  Species     : {result.species}")
        print(f"  Metadata    : {result.metadata}")
        print(f"  Annotation  : {result.annotation}")
        print(f"  Viz         : {result.visualization}")
        print(f"  Explanation : {result.explanation}")
        print(f"  Errors      : {result.errors}")
        assert result.species is not None
        assert result.species["assembly_id"] is None
        assert result.metadata is None
        assert result.annotation is None
        assert result.visualization is None
        assert len(result.errors) == 1
        assert "could not be resolved" in result.errors[0]
        print("  ✅ PASSED\n")

        # Test 6: protein_structure — escalate_to_platform (outside-agent NEEDS_AGENT)
        print(sep)
        print("TEST 6 — House mouse, protein_structure (escalate-outside-agent)")
        print(sep)
        result = await lg_orch.run(
            user_question="Predict the 3D protein structure for the house mouse.",
            species_name="house mouse",
            visualization_scope="protein_structure",
        )
        print(f"  Species     : {result.species}")
        print(f"  Metadata    : {result.metadata}")
        print(f"  Annotation  : {result.annotation}")
        print(f"  Viz         : {result.visualization}")
        print(f"  Explanation : {result.explanation}")
        print(f"  Errors      : {result.errors}")
        assert result.species is not None
        assert result.species["assembly_id"] == "GCF_000001635.27"
        assert result.visualization is not None
        assert result.visualization["status"] == "NEEDS_AGENT"
        assert result.visualization["target_agent"] == "Protein Structure Visualization Agent"
        assert result.explanation is not None
        print("  ✅ PASSED\n")

        # Test 7: Unknown scope — FAILED status → explanation_writer
        print(sep)
        print("TEST 7 — Tiger, unknown_scope (partial failure degrades gracefully)")
        print(sep)
        result = await lg_orch.run(
            user_question="Show me the genome size and genes of the tiger.",
            species_name="tiger",
            visualization_scope="unknown_scope",
        )
        print(f"  Species     : {result.species}")
        print(f"  Metadata    : {result.metadata}")
        print(f"  Annotation  : {result.annotation}")
        print(f"  Viz         : {result.visualization}")
        print(f"  Explanation : {result.explanation}")
        print(f"  Errors      : {result.errors}")
        assert result.species is not None
        assert result.species["assembly_id"] == "GCF_000464555.1"
        assert result.metadata is not None
        assert result.metadata["chromosome_count"] == 38
        assert result.annotation is not None
        assert result.annotation["gene_list"] == ["Mc1r"]
        assert result.visualization is not None
        assert result.visualization["status"] == "FAILED"
        assert result.explanation is not None
        print("  ✅ PASSED\n")

        # Test 8: Partial failure — annotation empty but metadata present
        print(sep)
        print("TEST 8 — Asian elephant (empty gene_list → partial results)")
        print(sep)
        result = await lg_orch.run(
            user_question="Show me the genome size and genes of the asian elephant.",
            species_name="asian elephant",
            visualization_scope="chromosome_map",
        )
        print(f"  Species     : {result.species}")
        print(f"  Metadata    : {result.metadata}")
        print(f"  Annotation  : {result.annotation}")
        print(f"  Viz         : {result.visualization}")
        print(f"  Explanation : {result.explanation}")
        print(f"  Errors      : {result.errors}")
        assert result.species is not None
        assert result.species["assembly_id"] == "GCA_024166365.1"
        assert result.metadata is not None
        assert result.metadata["genome_size_bp"] == 3200000000
        assert result.annotation is not None
        assert result.annotation["gene_list"] == []
        assert result.visualization is not None
        assert result.visualization["status"] == "COMPLETED"
        assert result.explanation is not None
        assert len(result.errors) == 1
        assert "no genes" in result.errors[0]
        print("  ✅ PASSED\n")

        print(sep)
        print("All LangGraph orchestrator tests passed ✅")
        print(sep)

    asyncio.run(_run_task3_tests())
