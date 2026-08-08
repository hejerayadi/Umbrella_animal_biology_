"""The protein workflow as a LangGraph ``StateGraph``.

The shape of the graph, and which node owns which decision, is documented in
``docs/workflow.md``. In short:

1. ``validate_input`` either stops with a clarification or resolves identity.
2. ``resolve_protein_identity`` is the anchor; without it the run abstains.
3. Structures, annotations and knowledge are then fetched in parallel.
4. AlphaFold is reached only when no PDB entry survives ``evaluate_pdb_results``.
5. ``join_results`` waits for every branch, then SIFTS runs if a position was asked for.
6. Evidence, scene and explanation are built, and the critic decides the outcome.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.Protein_visualization.app.orchestrators.protein import routers
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes import (
    ProteinNodes,
    complete,
    return_needs_clarification,
    return_partial,
    scientific_abstain,
    validate_input,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import (
    BUILD_EVIDENCE,
    BUILD_SCENE,
    COMPLETE,
    EVALUATE_PDB,
    FETCH_ANNOTATIONS,
    GENERATE_EXPLANATION,
    JOIN_RESULTS,
    MAP_RESIDUES,
    NEEDS_CLARIFICATION,
    RESOLVE_IDENTITY,
    RETRIEVE_KNOWLEDGE,
    RETURN_PARTIAL,
    RUN_CRITIC,
    SCIENTIFIC_ABSTAIN,
    SEARCH_ALPHAFOLD,
    SEARCH_EXPERIMENTAL,
    SELECT_STRUCTURE,
    VALIDATE_INPUT,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


async def join_results(state: ProteinWorkflowState) -> dict[str, object]:
    """Barrier node.

    Declared with ``defer=True`` so it runs once, after every parallel branch has
    settled. Without it the structure branch — which is two supersteps longer when
    the AlphaFold fallback fires — would trigger the tail of the graph a second time.
    """
    return {"executed_nodes": {JOIN_RESULTS}}


def build_graph(nodes: ProteinNodes) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(ProteinWorkflowState)

    graph.add_node(VALIDATE_INPUT, validate_input)
    graph.add_node(RESOLVE_IDENTITY, nodes.identity)
    graph.add_node(SEARCH_EXPERIMENTAL, nodes.structures.search_experimental)
    graph.add_node(EVALUATE_PDB, nodes.structures.evaluate_pdb)
    graph.add_node(SEARCH_ALPHAFOLD, nodes.structures.search_alphafold)
    graph.add_node(SELECT_STRUCTURE, nodes.structures.select_structure)
    graph.add_node(FETCH_ANNOTATIONS, nodes.annotations)
    graph.add_node(RETRIEVE_KNOWLEDGE, nodes.retrieval)
    graph.add_node(JOIN_RESULTS, join_results, defer=True)
    graph.add_node(MAP_RESIDUES, nodes.residue_mapping)
    graph.add_node(BUILD_EVIDENCE, nodes.evidence)
    graph.add_node(BUILD_SCENE, nodes.scene)
    graph.add_node(GENERATE_EXPLANATION, nodes.explanation)
    graph.add_node(RUN_CRITIC, nodes.critic)
    graph.add_node(COMPLETE, complete)
    graph.add_node(RETURN_PARTIAL, return_partial)
    graph.add_node(NEEDS_CLARIFICATION, return_needs_clarification)
    graph.add_node(SCIENTIFIC_ABSTAIN, scientific_abstain)

    graph.add_edge(START, VALIDATE_INPUT)
    graph.add_conditional_edges(
        VALIDATE_INPUT,
        routers.route_after_validation,
        [RESOLVE_IDENTITY, NEEDS_CLARIFICATION],
    )
    graph.add_conditional_edges(
        RESOLVE_IDENTITY,
        routers.route_after_identity,
        [*routers.PARALLEL_RETRIEVAL, SCIENTIFIC_ABSTAIN],
    )

    graph.add_edge(SEARCH_EXPERIMENTAL, EVALUATE_PDB)
    graph.add_conditional_edges(
        EVALUATE_PDB,
        routers.route_after_pdb_evaluation,
        [SELECT_STRUCTURE, SEARCH_ALPHAFOLD],
    )
    graph.add_edge(SEARCH_ALPHAFOLD, SELECT_STRUCTURE)

    graph.add_edge(SELECT_STRUCTURE, JOIN_RESULTS)
    graph.add_edge(FETCH_ANNOTATIONS, JOIN_RESULTS)
    graph.add_edge(RETRIEVE_KNOWLEDGE, JOIN_RESULTS)

    graph.add_conditional_edges(
        JOIN_RESULTS,
        routers.route_mapping_required,
        [MAP_RESIDUES, BUILD_EVIDENCE],
    )
    graph.add_edge(MAP_RESIDUES, BUILD_EVIDENCE)
    graph.add_edge(BUILD_EVIDENCE, BUILD_SCENE)
    graph.add_conditional_edges(
        BUILD_SCENE,
        routers.route_explanation,
        [GENERATE_EXPLANATION, RUN_CRITIC],
    )
    graph.add_edge(GENERATE_EXPLANATION, RUN_CRITIC)
    graph.add_conditional_edges(
        RUN_CRITIC,
        routers.route_after_critic,
        [COMPLETE, RETURN_PARTIAL, SCIENTIFIC_ABSTAIN],
    )

    for terminal in (COMPLETE, RETURN_PARTIAL, NEEDS_CLARIFICATION, SCIENTIFIC_ABSTAIN):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=MemorySaver())
