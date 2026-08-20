import logging

from langgraph.graph import END, START, StateGraph

from schemas.common import AgentStatus
from workflows.explanation_writer import write_explanation
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.nodes.escalation_nodes import escalate_genome_agent_node, escalate_literature_agent_node
from workflows.nodes.gene_mapper_node import gene_mapper_node
from workflows.nodes.literature_support_node import literature_support_node
from workflows.state import FunctionalEvidenceState, TraitDiscoveryState

logger = logging.getLogger(__name__)


async def aggregate_node(state: TraitDiscoveryState) -> dict:
    genes = ", ".join(a.gene_symbol for a in state.go_annotations)
    pathways = ", ".join(p.pathway_name for p in state.pathway_data)
    proteins = ", ".join(p.protein_name for p in state.protein_data)
    evidence = "; ".join(e.short_summary for e in state.evidence)

    explanation = await write_explanation(
        trait_name=state.trait_name,
        species_name=state.species_name,
        genes=genes,
        pathways=pathways,
        proteins=proteins,
        evidence=evidence,
    )
    logger.info("aggregate explanation=%s", explanation)
    return {"status": AgentStatus.COMPLETED, "explanation": explanation}


async def check_gene_list_node(state: TraitDiscoveryState) -> dict:
    logger.info("check_gene_list incoming gene list=%s", state.gene_list)
    if state.gene_list:
        return {"gene_list": state.gene_list}
    return {"status": AgentStatus.NEEDS_AGENT}


async def failed_node(state: TraitDiscoveryState) -> dict:
    logger.info("workflow terminated in failed node")
    return {"status": AgentStatus.FAILED}


async def join_and_route_node(state: TraitDiscoveryState) -> dict:
    logger.info(
        "join_and_route statuses gene_mapper=%s functional_evidence=%s literature=%s",
        state.gene_mapper_status,
        state.functional_evidence_status,
        state.literature_status,
    )
    if state.gene_mapper_status == AgentStatus.FAILED:
        return {"status": AgentStatus.FAILED}
    if state.literature_status == AgentStatus.NEEDS_AGENT:
        return {"status": AgentStatus.NEEDS_AGENT}
    if state.functional_evidence_status == AgentStatus.FAILED:
        return {"status": AgentStatus.FAILED}
    return {"status": AgentStatus.COMPLETED}


def route_after_check_gene_list(state: TraitDiscoveryState) -> str:
    return "escalate_genome_agent" if state.status == AgentStatus.NEEDS_AGENT else "gene_mapper"


def route_after_join(state: TraitDiscoveryState) -> str:
    if state.status == AgentStatus.NEEDS_AGENT:
        return "escalate_literature_agent"
    if state.status == AgentStatus.FAILED:
        return "failed"
    return "aggregate"


def build_trait_discovery_graph():
    graph = StateGraph(TraitDiscoveryState)
    functional_evidence_graph = build_functional_evidence_graph()

    async def functional_evidence_node(state: TraitDiscoveryState) -> dict:
        logger.info("functional_evidence input gene list=%s", state.gene_list)
        sub_result = await functional_evidence_graph.ainvoke(FunctionalEvidenceState(
            gene_list=state.gene_list,
            instruction=state.instruction,
            context=state.context,
        ))
        logger.info("functional_evidence output=%s", sub_result)
        return {
            "pathway_data": sub_result.get("pathway_data", []),
            "protein_data": sub_result.get("protein_data", []),
            "functional_evidence_status": sub_result.get("status"),
        }

    graph.add_node("check_gene_list", check_gene_list_node)
    graph.add_node("gene_mapper", gene_mapper_node)
    graph.add_node("functional_evidence", functional_evidence_node)
    graph.add_node("literature_support", literature_support_node)
    graph.add_node("join_and_route", join_and_route_node)
    graph.add_node("escalate_genome_agent", escalate_genome_agent_node)
    graph.add_node("escalate_literature_agent", escalate_literature_agent_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("failed", failed_node)

    graph.add_edge(START, "check_gene_list")
    graph.add_conditional_edges("check_gene_list", route_after_check_gene_list, {
        "escalate_genome_agent": "escalate_genome_agent",
        "gene_mapper": "gene_mapper",
    })
    graph.add_edge("gene_mapper", "functional_evidence")
    graph.add_edge("functional_evidence", "literature_support")
    graph.add_edge("literature_support", "join_and_route")
    graph.add_conditional_edges("join_and_route", route_after_join, {
        "escalate_literature_agent": "escalate_literature_agent",
        "failed": "failed",
        "aggregate": "aggregate",
    })
    graph.add_edge("escalate_genome_agent", END)
    graph.add_edge("escalate_literature_agent", END)
    graph.add_edge("aggregate", END)
    graph.add_edge("failed", END)
    return graph.compile()