from langgraph.graph import StateGraph, START, END

from workflows.state import TraitDiscoveryState, FunctionalEvidenceState
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.nodes.gene_mapper_node import gene_mapper_node
from workflows.nodes.literature_support_node import literature_support_node
from workflows.nodes.escalation_nodes import escalate_genome_agent_node, escalate_literature_agent_node
from schemas.common import AgentStatus

_functional_evidence_app = build_functional_evidence_graph()


async def check_gene_list_node(state: TraitDiscoveryState) -> dict:
    """Entry node. Pulls gene_list out of context, same as Task 1 Step 0."""
    gene_list = state.context.get("gene_list", [])
    return {"gene_list": gene_list}


def route_after_gene_list_check(state: TraitDiscoveryState) -> str:
    return "gene_mapper" if state.gene_list else "escalate_genome_agent"


def route_after_gene_mapper(state: TraitDiscoveryState) -> str:
    return "failed" if state.gene_mapper_status == AgentStatus.FAILED else "continue"


async def functional_evidence_node(state: TraitDiscoveryState) -> dict:
    """Invokes the compiled sub-orchestrator graph as a single node in the parent graph."""
    sub_state = FunctionalEvidenceState(
        gene_list=state.gene_list, instruction=state.instruction, context=state.context,
    )
    result = await _functional_evidence_app.ainvoke(sub_state)
    return {
        "pathway_data": result["pathway_data"],
        "protein_data": result["protein_data"],
        "functional_evidence_status": result["status"],
    }


async def join_and_route_node(state: TraitDiscoveryState) -> dict:
    """No-op join point — gives both parallel branches a single place to land before the
    conditional edge decides where to go next."""
    return None

async def gene_mapper_ok_node(state: TraitDiscoveryState) -> dict:
    """No-op fan-out gate — only reachable when gene_mapper succeeded, then splits into
    the two parallel branches."""
    return None

def route_after_join(state: TraitDiscoveryState) -> str:
    if state.literature_status == AgentStatus.NEEDS_AGENT:
        return "escalate_literature_agent"
    if (state.functional_evidence_status == AgentStatus.FAILED
            and state.literature_status == AgentStatus.FAILED):
        return "failed"
    return "aggregate"


async def aggregate_node(state: TraitDiscoveryState) -> dict:
    genes = ", ".join(a.gene_symbol for a in state.go_annotations) or "no genes matched"
    pathways = ", ".join(p.pathway_name for p in state.pathway_data) or "no pathways found"
    explanation = (
        f"Trait '{state.trait_name}' is associated with genes: {genes}. "
        f"Relevant pathways: {pathways}. "
        f"{len(state.evidence)} supporting literature reference(s) found."
    )
    return {"status": AgentStatus.COMPLETED, "explanation": explanation}


async def failed_node(state: TraitDiscoveryState) -> dict:
    return {"status": AgentStatus.FAILED}


def build_trait_discovery_graph():
    graph = StateGraph(TraitDiscoveryState)

    graph.add_node("check_gene_list", check_gene_list_node)
    graph.add_node("escalate_genome_agent", escalate_genome_agent_node)
    graph.add_node("gene_mapper", gene_mapper_node)
    graph.add_node("failed", failed_node)
    graph.add_node("functional_evidence", functional_evidence_node)
    graph.add_node("literature_support", literature_support_node)
    graph.add_node("join_and_route", join_and_route_node)
    graph.add_node("escalate_literature_agent", escalate_literature_agent_node)
    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "check_gene_list")
    graph.add_conditional_edges(
        "check_gene_list", route_after_gene_list_check,
        {"gene_mapper": "gene_mapper", "escalate_genome_agent": "escalate_genome_agent"},
    )
    graph.add_edge("escalate_genome_agent", END)

    graph.add_conditional_edges(
        "gene_mapper", route_after_gene_mapper,
        {"failed": "failed", "continue": "gene_mapper_ok"},
    )
    graph.add_edge("failed", END)

    # fan-out node: only reached when gene_mapper succeeded, then splits into the
    # two parallel branches
    graph.add_node("gene_mapper_ok", gene_mapper_ok_node)
    graph.add_edge("gene_mapper_ok", "functional_evidence")
    graph.add_edge("gene_mapper_ok", "literature_support")
    graph.add_edge("functional_evidence", "join_and_route")
    graph.add_edge("literature_support", "join_and_route")

    graph.add_conditional_edges(
        "join_and_route", route_after_join,
        {
            "escalate_literature_agent": "escalate_literature_agent",
            "failed": "failed",
            "aggregate": "aggregate",
        },
    )
    graph.add_edge("escalate_literature_agent", END)
    graph.add_edge("aggregate", END)

    return graph.compile()