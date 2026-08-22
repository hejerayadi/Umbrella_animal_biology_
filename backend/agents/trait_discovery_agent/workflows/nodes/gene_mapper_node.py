import logging

from workflows.state import TraitDiscoveryState
from schemas.inputs import GeneMapperInput
from schemas.common import AgentStatus
from subagents.gene_mapper import gene_mapper_agent  # real agent

logger = logging.getLogger(__name__)

async def gene_mapper_node(state: TraitDiscoveryState) -> dict:
    logger.info("gene_mapper input gene list=%s", state.gene_list)
    out = await gene_mapper_agent(GeneMapperInput(
        trait_name=state.trait_name,
        gene_list=state.gene_list,
        species_name=state.species_name,
        instruction=state.instruction,
        context=state.context,
    ))
    logger.info("gene_mapper output=%s", out)
    # §0.2/§8: the subagent's own status is FAILED whenever ANY gene is
    # unmatched (its contract, tested independently in test_gene_mapper.py),
    # which conflates "resolved nothing" with "resolved most of them". The
    # orchestrator cares about a coarser distinction: resolving nothing is
    # the one Gene Mapper failure mode critical enough to halt the whole run
    # (see join_and_route_node) — a partial match should still flow
    # go_annotations + unmatched_genes downstream rather than crash.
    routing_status = (
        AgentStatus.FAILED if not out.go_annotations else AgentStatus.COMPLETED
    )
    return {
        "go_annotations": out.go_annotations,
        "unmatched_genes": out.unmatched_genes,
        "gene_mapper_status": routing_status,
    }