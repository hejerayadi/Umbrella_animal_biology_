import logging

from workflows.state import TraitDiscoveryState
from schemas.inputs import GeneMapperInput
from subagents.gene_mapper import mock_gene_mapper

logger = logging.getLogger(__name__)


async def gene_mapper_node(state: TraitDiscoveryState) -> dict:
    logger.info("gene_mapper input gene list=%s", state.gene_list)
    out = await mock_gene_mapper(GeneMapperInput(
        trait_name=state.trait_name,
        gene_list=state.gene_list,
        species_name=state.species_name,
        instruction=state.instruction,
        context=state.context,
    ))
    logger.info("gene_mapper output=%s", out)
    return {"go_annotations": out.go_annotations, "gene_mapper_status": out.status}