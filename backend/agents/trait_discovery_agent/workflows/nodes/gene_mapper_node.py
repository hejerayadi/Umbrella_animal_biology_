from workflows.state import TraitDiscoveryState
from schemas.inputs import GeneMapperInput
from subagents.gene_mapper import mock_gene_mapper


async def gene_mapper_node(state: TraitDiscoveryState) -> dict:
    out = await mock_gene_mapper(GeneMapperInput(
        trait_name=state.trait_name,
        gene_list=state.gene_list,
        species_name=state.species_name,
        instruction=state.instruction,
        context=state.context,
    ))
    return {"go_annotations": out.go_annotations, "gene_mapper_status": out.status}