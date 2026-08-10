"""Species name -> NCBI taxonomy id.

`AgentTask` requires a `taxon_id`, but a caller usually starts from the name a
user typed. `/execute` resolves that internally; this exposes the same step on
its own, so anything building an `AgentTask` by hand - a script, another
service, the frontend - can fill the field in without guessing or running a
whole analysis to find out.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.agents.Protein_visualization.app.api.v1.dependencies import get_taxonomy_capability
from backend.agents.Protein_visualization.app.capabilities.taxonomy import TaxonomyCapability
from backend.agents.Protein_visualization.app.contracts.envelope import ApiResponse, success
from backend.agents.Protein_visualization.app.contracts.protein_request import SpeciesContract
from backend.agents.Protein_visualization.app.observability.logging import log_stage

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])

logger = logging.getLogger("app.taxonomy")


@router.get("", response_model=ApiResponse[SpeciesContract])
async def resolve_species(
    taxonomy: Annotated[TaxonomyCapability, Depends(get_taxonomy_capability)],
    name: Annotated[str, Query(min_length=2, description="Scientific or common species name")],
) -> ApiResponse[SpeciesContract]:
    """404s through `ProteinNotFoundError` when the name is unknown or ambiguous."""
    with log_stage(logger, "taxonomy_resolve", capability="taxonomy") as outcome:
        species = await taxonomy.resolve(name)
        outcome["taxon_id"] = species.taxon_id
    return success(SpeciesContract(scientific_name=species.scientific_name, taxon_id=species.taxon_id))
