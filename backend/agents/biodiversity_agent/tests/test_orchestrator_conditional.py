"""Conditional path - species-name normalization short-circuits into
FAILED when the species is missing on a species-scoped feature."""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentStatus,
    BiodiversityFeature,
)


@pytest.mark.asyncio
async def test_distribution_without_species_short_circuits(orchestrator) -> None:
    request = AgentRequest(
        instruction="Where does it live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        # species_name deliberately omitted
    )
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.FAILED
    # The orchestrator's own message, not the worker's
    assert "no valid feature" in str(result.output).lower() or "orchestrator" in str(result.output).lower()


@pytest.mark.asyncio
async def test_taxonomy_normalizes_common_to_scientific(orchestrator) -> None:
    """A common name in French must still land on the right worker."""

    request = AgentRequest(
        instruction="Ou vivent les elephants d'Afrique ?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="elephant d'afrique",
    )
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.COMPLETED
    payload = result.output
    # The normalized species name must be the scientific one
    assert payload.species_name == "Loxodonta africana"
