"""Parallel path - multiple features dispatched concurrently."""

from __future__ import annotations

import asyncio

import pytest

from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentStatus,
    BiodiversityFeature,
)


@pytest.mark.asyncio
async def test_parallel_distribution_and_migration_are_aggregated(orchestrator) -> None:
    request = AgentRequest(
        instruction="Where do African elephants live and how do they migrate?",
        context={
            "features": [
                BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
                BiodiversityFeature.MIGRATION_ANALYSIS.value,
            ]
        },
        species_name="Loxodonta africana",
    )
    result = await orchestrator.run(request)

    assert result.status is AgentStatus.COMPLETED
    # Both worker outputs must be present in the combined payload
    assert isinstance(result.output, dict)
    assert BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value in result.output
    assert BiodiversityFeature.MIGRATION_ANALYSIS.value in result.output
    assert result.migration_route  # populated by the migration worker
    assert result.observation_count  # populated by the distribution worker
    assert "Species Distribution Agent" in result.source_agents
    assert "Migration Analysis Agent" in result.source_agents


@pytest.mark.asyncio
async def test_parallel_partial_failure_still_returns_completed(orchestrator) -> None:
    """One worker COMPLETED + one FAILED should still surface a COMPLETED
    aggregate (partial success)."""

    request = AgentRequest(
        instruction="Where do polar bears live and how do they migrate?",
        context={
            "features": [
                BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
                BiodiversityFeature.MIGRATION_ANALYSIS.value,
            ]
        },
        species_name="Ursus maritimus",  # distribution mock knows it, migration mock does not
    )
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.COMPLETED
    assert result.observation_count  # distribution succeeded
    assert result.migration_route is None  # migration failed


@pytest.mark.asyncio
async def test_parallel_all_failed_returns_failed(orchestrator) -> None:
    request = AgentRequest(
        instruction="anything",
        context={
            "features": [
                BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
                BiodiversityFeature.MIGRATION_ANALYSIS.value,
            ]
        },
        species_name="Draco magicus",  # unknown to every mock
    )
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.FAILED
