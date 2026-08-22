from __future__ import annotations

import os

import pytest

from ..subagents.gene_annotation import get_gene_annotation
from ..subagents.genome_metadata import get_genome_metadata
from ..subagents.species_resolver import resolve_species

# These hit the real NCBI eutils API, so they're opt-in the same way
# test_llm_smoke.py is opt-in for the NVIDIA endpoint: skipped by default,
# run explicitly (e.g. in CI on a schedule, or locally) by setting
# RUN_NCBI_LIVE_TESTS=1. Unlike the NVIDIA smoke test there's no API key to
# check for — NCBI eutils is unauthenticated — so a separate opt-in flag
# is used instead of an env-var-presence check.
_skip_reason = "RUN_NCBI_LIVE_TESTS not set — skipping live NCBI eutils tests"
_run_live = pytest.mark.skipif(not os.getenv("RUN_NCBI_LIVE_TESTS"), reason=_skip_reason)


@_run_live
@pytest.mark.asyncio
async def test_resolve_species_known():
    result = await resolve_species("tiger")
    assert result["assembly_id"] is not None
    assert result["assembly_id"].startswith(("GCF_", "GCA_"))
    assert result["scientific_name"] is not None


@_run_live
@pytest.mark.asyncio
async def test_resolve_species_unknown():
    result = await resolve_species("definitely not a real species xyzzy123")
    assert result["assembly_id"] is None
    assert result["confidence"] == 0.0


@_run_live
@pytest.mark.asyncio
async def test_get_genome_metadata_known_assembly():
    species = await resolve_species("house mouse")
    assert species["assembly_id"] is not None

    metadata = await get_genome_metadata(species["assembly_id"])
    assert metadata["genome_size_bp"] is not None
    assert metadata["genome_size_bp"] > 0


@_run_live
@pytest.mark.asyncio
async def test_get_gene_annotation_known_assembly():
    species = await resolve_species("house mouse")
    assert species["assembly_id"] is not None

    annotation = await get_gene_annotation(species["assembly_id"])
    assert isinstance(annotation["gene_table"], list)
    assert isinstance(annotation["gene_list"], list)
