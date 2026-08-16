from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..subagents.species_resolver import resolve_species_llm
from ..schemas.outputs import SpeciesResolverOutput


def _make_llm_response(tool_calls=None, content=""):
    response = MagicMock()
    response.tool_calls = tool_calls or []
    response.content = content
    return response


def _make_tool_call(name, args, call_id="call_1"):
    return {
        "id": call_id,
        "name": name,
        "args": args,
        "type": "tool_call",
    }


@pytest.mark.asyncio
async def test_llm_client_raises_triggers_fallback():
    """When get_llm_client raises, resolve_species_llm returns None
    so the node can fall back to deterministic resolution."""
    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", side_effect=EnvironmentError("No key")):
        result = await resolve_species_llm("tiger")
    assert result is None


@pytest.mark.asyncio
async def test_llm_grounded_assembly_id_accepted():
    """When the LLM submits an assembly_id that appears in a prior tool result,
    resolve_species_llm returns the parsed output."""
    assembly_results = [
        {"assembly_id": "GCF_000001545.1", "scientific_name": "Panthera tigris", "assembly_level": "Chromosome"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    call1 = _make_tool_call("search_assembly_by_taxid", {"tax_id": "9685"})
    call2 = _make_tool_call(
        "SpeciesResolverOutput",
        {
            "assembly_id": "GCF_000001545.1",
            "scientific_name": "Panthera tigris",
            "common_name": "tiger",
            "confidence": 1.0,
            "reasoning": "RefSeq assembly found",
        },
        call_id="call_2",
    )
    client.invoke.side_effect = [
        _make_llm_response([call1]),
        _make_llm_response([call2]),
    ]

    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.species_resolver.search_assembly_by_taxid", MagicMock(ainvoke=AsyncMock(return_value=assembly_results))):
            result = await resolve_species_llm("tiger")

    assert result is not None
    assert result["assembly_id"] == "GCF_000001545.1"
    assert result["confidence"] == 1.0
    assert result["reasoning"] == "RefSeq assembly found"


@pytest.mark.asyncio
async def test_llm_fabricated_assembly_id_is_rejected_and_retried():
    """If the LLM submits an assembly_id not present in any tool result,
    resolve_species_llm sends a rejection message and continues the loop.
    It only succeeds after the LLM uses a grounded ID."""
    assembly_results = [
        {"assembly_id": "GCF_000001545.1", "scientific_name": "Panthera tigris", "assembly_level": "Chromosome"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    fabricated_call = _make_tool_call(
        "SpeciesResolverOutput",
        {
            "assembly_id": "GCF_FAKE_ID.1",
            "scientific_name": "Panthera tigris",
            "common_name": "tiger",
            "confidence": 1.0,
            "reasoning": "fabricated",
        },
        call_id="call_fabricated",
    )
    asm_call = _make_tool_call("search_assembly_by_taxid", {"tax_id": "9685"}, call_id="call_asm")
    grounded_call = _make_tool_call(
        "SpeciesResolverOutput",
        {
            "assembly_id": "GCF_000001545.1",
            "scientific_name": "Panthera tigris",
            "common_name": "tiger",
            "confidence": 1.0,
            "reasoning": "grounded",
        },
        call_id="call_grounded",
    )

    client.invoke.side_effect = [
        _make_llm_response([fabricated_call]),
        _make_llm_response([asm_call]),
        _make_llm_response([grounded_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.species_resolver.search_assembly_by_taxid", MagicMock(ainvoke=AsyncMock(return_value=assembly_results))):
            result = await resolve_species_llm("tiger")

    assert result is not None
    assert result["assembly_id"] == "GCF_000001545.1"
    assert result["reasoning"] == "grounded"
    assert client.invoke.call_count == 3


@pytest.mark.asyncio
async def test_empty_search_then_reformulated_retries_once():
    """If the first tool call returns empty results, the LLM should retry with a
    reformulated query. We assert the loop makes exactly 2 LLM calls."""
    empty_results = []
    assembly_results = [
        {"assembly_id": "GCF_000001545.1", "scientific_name": "Panthera tigris", "assembly_level": "Chromosome"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    empty_tax_call = _make_tool_call("search_taxonomy", {"query": "Panthera tigris altaica"}, call_id="call_empty")
    good_tax_call = _make_tool_call("search_taxonomy", {"query": "Panthera tigris"}, call_id="call_good")
    asm_call = _make_tool_call("search_assembly_by_taxid", {"tax_id": "9691"}, call_id="call_asm")
    final_output_call = _make_tool_call(
        "SpeciesResolverOutput",
        {
            "assembly_id": "GCF_000001545.1",
            "scientific_name": "Panthera tigris",
            "common_name": "tiger",
            "confidence": 1.0,
            "reasoning": "found",
        },
        call_id="call_out",
    )

    client.invoke.side_effect = [
        _make_llm_response([empty_tax_call]),
        _make_llm_response([good_tax_call]),
        _make_llm_response([asm_call]),
        _make_llm_response([final_output_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.species_resolver.search_taxonomy", MagicMock(ainvoke=AsyncMock(side_effect=[empty_results, [{"tax_id": "9691", "scientific_name": "Panthera tigris", "common_name": "tiger", "rank": "species"}]]))):
            with patch("backend.agents.genome_agent.subagents.species_resolver.search_assembly_by_taxid", MagicMock(ainvoke=AsyncMock(return_value=assembly_results))):
                result = await resolve_species_llm("Panthera tigris altaica")

    assert result is not None
    assert result["assembly_id"] == "GCF_000001545.1"
    assert client.invoke.call_count == 4


@pytest.mark.asyncio
async def test_ambiguous_elephant_returns_lowered_confidence():
    """With multiple taxonomy candidates, the LLM should return a valid
    assembly but with confidence < 1.0 to signal ambiguity."""
    tax_results = [
        {"tax_id": "9783", "scientific_name": "Elephas maximus", "common_name": "Asian elephant", "rank": "species"},
        {"tax_id": "9813", "scientific_name": "Loxodonta africana", "common_name": "African elephant", "rank": "species"},
        {"tax_id": "1229059", "scientific_name": "Elephantidae", "common_name": "elephants", "rank": "family"},
    ]
    assembly_results = [
        {"assembly_id": "GCF_000001895.1", "scientific_name": "Elephas maximus", "assembly_level": "Chromosome"},
        {"assembly_id": "GCF_000001905.1", "scientific_name": "Loxodonta africana", "assembly_level": "Chromosome"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    tax_call = _make_tool_call("search_taxonomy", {"query": "elephant"}, call_id="call_tax")
    asm_call = _make_tool_call("search_assembly_by_taxid", {"tax_id": "9783"}, call_id="call_asm")
    output_call = _make_tool_call(
        "SpeciesResolverOutput",
        {
            "assembly_id": "GCF_000001895.1",
            "scientific_name": "Elephas maximus",
            "common_name": "Asian elephant",
            "confidence": 0.7,
            "reasoning": "Multiple elephant candidates; selected Elephas maximus based on 'asian elephant' common name",
        },
        call_id="call_out",
    )

    client.invoke.side_effect = [
        _make_llm_response([tax_call]),
        _make_llm_response([asm_call]),
        _make_llm_response([output_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.species_resolver.search_taxonomy", MagicMock(ainvoke=AsyncMock(return_value=tax_results))):
            with patch("backend.agents.genome_agent.subagents.species_resolver.search_assembly_by_taxid", MagicMock(ainvoke=AsyncMock(return_value=assembly_results))):
                result = await resolve_species_llm("elephant")

    assert result is not None
    assert result["assembly_id"] == "GCF_000001895.1"
    assert result["confidence"] == 0.7
    assert "asian elephant" in result["reasoning"]


@pytest.mark.asyncio
async def test_house_mouse_clean_single_match():
    """Clean single match returns confidence == 1.0 and non-empty reasoning."""
    tax_results = [
        {"tax_id": "10090", "scientific_name": "Mus musculus", "common_name": "house mouse", "rank": "species"},
    ]
    assembly_results = [
        {"assembly_id": "GCF_000001635.1", "scientific_name": "Mus musculus", "assembly_level": "Chromosome"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    tax_call = _make_tool_call("search_taxonomy", {"query": "house mouse"}, call_id="call_tax")
    asm_call = _make_tool_call("search_assembly_by_taxid", {"tax_id": "10090"}, call_id="call_asm")
    output_call = _make_tool_call(
        "SpeciesResolverOutput",
        {
            "assembly_id": "GCF_000001635.1",
            "scientific_name": "Mus musculus",
            "common_name": "house mouse",
            "confidence": 1.0,
            "reasoning": "Unambiguous single candidate",
        },
        call_id="call_out",
    )

    client.invoke.side_effect = [
        _make_llm_response([tax_call]),
        _make_llm_response([asm_call]),
        _make_llm_response([output_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.species_resolver.search_taxonomy", MagicMock(ainvoke=AsyncMock(return_value=tax_results))):
            with patch("backend.agents.genome_agent.subagents.species_resolver.search_assembly_by_taxid", MagicMock(ainvoke=AsyncMock(return_value=assembly_results))):
                result = await resolve_species_llm("house mouse")

    assert result is not None
    assert result["confidence"] == 1.0
    assert len(result["reasoning"]) > 0


@pytest.mark.asyncio
async def test_no_match_after_reformulation_returns_none():
    """When search_taxonomy returns empty results for both the original and
    a reformulated query, resolve_species_llm exhausts its loop and returns
    None — not a fabricated assembly_id."""
    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    empty_results = []

    # LLM calls search_taxonomy twice (original + reformulated), both empty.
    # The loop runs up to 4 times; after two empty results the LLM may
    # make additional calls with no tool_calls before giving up.
    tax_call_1 = _make_tool_call("search_taxonomy", {"query": "definitely not a real species xyzzy123"}, call_id="call_1")
    tax_call_2 = _make_tool_call("search_taxonomy", {"query": "xyzzy123 species"}, call_id="call_2")

    client.invoke.side_effect = [
        _make_llm_response([tax_call_1]),
        _make_llm_response([tax_call_2]),
        _make_llm_response([]),
        _make_llm_response([]),
    ]

    with patch("backend.agents.genome_agent.subagents.species_resolver.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.species_resolver.search_taxonomy", MagicMock(ainvoke=AsyncMock(side_effect=[empty_results, empty_results]))):
            result = await resolve_species_llm("definitely not a real species xyzzy123")

    assert result is None
    assert client.invoke.call_count == 4
