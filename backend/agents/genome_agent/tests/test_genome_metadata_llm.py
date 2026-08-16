from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..subagents.genome_metadata import (
    _check_duplicate_tool_call,
    fetch_assembly_stats,
    fetch_metadata_fallback,
    resolve_metadata_llm,
)


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
async def test_single_assembly_no_substitution():
    """Single-assembly case: no substitution, assembly_id_used == input."""
    assembly_id = "GCF_000001545.1"
    stats = {
        "assembly_id": assembly_id,
        "scientific_name": "Panthera tigris",
        "assembly_level": "Chromosome",
        "genome_size_bp": 2503838624,
        "chromosome_count": 1,
        "tax_id": "9685",
        "submission_date": "2023/01/01",
    }
    alternates = [
        {"assembly_id": assembly_id, "assembly_level": "Chromosome", "submission_date": "2023/01/01"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    fetch_stats_call = _make_tool_call("fetch_assembly_stats", {"assembly_id": assembly_id}, call_id="call_1")
    list_alt_call = _make_tool_call("list_alternate_assemblies", {"tax_id": "9685"}, call_id="call_2")
    output_call = _make_tool_call(
        "GenomeMetadataOutput",
        {
            "genome_size_bp": 2503838624,
            "chromosome_count": 1,
            "karyotype": None,
            "assembly_level": "Chromosome",
            "assembly_id_used": assembly_id,
            "reasoning": "Single assembly available",
        },
        call_id="call_3",
    )

    client.invoke.side_effect = [
        _make_llm_response([fetch_stats_call]),
        _make_llm_response([list_alt_call]),
        _make_llm_response([output_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.genome_metadata.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.genome_metadata.fetch_assembly_stats", MagicMock(ainvoke=AsyncMock(return_value=stats))):
            with patch("backend.agents.genome_agent.subagents.genome_metadata.list_alternate_assemblies", MagicMock(ainvoke=AsyncMock(return_value=alternates))):
                result = await resolve_metadata_llm("tiger", assembly_id)

    assert result is not None
    assert result["assembly_id_used"] == assembly_id
    assert result["genome_size_bp"] == 2503838624
    assert result["chromosome_count"] == 1
    assert result["assembly_level"] == "Chromosome"


@pytest.mark.asyncio
async def test_multi_assembly_better_option_triggers_substitution():
    """Multi-assembly case with a genuinely better option: substitution happens."""
    given_id = "GCA_000001545.1"
    better_id = "GCF_000001545.1"

    given_stats = {
        "assembly_id": given_id,
        "scientific_name": "Panthera tigris",
        "assembly_level": "Scaffold",
        "genome_size_bp": 2503838624,
        "chromosome_count": None,
        "tax_id": "9685",
        "submission_date": "2015/06/01",
    }
    better_stats = {
        "assembly_id": better_id,
        "scientific_name": "Panthera tigris",
        "assembly_level": "Chromosome",
        "genome_size_bp": 2503838624,
        "chromosome_count": 1,
        "tax_id": "9685",
        "submission_date": "2023/01/01",
    }
    alternates = [
        {"assembly_id": given_id, "assembly_level": "Scaffold", "submission_date": "2015/06/01"},
        {"assembly_id": better_id, "assembly_level": "Chromosome", "submission_date": "2023/01/01"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    fetch_stats_call_1 = _make_tool_call("fetch_assembly_stats", {"assembly_id": given_id}, call_id="call_1")
    list_alt_call = _make_tool_call("list_alternate_assemblies", {"tax_id": "9685"}, call_id="call_2")
    fetch_stats_call_2 = _make_tool_call("fetch_assembly_stats", {"assembly_id": better_id}, call_id="call_3")
    output_call = _make_tool_call(
        "GenomeMetadataOutput",
        {
            "genome_size_bp": 2503838624,
            "chromosome_count": 1,
            "karyotype": None,
            "assembly_level": "Chromosome",
            "assembly_id_used": better_id,
            "reasoning": "Substituted GCA_ with GCF_ RefSeq chromosome-level assembly",
        },
        call_id="call_4",
    )

    client.invoke.side_effect = [
        _make_llm_response([fetch_stats_call_1]),
        _make_llm_response([list_alt_call]),
        _make_llm_response([fetch_stats_call_2]),
        _make_llm_response([output_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.genome_metadata.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.genome_metadata.fetch_assembly_stats", MagicMock(ainvoke=AsyncMock(side_effect=[given_stats, better_stats]))):
            with patch("backend.agents.genome_agent.subagents.genome_metadata.list_alternate_assemblies", MagicMock(ainvoke=AsyncMock(return_value=alternates))):
                result = await resolve_metadata_llm("tiger", given_id)

    assert result is not None
    assert result["assembly_id_used"] == better_id
    assert result["assembly_level"] == "Chromosome"
    assert "GCF_" in result["reasoning"] or "RefSeq" in result["reasoning"]


@pytest.mark.asyncio
async def test_llm_unavailable_uses_fallback():
    """LLM unavailable: deterministic fallback returns usable stats."""
    assembly_id = "GCF_000001545.1"
    stats = {
        "assembly_id": assembly_id,
        "scientific_name": "Panthera tigris",
        "assembly_level": "Chromosome",
        "genome_size_bp": 2503838624,
        "chromosome_count": 1,
        "tax_id": "9685",
        "submission_date": "2023/01/01",
    }

    with patch("backend.agents.genome_agent.subagents.genome_metadata.get_genome_metadata", AsyncMock(return_value=stats)):
        result = await fetch_metadata_fallback(assembly_id)

    assert result is not None
    assert result["assembly_id_used"] == assembly_id
    assert result["genome_size_bp"] == 2503838624
    assert "fallback" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_garbage_genome_size_surfaces_as_null():
    """Garbage total_length in tool result surfaces as null, never as -1."""
    assembly_id = "GCF_000001545.1"
    stats = {
        "assembly_id": assembly_id,
        "scientific_name": "Panthera tigris",
        "assembly_level": "Chromosome",
        "genome_size_bp": -1,
        "chromosome_count": 1,
        "tax_id": "9685",
        "submission_date": "2023/01/01",
    }
    alternates = [
        {"assembly_id": assembly_id, "assembly_level": "Chromosome", "submission_date": "2023/01/01"},
    ]

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = MagicMock()

    fetch_stats_call = _make_tool_call("fetch_assembly_stats", {"assembly_id": assembly_id}, call_id="call_1")
    output_call = _make_tool_call(
        "GenomeMetadataOutput",
        {
            "genome_size_bp": -1,
            "chromosome_count": 1,
            "karyotype": None,
            "assembly_level": "Chromosome",
            "assembly_id_used": assembly_id,
            "reasoning": "invalid size",
        },
        call_id="call_2",
    )

    client.invoke.side_effect = [
        _make_llm_response([fetch_stats_call]),
        _make_llm_response([output_call]),
    ]

    with patch("backend.agents.genome_agent.subagents.genome_metadata.get_llm_client", return_value=client):
        with patch("backend.agents.genome_agent.subagents.genome_metadata.fetch_assembly_stats", MagicMock(ainvoke=AsyncMock(return_value=stats))):
            with patch("backend.agents.genome_agent.subagents.genome_metadata.list_alternate_assemblies", MagicMock(ainvoke=AsyncMock(return_value=alternates))):
                result = await resolve_metadata_llm("tiger", assembly_id)

    assert result is not None
    assert result["genome_size_bp"] is None
    assert result["chromosome_count"] == 1


@pytest.mark.asyncio
async def test_chromosome_count_zero_scaffold_becomes_null():
    """Scaffold-level chromosome_count=0 becomes None; chromosome-level 0 stays 0."""
    fake_esearch = MagicMock()
    fake_esearch.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

    # Case 1: Scaffold assembly with chromosome_count=0 -> null
    fake_esummary_scaffold = MagicMock()
    fake_esummary_scaffold.json.return_value = {
        "result": {
            "12345": {
                "assemblyaccession": "GCA_029237445.1",
                "organism": "Mus musculus",
                "assemblystatus": "Scaffold",
                "taxid": "10090",
                "submissiondate": "2023/03/17",
                "meta": "<Stats><Stat category=\"chromosome_count\" sequence_tag=\"all\">0</Stat><Stat category=\"total_length\" sequence_tag=\"all\">2528201171</Stat></Stats>",
            }
        }
    }

    with patch("backend.agents.genome_agent.subagents.genome_metadata.ncbi_get", side_effect=[fake_esearch, fake_esummary_scaffold]):
        result = await fetch_assembly_stats.ainvoke({"assembly_id": "GCA_029237445.1"})

    assert result["chromosome_count"] is None
    assert result["assembly_level"] == "Scaffold"

    # Case 2: Chromosome assembly with chromosome_count=0 -> stays 0
    fake_esummary_chromosome = MagicMock()
    fake_esummary_chromosome.json.return_value = {
        "result": {
            "12345": {
                "assemblyaccession": "GCF_000001635.27",
                "organism": "Mus musculus",
                "assemblystatus": "Chromosome",
                "taxid": "10090",
                "submissiondate": "2023/01/01",
                "meta": "<Stats><Stat category=\"chromosome_count\" sequence_tag=\"all\">0</Stat><Stat category=\"total_length\" sequence_tag=\"all\">2725821377</Stat></Stats>",
            }
        }
    }

    with patch("backend.agents.genome_agent.subagents.genome_metadata.ncbi_get", side_effect=[fake_esearch, fake_esummary_chromosome]):
        result2 = await fetch_assembly_stats.ainvoke({"assembly_id": "GCF_000001635.27"})

    assert result2["chromosome_count"] == 0
    assert result2["assembly_level"] == "Chromosome"


def test_check_duplicate_tool_call_detects_repeat():
    """_check_duplicate_tool_call returns (True, cached_result) for identical call."""
    from langchain_core.messages import AIMessage, ToolMessage

    # Build a fake history: one prior tool call with its result
    prior_call = {
        "id": "call_1",
        "name": "fetch_assembly_stats",
        "args": {"assembly_id": "GCA_029237445.1"},
        "type": "tool_call",
    }
    prior_result = ToolMessage(
        content='{"assembly_id": "GCA_029237445.1", "assembly_level": "Scaffold"}',
        tool_call_id="call_1",
    )
    messages = [
        AIMessage(content="", tool_calls=[prior_call]),
        prior_result,
    ]

    # Same call again — should hit cache
    is_dup, cached = _check_duplicate_tool_call(
        messages, "fetch_assembly_stats", {"assembly_id": "GCA_029237445.1"}
    )
    assert is_dup is True
    assert cached is not None
    assert "GCA_029237445.1" in cached

    # Different call — should miss
    is_dup2, cached2 = _check_duplicate_tool_call(
        messages, "fetch_assembly_stats", {"assembly_id": "GCF_000001635.27"}
    )
    assert is_dup2 is False
    assert cached2 is None

    # Different tool name — should miss
    is_dup3, cached3 = _check_duplicate_tool_call(
        messages, "list_alternate_assemblies", {"tax_id": "10090"}
    )
    assert is_dup3 is False
    assert cached3 is None
