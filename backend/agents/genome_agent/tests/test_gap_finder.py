"""
tests/test_gap_finder.py
=========================
Unit tests for the new gap-finder subagent (subagents/gap_finder.py) and its
orchestrator node (workflows/nodes/gap_finder_node.py).

Feature-table parsing is pure/offline and tested directly. Everything that
hits NCBI (`find_target_gaps`) is mocked at the module boundary, matching the
pattern used elsewhere in this suite (see test_reconstruction_path.py).
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genome_agent.subagents.gap_finder import (
    GapFinderError,
    _extract_fasta_sequence,
    _parse_feature_table_gaps,
)


# ===========================================================================
# 1. FEATURE TABLE PARSING (pure, offline)
# ===========================================================================

_SAMPLE_FT = """\
>Feature NW_007907101.1
1\t100\tgene
\t\t\tgene\tSOME1
125430\t125474\tgap
\t\t\tgap_type\twithin scaffold
\t\t\testimated_length\t45
200\t300\tCDS
\t\t\tproduct\thypothetical protein
891203\t891272\tassembly_gap
\t\t\testimated_length\tunknown
"""


class TestParseFeatureTableGaps:
    def test_finds_all_gaps(self):
        gaps = _parse_feature_table_gaps(_SAMPLE_FT)
        assert len(gaps) == 2

    def test_prefers_estimated_length_qualifier(self):
        gaps = _parse_feature_table_gaps(_SAMPLE_FT)
        first = gaps[0]
        assert first["start"] == 125430
        assert first["end"] == 125474
        assert first["length"] == 45

    def test_falls_back_to_span_when_length_unknown(self):
        gaps = _parse_feature_table_gaps(_SAMPLE_FT)
        second = gaps[1]
        assert second["start"] == 891203
        assert second["end"] == 891272
        # "unknown" isn't digits, so length falls back to the coordinate span
        assert second["length"] == 891272 - 891203 + 1

    def test_non_gap_features_are_ignored(self):
        gaps = _parse_feature_table_gaps(_SAMPLE_FT)
        starts = {g["start"] for g in gaps}
        assert 1 not in starts
        assert 200 not in starts

    def test_no_gaps_returns_empty_list(self):
        text = ">Feature NW_000000001.1\n1\t100\tgene\n\t\t\tgene\tFOO\n"
        assert _parse_feature_table_gaps(text) == []

    def test_empty_text_returns_empty_list(self):
        assert _parse_feature_table_gaps("") == []

    def test_reversed_coordinates_are_normalised(self):
        text = "500\t450\tgap\n\t\t\testimated_length\t51\n"
        gaps = _parse_feature_table_gaps(text)
        assert gaps == [{"start": 450, "end": 500, "length": 51}]

    def test_trailing_gap_without_following_feature_is_captured(self):
        """A gap as the very last feature in the file (no feature line after
        it) must still be appended - regression guard for the end-of-loop
        flush."""
        text = "10\t20\tgap\n\t\t\testimated_length\t11\n"
        gaps = _parse_feature_table_gaps(text)
        assert gaps == [{"start": 10, "end": 20, "length": 11}]


# ===========================================================================
# 2. FASTA STRIPPING (pure, offline)
# ===========================================================================

class TestExtractFastaSequence:
    def test_strips_header_and_joins_lines(self):
        fasta = ">NW_007907101.1:125380-125429\nACTGACTGAC\nTGACTGACTG\n"
        assert _extract_fasta_sequence(fasta) == "ACTGACTGACTGACTGACTG"

    def test_handles_no_trailing_newline(self):
        fasta = ">header\nACGT"
        assert _extract_fasta_sequence(fasta) == "ACGT"

    def test_empty_input(self):
        assert _extract_fasta_sequence("") == ""


# ===========================================================================
# 3. find_target_gaps — orchestration wiring (mocked I/O)
# ===========================================================================

class TestFindTargetGaps:
    def test_raises_gap_finder_error_when_assembly_unresolvable(self):
        from genome_agent.subagents import gap_finder

        with patch.object(gap_finder, "_resolve_assembly_uid", new=AsyncMock(return_value=None)):
            with pytest.raises(GapFinderError):
                asyncio.run(gap_finder.find_target_gaps("GCF_doesnotexist.1"))

    def test_empty_gaps_is_not_an_error(self):
        from genome_agent.subagents import gap_finder

        with (
            patch.object(gap_finder, "_resolve_assembly_uid", new=AsyncMock(return_value="111")),
            patch.object(gap_finder, "_resolve_nuccore_id", new=AsyncMock(return_value="222")),
            patch.object(gap_finder, "_fetch_accession_version", new=AsyncMock(return_value="NW_000000001.1")),
            patch.object(gap_finder, "_fetch_feature_table", new=AsyncMock(return_value="1\t100\tgene\n")),
        ):
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1"))

        assert result == {"sequence_accession": "NW_000000001.1", "target_gaps": []}

    def test_gaps_are_enriched_with_flanks(self):
        from genome_agent.subagents import gap_finder

        async def _fake_window(assembly_id, start, stop):
            return f">window\n{'L' if stop < 125430 else 'R'}" * 1 + "\nSEQ\n"

        with (
            patch.object(gap_finder, "_resolve_assembly_uid", new=AsyncMock(return_value="111")),
            patch.object(gap_finder, "_resolve_nuccore_id", new=AsyncMock(return_value="222")),
            patch.object(gap_finder, "_fetch_accession_version", new=AsyncMock(return_value="NW_007907101.1")),
            patch.object(
                gap_finder,
                "_fetch_feature_table",
                new=AsyncMock(return_value=_SAMPLE_FT),
            ),
            patch.object(gap_finder, "fetch_sequence_window", new=_fake_window),
        ):
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1", max_gaps=5))

        assert result["sequence_accession"] == "NW_007907101.1"
        assert len(result["target_gaps"]) == 2
        for gap in result["target_gaps"]:
            assert "left_flank" in gap
            assert "right_flank" in gap

    def test_max_gaps_caps_results(self):
        from genome_agent.subagents import gap_finder

        many_gaps_ft = "\n".join(
            f"{100 * i}\t{100 * i + 10}\tgap\n\t\t\testimated_length\t11" for i in range(1, 10)
        )

        async def _fake_window(assembly_id, start, stop):
            return ">w\nAAAA\n"

        with (
            patch.object(gap_finder, "_resolve_assembly_uid", new=AsyncMock(return_value="111")),
            patch.object(gap_finder, "_resolve_nuccore_id", new=AsyncMock(return_value="222")),
            patch.object(gap_finder, "_fetch_accession_version", new=AsyncMock(return_value="NW_x.1")),
            patch.object(gap_finder, "_fetch_feature_table", new=AsyncMock(return_value=many_gaps_ft)),
            patch.object(gap_finder, "fetch_sequence_window", new=_fake_window),
        ):
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1", max_gaps=3))

        assert len(result["target_gaps"]) == 3


# ===========================================================================
# 4. find_target_gaps_node — degrades gracefully on failure
# ===========================================================================

class TestFindTargetGapsNode:
    def test_success_populates_state_fields(self):
        from genome_agent.workflows.nodes.gap_finder_node import find_target_gaps_node
        from genome_agent.workflows.state import GenomeAgentState

        state = GenomeAgentState(assembly_id="GCF_x.1")
        fake_result = {
            "sequence_accession": "NW_007907101.1",
            "target_gaps": [{"start": 1, "end": 10, "length": 10, "left_flank": "", "right_flank": "AC"}],
        }
        with patch(
            "genome_agent.workflows.nodes.gap_finder_node.find_target_gaps",
            new=AsyncMock(return_value=fake_result),
        ):
            result = asyncio.run(find_target_gaps_node(state))

        assert result["sequence_accession"] == "NW_007907101.1"
        assert result["target_gaps"] == fake_result["target_gaps"]

    def test_exception_degrades_to_empty_with_warning(self):
        from genome_agent.workflows.nodes.gap_finder_node import find_target_gaps_node
        from genome_agent.workflows.state import GenomeAgentState

        state = GenomeAgentState(assembly_id="GCF_x.1")
        with patch(
            "genome_agent.workflows.nodes.gap_finder_node.find_target_gaps",
            new=AsyncMock(side_effect=RuntimeError("network down")),
        ):
            result = asyncio.run(find_target_gaps_node(state))

        assert result["sequence_accession"] is None
        assert result["target_gaps"] == []
        assert any("find_target_gaps" in e for e in result["errors"])
