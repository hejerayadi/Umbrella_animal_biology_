"""
tests/test_gap_finder.py
=========================
Unit tests for the gap-finder subagent (subagents/gap_finder.py) and its
orchestrator node (workflows/nodes/gap_finder_node.py).

Gap detection and flank slicing are pure/offline and tested directly.
Everything that hits NCBI (`find_target_gaps`) is mocked at the module
boundary, matching the pattern used elsewhere in this suite (see
test_reconstruction_path.py).
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genome_agent.subagents.gap_finder import (
    MIN_GAP_BP,
    GapFinderError,
    _attach_flanks,
    _extract_fasta_sequence,
    _find_n_runs,
    _is_wgs_master,
)


# ===========================================================================
# 0. WGS MASTER RECORDS (pure, offline)
# ===========================================================================

class TestIsWgsMaster:
    @pytest.mark.parametrize(
        "accession", ["LVCL000000000.1", "PVIE00000000.1", "JALPRZ000000000.1", "LVCL000000000"]
    )
    def test_master_records_are_recognised(self, accession):
        """Padding stubs, not assembled bases - scanning one yields a megabase
        run of N that is not a gap in anything."""
        assert _is_wgs_master(accession) is True

    @pytest.mark.parametrize(
        "accession", ["NW_024426341.1", "NW_020955186.1", "LVCL01000001.1", "NC_007605.1"]
    )
    def test_real_sequences_are_not(self, accession):
        assert _is_wgs_master(accession) is False


# ===========================================================================
# 1. N-RUN DETECTION (pure, offline)
# ===========================================================================

# 20 resolved bases, a 45-base run of N, 20 more resolved bases. The run
# therefore occupies 1-based inclusive [21, 65] - the same fixture shape the
# Reconstruction Agent's own contract test uses.
_SEQ = "ACGT" * 5 + "N" * 45 + "TGCA" * 5


class TestFindNRuns:
    def test_finds_the_run(self):
        assert _find_n_runs(_SEQ) == [{"start": 21, "end": 65, "length": 45}]

    def test_coordinates_are_one_based_inclusive(self):
        """`start` is the first N and `end` the last, counting from 1.

        This is the convention the Reconstruction Agent documents this agent
        as using; getting it wrong leaves the first N unrepaired and
        overwrites the first base of the right flank.
        """
        (gap,) = _find_n_runs(_SEQ)
        # Python slicing is 0-based half-open, so the same region is [20, 65).
        assert _SEQ[gap["start"] - 1 : gap["end"]] == "N" * 45
        assert _SEQ[gap["start"] - 2] != "N"
        assert _SEQ[gap["end"]] != "N"

    def test_length_always_agrees_with_the_coordinates(self):
        for gap in _find_n_runs("A" * 10 + "N" * 30 + "C" * 5 + "N" * 12 + "T" * 3):
            assert gap["length"] == gap["end"] - gap["start"] + 1

    def test_finds_multiple_runs(self):
        gaps = _find_n_runs("A" * 10 + "N" * 30 + "C" * 5 + "N" * 12 + "T" * 3)
        assert [(g["start"], g["end"]) for g in gaps] == [(11, 40), (46, 57)]

    def test_run_exactly_at_the_floor_is_kept(self):
        assert len(_find_n_runs("ACGT" + "N" * MIN_GAP_BP + "ACGT")) == 1

    def test_very_short_runs_are_reported(self):
        """Short runs are the ones the Reconstruction Agent actually closes -
        both gaps it resolved unaided on NW_024426341.1 were a handful of
        bases - so they must survive to the handoff."""
        gaps = _find_n_runs("ACGT" * 5 + "N" * 5 + "ACGT" * 5)
        assert [g["length"] for g in gaps] == [5]

    def test_an_explicit_floor_still_filters(self):
        assert _find_n_runs("ACGT" + "N" * 4 + "ACGT", min_length=5) == []

    def test_run_at_the_very_start(self):
        gaps = _find_n_runs("N" * 20 + "ACGT")
        assert gaps == [{"start": 1, "end": 20, "length": 20}]

    def test_run_at_the_very_end(self):
        gaps = _find_n_runs("ACGT" + "N" * 20)
        assert gaps == [{"start": 5, "end": 24, "length": 20}]

    def test_no_runs_returns_empty_list(self):
        assert _find_n_runs("ACGT" * 100) == []

    def test_empty_sequence_returns_empty_list(self):
        assert _find_n_runs("") == []


# ===========================================================================
# 2. FLANK SLICING (pure, offline)
# ===========================================================================

class TestAttachFlanks:
    def test_slices_the_bases_either_side(self):
        gap = {"start": 21, "end": 65, "length": 45}
        enriched = _attach_flanks(gap, _SEQ, flank_bp=8)
        assert enriched["left_flank"] == _SEQ[12:20]
        assert enriched["right_flank"] == _SEQ[65:73]

    def test_flanks_never_contain_the_gap(self):
        enriched = _attach_flanks({"start": 21, "end": 65, "length": 45}, _SEQ, 50)
        assert "N" not in enriched["left_flank"]
        assert "N" not in enriched["right_flank"]

    def test_flank_is_clamped_at_the_start_of_the_record(self):
        """A gap 5 bases in cannot have a 50-base left flank."""
        seq = "ACGTA" + "N" * 20 + "TGCAT" * 20
        enriched = _attach_flanks({"start": 6, "end": 25, "length": 20}, seq, 50)
        assert enriched["left_flank"] == "ACGTA"

    def test_flank_is_clamped_at_the_end_of_the_record(self):
        seq = "ACGTA" * 20 + "N" * 20 + "TGCAT"
        enriched = _attach_flanks({"start": 101, "end": 120, "length": 20}, seq, 50)
        assert enriched["right_flank"] == "TGCAT"

    def test_gap_touching_the_first_base_has_an_empty_left_flank(self):
        enriched = _attach_flanks({"start": 1, "end": 20, "length": 20}, "N" * 20 + "ACGT", 10)
        assert enriched["left_flank"] == ""
        assert enriched["right_flank"] == "ACGT"

    def test_original_keys_are_preserved(self):
        enriched = _attach_flanks({"start": 21, "end": 65, "length": 45}, _SEQ, 8)
        assert enriched["start"] == 21
        assert enriched["end"] == 65
        assert enriched["length"] == 45


# ===========================================================================
# 3. FASTA STRIPPING (pure, offline)
# ===========================================================================

class TestExtractFastaSequence:
    def test_strips_header_and_joins_lines(self):
        fasta = ">NW_007907101.1:125380-125429\nACTGACTGAC\nTGACTGACTG\n"
        assert _extract_fasta_sequence(fasta) == "ACTGACTGACTGACTGACTG"

    def test_handles_no_trailing_newline(self):
        assert _extract_fasta_sequence(">header\nACGT") == "ACGT"

    def test_empty_input(self):
        assert _extract_fasta_sequence("") == ""


# ===========================================================================
# 4. find_target_gaps — orchestration wiring (mocked I/O)
# ===========================================================================

def _patched(sequence: str, accession: str = "NW_TEST.1", length: int | None = None):
    """Patch the two NCBI boundaries `find_target_gaps` reaches through."""
    from genome_agent.subagents import gap_finder

    async def _fake_window(_accession, start, stop):
        return f">window\n{sequence[start - 1 : stop]}\n"

    return (
        patch.object(
            gap_finder,
            "_select_target_record",
            new=AsyncMock(return_value=(accession, length if length is not None else len(sequence))),
        ),
        patch.object(gap_finder, "fetch_window_by_accession", new=_fake_window),
        # The census is a third NCBI boundary, patched here for the same reason
        # as the other two: nothing in this file may reach the network.
        patch.object(
            gap_finder,
            "_record_census",
            new=AsyncMock(
                return_value={"records_in_assembly": 3899, "records_over_size_ceiling": 37}
            ),
        ),
    )


class TestFindTargetGaps:
    def test_raises_when_no_record_can_be_selected(self):
        from genome_agent.subagents import gap_finder

        with patch.object(
            gap_finder, "_largest_record_uid", new=AsyncMock(return_value=None)
        ):
            with pytest.raises(GapFinderError):
                asyncio.run(gap_finder.find_target_gaps("GCF_doesnotexist.1"))

    def test_empty_gaps_is_not_an_error(self):
        from genome_agent.subagents import gap_finder

        select, window, census = _patched("ACGT" * 100, accession="NW_000000001.1")
        with select, window, census:
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1"))

        assert result["sequence_accession"] == "NW_000000001.1"
        assert result["target_gaps"] == []
        assert result["gaps_found"] == 0
        assert result["gaps_selected"] == 0

    def test_gaps_are_enriched_with_flanks(self):
        from genome_agent.subagents import gap_finder

        select, window, census = _patched(_SEQ, accession="NW_007907101.1")
        with select, window, census:
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1", flank_bp=8))

        assert result["sequence_accession"] == "NW_007907101.1"
        (gap,) = result["target_gaps"]
        assert (gap["start"], gap["end"], gap["length"]) == (21, 65, 45)
        assert gap["left_flank"] == _SEQ[12:20]
        assert gap["right_flank"] == _SEQ[65:73]

    def test_max_gaps_caps_results(self):
        from genome_agent.subagents import gap_finder

        sequence = "".join("ACGT" * 5 + "N" * (20 + i) for i in range(9))
        select, window, census = _patched(sequence)
        with select, window, census:
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1", max_gaps=3))

        assert len(result["target_gaps"]) == 3

    def test_shortest_gaps_are_kept_when_capping(self):
        """Not whichever come first in the record, and deliberately not the
        longest: the short runs are the ones that get resolved downstream."""
        from genome_agent.subagents import gap_finder

        sequence = "ACGT" * 5 + "N" * 20 + "ACGT" * 5 + "N" * 900 + "ACGT" * 5 + "N" * 40
        select, window, census = _patched(sequence)
        with select, window, census:
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1", max_gaps=2))

        assert [gap["length"] for gap in result["target_gaps"]] == [20, 40]

    def test_a_record_longer_than_one_window_is_read_in_full(self):
        """Regression guard for the windowed read: a gap past the first window
        boundary must still be found, with its coordinates counted from the
        start of the record rather than from the start of its window."""
        from genome_agent.subagents import gap_finder

        # Two windows' worth, with the run of N sitting in the second.
        head = "ACGT" * (gap_finder.MAX_WINDOW_BP // 4)
        sequence = head + "ACGT" * 25 + "N" * 60 + "ACGT" * 25
        select, window, census = _patched(sequence)
        with select, window, census:
            result = asyncio.run(gap_finder.find_target_gaps("GCF_x.1"))

        (gap,) = result["target_gaps"]
        assert gap["length"] == 60
        assert gap["start"] == len(head) + 101
        assert sequence[gap["start"] - 1 : gap["end"]] == "N" * 60


# ===========================================================================
# 5. find_target_gaps_node — degrades gracefully on failure
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
