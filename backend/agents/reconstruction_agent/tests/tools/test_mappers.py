"""Parsing external-service payloads into domain objects, offline."""
from __future__ import annotations

import json

import pytest

from reconstruction_agent.tools.blast.mapper import to_references as blast_references
from reconstruction_agent.tools.evo.mapper import (
    heuristic_relatedness,
    parse_evolution_agent_reply,
)
from reconstruction_agent.tools.mafft.mapper import build_fasta, to_alignment
from reconstruction_agent.tools.ncbi.mapper import parse_fasta, to_references

FASTA = """\
>NC_007596.2 Mammuthus primigenius mitochondrion, complete genome
ACGTACGTAC
GTACGTACGT
>NC_000934.1 Loxodonta africana mitochondrion
TTTTAAAACC
"""


class TestNCBIMapper:
    def test_splits_multi_fasta_into_records(self) -> None:
        records = parse_fasta(FASTA)

        assert len(records) == 2
        assert records[0][0] == "NC_007596.2"
        # Wrapped lines are joined back into one sequence.
        assert records[0][2] == "ACGTACGTACGTACGTACGT"

    def test_extracts_the_organism_from_the_header(self) -> None:
        references = to_references(FASTA)

        assert references[0].organism == "Mammuthus primigenius"
        assert references[1].organism == "Loxodonta africana"

    def test_empty_input_yields_no_records(self) -> None:
        assert parse_fasta("") == []


class TestBlastMapper:
    def test_converts_percentages_to_fractions(self) -> None:
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "REF_1",
                        "hit_os": "Loxodonta africana",
                        "hit_hsps": [{"hsp_identity": 92.5, "hsp_expect": 1e-40}],
                    }
                ]
            }
        )

        references = blast_references(raw)

        assert references[0].identity == pytest.approx(0.925)

    def test_derives_coverage_from_alignment_length(self) -> None:
        raw = json.dumps(
            {"hits": [{"hit_acc": "REF_1", "hit_hsps": [{"hsp_align_len": 80}]}]}
        )

        references = blast_references(raw, query_length=100)

        assert references[0].coverage == pytest.approx(0.8)

    def test_malformed_json_yields_no_hits(self) -> None:
        """A search that produced nothing usable is a normal outcome."""
        assert blast_references("<html>Service Unavailable</html>") == []


class TestMafftMapper:
    def test_locates_the_gap_columns_between_the_flanks(self) -> None:
        """The columns the aligner inserted at the flank junction are the gap."""
        aligned = (
            ">target\nAAAA----CCCC\n"
            ">REF_1\nAAAAGGGGCCCC\n"
            ">REF_2\nAAAAGGGGCCCC\n"
        )

        alignment = to_alignment(
            aligned, gap_id="gap_1", target_id="target", left_flank_length=4
        )

        assert alignment is not None
        assert alignment.spans_gap
        assert (alignment.gap_column_start, alignment.gap_column_end) == (4, 8)
        assert alignment.reference_count == 2

    def test_no_inserted_columns_means_no_gap_to_read(self) -> None:
        """References agreeing the target lacks nothing is a real answer."""
        aligned = ">target\nAAAACCCC\n>REF_1\nAAAACCCC\n"

        alignment = to_alignment(
            aligned, gap_id="gap_1", target_id="target", left_flank_length=4
        )

        assert alignment is not None
        assert not alignment.spans_gap

    def test_missing_target_row_yields_nothing(self) -> None:
        aligned = ">REF_1\nAAAAGGGGCCCC\n"

        assert (
            to_alignment(aligned, gap_id="gap_1", target_id="target", left_flank_length=4)
            is None
        )

    def test_identity_ignores_columns_with_alignment_gaps(self) -> None:
        aligned = ">target\nAAAA----CCCC\n>REF_1\nAAAAGGGGCCCC\n"

        alignment = to_alignment(
            aligned, gap_id="gap_1", target_id="target", left_flank_length=4
        )

        assert alignment is not None
        # All eight comparable columns match; the four gap columns are excluded.
        assert alignment.pairs[0].identity == pytest.approx(1.0)

    def test_build_fasta_puts_the_target_first(self) -> None:
        fasta = build_fasta("target", "ACGT", {"REF_1": "ACGA", "REF_2": "ACGC"})

        assert fasta.startswith(">target\nACGT")
        assert ">REF_1" in fasta and ">REF_2" in fasta


class TestEvoMapper:
    def test_same_species_scores_highest(self) -> None:
        assert heuristic_relatedness("Loxodonta africana", "Loxodonta africana") == 1.0

    def test_same_genus_scores_high(self) -> None:
        assert heuristic_relatedness("Loxodonta africana", "Loxodonta cyclotis") == 0.8

    def test_different_genus_scores_low(self) -> None:
        """The heuristic cannot see that these are in fact close relatives."""
        assert heuristic_relatedness("Loxodonta africana", "Mammuthus primigenius") == 0.3

    def test_parses_a_direct_relatedness_map(self) -> None:
        assert parse_evolution_agent_reply({"Loxodonta africana": 0.9}) == {
            "Loxodonta africana": 0.9
        }

    def test_parses_a_wrapped_relatedness_map(self) -> None:
        assert parse_evolution_agent_reply({"relatedness": {"A": 0.5}}) == {"A": 0.5}

    def test_unparseable_payload_yields_an_empty_map(self) -> None:
        assert parse_evolution_agent_reply("not a mapping") == {}
