"""Parsing external-service payloads into domain objects, offline."""
from __future__ import annotations

import json

import pytest

from tools.blast.mapper import to_references as blast_references
from tools.evo.mapper import (
    heuristic_relatedness,
    parse_evolution_agent_reply,
)
from tools.mafft.mapper import build_fasta, to_alignment
from tools.ncbi.mapper import parse_fasta, to_references

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
    """Parsing hits, and the residues that make them alignable.

    A BLAST hit used to arrive as metadata only, which the alignment step then
    filtered out for having no sequence - so the default plan could never align
    anything. These tests pin the retrieval path as much as the parsing.
    """

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

        references, _ = blast_references(raw)

        assert references[0].identity == pytest.approx(0.925)

    def test_derives_coverage_from_every_hsp_not_just_the_first(self) -> None:
        """A hit matching both flanks separately covers both of them."""
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "REF_1",
                        "hit_hsps": [
                            {"hsp_align_len": 40, "hsp_hit_from": 1, "hsp_hit_to": 40},
                            {"hsp_align_len": 40, "hsp_hit_from": 90, "hsp_hit_to": 130},
                        ],
                    }
                ]
            }
        )

        references, _ = blast_references(raw, query_length=100)

        assert references[0].coverage == pytest.approx(0.8)

    def test_malformed_json_yields_no_hits(self) -> None:
        """A search that produced nothing usable is a normal outcome."""
        assert blast_references("<html>Service Unavailable</html>") == ([], [])

    def test_single_hsp_carries_its_residues(self) -> None:
        """No fetch is needed when one HSP already holds the subject."""
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "REF_1",
                        "hit_hsps": [
                            {
                                "hsp_hseq": "ACGT-ACGT",
                                "hsp_hit_from": 1,
                                "hsp_hit_to": 8,
                                "hsp_strand": "plus/plus",
                            }
                        ],
                    }
                ]
            }
        )

        references, pending = blast_references(raw)

        # Alignment gaps are stripped: MAFFT reintroduces whatever it needs.
        assert references[0].residues == "ACGTACGT"
        assert pending == []

    def test_minus_strand_hit_is_brought_onto_the_target_strand(self) -> None:
        """Aligned as reported, a minus-strand hit is noise rather than evidence."""
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "REF_1",
                        "hit_hsps": [
                            {
                                "hsp_hseq": "AAAACGT",
                                "hsp_hit_from": 1,
                                "hsp_hit_to": 7,
                                "hsp_strand": "plus/minus",
                            }
                        ],
                    }
                ]
            }
        )

        references, _ = blast_references(raw)

        assert references[0].strand == -1
        assert references[0].residues == "ACGTTTT"

    def test_bracketing_hsps_request_the_region_between_them(self) -> None:
        """The whole point: for a real gap the missing segment is in no HSP.

        Two HSPs, one per flank, mean the subject carries something the query
        does not. That something is what the reconstruction needs, and it can
        only be had by fetching the subject range the HSPs bracket.
        """
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "REF_1",
                        "hit_hsps": [
                            {"hsp_hseq": "ACGT", "hsp_hit_from": 100, "hsp_hit_to": 140},
                            {"hsp_hseq": "TTTT", "hsp_hit_from": 200, "hsp_hit_to": 240},
                        ],
                    }
                ]
            }
        )

        references, pending = blast_references(raw, gap_length=30)

        # Neither HSP's residues are used: they are the flanks, not the fill.
        assert references[0].residues is None
        assert len(pending) == 1
        span = pending[0]
        assert span.accession == "REF_1"
        # Widened by the gap length on both sides, so the fetched region
        # actually contains the segment sitting between the flanks.
        assert span.start == 70
        assert span.stop == 270

    def test_span_never_starts_before_the_first_base(self) -> None:
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "REF_1",
                        "hit_hsps": [
                            {"hsp_hit_from": 5, "hsp_hit_to": 20},
                            {"hsp_hit_from": 60, "hsp_hit_to": 80},
                        ],
                    }
                ]
            }
        )

        _, pending = blast_references(raw, gap_length=400)

        assert pending[0].start == 1


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


class TestGapCarryingReferences:
    """Which BLAST hits actually carry bases across the gap.

    The query is the two flanks joined, so a reference that still holds the
    missing segment aligns with a gap run in the *query* row at the junction.
    Hits that merely match a flank carry nothing, and aligning them alongside
    the donors is what made MAFFT open no gap columns at all.
    """

    @staticmethod
    def _result(qseq: str, hseq: str, query_from: int = 1) -> str:
        import json

        return json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "TEST001.1",
                        "hit_hsps": [
                            {
                                "hsp_qseq": qseq,
                                "hsp_hseq": hseq,
                                "hsp_query_from": query_from,
                                "hsp_query_to": query_from + len(qseq.replace("-", "")) - 1,
                                "hsp_hit_from": 1,
                                "hsp_hit_to": len(hseq.replace("-", "")),
                                "hsp_identity": 95.0,
                                "hsp_expect": 1e-40,
                            }
                        ],
                    }
                ]
            }
        )

    def test_a_reference_carrying_the_segment_is_measured(self) -> None:
        from tools.blast.mapper import to_references

        # 10 query bases, a 4-base insertion the query lacks, then 10 more.
        qseq = "ACGTACGTAC" + "----" + "GTACGTACGT"
        hseq = "ACGTACGTAC" + "TTTT" + "GTACGTACGT"
        refs, _ = to_references(self._result(qseq, hseq), left_flank_length=10)

        assert refs[0].gap_bases == 4
        assert refs[0].carries_gap is True

    def test_a_reference_matching_only_a_flank_carries_nothing(self) -> None:
        from tools.blast.mapper import to_references

        qseq = "ACGTACGTACGTACGTACGT"
        hseq = "ACGTACGTACGTACGTACGT"
        refs, _ = to_references(self._result(qseq, hseq), left_flank_length=10)

        assert refs[0].gap_bases == 0
        assert refs[0].carries_gap is False

    def test_an_insertion_far_from_the_junction_is_not_the_gap(self) -> None:
        from tools.blast.mapper import to_references

        # The insertion sits at query position 2, nowhere near a junction at 40.
        qseq = "AC" + "----" + "GTACGTACGTACGTACGTAC"
        hseq = "AC" + "TTTT" + "GTACGTACGTACGTACGTAC"
        refs, _ = to_references(self._result(qseq, hseq), left_flank_length=40)

        assert refs[0].gap_bases == 0

    def test_unmeasured_when_no_junction_is_supplied(self) -> None:
        from tools.blast.mapper import to_references

        qseq = "ACGTACGTAC" + "----" + "GTACGTACGT"
        hseq = "ACGTACGTAC" + "TTTT" + "GTACGTACGT"
        refs, _ = to_references(self._result(qseq, hseq))

        # None, not 0: an NCBI record that never went through a search is
        # unmeasured, and that must not read as "carries nothing".
        assert refs[0].gap_bases is None


class TestBracketingReferencesCarryTheGap:
    """The other shape a donor arrives in - two HSPs, one per flank.

    A 45-base insertion costs far more under an affine gap penalty than ending
    one alignment and starting another, so BLAST reports a real donor as two
    HSPs bracketing the missing segment rather than as one gapped HSP. Measuring
    only the gapped shape scored every one of these at zero, so `carries_gap`
    was False for exactly the hits the mapper had just decided to fetch a
    subject region for - and the selector, finding no carriers at all, aligned
    the non-carriers instead and MAFFT opened no column at the junction.
    """

    @staticmethod
    def _bracketing(
        *,
        left_hit: tuple[int, int],
        right_hit: tuple[int, int],
        left_flank_length: int = 500,
    ) -> str:
        """One hit whose two HSPs meet exactly at the query junction."""
        import json

        return json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "TEST002.1",
                        "hit_hsps": [
                            {
                                "hsp_query_from": 1,
                                "hsp_query_to": left_flank_length,
                                "hsp_hit_from": left_hit[0],
                                "hsp_hit_to": left_hit[1],
                                "hsp_identity": 96.0,
                                "hsp_expect": 1e-50,
                            },
                            {
                                "hsp_query_from": left_flank_length + 1,
                                "hsp_query_to": left_flank_length * 2,
                                "hsp_hit_from": right_hit[0],
                                "hsp_hit_to": right_hit[1],
                                "hsp_identity": 95.0,
                                "hsp_expect": 1e-48,
                            },
                        ],
                    }
                ]
            }
        )

    def test_bracketed_subject_bases_are_counted_as_carried(self) -> None:
        from tools.blast.mapper import to_references

        # The subject runs 1000..1499, then 45 bases we do not have, then
        # 1545..2044. Those 45 are the fill.
        refs, _ = to_references(
            self._bracketing(left_hit=(1000, 1499), right_hit=(1545, 2044)),
            left_flank_length=500,
        )

        assert refs[0].gap_bases == 45
        assert refs[0].carries_gap is True

    def test_a_minus_strand_donor_is_measured_the_same(self) -> None:
        from tools.blast.mapper import to_references

        # EBI reports a minus-strand HSP with hit_from above hit_to. Subtracting
        # raw would give a negative width for exactly these hits.
        refs, _ = to_references(
            self._bracketing(left_hit=(2044, 1545), right_hit=(1499, 1000)),
            left_flank_length=500,
        )

        assert refs[0].gap_bases == 45
        assert refs[0].carries_gap is True

    def test_abutting_hsps_carry_nothing(self) -> None:
        from tools.blast.mapper import to_references

        # The subject continues straight through: it has no extra bases, so it
        # is homologous but not a donor.
        refs, _ = to_references(
            self._bracketing(left_hit=(1000, 1499), right_hit=(1500, 1999)),
            left_flank_length=500,
        )

        assert refs[0].gap_bases == 0
        assert refs[0].carries_gap is False

    def test_hsps_that_do_not_meet_at_the_junction_are_not_a_bracket(self) -> None:
        import json

        from tools.blast.mapper import to_references

        # Two HSPs against the same subject, but both inside the left flank -
        # a repeat, not a donor. Nothing here spans the junction at 500.
        raw = json.dumps(
            {
                "hits": [
                    {
                        "hit_acc": "TEST003.1",
                        "hit_hsps": [
                            {
                                "hsp_query_from": 1,
                                "hsp_query_to": 100,
                                "hsp_hit_from": 1000,
                                "hsp_hit_to": 1099,
                                "hsp_expect": 1e-20,
                            },
                            {
                                "hsp_query_from": 150,
                                "hsp_query_to": 250,
                                "hsp_hit_from": 1200,
                                "hsp_hit_to": 1300,
                                "hsp_expect": 1e-20,
                            },
                        ],
                    }
                ]
            }
        )
        refs, _ = to_references(raw, left_flank_length=500)

        assert refs[0].gap_bases == 0
        assert refs[0].carries_gap is False
        assert refs[0].carries_gap is False
