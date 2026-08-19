"""Evidence deepening across rounds instead of being replaced by the latest one.

Confidence scales with how many references support a fill, so a reducer that
overwrote a gap's list made the agent *worse* the more it searched: a second
round returning fewer hits shrank the pool and lowered the score. These tests
pin the accumulate-and-deduplicate behaviour that replaced it.
"""
from __future__ import annotations

from agent.state.reducers import accumulate_references
from domain.models import Reference


def reference(accession: str, **kwargs: object) -> Reference:
    return Reference(accession=accession, **kwargs)  # type: ignore[arg-type]


class TestAccumulation:
    def test_a_later_round_adds_to_a_gap_rather_than_replacing_it(self) -> None:
        first = {"gap_1": [reference("REF_1")]}
        second = {"gap_1": [reference("REF_2")]}

        merged = accumulate_references(first, second)

        assert {r.accession for r in merged["gap_1"]} == {"REF_1", "REF_2"}

    def test_gaps_are_kept_apart(self) -> None:
        merged = accumulate_references(
            {"gap_1": [reference("REF_1")]}, {"gap_2": [reference("REF_2")]}
        )

        assert [r.accession for r in merged["gap_1"]] == ["REF_1"]
        assert [r.accession for r in merged["gap_2"]] == ["REF_2"]

    def test_the_same_accession_is_not_counted_twice(self) -> None:
        """Depth is evidence; a duplicate would inflate it without adding any."""
        merged = accumulate_references(
            {"gap_1": [reference("REF_1")]}, {"gap_1": [reference("REF_1")]}
        )

        assert len(merged["gap_1"]) == 1

    def test_residues_beat_metadata_for_the_same_accession(self) -> None:
        """A reference that cannot be aligned is not evidence, however well scored."""
        metadata_only = reference("REF_1", identity=0.99, coverage=0.99)
        alignable = reference("REF_1", residues="ACGT", identity=0.70, coverage=0.70)

        merged = accumulate_references({"gap_1": [metadata_only]}, {"gap_1": [alignable]})

        assert merged["gap_1"][0].residues == "ACGT"

    def test_residues_are_not_lost_when_the_weaker_copy_arrives_second(self) -> None:
        alignable = reference("REF_1", residues="ACGT", identity=0.70)
        metadata_only = reference("REF_1", identity=0.99)

        merged = accumulate_references({"gap_1": [alignable]}, {"gap_1": [metadata_only]})

        assert merged["gap_1"][0].residues == "ACGT"

    def test_the_better_measured_copy_wins_when_both_are_alignable(self) -> None:
        weak = reference("REF_1", residues="ACGT", identity=0.70, coverage=0.70)
        strong = reference("REF_1", residues="ACGT", identity=0.99, coverage=0.99)

        merged = accumulate_references({"gap_1": [weak]}, {"gap_1": [strong]})

        assert merged["gap_1"][0].identity == 0.99

    def test_an_empty_round_does_not_erase_what_was_gathered(self) -> None:
        held = {"gap_1": [reference("REF_1"), reference("REF_2")]}

        assert len(accumulate_references(held, {})["gap_1"]) == 2
        assert len(accumulate_references(held, None)["gap_1"]) == 2

    def test_starting_from_nothing(self) -> None:
        merged = accumulate_references(None, {"gap_1": [reference("REF_1")]})

        assert [r.accession for r in merged["gap_1"]] == ["REF_1"]

    def test_the_inputs_are_left_alone(self) -> None:
        """The reducer runs on checkpointed state; mutating it would corrupt a resume."""
        first = {"gap_1": [reference("REF_1")]}

        accumulate_references(first, {"gap_1": [reference("REF_2")]})

        assert [r.accession for r in first["gap_1"]] == ["REF_1"]


class TestProvenance:
    def test_a_reference_records_which_round_produced_it(self) -> None:
        tagged = reference("REF_1", iteration=2, attempt=1, source="blast")

        assert tagged.iteration == 2
        assert tagged.attempt == 1
        assert tagged.source == "blast"

    def test_quality_prefers_measured_homology_over_none(self) -> None:
        measured = reference("REF_1", identity=0.9, coverage=0.8)
        unmeasured = reference("REF_2")

        assert measured.quality > unmeasured.quality
        assert unmeasured.quality == 0.0

    def test_quality_uses_whichever_signals_exist(self) -> None:
        """NCBI records carry no alignment statistics; BLAST hits carry both."""
        identity_only = reference("REF_1", identity=0.8)

        assert identity_only.quality == 0.8
