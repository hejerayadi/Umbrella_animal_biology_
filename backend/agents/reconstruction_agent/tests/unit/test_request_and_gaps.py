"""Reading the request, and finding what needs repairing.

Both run before anything expensive. A malformed request discovered here costs
nothing; discovered two external services and two hundred seconds into a run it
costs the whole run.
"""

from __future__ import annotations

import pytest

from reconstruction_agent.domain.exceptions import (
    InvalidGapCoordinatesError,
    InvalidRequestError,
)
from reconstruction_agent.domain.models.request import ReconstructionRequest
from reconstruction_agent.domain.models.sequence import Gap, SequenceRecord
from reconstruction_agent.services.sequence.sequence_service import (
    build_context,
    detect_gaps,
    triage,
)


class TestReadingTheRequest:
    def test_a_request_naming_no_target_is_rejected(self) -> None:
        """Guessing a target would mean repairing a sequence nobody named."""
        with pytest.raises(InvalidRequestError, match="No sequence to reconstruct"):
            ReconstructionRequest.from_agent_request("Fix the genome", {})

    @pytest.mark.parametrize(
        "key", ["sequence_accession", "accession", "nucleotide_accession", "refseq_accession"]
    )
    def test_an_accession_is_read_from_any_spelling_other_agents_use(self, key: str) -> None:
        """The context is shared and the Genome agent seeds several spellings."""
        parsed = ReconstructionRequest.from_agent_request("x", {key: "NC_003428.1"})
        assert parsed.sequence_accession == "NC_003428.1"

    def test_a_pasted_sequence_is_accepted_and_cleaned(self) -> None:
        """An interactive user pastes newlines and lowercase along with it."""
        parsed = ReconstructionRequest.from_agent_request("x", {"sequence": "acgt\nnnnnn\tacgt "})
        assert parsed.residues == "ACGTNNNNNACGT"

    def test_a_nested_sequence_object_is_accepted(self) -> None:
        parsed = ReconstructionRequest.from_agent_request(
            "x", {"sequence": {"residues": "ACGTNNNNNACGT"}}
        )
        assert parsed.residues == "ACGTNNNNNACGT"

    def test_organism_and_assembly_travel_when_present(self) -> None:
        parsed = ReconstructionRequest.from_agent_request(
            "x",
            {
                "accession": "NC_1",
                "species": "Ursus maritimus",
                "assembly_id": "GCF_000687225.1",
            },
        )
        assert parsed.scientific_name == "Ursus maritimus"
        assert parsed.assembly_id == "GCF_000687225.1"

    def test_a_context_of_none_is_survivable(self) -> None:
        with pytest.raises(InvalidRequestError):
            ReconstructionRequest.from_agent_request("x", None)


class TestGapCoordinatesAreCheckedUpFront:
    """These decide which bases get replaced. An off-by-one accepted here is
    undetectable downstream."""

    def test_inverted_coordinates_are_rejected(self) -> None:
        with pytest.raises(InvalidGapCoordinatesError, match="end"):
            ReconstructionRequest.from_agent_request(
                "x", {"accession": "NC_1", "target_gaps": [{"start": 100, "end": 50}]}
            )

    def test_non_integer_coordinates_are_rejected(self) -> None:
        with pytest.raises(InvalidGapCoordinatesError, match="integer"):
            ReconstructionRequest.from_agent_request(
                "x", {"accession": "NC_1", "target_gaps": [{"start": "a", "end": "b"}]}
            )

    def test_a_missing_coordinate_is_rejected(self) -> None:
        with pytest.raises(InvalidGapCoordinatesError):
            ReconstructionRequest.from_agent_request(
                "x", {"accession": "NC_1", "target_gaps": [{"start": 10}]}
            )

    def test_target_gaps_of_the_wrong_shape_is_rejected(self) -> None:
        with pytest.raises(InvalidGapCoordinatesError, match="list"):
            ReconstructionRequest.from_agent_request(
                "x", {"accession": "NC_1", "target_gaps": {"start": 1, "end": 2}}
            )

    def test_well_formed_coordinates_are_kept(self) -> None:
        parsed = ReconstructionRequest.from_agent_request(
            "x",
            {"accession": "NC_1", "target_gaps": [{"start": 10, "end": 20}]},
        )
        assert parsed.gaps[0].start == 10
        assert parsed.gaps[0].length == 10
        assert parsed.target_gaps[0].notes == ()

    def test_a_stated_length_that_contradicts_the_coordinates_is_reported(self) -> None:
        """The coordinates win - they are what gets replaced - but silently
        discarding the contradiction would hide a counting-origin mismatch."""
        parsed = ReconstructionRequest.from_agent_request(
            "x",
            {"accession": "NC_1", "target_gaps": [{"start": 10, "end": 20, "length": 11}]},
        )
        assert parsed.gaps[0].length == 10
        assert "disagrees" in parsed.target_gaps[0].notes[0]

    def test_a_truncated_placeholder_flank_is_not_taken_for_sequence(self) -> None:
        """Callers send "ACGT..." as a display string; anchoring an alignment on
        it would use characters that were never in the record."""
        parsed = ReconstructionRequest.from_agent_request(
            "x",
            {
                "accession": "NC_1",
                "target_gaps": [
                    {"start": 10, "end": 20, "left_flank": "ACGT...", "right_flank": "acgtn"}
                ],
            },
        )
        assert parsed.target_gaps[0].left_flank == ""
        assert parsed.target_gaps[0].right_flank == "ACGTN"


class TestFindingGaps:
    def test_runs_of_ambiguity_are_found(self) -> None:
        record = SequenceRecord(accession="X", residues="ACGT" + "N" * 10 + "ACGT")
        (gap,) = detect_gaps(record)

        assert (gap.start, gap.end, gap.length) == (4, 14, 10)

    def test_a_single_uncertain_base_is_not_an_assembly_gap(self) -> None:
        """One ambiguous base is a sequencing artefact, not a hole worth a search."""
        record = SequenceRecord(accession="X", residues="ACGTNACGT")
        assert detect_gaps(record) == ()

    def test_other_ambiguity_characters_count(self) -> None:
        """Assemblers emit more than N for an unresolved position."""
        record = SequenceRecord(accession="X", residues="ACGT" + "X" * 8 + "ACGT")
        assert len(detect_gaps(record)) == 1

    def test_a_complete_sequence_has_no_gaps(self) -> None:
        """Answerable without touching a single external service."""
        record = SequenceRecord(accession="X", residues="ACGTACGTACGT")
        assert detect_gaps(record) == ()

    def test_several_gaps_are_returned_in_coordinate_order(self) -> None:
        record = SequenceRecord(
            accession="X", residues="AAAA" + "N" * 6 + "CCCC" + "N" * 8 + "GGGG"
        )
        gaps = detect_gaps(record)

        assert [gap.start for gaps_ in [gaps] for gap in gaps_] == [4, 14]


class TestFlanks:
    def test_flanks_exclude_the_gap_itself(self) -> None:
        record = SequenceRecord(accession="X", residues="AAAA" + "NNNNN" + "CCCC")
        context = build_context(record, Gap(gap_id="g1", start=4, end=9), flank_size=4)

        assert context.left_flank == "AAAA"
        assert context.right_flank == "CCCC"
        assert "N" not in context.query_sequence()

    def test_flanks_clamp_at_the_record_edges(self) -> None:
        """A gap near the start has a short left flank, not a negative slice."""
        record = SequenceRecord(accession="X", residues="AA" + "NNNNN" + "CCCC")
        context = build_context(record, Gap(gap_id="g1", start=2, end=7), flank_size=500)

        assert context.left_flank == "AA"
        assert context.right_flank == "CCCC"


class TestTriage:
    """A draft scaffold can carry hundreds of gaps and the run has one deadline.
    Attempting all of them finishes none."""

    def test_the_run_commits_to_what_it_can_finish(self) -> None:
        gaps = tuple(Gap(gap_id=f"g{i}", start=i * 100, end=i * 100 + 10 + i) for i in range(20))
        attempted, deferred = triage(gaps, max_gaps=5, max_gap_length=5000)

        assert len(attempted) == 5
        assert len(deferred) == 15

    def test_shorter_gaps_are_attempted_first(self) -> None:
        """Cheaper to resolve and likelier to be resolvable, so this maximises
        the number of regions actually recovered inside the budget."""
        gaps = (
            Gap(gap_id="long", start=0, end=900),
            Gap(gap_id="short", start=1000, end=1020),
            Gap(gap_id="medium", start=2000, end=2200),
        )
        attempted, _ = triage(gaps, max_gaps=2, max_gap_length=5000)

        assert [gap.gap_id for gap in attempted] == ["short", "medium"]

    def test_a_gap_beyond_the_length_ceiling_is_never_attempted(self) -> None:
        gaps = (
            Gap(gap_id="huge", start=0, end=100_000),
            Gap(gap_id="small", start=200_000, end=200_050),
        )
        attempted, deferred = triage(gaps, max_gaps=10, max_gap_length=5000)

        assert [gap.gap_id for gap in attempted] == ["small"]
        assert [gap.gap_id for gap in deferred] == ["huge"]

    def test_nothing_is_lost_between_attempted_and_deferred(self) -> None:
        """Every requested gap must be reported, one way or the other."""
        gaps = tuple(Gap(gap_id=f"g{i}", start=i * 100, end=i * 100 + 20) for i in range(12))
        attempted, deferred = triage(gaps, max_gaps=4, max_gap_length=5000)

        assert len(attempted) + len(deferred) == len(gaps)
        assert {g.gap_id for g in attempted} | {g.gap_id for g in deferred} == {
            g.gap_id for g in gaps
        }
