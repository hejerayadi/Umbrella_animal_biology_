"""The payload the Genome Agent actually sends, parsed the way it means it.

`genome_agent/subagents/gap_finder.py` reads NCBI's feature table and forwards
its coordinates unconverted: `{"start": lo, "end": hi, "length": hi - lo + 1}`,
where `lo` is the first unresolved base and `hi` the last, both 1-based. This
agent's `Gap` is 0-based with an exclusive end. Every test here exists because
applying one convention to the other is undetectable downstream - it would
leave the first N unrepaired and overwrite the first base of the right flank.
"""

from __future__ import annotations

from reconstruction_agent.domain.models.request import ReconstructionRequest
from reconstruction_agent.domain.models.sequence import SequenceRecord
from reconstruction_agent.services.sequence.sequence_service import reconcile_requested_gaps

#: 20 resolved bases, a 45-base unresolved run, 20 more resolved bases. The run
#: therefore occupies 0-based [20, 65) and 1-based inclusive [21, 65].
_RECORD = SequenceRecord(accession="NC_TEST.1", residues="ACGT" * 5 + "N" * 45 + "TGCA" * 5)


def _request(gap: dict[str, object]) -> ReconstructionRequest:
    return ReconstructionRequest.from_agent_request(
        "Reconstruct the selected unresolved regions.",
        {
            "scientific_name": "Ursus maritimus",
            "assembly_id": "GCF_000687225.1",
            "sequence_accession": "NW_007907101",
            "assembly_level": "Scaffold",
            "target_gaps": [gap],
        },
    )


class TestTheGenomeAgentContract:
    def test_every_key_the_genome_agent_sends_is_read(self) -> None:
        parsed = _request({"start": 21, "end": 65, "length": 45})

        assert parsed.sequence_accession == "NW_007907101"
        assert parsed.scientific_name == "Ursus maritimus"
        assert parsed.assembly_id == "GCF_000687225.1"
        assert parsed.assembly_level == "Scaffold"

    def test_a_warnings_key_does_not_break_the_parse(self) -> None:
        """`gap_finder` failing is non-fatal upstream; it adds `warnings` and
        sends empty gaps rather than dropping the escalation."""
        parsed = ReconstructionRequest.from_agent_request(
            "Reconstruct the selected unresolved regions.",
            {
                "sequence_accession": "NW_007907101",
                "target_gaps": [],
                "warnings": ["gap finding failed: NCBI unreachable"],
            },
        )
        assert parsed.target_gaps == ()


class TestCoordinateOriginIsResolvedAgainstTheRecord:
    def test_one_based_inclusive_coordinates_land_on_the_real_run(self) -> None:
        """The Genome Agent's own convention, unconverted, must still repair
        exactly the 45 N bases - not 44 shifted by one."""
        (gap,), notes = reconcile_requested_gaps(
            _request({"start": 21, "end": 65}).target_gaps, _RECORD
        )

        assert (gap.start, gap.end, gap.length) == (20, 65, 45)
        assert _RECORD.residues[gap.start : gap.end] == "N" * 45
        assert "1-based" in notes[0]

    def test_zero_based_half_open_coordinates_are_left_alone(self) -> None:
        """This agent's own convention must not be 'corrected' into a shift."""
        (gap,), notes = reconcile_requested_gaps(
            _request({"start": 20, "end": 65}).target_gaps, _RECORD
        )

        assert (gap.start, gap.end) == (20, 65)
        assert notes == ()

    def test_a_placeholder_span_is_snapped_onto_the_whole_run(self) -> None:
        """NCBI records some gaps with a placeholder span and the true size in
        an `estimated_length` qualifier, so the two disagree legitimately."""
        (gap,), notes = reconcile_requested_gaps(
            _request({"start": 30, "end": 40, "length": 45}).target_gaps, _RECORD
        )

        assert (gap.start, gap.end, gap.length) == (20, 65, 45)
        assert any("disagrees" in note for note in notes)

    def test_coordinates_matching_no_run_are_kept_and_reported(self) -> None:
        """The caller may know something this copy of the record does not, but
        reconstructing resolved bases silently is never acceptable."""
        (gap,), notes = reconcile_requested_gaps(
            _request({"start": 2, "end": 8}).target_gaps, _RECORD
        )

        assert (gap.start, gap.end) == (2, 8)
        assert any("match no unresolved region" in note for note in notes)

    def test_the_right_run_is_chosen_when_the_record_has_several(self) -> None:
        record = SequenceRecord(
            accession="X", residues="ACGT" * 5 + "N" * 10 + "ACGT" * 5 + "N" * 12 + "ACGT" * 5
        )
        # Second run: 0-based [50, 62); 1-based inclusive [51, 62].
        (gap,), _ = reconcile_requested_gaps(_request({"start": 51, "end": 62}).target_gaps, record)

        assert (gap.start, gap.end, gap.length) == (50, 62, 12)
        assert record.residues[gap.start : gap.end] == "N" * 12
