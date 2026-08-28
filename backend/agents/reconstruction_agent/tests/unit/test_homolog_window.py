"""Only the aligned window of a homologue is fetched, not its whole record.

Found by a live run. On NW_007907101 a BLAST hit reported an 800-base aligned
region inside a 181,395-base genomic record, and the agent stored and shipped
all 181,395 of them. With six such hits the FASTA passed a megabyte, and MAFFT
was being asked to align a 1 kb query against a chromosome - an alignment in
which the query is a rounding error and the gap columns read off it mean
nothing.

The margin is as load-bearing as the window: the gap sits between the two
flanks, so the subject must extend past the aligned region on both sides for
anything to align across the junction being reconstructed.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.sequence import SequenceRecord
from reconstruction_agent.services.sequence.sequence_service import (
    HOMOLOG_WINDOW_MARGIN,
    SequenceService,
)


class _Ncbi:
    """Records which fetch the service chose, and with what bounds."""

    def __init__(self, *, window: str = "ACGT" * 250, whole: str = "TTTT" * 50_000) -> None:
        self._window = window
        self._whole = whole
        self.window_calls: list[tuple[str, int, int]] = []
        self.whole_calls: list[str] = []

    async def fetch_sequence_window(self, accession: str, start: int, end: int) -> str:
        self.window_calls.append((accession, start, end))
        return self._window

    async def fetch_sequence(self, accession: str) -> SequenceRecord:
        self.whole_calls.append(accession)
        return SequenceRecord(accession=accession, residues=self._whole)


def _service(ncbi: _Ncbi) -> SequenceService:
    return SequenceService(ncbi)  # type: ignore[arg-type]


def _hit(**overrides: Any) -> HomologHit:
    base: dict[str, Any] = {
        "accession": "AC186203",
        "identity": 0.858,
        "subject_start": 59218,
        "subject_end": 60016,
    }
    return HomologHit(**{**base, **overrides})


class TestOnlyTheAlignedRegionIsFetched:
    async def test_a_hit_with_coordinates_fetches_a_window(self) -> None:
        ncbi = _Ncbi()
        residues = await _service(ncbi)._residues_for(_hit())

        assert ncbi.window_calls, "the whole record must not be fetched"
        assert not ncbi.whole_calls
        assert len(residues) < 10_000

    async def test_the_window_extends_past_the_aligned_region_on_both_sides(self) -> None:
        """Trimmed exactly to the HSP, the subject would stop at the very
        junction being reconstructed."""
        ncbi = _Ncbi()
        await _service(ncbi)._residues_for(_hit())
        _, start, end = ncbi.window_calls[0]

        assert start == 59218 - HOMOLOG_WINDOW_MARGIN
        assert end == 60016 + HOMOLOG_WINDOW_MARGIN

    async def test_a_minus_strand_hit_is_ordered_before_fetching(self) -> None:
        """BLAST reports the subject range reversed for a minus-strand match;
        an inverted range fetches nothing."""
        ncbi = _Ncbi()
        await _service(ncbi)._residues_for(_hit(subject_start=60016, subject_end=59218))
        _, start, end = ncbi.window_calls[0]

        assert start < end


class TestFallingBackWithoutLosingAHomologue:
    async def test_a_hit_with_no_coordinates_takes_the_whole_record(self) -> None:
        ncbi = _Ncbi()
        await _service(ncbi)._residues_for(_hit(subject_start=0, subject_end=0))

        assert ncbi.whole_calls == ["AC186203"]
        assert not ncbi.window_calls

    async def test_an_empty_window_falls_back_rather_than_dropping_the_hit(self) -> None:
        """BLAST said this homologue spans the gap; a rejected range request is
        no reason to discard that measurement."""
        ncbi = _Ncbi(window="")
        residues = await _service(ncbi)._residues_for(_hit())

        assert ncbi.window_calls
        assert ncbi.whole_calls == ["AC186203"]
        assert residues


class TestTheFetchLoopStaysHonest:
    async def test_a_hit_whose_sequence_cannot_be_read_is_dropped(self) -> None:
        class _Empty(_Ncbi):
            async def fetch_sequence(self, accession: str) -> SequenceRecord:
                return SequenceRecord(accession=accession, residues="")

        ncbi = _Empty(window="")
        fetched = await _service(ncbi).fetch_homolog_sequences((_hit(),))

        assert fetched == ()

    async def test_a_failing_accession_does_not_fail_the_round(self) -> None:
        """The remaining homologues are still usable evidence."""

        class _Broken(_Ncbi):
            async def fetch_sequence_window(self, accession: str, start: int, end: int) -> str:
                if accession == "BAD":
                    raise RuntimeError("no such record")
                return await super().fetch_sequence_window(accession, start, end)

            async def fetch_sequence(self, accession: str) -> SequenceRecord:
                if accession == "BAD":
                    raise RuntimeError("no such record")
                return await super().fetch_sequence(accession)

        fetched = await _service(_Broken()).fetch_homolog_sequences(
            (_hit(accession="BAD"), _hit(accession="GOOD"))
        )

        assert [hit.accession for hit in fetched] == ["GOOD"]
