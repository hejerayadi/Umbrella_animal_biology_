"""Getting the target sequence, finding its holes, and fetching homologues.

Gap detection runs offline and first. It costs nothing, it establishes whether
there is anything to reconstruct at all, and a request naming a record with no
unresolved regions can be answered without touching a single external service.

Triage matters more than it looks. A draft scaffold can carry hundreds of
N-runs - one polar bear scaffold has around five hundred - while the run has a
few hundred seconds and a bounded search budget. Attempting all of them
finishes none. The run commits to the gaps it can actually finish and reports
the rest honestly as not attempted.
"""

from __future__ import annotations

import re

from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.request import RequestedGap
from reconstruction_agent.domain.models.sequence import Gap, GapContext, SequenceRecord
from reconstruction_agent.integrations.ncbi.client import NcbiClient
from reconstruction_agent.observability.logger import get_logger
from reconstruction_agent.orchestration.deadline import Deadline

_log = get_logger(__name__)

#: Runs of any ambiguity character, not just N: assemblers also emit other
#: IUPAC codes for an unresolved position.
_AMBIGUOUS_RUN = re.compile(r"[NnXx]+")

#: Shorter runs than this are usually a single uncertain base rather than an
#: assembly gap, and are not worth a search round.
DEFAULT_MIN_GAP_LENGTH = 5

#: Bases taken either side of a hit's aligned region. Wide enough that the
#: subject still extends past both flanks of the query - the alignment has to
#: cross the junction being reconstructed - and narrow enough that a genomic
#: hit stays a few kilobases rather than a whole chromosome.
HOMOLOG_WINDOW_MARGIN = 1000


class SequenceService:
    """Retrieval and preparation of the sequence being repaired."""

    def __init__(self, ncbi: NcbiClient) -> None:
        self._ncbi = ncbi

    async def fetch(self, accession: str) -> SequenceRecord:
        """The record named by `accession`."""
        return await self._ncbi.fetch_sequence(accession)

    async def fetch_homolog_sequences(
        self,
        hits: tuple[HomologHit, ...],
        *,
        limit: int = 12,
        deadline: Deadline | None = None,
    ) -> tuple[HomologHit, ...]:
        """Retrieve the residues for the best hits, so they can be aligned.

        Capped, because alignment cost grows with the number of sequences and
        the top homologues carry nearly all the information. Fetching fifty to
        align fifty spends the run deadline on evidence that changes no answer.

        A hit whose sequence cannot be retrieved is dropped rather than failing
        the round: the remaining homologues are still usable evidence.

        Bounded by the run clock as well as by count. These are a dozen
        sequential NCBI calls after a search that may already have consumed
        most of the deadline, and leaving them unbounded is how a run overruns
        the budget it was supposed to answer within.
        """
        resolved: list[HomologHit] = []
        for hit in hits[:limit]:
            if deadline is not None and deadline.expired():
                _log.info(
                    "homolog_fetch_stopped_at_deadline",
                    fetched=len(resolved),
                    requested=min(len(hits), limit),
                )
                break
            try:
                residues = await self._residues_for(hit)
            except Exception as error:  # noqa: BLE001 - one bad accession is not fatal
                _log.warning("homolog_fetch_failed", accession=hit.accession, error=str(error))
                continue
            if not residues:
                continue
            resolved.append(hit.model_copy(update={"subject_sequence": residues}))
        return tuple(resolved)

    async def _residues_for(self, hit: HomologHit) -> str:
        """The part of a homologue worth aligning.

        Only the aligned window plus a margin, whenever BLAST reported one. A
        genomic hit names a record that can be hundreds of kilobases long while
        the region that aligned is a few hundred bases; taking the whole thing
        makes the query a rounding error in the alignment and the gap columns
        read off it meaningless. Measured on NW_007907101: an 800-base HSP
        inside a 181,395-base record.

        The margin matters as much as the window. The gap sits between the two
        flanks, so the subject has to extend past the HSP on both sides for
        anything to align across it - a window trimmed exactly to the HSP would
        stop at the very junction being reconstructed.
        """
        if hit.subject_start and hit.subject_end:
            low, high = sorted((hit.subject_start, hit.subject_end))
            window = await self._ncbi.fetch_sequence_window(
                hit.accession, low - HOMOLOG_WINDOW_MARGIN, high + HOMOLOG_WINDOW_MARGIN
            )
            if window:
                return window
            # The range was rejected or empty; fall through to the whole record
            # rather than dropping a homologue that BLAST said spans the gap.
        record = await self._ncbi.fetch_sequence(hit.accession)
        return record.residues


def detect_gaps(
    record: SequenceRecord, *, min_length: int = DEFAULT_MIN_GAP_LENGTH
) -> tuple[Gap, ...]:
    """Every unresolved region in `record`, in coordinate order."""
    return tuple(
        Gap(gap_id=f"gap_{index}", start=match.start(), end=match.end())
        for index, match in enumerate(_AMBIGUOUS_RUN.finditer(record.residues), start=1)
        if match.end() - match.start() >= min_length
    )


def build_context(record: SequenceRecord, gap: Gap, *, flank_size: int) -> GapContext:
    """A gap together with the flanking sequence that will be searched."""
    return GapContext(
        gap=gap,
        source_accession=record.accession,
        left_flank=record.left_flank_of(gap, flank_size),
        right_flank=record.right_flank_of(gap, flank_size),
    )


def triage(
    gaps: tuple[Gap, ...], *, max_gaps: int, max_gap_length: int
) -> tuple[tuple[Gap, ...], tuple[Gap, ...]]:
    """Split gaps into those this run will attempt and those it will not.

    Shorter gaps first. Not arbitrary: a short gap is both cheaper to resolve
    and far more likely to be resolvable from homology, so ordering this way
    maximises the number of regions actually recovered within the budget.
    """
    attemptable = sorted(
        (gap for gap in gaps if gap.length <= max_gap_length), key=lambda gap: gap.length
    )
    too_long = [gap for gap in gaps if gap.length > max_gap_length]

    selected = attemptable[:max_gaps]
    deferred = attemptable[max_gaps:] + too_long
    return tuple(selected), tuple(deferred)


def reconcile_requested_gaps(
    requested: tuple[RequestedGap, ...], record: SequenceRecord
) -> tuple[tuple[Gap, ...], tuple[str, ...]]:
    """Snap caller-supplied gap coordinates onto the record's real N-runs.

    Callers do not agree on a coordinate origin. The Genome Agent reads NCBI's
    feature table, which is 1-based with an inclusive end, and forwards it
    unconverted; this agent's `Gap` is 0-based with an exclusive end. Applying
    one convention to the other shifts the region by a base at the start and
    drops one at the end - it would leave the first N unrepaired and overwrite
    the first base of the right flank, and nothing downstream could detect it.

    Rather than guess the convention, this resolves it against the record,
    which is the only authority on where the ambiguity actually is: the caller
    is treated as naming *which* run it means, and the run's own boundaries are
    what get used. That absorbs the origin mismatch, an off-by-one from any
    other source, and NCBI's placeholder spans for gaps whose true size is
    recorded only in an `estimated_length` qualifier.

    A gap that matches no run is kept exactly as supplied - the caller may know
    something about the record that this copy of it does not - but it is
    reported, because the alternative is silently reconstructing the wrong
    bases.
    """
    runs = tuple(
        Gap(gap_id="run", start=match.start(), end=match.end())
        for match in _AMBIGUOUS_RUN.finditer(record.residues)
    )

    gaps: list[Gap] = []
    notes: list[str] = list(note for item in requested for note in item.notes)

    for item in requested:
        supplied = item.gap
        run = _run_for(supplied, runs)

        if run is None:
            notes.append(
                f"{supplied.gap_id}: the supplied coordinates "
                f"({supplied.start}-{supplied.end}) match no unresolved region in "
                f"{record.accession}; they were used unchanged."
            )
            gaps.append(supplied)
            continue

        if (run.start, run.end) == (supplied.start, supplied.end):
            gaps.append(supplied)
            continue

        notes.append(
            f"{supplied.gap_id}: supplied as {supplied.start}-{supplied.end} "
            f"({supplied.length} bases) and matched to the unresolved run at "
            f"{run.start}-{run.end} ({run.length} bases) in {record.accession}"
            f"{_origin_hint(supplied, run)}."
        )
        gaps.append(Gap(gap_id=supplied.gap_id, start=run.start, end=run.end))

    return tuple(gaps), tuple(notes)


def _run_for(supplied: Gap, runs: tuple[Gap, ...]) -> Gap | None:
    """The unresolved run the caller was pointing at, under either convention.

    Both readings of the supplied coordinates are tried, and the run sharing
    the most bases with either wins. Ties go to the earlier run, which is the
    one the caller's own start value is closest to.
    """
    readings = (
        (supplied.start, supplied.end),
        # The same region read as 1-based with an inclusive end.
        (supplied.start - 1, supplied.end),
    )

    best: Gap | None = None
    best_overlap = 0
    for run in runs:
        overlap = max(min(run.end, end) - max(run.start, start) for start, end in readings)
        if overlap > best_overlap:
            best, best_overlap = run, overlap
    return best


def _origin_hint(supplied: Gap, run: Gap) -> str:
    """Name the coordinate convention when the shift identifies one exactly."""
    if (supplied.start - 1, supplied.end) == (run.start, run.end):
        return ", which is that region read as 1-based with an inclusive end"
    return ""
