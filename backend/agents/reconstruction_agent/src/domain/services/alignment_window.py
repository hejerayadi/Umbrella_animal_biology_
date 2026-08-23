"""Cutting a reference down to the part worth aligning.

MAFFT is submitted as a job and polled, and one slice grants it at most ~75
seconds. A BLAST-derived reference is already short - `BlastSearchTool` fetches
only the subject span the hit covers - but an `ncbi_search` record is the whole
sequence it was published as, which for a mitogenome is ~16,000 bases. Aligning
a 400-base target against several of those runs past the slice deadline and is
killed, every time, for as many slices as the run is granted. Measured: the same
prompt finishes in 8.9 s when the planner opens with BLAST and never finishes
when it opens with `ncbi_search`.

The excess is not evidence. Only the region homologous to the flanks can say
anything about what sits between them; sequence tens of kilobases away cannot,
whatever the aligner does with it. So the window here is a performance fix that
costs no information - provided the window is actually placed on the homologous
region, which is what `locate_window` is careful about.

Nothing is truncated blindly. A reference that cannot be anchored is returned
unchanged and left for the caller to decide about, because a window placed at
the wrong offset would be worse than a slow alignment: it would quietly align
the target against unrelated sequence and produce a confident, wrong fill.
"""
from __future__ import annotations

from dataclasses import dataclass

#: How much sequence either side of the anchor is kept. Generous on purpose:
#: the aligner needs room to place the flanks and the gap, and the cost of a
#: few hundred extra bases is nothing next to a whole genome.
DEFAULT_MARGIN = 400

#: Only references longer than this are windowed at all. Below it there is
#: nothing to gain and an anchoring mistake would be pure loss.
DEFAULT_MAX_RESIDUES = 3_000

#: Length of the probe taken from a flank to locate the homologous region.
#: Long enough to be specific in a genome (4^24 is astronomically more than any
#: genome length), short enough to still match across species at conserved
#: positions.
_PROBE = 24


@dataclass(frozen=True, slots=True)
class Window:
    """Where a reference was cut, and why."""

    start: int
    stop: int
    #: "subject_span" (BLAST told us), "probe" (a flank was located), or
    #: "none" (nothing to go on).
    anchor: str


def _span_from_metadata(metadata: dict[str, str], length: int) -> tuple[int, int] | None:
    """The subject region a BLAST hit reported, as 0-based Python bounds.

    Stored by `tools.blast.mapper._metadata` as "start-stop", 1-based and
    inclusive the way BLAST reports coordinates.
    """
    raw = (metadata or {}).get("subject_span")
    if not raw or "-" not in raw:
        return None
    head, _, tail = raw.partition("-")
    try:
        start, stop = int(head), int(tail)
    except ValueError:
        return None
    if start < 1 or stop < start:
        return None
    return min(start - 1, length), min(stop, length)


def _probe_offset(residues: str, flank: str, *, from_end: bool) -> int | None:
    """Where a flank's most distinctive end sits inside `residues`, if it does.

    The probe is taken from the flank end nearest the gap - the last bases of
    the left flank, the first of the right - because those are the ones that
    bracket the missing segment. An exact match is required: a probe that
    matched approximately would be evidence of nothing in particular, and the
    caller's fallback (leave the reference alone) is safe.
    """
    if len(flank) < _PROBE:
        return None
    probe = flank[-_PROBE:] if from_end else flank[:_PROBE]
    found = residues.find(probe)
    return None if found < 0 else found


def locate_window(
    residues: str,
    *,
    metadata: dict[str, str] | None = None,
    left_flank: str = "",
    right_flank: str = "",
    margin: int = DEFAULT_MARGIN,
    max_residues: int = DEFAULT_MAX_RESIDUES,
) -> Window | None:
    """The region of `residues` worth aligning, or None to keep all of it.

    None means "do not cut": either the reference is already short enough, or
    nothing located the homologous region and guessing would risk aligning
    against the wrong part of a genome.
    """
    length = len(residues)
    if length <= max_residues:
        return None

    span = _span_from_metadata(metadata or {}, length)
    if span is not None:
        start, stop = span
        return Window(max(0, start - margin), min(length, stop + margin), "subject_span")

    # No span: find the flanks ourselves. Either one is enough to place the
    # window, and having both lets it bracket the gap exactly.
    left_at = _probe_offset(residues, left_flank, from_end=True)
    right_at = _probe_offset(residues, right_flank, from_end=False)

    hits = [offset for offset in (left_at, right_at) if offset is not None]
    if not hits:
        return None

    start = max(0, min(hits) - margin)
    stop = min(length, max(hits) + _PROBE + margin)
    return Window(start, stop, "probe")


def window_residues(
    residues: str,
    *,
    metadata: dict[str, str] | None = None,
    left_flank: str = "",
    right_flank: str = "",
    margin: int = DEFAULT_MARGIN,
    max_residues: int = DEFAULT_MAX_RESIDUES,
) -> tuple[str, str]:
    """`residues` cut to its useful window, with the anchor that placed it.

    Returns the residues unchanged with anchor "none" when no cut was made, so
    a caller can log how each reference was treated without special-casing.
    """
    window = locate_window(
        residues,
        metadata=metadata,
        left_flank=left_flank,
        right_flank=right_flank,
        margin=margin,
        max_residues=max_residues,
    )
    if window is None:
        return residues, "none"
    return residues[window.start : window.stop], window.anchor
