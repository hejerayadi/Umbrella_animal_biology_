"""How concurrent node updates get merged into the state.

LangGraph runs independent branches in parallel - several gaps are searched at
once - and then has to combine what each returned. Without an explicit reducer
the last writer wins, which for a dict keyed by gap id means losing every gap
but one.
"""
from __future__ import annotations

from typing import TypeVar

from domain.models import Reference

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def replace(current: T, incoming: T) -> T:
    """Last write wins.

    For fields a single node owns outright - the gap list, the current plan -
    where a later value is a genuine correction rather than a parallel result.
    """
    return incoming


def merge_by_gap(current: dict[K, V] | None, incoming: dict[K, V] | None) -> dict[K, V]:
    """Shallow dict merge, incoming keys winning.

    The workhorse for the per-gap evidence maps. Two branches working on
    different gaps contribute disjoint keys and both survive; a branch
    revisiting its own gap in a later iteration overwrites just that entry.
    """
    if not current:
        return dict(incoming or {})
    if not incoming:
        return dict(current)
    return {**current, **incoming}


def extend_by_gap(
    current: dict[K, list[T]] | None, incoming: dict[K, list[T]] | None
) -> dict[K, list[T]]:
    """Dict merge that concatenates the lists under shared keys.

    For evidence that accumulates across iterations rather than being replaced
    by it - a second BLAST round adding references to a gap that already has
    some, instead of discarding the first round's.
    """
    merged: dict[K, list[T]] = {key: list(value) for key, value in (current or {}).items()}
    for key, value in (incoming or {}).items():
        merged.setdefault(key, []).extend(value)
    return merged


def accumulate_references(
    current: dict[str, list[Reference]] | None,
    incoming: dict[str, list[Reference]] | None,
) -> dict[str, list[Reference]]:
    """Merge reference lists per gap, keeping the best version of each accession.

    Evidence has to *deepen* across iterations and slices, not be replaced by
    the latest round. With a plain overwrite, a second search that happened to
    return fewer hits shrank the pool - and since confidence scales with how
    many references support a fill, a round that added evidence could lower the
    score.

    Deduplication is by accession, because the same record legitimately arrives
    from several tools: BLAST finds it and measures its homology, NCBI fetches
    it and knows only its name. `Reference.quality` decides which copy wins, and
    residues always beat no residues - a reference that cannot be aligned is not
    evidence, whatever its metadata says.
    """
    merged: dict[str, list[Reference]] = {
        gap: list(references) for gap, references in (current or {}).items()
    }

    for gap, references in (incoming or {}).items():
        by_accession: dict[str, Reference] = {}
        for reference in [*merged.get(gap, []), *references]:
            existing = by_accession.get(reference.accession)
            if existing is None or _prefer(reference, existing):
                by_accession[reference.accession] = reference
        merged[gap] = list(by_accession.values())

    return merged


def _prefer(candidate: Reference, existing: Reference) -> bool:
    """Whether `candidate` is the better copy of an accession already held."""
    if candidate.has_sequence != existing.has_sequence:
        return candidate.has_sequence
    return candidate.quality > existing.quality


def unique_extend(current: list[T] | None, incoming: list[T] | None) -> list[T]:
    """Append, dropping values already present and preserving order.

    Used for `tools_used`, where the same tool running on five gaps should
    appear once.
    """
    seen = list(current or [])
    for item in incoming or []:
        if item not in seen:
            seen.append(item)
    return seen
