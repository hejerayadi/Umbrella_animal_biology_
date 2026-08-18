"""How concurrent node updates get merged into the state.

LangGraph runs independent branches in parallel - several gaps are searched at
once - and then has to combine what each returned. Without an explicit reducer
the last writer wins, which for a dict keyed by gap id means losing every gap
but one.
"""
from __future__ import annotations

from typing import TypeVar

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
