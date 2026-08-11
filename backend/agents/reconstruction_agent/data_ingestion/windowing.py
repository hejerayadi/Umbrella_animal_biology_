from __future__ import annotations

import random

from . import CleanedScaffold, MaskedTriplet, Window


def extract_windows(scaffold: CleanedScaffold, step: int = 1000) -> list[Window]:
    """Slide 2000 bp windows across a scaffold and discard terminal windows smaller than 2000 bp."""
    seq = scaffold.sequence
    windows: list[Window] = []
    for start in range(0, len(seq) - 1999, step):
        end = start + 2000
        windows.append(
            Window(
                scaffold_id=getattr(scaffold, "scaffold_id", "unknown"),
                accession=getattr(scaffold, "accession", ""),
                species_id=getattr(scaffold, "species_id", ""),
                sequence=seq[start:end],
                start=start,
                end=end,
            )
        )
    return windows


def mask_window(window: Window, seed: int | None = None) -> MaskedTriplet:
    """Mask a contiguous segment from the center of a 2000 bp window."""
    if len(window.sequence) != 2000:
        raise ValueError("Window sequence must be exactly 2000 bp")

    rng = random.Random(seed) if seed is not None else random.Random()
    masked_length = rng.randint(10, min(500, len(window.sequence)))
    center = len(window.sequence) // 2
    start = max(0, center - masked_length // 2)
    end = min(len(window.sequence), start + masked_length)
    masked_region = window.sequence[start:end]
    left_context = window.sequence[:start]
    right_context = window.sequence[end:]
    assert len(left_context) + len(masked_region) + len(right_context) == 2000
    return MaskedTriplet(
        window_id=f"{window.scaffold_id}:{window.start}-{window.end}",
        left_context=left_context,
        masked_region=masked_region,
        right_context=right_context,
    )
