"""M3 stage T3 - index. Turn per-cell counts into comparable measures.

Doc §5.3. Four measures, because they answer different questions and can rank
the same cells differently - which is why the index is a user-selectable input
rather than a hard-coded choice.
"""

from __future__ import annotations

import math

import pandas as pd


def cell_indices(species_counts: dict[int, int]) -> dict[str, float]:
    """Richness, Shannon, Simpson and Chao1 for one cell.

    ``species_counts`` maps species_key -> number of records in this cell.
    """

    total = sum(species_counts.values())
    if total == 0:
        return {"richness": 0, "shannon": 0.0, "simpson": 0.0, "chao1": 0.0}

    # Richness counts DISTINCT species, not records. A cell with 100 sightings of
    # one bird has a richness of 1.
    richness = len(species_counts)

    proportions = [count / total for count in species_counts.values()]

    # Shannon rewards evenness as well as count: two cells with 10 species each
    # score differently if one is dominated by a single species.
    shannon = -sum(p * math.log(p) for p in proportions)

    # Simpson: the chance that two records drawn at random are different species.
    simpson = 1.0 - sum(p * p for p in proportions)

    # Chao1 estimates species present but not yet observed, from the number seen
    # exactly once (f1) and exactly twice (f2). Many singletons means the survey
    # is plainly incomplete, so others must be there.
    f1 = sum(1 for c in species_counts.values() if c == 1)
    f2 = sum(1 for c in species_counts.values() if c == 2)
    chao1 = (richness + (f1 * f1) / (2 * f2)) if f2 > 0 else richness + f1 * (f1 - 1) / 2

    return {"richness": richness, "shannon": shannon,
            "simpson": simpson, "chao1": chao1}


def build_cell_table(
    table: pd.DataFrame,
    cell_size_km: float,
    min_records_per_cell: int,
) -> tuple[pd.DataFrame, int]:
    """One row per cell, with its position, effort and indices.

    Returns the table plus how many cells the noise floor removed. The floor is
    applied here, before anything else: doc Table 13 is explicit that one stray
    record must never be able to seed a hotspot.
    """

    if table.empty:
        return pd.DataFrame(), 0

    rows = []
    for cell_id, group in table.groupby("cell_id"):
        counts = group["speciesKey"].value_counts().to_dict()
        measures = cell_indices(counts)
        rows.append({
            "cell_id": cell_id,
            # The cell's position is the mean of its records rather than the
            # geometric centre, so a marker sits where the observations are.
            "lat": float(group["decimalLatitude"].mean()),
            "lon": float(group["decimalLongitude"].mean()),
            "n_species": int(measures["richness"]),
            "n_records": int(len(group)),          # the effort proxy for this cell
            "shannon": float(measures["shannon"]),
            "simpson": float(measures["simpson"]),
            "chao1": float(measures["chao1"]),
            "area_km2": cell_size_km ** 2,
            "country": (group["countryCode"].mode().iat[0]
                        if group["countryCode"].notna().any() else None),
        })

    cells = pd.DataFrame(rows)
    before = len(cells)
    cells = cells[cells["n_records"] >= min_records_per_cell].reset_index(drop=True)
    return cells, before - len(cells)
