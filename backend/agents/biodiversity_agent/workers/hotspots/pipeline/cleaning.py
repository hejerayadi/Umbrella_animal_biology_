"""M3 stage T1 - filter. Keep only records that can carry richness.

Doc §5.2 stage T1 and §3.5. Five rules, each counted, because a deleted row must
stay explainable: the report travels up to the orchestrator in the output.
"""

from __future__ import annotations

import pandas as pd


# Ranks at or below species. A record identified to subspecies is still a record
# of the species - it is only useless when the question is "how many species are
# here", which is why the region path excludes it and the species path does not.
INFRASPECIFIC = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")


def clean_occurrences(
    table: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    *,
    allow_infraspecific: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the five documented rules; return the clean table and a report.

    ``allow_infraspecific`` widens rule 2 for the species path. Measured on
    Panthera tigris: 4,552 of 5,008 GBIF records are at SUBSPECIES rank, so
    keeping the rule strict there would throw away 91% of the tiger data.
    """

    if table.empty:
        return table, [{"rule": "input", "removed": 0, "kept": 0}]

    start = len(table)
    report: list[dict] = []

    def _step(rule: str, before: int, after: pd.DataFrame) -> pd.DataFrame:
        report.append({"rule": rule, "removed": before - len(after)})
        return after

    # 1. A record without a coordinate or a species key cannot be placed or counted.
    table = _step("no coordinate or no species key", len(table),
                  table.dropna(subset=["decimalLatitude", "decimalLongitude", "speciesKey"]))

    # 2. Identified to species at least: "bird" is not a species, Loxodonta
    #    africana is. Doc §3.5 enforces this both in the predicate and again here.
    if allow_infraspecific:
        table = _step("not identified to species rank or below", len(table),
                      table[table["taxonRank"].isin(INFRASPECIFIC)])
    else:
        table = _step("not identified to species rank", len(table),
                      table[table["taxonRank"] == "SPECIES"])

    # 3. The same observation is often republished by several aggregators, which
    #    would inflate both richness and effort.
    table = _step("duplicate observations", len(table),
                  table.drop_duplicates(subset=["speciesKey", "decimalLatitude",
                                                "decimalLongitude", "year"]))

    # 4. "Null island" - a missing coordinate written as zero lands at (0, 0) in
    #    the Gulf of Guinea.
    on_null_island = ((table["decimalLatitude"].abs() < 1e-6) &
                      (table["decimalLongitude"].abs() < 1e-6))
    table = _step("null island (0, 0)", len(table), table[~on_null_island])

    # 5. GBIF can return points slightly outside the requested polygon.
    lon_min, lat_min, lon_max, lat_max = bbox
    inside = (table["decimalLongitude"].between(lon_min, lon_max) &
              table["decimalLatitude"].between(lat_min, lat_max))
    table = _step("outside the study area", len(table), table[inside])

    report.append({
        "rule": "kept",
        "removed": 0,
        "kept": len(table),
        "share_of_input": round(len(table) / start, 3) if start else 0.0,
    })
    return table.reset_index(drop=True), report
