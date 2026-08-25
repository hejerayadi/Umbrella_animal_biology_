"""M3 pipeline unit tests. No network, no GBIF, fully deterministic.

The important one is ``test_weights_are_normalised_so_noise_stays_reachable``:
it pins the defect found in the design document's own code sample, which is the
single most consequential thing to be able to demonstrate about this module.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.agents.biodiversity_agent.schema import AgentRequest, AgentStatus
from backend.agents.biodiversity_agent.workers.hotspots.pipeline import (
    clustering,
    effort,
    grid,
    indices,
)
from backend.agents.biodiversity_agent.workers.hotspots.pipeline.cleaning import (
    clean_occurrences,
)
from backend.agents.biodiversity_agent.workers.hotspots.status import (
    M3Outcome,
    outcome_of,
)
from backend.agents.biodiversity_agent.workers.hotspots.worker import HotspotsWorker
from backend.agents.biodiversity_agent.tests.m3_gazetteer import (
    patch_gazetteer,
)

@pytest.fixture(autouse=True)
def _offline_gazetteer(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in the suite: place lookups use the fixed table."""

    patch_gazetteer(monkeypatch)


BBOX = (0.0, 0.0, 10.0, 10.0)


def _records(rows: list[dict]) -> pd.DataFrame:
    """A GBIF-shaped frame with sensible defaults per row."""

    base = {"speciesKey": 1.0, "species": "Aaa bbb", "decimalLatitude": 1.0,
            "decimalLongitude": 1.0, "year": 2020, "countryCode": "KE",
            "taxonRank": "SPECIES"}
    return pd.DataFrame([{**base, **row} for row in rows])


# ----------------------------------------------------------------- cleaning


def test_each_cleaning_rule_removes_exactly_its_own_rows() -> None:
    table = _records([
        {},                                                    # keeper
        {"decimalLatitude": None},                             # rule 1
        {"speciesKey": None},                                  # rule 1
        {"taxonRank": "GENUS"},                                # rule 2
        {},                                                    # rule 3 duplicate
        {"decimalLatitude": 0.0, "decimalLongitude": 0.0},     # rule 4 null island
        {"decimalLatitude": 80.0},                             # rule 5 outside bbox
    ])

    clean, report = clean_occurrences(table, BBOX)
    removed = {entry["rule"]: entry["removed"] for entry in report if "kept" not in entry}

    assert removed["no coordinate or no species key"] == 2
    assert removed["not identified to species rank"] == 1
    assert removed["duplicate observations"] == 1
    assert removed["null island (0, 0)"] == 1
    assert removed["outside the study area"] == 1
    assert len(clean) == 1


# ----------------------------------------------------------------- indices


def test_richness_counts_species_not_records() -> None:
    # 100 sightings of one bird is a richness of 1, not 100.
    assert indices.cell_indices({7: 100})["richness"] == 1


def test_shannon_rewards_evenness_at_equal_richness() -> None:
    balanced = indices.cell_indices({i: 10 for i in range(10)})
    skewed = indices.cell_indices({0: 91, **{i: 1 for i in range(1, 10)}})

    assert balanced["richness"] == skewed["richness"] == 10
    assert balanced["shannon"] > skewed["shannon"]
    # A perfectly even assemblage of n species has Shannon = ln(n).
    assert balanced["shannon"] == pytest.approx(math.log(10), abs=1e-9)


def test_chao1_adds_more_when_singletons_dominate() -> None:
    # f1=4, f2=1 -> 5 + 16/2 = 13; many singletons means more unseen species.
    many_singletons = indices.cell_indices({1: 1, 2: 1, 3: 1, 4: 1, 5: 2})
    assert many_singletons["chao1"] == pytest.approx(13.0)

    well_sampled = indices.cell_indices({1: 30, 2: 25, 3: 40})
    assert well_sampled["chao1"] == pytest.approx(well_sampled["richness"])


def test_noise_floor_drops_thin_cells() -> None:
    table = grid.assign_cells(_records(
        [{"decimalLatitude": 1.0, "decimalLongitude": 1.0, "speciesKey": float(i)}
         for i in range(12)] +
        [{"decimalLatitude": 8.0, "decimalLongitude": 8.0, "speciesKey": 99.0}]
    ), cell_size_km=42.0)

    cells, dropped = indices.build_cell_table(table, 42.0, min_records_per_cell=10)
    assert len(cells) == 1          # only the well-sampled cell survives
    assert dropped == 1


# ----------------------------------------------------------------- grid


@pytest.mark.parametrize("latitude", [0.0, 45.0, 70.0])
def test_one_cell_step_is_the_same_ground_distance_at_any_latitude(latitude) -> None:
    """The whole point of the sinusoidal projection (problem P3).

    On a degree grid a polar cell covers a quarter of an equatorial one, so a
    richness map would report an artefact of latitude next to the biology. Here,
    three cell widths of *ground distance* must always span three cell steps,
    whatever the latitude.

    Each pair starts at longitude 0, which projects to x = 0 exactly, so the
    column index starts at 0 and floor() boundaries cannot muddy the comparison.
    """

    cell_size_km = 42.0
    three_cells_km = 3 * cell_size_km

    # Invert x = R * lon_rad * cos(lat) for the longitude three cells east.
    lon_rad = three_cells_km / (grid.EARTH_RADIUS_KM * math.cos(math.radians(latitude)))
    lon_deg = math.degrees(lon_rad)

    binned = grid.assign_cells(_records([
        {"decimalLatitude": latitude, "decimalLongitude": 0.0},
        {"decimalLatitude": latitude, "decimalLongitude": lon_deg},
    ]), cell_size_km=cell_size_km)

    columns = [int(cell_id.split("_")[0]) for cell_id in binned["cell_id"]]
    assert columns[1] - columns[0] == 3


def test_study_area_shrinks_towards_the_pole() -> None:
    equatorial = grid.study_area_km2((0.0, 0.0, 10.0, 10.0))
    polar = grid.study_area_km2((0.0, 70.0, 10.0, 80.0))
    assert equatorial > polar * 2


# ----------------------------------------------------------------- effort


def _cells(records: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(len(records))],
        "lat": np.linspace(0, 1, len(records)),
        "lon": np.linspace(0, 1, len(records)),
        "n_species": [10] * len(records),
        "n_records": records,
        "shannon": [1.5] * len(records),
        "simpson": [0.6] * len(records),
        "chao1": [12.0] * len(records),
        "area_km2": [1764.0] * len(records),
        "country": ["KE"] * len(records),
    })


def test_effort_correction_lifts_undersampled_and_damps_oversampled() -> None:
    cells, record = effort.correct_for_effort(_cells([10, 100, 1000]))

    assert record.applied is True
    assert record.median_effort == 100
    # Equal richness, so the correction alone must reorder them.
    assert cells.loc[0, "corrected_species"] > cells.loc[1, "corrected_species"]
    assert cells.loc[2, "corrected_species"] < cells.loc[1, "corrected_species"]


def test_effort_boost_is_capped_so_one_record_cannot_win() -> None:
    cells, _ = effort.correct_for_effort(_cells([1, 10_000]))
    assert cells["effort_factor"].max() <= effort.CORRECTION_CAP


def test_skipping_the_correction_still_reports_that_it_was_skipped() -> None:
    cells, record = effort.correct_for_effort(_cells([10, 100]), applied=False)
    assert record.applied is False
    assert "raw" in record.note
    assert (cells["corrected_species"] == cells["n_species"]).all()


# ----------------------------------------------------------------- clustering


def _two_clusters_plus_outliers() -> pd.DataFrame:
    """Two tight groups about 900 km apart, plus three isolated cells."""

    rows = []
    for index in range(10):
        rows.append({"lat": 0.0 + index * 0.05, "lon": 0.0, "n_species": 40})
    for index in range(10):
        rows.append({"lat": 8.0 + index * 0.05, "lon": 0.0, "n_species": 35})
    for lat in (20.0, 30.0, 40.0):
        rows.append({"lat": lat, "lon": 25.0, "n_species": 5})

    frame = pd.DataFrame(rows)
    frame["cell_id"] = [f"c{i}" for i in range(len(frame))]
    frame["n_records"] = 50
    frame["shannon"] = 1.5
    frame["simpson"] = 0.6
    frame["chao1"] = 45.0
    frame["area_km2"] = 1764.0
    frame["country"] = "KE"
    corrected, _ = effort.correct_for_effort(frame)
    return corrected


def test_weights_are_normalised_so_noise_stays_reachable() -> None:
    """The defect in the design document's own §4.4 snippet.

    scikit-learn treats a sample whose weight is at least ``min_samples`` as a
    core sample on its own. Corrected richness is far above min_samples, so
    passing it raw makes every cell a core sample: noise becomes unreachable and
    even a desert cell is reported as a hotspot. Normalising by the median is the
    fix, and this test fails if anyone removes it.
    """

    cells = _two_clusters_plus_outliers()
    _, labels = clustering.run_dbscan(cells, eps_km=120.0, min_samples=5)

    quality = clustering.measure_quality(cells, labels, eps_km=120.0, min_samples=5)

    assert quality.n_clusters >= 2, "the two dense groups must be separate clusters"
    assert quality.noise_share > 0, "the isolated cells must be labelled noise"
    assert quality.silhouette is not None and quality.silhouette > 0.5


def test_dbscan_is_reproducible() -> None:
    cells = _two_clusters_plus_outliers()
    _, first = clustering.run_dbscan(cells, 120.0, 5)
    _, second = clustering.run_dbscan(cells, 120.0, 5)
    assert (first == second).all()


def test_k_distance_curve_is_sorted_and_in_km() -> None:
    curve = clustering.k_distance_curve(_two_clusters_plus_outliers(), k=5)
    assert len(curve) > 0
    assert (np.diff(curve) >= -1e-9).all()
    assert curve.max() < 20_000            # half the Earth's circumference


def test_tuning_only_accepts_configurations_inside_the_noise_band() -> None:
    eps_km, min_samples, scan, tuned = clustering.tune(
        _two_clusters_plus_outliers(), fallback_eps=120.0, fallback_min_samples=5)

    assert scan, "the scan must record every combination it tried"
    if tuned:
        chosen = [row for row in scan
                  if row["eps_km"] == eps_km and row["min_samples"] == min_samples][0]
        assert chosen["admissible"] is True
        assert 0.05 < chosen["noise_share"] < 0.80


# ----------------------------------------------------------------- worker


def test_unknown_region_asks_instead_of_guessing() -> None:
    result = HotspotsWorker().run(
        AgentRequest(instruction="hotspots in Atlantis", context={}, region="atlantis"))

    # NEEDS_CLARIFICATION narrows to FAILED at the platform boundary; the
    # module's own outcome is preserved in the payload.
    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert result.status is AgentStatus.FAILED
    assert "atlantis" in result.output["message"].lower()
    assert "congo basin" in result.output["known_regions"]


def test_global_is_refused_as_a_study_area() -> None:
    """Asked for the whole planet, M3 says why it cannot grid it."""

    result = HotspotsWorker().run(
        AgentRequest(instruction="show hotspots globally", context={},
                     region="global"))
    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert "too large" in result.output["message"]


def test_the_default_region_does_not_speak_for_the_user() -> None:
    """``AgentRequest.region`` defaults to "global" - that is not a request.

    A question with no place in it at all must not be answered with "the whole
    globe is too large", which replies to something nobody asked.
    """

    result = HotspotsWorker().run(AgentRequest(
        instruction="show me the hotspots", context={}))

    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert result.output["message"] == "I could not tell which study area you mean."


def test_a_continent_is_too_large_to_grid() -> None:
    """"Which part of Africa" names a real place, and gets a real reason back."""

    result = HotspotsWorker().run(AgentRequest(
        instruction="Which part of Africa has the most species?", context={}))

    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert "too large" in result.output["message"]
    assert "congo basin" in result.output["known_regions"], "and what it can do"


# ------------------------------- the grid fits the place, not the other way round


def test_a_small_island_gets_a_finer_grid() -> None:
    """Pantelleria is 136 km2: the documented 42 km cell is one cell.

    Refusing it would be refusing a real question because of a default - and the
    island has 1,500 GBIF records in it.
    """

    worker = HotspotsWorker()
    params = worker.build_params({"bbox": (11.92, 36.73, 12.07, 36.85),
                                  "region_name": "Pantelleria"})

    assert params.cell_size_km < 42.0
    assert params.cell_size_km >= 1.0
    assert params.cell_size_note and "refined" in params.cell_size_note


def test_a_documented_region_keeps_the_documented_grid() -> None:
    """The report's numbers depend on the 42 km cell, so nothing may move it."""

    params = HotspotsWorker().build_params({"region_name": "congo basin"})

    assert params.cell_size_km == 42.0
    assert params.cell_size_note is None


def test_an_explicit_cell_size_is_never_overridden() -> None:
    """A caller who asks for 42 km over a small island gets 42 km, and one cell."""

    params = HotspotsWorker().build_params({"bbox": (11.92, 36.73, 12.07, 36.85),
                                            "region_name": "Pantelleria",
                                            "cell_size_km": 42.0})

    assert params.cell_size_km == 42.0
    assert params.cell_size_note is None


def test_eps_scales_with_a_refined_grid() -> None:
    """eps is a distance between cell centres, so it moves with the cell size.

    The documented 120 km eps across a 27 km island puts every cell in one
    cluster by construction, and the scan then reports that nothing qualified for
    a reason that has nothing to do with the data.
    """

    params = HotspotsWorker().build_params({"bbox": (14.18, 35.79, 14.58, 36.08),
                                            "region_name": "Malta"})

    assert params.cell_size_km < 42.0
    assert params.eps_km < 120.0
    # The documented ratio is kept: 120 / 42.
    assert abs(params.eps_km / params.cell_size_km - 120 / 42) < 0.05


def test_a_documented_region_keeps_the_documented_eps() -> None:
    params = HotspotsWorker().build_params({"region_name": "amazon"})

    assert params.cell_size_km == 42.0
    assert params.eps_km == 120.0


def test_the_scan_grid_scales_only_for_a_refined_cell() -> None:
    from backend.agents.biodiversity_agent.workers.hotspots.worker import (
        _eps_grid_for,
    )

    assert _eps_grid_for(42.0) is None, "documented cell scans documented values"
    scaled = _eps_grid_for(10.0)
    assert scaled is not None and max(scaled) < 120


# ---------------------------------------- the cells drawn are the cells measured


def test_drawn_cells_tile_without_overlapping() -> None:
    """A cell rectangle comes from its id, not from where its records happen to sit.

    Drawing each box around the mean of its records put the boxes off the cells
    they described and made them overlap - a dozen translucent squares layered
    over each other on Malta. The id encodes the projection, so it inverts
    exactly.
    """

    import itertools

    from backend.agents.biodiversity_agent.workers.hotspots.pipeline.grid import (
        assign_cells,
    )
    from backend.agents.biodiversity_agent.workers.hotspots.pipeline.render import (
        _cell_bounds,
    )

    rng = np.random.default_rng(0)
    records = pd.DataFrame({
        "decimalLatitude": 35.9 + rng.normal(0, 0.08, 400),
        "decimalLongitude": 14.4 + rng.normal(0, 0.08, 400),
    })
    gridded = assign_cells(records, 10.1)
    boxes = [_cell_bounds(cell_id, 10.1, 0.0, 0.0)
             for cell_id in gridded["cell_id"].unique()]

    def overlap(one, two) -> bool:
        (south1, west1), (north1, east1) = one
        (south2, west2), (north2, east2) = two
        return (min(north1, north2) - max(south1, south2) > 1e-9
                and min(east1, east2) - max(west1, west2) > 1e-9)

    assert not any(overlap(a, b) for a, b in itertools.combinations(boxes, 2))

    # And each box is the size it claims to be.
    (south, west), (north, east) = boxes[0]
    height = abs(north - south) * 110.57
    width = (abs(east - west) * 111.32
             * math.cos(math.radians((south + north) / 2)))
    assert abs(height - 10.1) < 0.3
    assert abs(width - 10.1) < 0.3
