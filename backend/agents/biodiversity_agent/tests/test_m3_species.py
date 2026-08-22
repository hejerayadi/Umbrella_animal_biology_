"""The species-scoped path: one species, occurrence clusters, honest labels.

Everything here is offline. GBIF is replaced at the module boundary, so what is
under test is the module's own arithmetic and its wording - not the network.

The distinction these tests defend: with one species, richness is 1 everywhere,
so the answer must be ranked and described as *occurrence density*. A test that
let "biodiversity hotspot" through for a single species would be letting the
module make a claim the data cannot support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.agents.biodiversity_agent.schema import AgentRequest, AgentStatus
from backend.agents.biodiversity_agent.workers.common import gbif
from backend.agents.biodiversity_agent.workers.common.config import (
    WORLD_BBOX,
    species_in_text,
)
from backend.agents.biodiversity_agent.workers.hotspots import worker as worker_module
from backend.agents.biodiversity_agent.workers.hotspots.pipeline import points
from backend.agents.biodiversity_agent.workers.hotspots.worker import HotspotsWorker
from backend.agents.biodiversity_agent.tests.m3_gazetteer import (
    patch_gazetteer,
)


@pytest.fixture(autouse=True)
def _offline_gazetteer(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in the suite: place lookups use the fixed table."""

    patch_gazetteer(monkeypatch)

TIGER_KEY = 5219416


# ------------------------------------------------- reading the question


@pytest.mark.parametrize("question, expected", [
    ("Where are the biodiversity hotspots for tigers?", "tiger"),
    ("hotspots for Panthera tigris", "Panthera tigris"),
    ("where do polar bears live", "polar bear"),
    ("show me African elephants", "african elephant"),
])
def test_a_species_question_is_recognised(question: str, expected: str) -> None:
    assert species_in_text(question) == expected


@pytest.mark.parametrize("question", [
    "Where are the biodiversity hotspots in Madagascar?",
    "Which part of Africa has the most species?",
    "hotspots in the Congo basin",          # a region, not a genus and epithet
    "Show me a biodiversity heatmap of Southeast Asia",
    "",
])
def test_a_region_question_is_not_read_as_a_species(question: str) -> None:
    """A false positive here would silently answer about one animal."""

    assert species_in_text(question) is None


# ------------------------------------------------- resolving the name


def _fake_gbif(match: dict | None = None, search: list | None = None):
    """A stand-in for requests.get that answers the two GBIF endpoints."""

    class _Answer:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fetch(url, params):
        if url == gbif.MATCH_URL:
            name = (params.get("name") or "").lower()
            if match and name == match["_name"].lower():
                return _Answer(match)
            return _Answer({"matchType": "NONE"})
        return _Answer({"results": search or []})

    return fetch


def test_a_scientific_name_resolves() -> None:
    fetch = _fake_gbif(match={"_name": "Panthera tigris", "rank": "SPECIES",
                              "matchType": "EXACT", "confidence": 99,
                              "usageKey": TIGER_KEY, "species": "Panthera tigris"})
    assert gbif.match_species("Panthera tigris", fetch=fetch) == (
        TIGER_KEY, "Panthera tigris")


def test_a_common_name_resolves_through_the_table() -> None:
    """"tiger" must not depend on GBIF's ranking, which buries it - see gbif.py."""

    fetch = _fake_gbif(match={"_name": "Panthera tigris", "rank": "SPECIES",
                              "matchType": "EXACT", "confidence": 99,
                              "usageKey": TIGER_KEY, "species": "Panthera tigris"})
    assert gbif.match_species("tigers", fetch=fetch) == (TIGER_KEY, "Panthera tigris")


def test_a_low_confidence_match_is_refused() -> None:
    """A guessed species answers about the wrong animal, so it is not a match."""

    fetch = _fake_gbif(match={"_name": "pantera", "rank": "SPECIES",
                              "matchType": "FUZZY", "confidence": 40,
                              "usageKey": 1, "species": "Something else"})
    assert gbif.match_species("pantera", fetch=fetch) is None


def test_a_genus_is_refused() -> None:
    """Rank must be SPECIES: a genus key would widen the question silently."""

    fetch = _fake_gbif(match={"_name": "Panthera", "rank": "GENUS",
                              "matchType": "EXACT", "confidence": 99,
                              "usageKey": 2435099})
    assert gbif.match_species("Panthera", fetch=fetch) is None


def test_an_unresolvable_name_returns_none() -> None:
    assert gbif.match_species("qwertyuiop", fetch=_fake_gbif()) is None
    assert gbif.match_species("", fetch=_fake_gbif()) is None


# ------------------------------------------------- clustering the points


def _blob(lat: float, lon: float, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "speciesKey": TIGER_KEY,
        "species": "Panthera tigris",
        "decimalLatitude": lat + rng.normal(0, 0.15, count),
        "decimalLongitude": lon + rng.normal(0, 0.15, count),
        "year": 2015,
        "countryCode": "IN" if lat < 25 else "NP",
        "taxonRank": "SUBSPECIES",
    })


def _tiger_records() -> pd.DataFrame:
    """Two dense areas plus scatter - the shape DBSCAN is chosen for."""

    scattered = pd.DataFrame({
        "speciesKey": TIGER_KEY, "species": "Panthera tigris",
        "decimalLatitude": [10.0, 30.0, 45.0, 5.0],
        "decimalLongitude": [80.0, 95.0, 130.0, 100.0],
        "year": 2015, "countryCode": "XX", "taxonRank": "SPECIES",
    })
    return pd.concat([_blob(22.0, 78.0, 60, 1), _blob(27.5, 84.0, 40, 2), scattered],
                     ignore_index=True)


def test_dense_areas_cluster_and_strays_become_noise() -> None:
    records = _tiger_records()
    _, labels = points.cluster_occurrences(records, eps_km=60, min_samples=8)

    assert len(set(labels) - {-1}) == 2, "the two blobs"
    assert (labels == -1).sum() == 4, "the four scattered records"


def test_clusters_are_ranked_by_record_count_and_named_as_occurrences() -> None:
    records = _tiger_records()
    _, labels = points.cluster_occurrences(records, eps_km=60, min_samples=8)
    clusters, ranked = points.build_occurrence_clusters(
        records, labels, top_n=10, species_name="Panthera tigris")

    assert [c.record_count for c in ranked] == [60, 40], "busiest first"
    assert ranked[0].rank == 1 and ranked[1].rank == 2
    assert all("Occurrence cluster" in c.label for c in clusters), (
        "never 'biodiversity hotspot': one species has no richness")
    assert all(c.species_count == 1 for c in clusters)
    assert all(c.cell_count == 0 for c in clusters), "no grid on this path"


def test_the_extent_of_a_single_record_is_zero() -> None:
    one = _tiger_records().head(1)
    assert points.extent_km2(one) == 0.0
    assert points.extent_km2(_tiger_records()) > 0


def test_top_n_limits_the_ranked_list_but_not_the_clusters() -> None:
    records = _tiger_records()
    _, labels = points.cluster_occurrences(records, eps_km=60, min_samples=8)
    clusters, ranked = points.build_occurrence_clusters(
        records, labels, top_n=1, species_name="Panthera tigris")

    assert len(clusters) == 2 and len(ranked) == 1


# ------------------------------------------------- the worker, end to end


@pytest.fixture()
def offline_gbif(monkeypatch: pytest.MonkeyPatch) -> None:
    """GBIF replaced where the worker imports it. No network in these tests."""

    records = _tiger_records()
    monkeypatch.setattr(worker_module, "match_species",
                        lambda name: (TIGER_KEY, "Panthera tigris"))
    monkeypatch.setattr(worker_module, "count_probe", lambda *a, **k: (5000, None))
    monkeypatch.setattr(worker_module, "download_occurrences",
                        lambda *a, **k: records.copy())


def test_a_species_question_takes_the_species_path(offline_gbif) -> None:
    result = HotspotsWorker().run(AgentRequest(
        instruction="Where are the biodiversity hotspots for tigers?", context={}))

    assert result.status is AgentStatus.COMPLETED
    payload = result.output
    assert payload.parameters_used["mode"] == "species"
    assert payload.parameters_used["ranked_by"] == "record_count"
    assert payload.parameters_used["species_key"] == TIGER_KEY
    assert payload.species_analysed == 1
    assert payload.cells_analysed == 0, "the grid is not used here"
    assert payload.effort_correction.applied is False
    assert payload.region == "worldwide", "a species is not bound to a study area"


def test_the_species_answer_says_it_is_not_biodiversity(offline_gbif) -> None:
    """The one claim this path must never make."""

    result = HotspotsWorker().run(AgentRequest(
        instruction="hotspots for tigers", context={}))
    payload = result.output

    assert "observed most" in payload.summary
    assert any("not by species richness" in warning for warning in payload.warnings)
    assert "biodiversity hotspot" not in payload.summary.lower()


def test_a_species_inside_one_region_stays_inside_it(offline_gbif) -> None:
    result = HotspotsWorker().run(AgentRequest(
        instruction="hotspots for tigers in Southeast Asia", context={}))

    assert result.output.parameters_used["region"] == "southeast asia"
    assert result.output.parameters_used["bbox"] != WORLD_BBOX


def test_an_unresolvable_species_asks_for_the_scientific_name(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "match_species", lambda name: None)

    result = HotspotsWorker().run(AgentRequest(
        instruction="hotspots for tigers", context={}))

    assert result.status is AgentStatus.FAILED
    assert "scientific name" in result.output["message"]


def test_the_region_path_is_unaffected(offline_gbif) -> None:
    """A region question must not be pulled into the species path."""

    class _Guard:
        called = False

        @staticmethod
        def analyse(*args, **kwargs):
            _Guard.called = True
            raise AssertionError("the species path must not run here")

    worker = HotspotsWorker()
    monkey = worker._analyse_species
    worker._analyse_species = _Guard.analyse           # type: ignore[method-assign]
    try:
        result = worker.run(AgentRequest(
            instruction="Where are the biodiversity hotspots in Madagascar?",
            context={}))
    finally:
        worker._analyse_species = monkey               # type: ignore[method-assign]

    assert _Guard.called is False
    # The region path then runs on the stubbed records, which are all outside
    # Madagascar, so a FAILED here is the honest outcome - what matters is that
    # it took the region branch.
    assert result.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)


def test_the_species_name_is_never_geocoded(offline_gbif) -> None:
    """"hotspots for tigers" must not ask a gazetteer about "tigers".

    Two bugs lived here. Excluding only GBIF's canonical name left "tigers" in
    the sentence, and it resolves to a street in Frederiksberg; excluding the
    singular before the plural left an orphan "s", which resolves to something
    else again. Either way the answer came back about the wrong place.
    """

    asked: list[str] = []

    def _spy(name, **_kwargs):
        asked.append(name)
        return None

    from backend.agents.biodiversity_agent.workers.hotspots import worker as module
    original = module.resolve_place
    module.resolve_place = _spy
    try:
        result = HotspotsWorker().run(AgentRequest(
            instruction="hotspots for tigers", context={}))
    finally:
        module.resolve_place = original

    assert result.status is AgentStatus.COMPLETED
    assert result.output.region == "worldwide"
    for candidate in asked:
        assert "tiger" not in candidate.lower(), candidate
        assert len(candidate) >= 3, f"orphan token geocoded: {candidate!r}"


def test_a_species_and_a_place_together_use_the_place(offline_gbif) -> None:
    result = HotspotsWorker().run(AgentRequest(
        instruction="hotspots for tigers in India", context={}))

    assert result.status is AgentStatus.COMPLETED
    assert result.output.parameters_used["region"] == "India"
    assert result.output.parameters_used["mode"] == "species"
