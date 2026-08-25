"""Discovering and choosing a BLAST database, with no network.

The agent no longer names a database anywhere. It reads EBI's catalogue,
proposes candidates against a taxonomy prior, searches them in parallel and
keeps whichever produced references that actually cross the gap. These tests
cover the proposal and the choice; `test_target_profile` covers the prior.
"""
from __future__ import annotations

import asyncio

import pytest

from domain.exceptions import ExternalServiceError
from domain.services.target_profile import Division, Molecule
from tools.blast.advisor import DatabaseAdvisor, best_database, record_trial
from tools.blast.catalogue import EbiDatabaseCatalogue

URSUS = ("Eukaryota", "Metazoa", "Chordata", "Mammalia", "Ursus", "Ursus maritimus")


class FakeTaxonomy:
    def __init__(self, lineage=URSUS, error: Exception | None = None, hang: bool = False):
        self._lineage, self._error, self._hang = lineage, error, hang
        self.calls = 0

    async def lineage(self, organism: str):
        self.calls += 1
        if self._hang:
            await asyncio.sleep(60)
        if self._error:
            raise self._error
        return self._lineage


def advisor(**kwargs) -> DatabaseAdvisor:
    """Wired to the offline snapshot: real validation, no EBI call."""
    return DatabaseAdvisor(catalogue=EbiDatabaseCatalogue(None), **kwargs)


class TestCatalogue:
    async def test_an_invented_code_is_rejected(self) -> None:
        """`em_rel_vrt` was hardcoded and does not exist; EBI 400s on it, so
        every relaxed retry failed on submission for the agent's whole life."""
        accepted, rejected = await EbiDatabaseCatalogue(None).validate(
            ["em_std_mam", "em_rel_vrt"]
        )
        assert accepted == ["em_std_mam"]
        assert rejected == ["em_rel_vrt"]

    async def test_a_code_is_never_offered_twice(self) -> None:
        """Two identical searches would spend two of eight BLAST calls agreeing."""
        accepted, _ = await EbiDatabaseCatalogue(None).validate(
            ["em_std_mam", "em_std_mam"]
        )
        assert accepted == ["em_std_mam"]


class TestAdvice:
    async def test_a_mammal_scaffold_is_advised_the_mammal_division(self) -> None:
        advice = await advisor(taxonomy=FakeTaxonomy()).advise(
            organism="Ursus maritimus",
            description="Ursus maritimus isolate Baiyulong unplaced genomic scaffold",
            length=15_920_966,
        )
        assert advice.profile.division is Division.MAM
        assert advice.profile.molecule is Molecule.NUCLEAR
        assert advice.candidates[0] == "em_mam"

    async def test_the_prior_augments_the_planner_rather_than_losing_to_it(self) -> None:
        """If the model picks the division the organism is not filed under -
        the exact original defect - both are probed and the measurement decides.
        Dropping the prior would repeat the failure; dropping the model's choice
        would make asking it pointless."""
        advice = await advisor(taxonomy=FakeTaxonomy()).advise(
            organism="Ursus maritimus", proposed=["em_std_vrt"]
        )
        assert "em_std_vrt" in advice.candidates
        assert "em_std_mam" in advice.candidates

    async def test_a_database_measured_too_slow_is_dropped(self) -> None:
        """`em_std` had not finished at 813 s, against a 600 s slice."""
        advice = await advisor(taxonomy=FakeTaxonomy()).advise(
            organism="Ursus maritimus", proposed=["em_std"]
        )
        assert "em_std" not in advice.candidates

    async def test_an_unknown_organism_proposes_nothing(self) -> None:
        """The caller reports that it could not tell where to look. It does not
        guess: a confident wrong division returns fish and a plausible score."""
        advice = await advisor(taxonomy=FakeTaxonomy(lineage=())).advise(organism="???")
        assert advice.candidates == []

    @pytest.mark.parametrize(
        "taxonomy",
        [None, FakeTaxonomy(error=ExternalServiceError("ncbi", "down")), FakeTaxonomy(hang=True)],
    )
    async def test_taxonomy_trouble_never_stops_the_run(self, taxonomy) -> None:
        """No client, an outage, and a hang. Discovery is an enrichment: losing
        it costs the prior, not the run - and the hang must not eat the slice."""
        advice = await advisor(taxonomy=taxonomy).advise(organism="Ursus maritimus")
        assert advice.candidates == []
        assert advice.profile.division is None

    async def test_the_lineage_is_looked_up_once_per_run(self) -> None:
        """A scaffold has hundreds of gaps; one lookup each would be the run."""
        taxonomy = FakeTaxonomy()
        subject = advisor(taxonomy=taxonomy)
        await subject.advise(organism="Ursus maritimus")
        await subject.advise(organism="Ursus maritimus")
        assert taxonomy.calls == 1


class TestChoosingTheWinner:
    def test_carriers_decide_not_hit_count(self) -> None:
        """The whole point. `em_std_vrt` returned 50 hits on the polar bear and
        7 that crossed the gap; `em_std_mam` returned 50 and 50. A choice made
        on hit count cannot tell those apart."""
        trials = {}
        trials = record_trial(trials, "em_std_vrt", hits=50, carrying_gap=7, seconds=165)
        trials = record_trial(trials, "em_std_mam", hits=50, carrying_gap=50, seconds=213)
        assert best_database(trials) == "em_std_mam"

    def test_no_winner_when_nothing_crossed_the_gap(self) -> None:
        """Naming one would send every later gap to a database already shown
        not to work here."""
        trials = record_trial({}, "em_std_vrt", hits=50, carrying_gap=0, seconds=165)
        assert best_database(trials) is None

    def test_the_faster_database_wins_a_tie(self) -> None:
        trials = record_trial({}, "em_std_mam", hits=50, carrying_gap=50, seconds=213)
        trials = record_trial(trials, "em_mam", hits=50, carrying_gap=50, seconds=309)
        assert best_database(trials) == "em_std_mam"

    def test_trials_accumulate_across_gaps(self) -> None:
        trials = record_trial({}, "em_std_mam", hits=50, carrying_gap=50, seconds=213)
        trials = record_trial(trials, "em_std_mam", hits=40, carrying_gap=30, seconds=200)
        assert trials["em_std_mam"] == {
            "hits": 90, "carrying_gap": 80, "seconds": 413.0, "searches": 2,
        }
