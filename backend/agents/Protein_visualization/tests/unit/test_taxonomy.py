from typing import Any

import pytest

from backend.agents.Protein_visualization.app.capabilities.taxonomy import TaxonomyCapability
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinNotFoundError


class StubUniProt:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.calls = 0

    async def search_taxonomy(self, name: str) -> list[dict[str, Any]]:
        self.calls += 1
        return self.records


def _record(**overrides: Any) -> dict[str, Any]:
    return {
        "scientificName": "Homo sapiens",
        "commonName": "Human",
        "otherNames": [],
        "taxonId": 9606,
        "rank": "species",
        "active": True,
        **overrides,
    }


def _capability(records: list[dict[str, Any]]) -> tuple[TaxonomyCapability, StubUniProt]:
    client = StubUniProt(records)
    return TaxonomyCapability(client), client  # type: ignore[arg-type]


async def test_scientific_name_resolves_to_the_taxon_id() -> None:
    capability, _ = _capability([_record()])

    species = await capability.resolve("Homo sapiens")

    assert species.taxon_id == 9606
    assert species.scientific_name == "Homo sapiens"


async def test_a_common_name_resolves_through_other_names() -> None:
    capability, _ = _capability(
        [
            _record(scientificName="Mammuthus", taxonId=37348, rank="genus", commonName="mammoths"),
            _record(
                scientificName="Mammuthus primigenius",
                taxonId=37349,
                commonName="Siberian woolly mammoth",
                otherNames=["mammoth", "woolly mammoth"],
            ),
        ]
    )

    species = await capability.resolve("woolly mammoth")

    assert species.taxon_id == 37349


async def test_a_common_name_does_not_resolve_to_its_genus() -> None:
    """The real answer for "mouse": UniProt puts the genus Mus first and lists
    Mus musculus ninth. A genus id never matches a protein's organism, so
    picking it turns into "identity does not match the requested species" three
    nodes later."""
    capability, _ = _capability(
        [
            _record(
                scientificName="Mus", taxonId=10088, rank="genus", commonName="mice", otherNames=["mouse"]
            ),
            _record(
                scientificName="Microcebus murinus",
                taxonId=30608,
                commonName="Gray mouse lemur",
                otherNames=["gray mouse lemur"],
            ),
            _record(scientificName="Mouse cyclovirus", taxonId=1908802, commonName=None),
            _record(
                scientificName="Mus musculus",
                taxonId=10090,
                commonName="Mouse",
                otherNames=["house mouse", "mouse", "nude mice"],
            ),
            _record(
                scientificName="Mus musculus domesticus",
                taxonId=10092,
                rank="subspecies",
                commonName="western European house mouse",
            ),
        ]
    )

    species = await capability.resolve("mouse")

    assert species.taxon_id == 10090
    assert species.scientific_name == "Mus musculus"


async def test_a_binomial_beats_another_species_vernacular_name() -> None:
    capability, _ = _capability(
        [
            _record(
                scientificName="Peromyscus polionotus",
                taxonId=42413,
                commonName="Oldfield mouse",
                otherNames=["Mus musculus"],
            ),
            _record(scientificName="Mus musculus", taxonId=10090, commonName="Mouse"),
        ]
    )

    species = await capability.resolve("Mus musculus")

    assert species.taxon_id == 10090


async def test_the_exact_match_wins_over_the_first_result() -> None:
    """UniProt orders by relevance, which happily puts the genus above the species."""
    capability, _ = _capability(
        [
            _record(scientificName="Mammuthus", taxonId=37348, rank="genus", commonName=None),
            _record(scientificName="Mammuthus primigenius", taxonId=37349, commonName=None),
        ]
    )

    species = await capability.resolve("Mammuthus primigenius")

    assert species.taxon_id == 37349


async def test_several_inexact_species_are_left_unresolved() -> None:
    capability, _ = _capability(
        [
            _record(scientificName="Canis lupus", taxonId=9612, commonName=None),
            _record(scientificName="Canis latrans", taxonId=9614, commonName=None),
        ]
    )

    with pytest.raises(ProteinNotFoundError):
        await capability.resolve("canis")


async def test_an_unknown_name_raises_rather_than_guessing() -> None:
    capability, _ = _capability([])

    with pytest.raises(ProteinNotFoundError):
        await capability.resolve("not a species")


async def test_inactive_records_are_ignored() -> None:
    capability, _ = _capability([_record(active=False)])

    with pytest.raises(ProteinNotFoundError):
        await capability.resolve("Homo sapiens")


async def test_the_same_name_is_only_looked_up_once() -> None:
    capability, client = _capability([_record()])

    await capability.resolve("Homo sapiens")
    await capability.resolve("  homo sapiens  ")

    assert client.calls == 1
