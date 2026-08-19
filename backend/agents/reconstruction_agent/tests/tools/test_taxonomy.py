"""Real phylogenetic distance replacing the three-value name heuristic.

The heuristic returned 1.0 for the same species, 0.8 for the same genus, and a
flat 0.3 for everything else. Since almost every useful reference sits outside
the target's genus, that 0.3 was a constant carrying no information - yet the
confidence score weighted it as though it ranked something.

Nothing here reaches the network: the NCBI payloads are supplied as fixtures.
"""
from __future__ import annotations

import pytest

from domain.exceptions import ExternalServiceError
from infrastructure.ncbi.taxonomy import TaxonomyService, _parse_lineage, _shared_depth
from tools.evo.schemas import EvolutionaryContextInput
from tools.evo.tool import EvolutionaryContextTool

pytestmark = pytest.mark.asyncio

ELEPHANT = (
    "cellular organisms; Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; "
    "Mammalia; Eutheria; Afrotheria; Proboscidea; Elephantidae; Loxodonta"
)
MAMMOTH = (
    "cellular organisms; Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; "
    "Mammalia; Eutheria; Afrotheria; Proboscidea; Elephantidae; Mammuthus"
)
MOUSE = (
    "cellular organisms; Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; "
    "Mammalia; Eutheria; Euarchontoglires; Glires; Rodentia; Muridae; Mus"
)


def taxonomy_xml(lineage: str, name: str) -> str:
    return (
        f"<TaxaSet><Taxon><ScientificName>{name}</ScientificName>"
        f"<Lineage>{lineage}</Lineage></Taxon></TaxaSet>"
    )


class FakeNCBI:
    """Answers taxonomy lookups from a name -> (lineage, taxid) table."""

    def __init__(self, records: dict[str, str], fail: bool = False) -> None:
        self._records = records
        self._fail = fail
        self.searches = 0
        self.fetches = 0

    async def search(self, term: str, *, database: str = "nuccore", limit: int = 20) -> list[str]:
        self.searches += 1
        if self._fail:
            raise ExternalServiceError("ncbi", "simulated outage")
        return ["1234"] if term in self._records else []

    async def fetch_taxonomy(self, identifier: str) -> str:
        self.fetches += 1
        name = next(iter(self._records))
        return taxonomy_xml(self._records[name], name)


class TestLineageParsing:
    def test_reads_the_lineage_and_appends_the_organism(self) -> None:
        """Without the leaf, two species in one genus look identical."""
        parsed = _parse_lineage(taxonomy_xml(ELEPHANT, "Loxodonta africana"))

        assert parsed[-1] == "Loxodonta africana"
        assert parsed[0] == "cellular organisms"

    def test_a_payload_with_no_lineage_yields_nothing(self) -> None:
        assert _parse_lineage("<TaxaSet></TaxaSet>") == ()

    def test_the_leaf_is_not_duplicated(self) -> None:
        parsed = _parse_lineage(
            taxonomy_xml(ELEPHANT + "; Loxodonta africana", "Loxodonta africana")
        )

        assert parsed.count("Loxodonta africana") == 1


class TestSharedDepth:
    def test_identical_lineages_score_one(self) -> None:
        lineage = tuple(ELEPHANT.split("; "))

        assert _shared_depth(lineage, lineage) == 1.0

    def test_the_case_the_heuristic_could_not_see(self) -> None:
        """Loxodonta and Mammuthus are close; the name heuristic said 0.3."""
        elephant = tuple(ELEPHANT.split("; "))
        mammoth = tuple(MAMMOTH.split("; "))

        assert _shared_depth(elephant, mammoth) > 0.8

    def test_a_distant_relative_scores_lower_than_a_close_one(self) -> None:
        elephant = tuple(ELEPHANT.split("; "))

        close = _shared_depth(elephant, tuple(MAMMOTH.split("; ")))
        distant = _shared_depth(elephant, tuple(MOUSE.split("; ")))

        assert distant < close
        # Both are mammals, so it is emphatically not zero either - which is
        # the discrimination the flat 0.3 could never provide.
        assert 0.0 < distant < 0.7

    def test_an_empty_target_lineage_scores_nothing(self) -> None:
        assert _shared_depth((), tuple(MOUSE.split("; "))) == 0.0


class TestTaxonomyService:
    async def test_it_places_an_organism(self) -> None:
        service = TaxonomyService(FakeNCBI({"Loxodonta africana": ELEPHANT}))  # type: ignore[arg-type]

        assert await service.lineage("Loxodonta africana")

    async def test_lookups_are_cached(self) -> None:
        """A run asks about the same organisms once per reference per round."""
        client = FakeNCBI({"Loxodonta africana": ELEPHANT})
        service = TaxonomyService(client)  # type: ignore[arg-type]

        await service.lineage("Loxodonta africana")
        await service.lineage("Loxodonta africana")

        assert client.fetches == 1

    async def test_an_unknown_name_is_remembered_as_a_miss(self) -> None:
        client = FakeNCBI({"Loxodonta africana": ELEPHANT})
        service = TaxonomyService(client)  # type: ignore[arg-type]

        await service.lineage("Notareal organism")
        await service.lineage("Notareal organism")

        assert client.searches == 1

    async def test_an_outage_is_not_cached(self) -> None:
        """The name may be fine and the service merely down; retry next slice."""
        client = FakeNCBI({"Loxodonta africana": ELEPHANT}, fail=True)
        service = TaxonomyService(client)  # type: ignore[arg-type]

        assert await service.lineage("Loxodonta africana") == ()
        await service.lineage("Loxodonta africana")

        assert client.searches == 2


class TestToolFallsBackSafely:
    async def test_without_taxonomy_it_keeps_the_heuristic(self) -> None:
        output = await EvolutionaryContextTool().run(
            EvolutionaryContextInput(
                target_organism="Loxodonta africana",
                candidate_organisms=["Mammuthus primigenius"],
            )
        )

        assert output.succeeded
        assert output.source == "heuristic"

    async def test_an_outage_does_not_fail_the_tool(self) -> None:
        """Taxonomy enriches a ranking; it is never a precondition for one."""
        tool = EvolutionaryContextTool(
            TaxonomyService(FakeNCBI({"Loxodonta africana": ELEPHANT}, fail=True))  # type: ignore[arg-type]
        )

        output = await tool.run(
            EvolutionaryContextInput(
                target_organism="Loxodonta africana",
                candidate_organisms=["Mammuthus primigenius"],
            )
        )

        assert output.succeeded
        assert output.relatedness  # the heuristic still answered
