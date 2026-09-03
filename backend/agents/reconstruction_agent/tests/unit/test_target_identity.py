"""A named species must reach the search, not just the response.

Measured on the pasted-sequence path: "Reconstruct the missing bases in this
polar bear sequence: ..." returned the gap unresolved at confidence 0.07, the
same score the identical prompt produced with no species named at all. The name
was reaching the response as a label and nothing else - no record had been
fetched, so no tax id existed, so the lineage was empty, so `_scopes` had
nothing to derive a scope from and the homology round returned before
dispatching a single BLAST call. These tests pin the resolution that prevents
that.
"""

from __future__ import annotations

from reconstruction_agent.domain.models.taxonomy import TaxonNode
from reconstruction_agent.services.taxonomy.taxonomy_service import TaxonomyService

POLAR_BEAR = TaxonNode(tax_id=29073, name="Ursus maritimus", rank="species")
URSIDAE = TaxonNode(tax_id=9632, name="Ursidae", rank="family")


class _FakeClient:
    """Records what it was asked, so a missing lookup is a visible failure."""

    def __init__(self, *, resolves: dict[str, TaxonNode] | None = None) -> None:
        self._resolves = resolves or {}
        self.names_looked_up: list[str] = []
        self.lineages_fetched: list[int] = []

    async def resolve_name(self, name: str) -> TaxonNode | None:
        self.names_looked_up.append(name)
        return self._resolves.get(name)

    async def lineage(self, tax_id: int) -> tuple[TaxonNode, ...]:
        self.lineages_fetched.append(tax_id)
        return (URSIDAE, POLAR_BEAR)


class TestANameWithoutATaxIdIsResolved:
    async def test_the_name_becomes_a_tax_id_and_a_lineage(self) -> None:
        """The pasted-sequence case: a species was named, no record was
        fetched, and the scopes still have to come from somewhere."""
        client = _FakeClient(resolves={"polar bear": POLAR_BEAR})
        profile = await TaxonomyService(client).profile_for("polar bear", None)

        assert profile.tax_id == 29073
        assert client.lineages_fetched == [29073]
        # The family rank is what `_scopes` reads first; without it the search
        # is scoped to the species alone and misses every congener.
        assert profile.taxonomy_ranks.get("family") == "Ursidae"

    async def test_an_unresolvable_name_is_an_ordinary_answer(self) -> None:
        """Not every string names a taxon. The run continues unscoped rather
        than failing - the same outcome as before, reached deliberately."""
        client = _FakeClient()
        profile = await TaxonomyService(client).profile_for("not an organism", None)

        assert profile.tax_id is None
        assert profile.taxonomy_lineage == ()

    async def test_a_supplied_tax_id_is_never_second_guessed(self) -> None:
        """The Genome hand-off carries a tax id already. Re-resolving its name
        would let a vernacular collision overrule a resolved identity."""
        client = _FakeClient(resolves={"Ursus maritimus (polar bear)": URSIDAE})
        profile = await TaxonomyService(client).profile_for(
            "Ursus maritimus (polar bear)", 29073
        )

        assert profile.tax_id == 29073
        assert client.names_looked_up == []

    async def test_an_unnamed_target_costs_no_lookup(self) -> None:
        """A search for the placeholder can only ever miss, and NCBI is rate
        limited to one request every ten seconds."""
        client = _FakeClient()
        profile = await TaxonomyService(client).profile_for("", None)

        assert profile.scientific_name == "unknown organism"
        assert client.names_looked_up == []
