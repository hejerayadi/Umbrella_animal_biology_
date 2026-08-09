"""Resolve a species name to the NCBI taxonomy id the workflow is keyed by."""

from typing import Any

from backend.agents.Protein_visualization.app.domain.exceptions import ProteinNotFoundError
from backend.agents.Protein_visualization.app.domain.models import SpeciesRef
from backend.agents.Protein_visualization.app.tools.uniprot_client import UniProtClient

# Only species rank is usable. `IdentityCapability` compares a protein's
# organism taxon id to this one exactly, so a genus ("mouse" matches the genus
# Mus before it matches Mus musculus) or a subspecies would never match and
# would surface later as a spurious identity mismatch.
_SPECIES_RANK = "species"


class TaxonomyCapability:
    """Species name -> `SpeciesRef`, via UniProt's taxonomy index.

    `IdentityCapability` rejects a UniProt entry whose organism does not match
    the requested `taxon_id`, so guessing here would surface later as a spurious
    "identity does not match the requested species". An unresolvable or
    ambiguous name therefore raises rather than picking a plausible candidate.
    """

    def __init__(self, client: UniProtClient) -> None:
        self.client = client
        self._cache: dict[str, SpeciesRef] = {}

    async def resolve(self, name: str) -> SpeciesRef:
        key = name.strip().casefold()
        if not key:
            raise ProteinNotFoundError("No species name was supplied")
        if key in self._cache:
            return self._cache[key]

        records = await self.client.search_taxonomy(name)
        match = self._select(records, key)
        if match is None:
            raise ProteinNotFoundError(f"UniProt taxonomy has no species matching {name!r}")

        species = SpeciesRef(
            scientific_name=str(match["scientificName"]),
            taxon_id=int(match["taxonId"]),
        )
        self._cache[key] = species
        return species

    @classmethod
    def _select(cls, records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        """Pick the one species the name actually denotes, or nothing.

        UniProt orders its own results by relevance, and relevance alone is
        wrong here: searching "mouse" puts the genus Mus first and Mus musculus
        ninth. So the record has to earn the match by carrying the name itself,
        at species rank.

        The scientific name is tried before the vernacular ones, because a
        binomial is unambiguous and a common name is shared - "mouse" is listed
        under Mus musculus and under the mouse metagenome alike.
        """
        candidates = [
            record
            for record in records
            if record.get("active", True)
            and record.get("taxonId")
            and str(record.get("rank", "")).lower() == _SPECIES_RANK
        ]

        for names in (cls._scientific_name, cls._all_names):
            matches = [record for record in candidates if key in names(record)]
            if len(matches) == 1:
                return matches[0]
            if matches:
                # Two species answering to the same name: no basis to choose.
                return None

        # No name matched outright. A lone species-ranked result is still
        # unambiguous enough to act on; a list of them is not.
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _scientific_name(record: dict[str, Any]) -> set[str]:
        return {str(record["scientificName"]).casefold()} if record.get("scientificName") else set()

    @staticmethod
    def _all_names(record: dict[str, Any]) -> set[str]:
        return {
            str(value).casefold()
            for value in (
                record.get("scientificName"),
                record.get("commonName"),
                *record.get("otherNames", []),
            )
            if value
        }
