"""Resolve a species name to the NCBI taxonomy id the workflow is keyed by.

This backs `GET /api/v1/taxonomy`, for a caller that has a name and needs the
id `AgentTask` requires. It is not the path `/execute` takes: the Global
Orchestrator's requests go through `orchestrator_adapter.py`, which resolves
the gene and the species *together* against UniProtKB, and gets a taxon that
is true for the matched protein by construction.

The difference matters, because the two data sources do not carry the same
names. UniProtKB's `organism_name` filter knows "dog" and "cat"; the taxonomy
index this module searches does not surface them at all - a query for "dog"
returns Toxocara canis and a wolfdog hybrid, and never Canis lupus familiaris.
So a bare common name that resolves fine through the adapter can still be
unresolvable here, and this module raises rather than returning the roundworm.
"""

from typing import Any

from backend.agents.Protein_visualization.app.domain.exceptions import ProteinNotFoundError
from backend.agents.Protein_visualization.app.domain.models import SpeciesRef
from backend.agents.Protein_visualization.app.tools.uniprot_client import UniProtClient

# A genus is excluded on purpose: `IdentityCapability` compares a protein's
# organism taxon id to this one exactly, so the genus Mus - which "mouse"
# matches before Mus musculus does - would surface later as a spurious
# "identity does not match the requested species". A subspecies is a real
# organism a protein can be annotated with, so it is kept.
_ORGANISM_RANKS = frozenset({"species", "subspecies"})


def singular_candidates(name: str) -> list[str]:
    """The name as written, then its singular, to try in turn.

    Users and the orchestrator's extractor both write plurals - "show me MC1R
    in dogs". The taxonomy index stores the names a taxon is *known by*, and
    whether the plural is among them is luck: "humans" reaches only the genus
    Homo, while "human" reaches Homo sapiens.

    Singularising the last word only keeps "polar bears" -> "polar bear" while
    leaving the qualifier alone. The result is still used as an exact-match
    lookup, so a bad guess finds nothing rather than the wrong organism.
    """
    candidates = [name]
    head, _, last = name.rpartition(" ")

    # "foxes" -> "fox" before "dogs" -> "dog": the -es rule is the more specific.
    singular = None
    if last.endswith("es") and len(last) > 4:
        singular = last[:-2]
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        singular = last[:-1]

    if singular:
        candidates.append(f"{head} {singular}".strip())
    return candidates


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

        for candidate in singular_candidates(name.strip()):
            records = await self.client.search_taxonomy(candidate)
            match = self._select(records, candidate.casefold())
            if match is not None:
                species = SpeciesRef(
                    scientific_name=str(match["scientificName"]),
                    taxon_id=int(match["taxonId"]),
                )
                self._cache[key] = species
                return species

        raise ProteinNotFoundError(
            f"UniProt taxonomy has no species matching {name!r}. Try the scientific name."
        )

    @classmethod
    def _select(cls, records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        """Pick the one organism the name actually denotes, or nothing.

        UniProt orders its own results by relevance, and relevance alone is
        wrong here: searching "polar bear" puts Polar bear adenovirus 1 first
        and Ursus maritimus second. So the record has to earn the match by
        carrying the name itself, at an organism rank.

        The scientific name is tried before the vernacular ones, because a
        binomial is unambiguous and a common name is shared - "mouse" is listed
        under Mus musculus and under the mouse metagenome alike.
        """
        candidates = [
            record
            for record in records
            if record.get("active", True)
            and record.get("taxonId")
            and str(record.get("rank", "")).lower() in _ORGANISM_RANKS
        ]

        for names in (cls._scientific_name, cls._all_names):
            matches = [record for record in candidates if key in names(record)]
            if len(matches) == 1:
                return matches[0]
            if matches:
                # Two organisms answering to the same name: no basis to choose.
                return None

        # No name matched outright. A lone organism-ranked result is still
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
