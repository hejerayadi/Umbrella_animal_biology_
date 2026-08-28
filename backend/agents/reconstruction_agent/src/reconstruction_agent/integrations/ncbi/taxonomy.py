"""NCBI Taxonomy: lineages, ranks, and resolving a name to a taxon.

This is the single authority in the agent on how organisms relate to one
another. Two very different questions come here, and it matters that both do:

- What clade is the target in? (drives evolutionary distance scoring)
- What clade does a reference collection labelled "Mammal" cover?

The second is what keeps provider vocabulary out of this codebase. NCBI
Taxonomy already records the vernacular names that catalogue labels use, so the
correspondence between "Mammal" and Mammalia is looked up, not written down by
us. There is no hand-maintained table of group names here, and adding one would
reintroduce exactly the coupling this design removes.

Results are cached for the process lifetime: taxonomy does not change during a
run, and a lineage lookup repeated per hit would dominate the rate limit.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from xml.etree import ElementTree

from cachetools import TTLCache

from reconstruction_agent.domain.models.taxonomy import TaxonNode
from reconstruction_agent.integrations.ncbi.client import NcbiClient

#: Suffixes NCBI appends to disambiguate a name it also uses elsewhere, e.g.
#: "Ursus <bear>". Stripped before comparison, never used to infer anything.
_DISAMBIGUATOR = re.compile(r"\s*<[^>]*>\s*")


class TaxonomyClient:
    """Cached read access to NCBI Taxonomy."""

    def __init__(self, client: NcbiClient, *, cache_ttl: float = 3600.0) -> None:
        self._client = client
        self._lineages: TTLCache[int, tuple[TaxonNode, ...]] = TTLCache(maxsize=2048, ttl=cache_ttl)
        self._names: TTLCache[str, tuple[TaxonNode, ...]] = TTLCache(maxsize=2048, ttl=cache_ttl)

    async def lineage(self, tax_id: int) -> tuple[TaxonNode, ...]:
        """The full lineage for `tax_id`, ordered root -> organism.

        The taxon itself is the last element, so position in the tuple is
        specificity - which is what lets one clade be compared against another
        without anything knowing which is broader.
        """
        cached = self._lineages.get(tax_id)
        if cached is not None:
            return cached

        root = await self._client.efetch_xml("taxonomy", str(tax_id))
        taxon = root.find("Taxon")
        if taxon is None:
            self._lineages[tax_id] = ()
            return ()

        nodes = [
            node
            for element in taxon.findall("./LineageEx/Taxon")
            if (node := _node_from_element(element)) is not None
        ]
        if (own := _node_from_element(taxon)) is not None:
            nodes.append(own)

        lineage = tuple(nodes)
        self._lineages[tax_id] = lineage
        return lineage

    async def resolve_name(self, name: str) -> TaxonNode | None:
        """The taxon `name` most likely refers to, or None if unrecognised."""
        candidates = await self.resolve_candidates(name)
        return candidates[0] if candidates else None

    async def resolve_candidates(self, name: str, *, limit: int = 3) -> tuple[TaxonNode, ...]:
        """Every taxon `name` could refer to, best match first.

        `name` may be a scientific name ("Mammalia"), a vernacular one
        ("Mammals"), or free text lifted from a provider catalogue label. An
        empty result is an ordinary answer, not a failure: a collection whose
        label names no taxon simply gets no taxonomic prior and is ranked on
        measured evidence instead.

        More than one candidate is returned because scientific names collide
        with vernacular ones - searching "Bacteria" returns the insect genus
        *Bacteria* Latreille ahead of the domain. Resolving to a single best
        guess would hand the caller the wrong clade silently, so the choice is
        left to whoever knows what the name is being matched against.
        """
        key = _normalise(name)
        if not key:
            return ()
        cached = self._names.get(key)
        if cached is not None:
            return cached

        resolved = await self._lookup_name(key, limit)
        self._names[key] = resolved
        return resolved

    async def _lookup_name(self, key: str, limit: int) -> tuple[TaxonNode, ...]:
        """Search taxonomy for `key`, trying simple morphological variants.

        Needed because the two vocabularies differ in number: NCBI indexes the
        vernacular plural ("Mammals", "Vertebrates", "Plants") while EMBL-EBI
        labels its collections in the singular ("Mammal", "Vertebrate",
        "Plant"), so a literal lookup finds nothing for exactly the groups that
        matter most.

        This is English morphology, in the same category as trimming
        whitespace. It stays general - no term is mapped to a particular clade
        here, and every variant that does resolve is resolved by NCBI.
        """
        for variant in _variants(key):
            ids = await self._client.esearch("taxonomy", variant, retmax=limit)
            if not ids:
                continue
            nodes = [lineage[-1] for raw in ids if (lineage := await self.lineage(int(raw)))]
            if nodes:
                return tuple(nodes)
        return ()

    async def node_for(self, tax_id: int) -> TaxonNode | None:
        """The taxon `tax_id` itself, without its ancestors."""
        lineage = await self.lineage(tax_id)
        return lineage[-1] if lineage else None


def _variants(term: str) -> Iterator[str]:
    """`term` and the number-variants worth trying, in order, without repeats."""
    seen: set[str] = set()
    candidates = [term]

    lowered = term.lower()
    if lowered.endswith("y"):
        candidates.append(f"{term[:-1]}ies")
    elif lowered.endswith(("s", "x", "z", "ch", "sh")):
        candidates.append(f"{term}es")
    else:
        candidates.append(f"{term}s")

    if lowered.endswith("ies"):
        candidates.append(f"{term[:-3]}y")
    elif lowered.endswith("es"):
        candidates.append(term[:-2])
    if lowered.endswith("s"):
        candidates.append(term[:-1])

    for candidate in candidates:
        normalised = candidate.strip()
        if normalised and normalised not in seen:
            seen.add(normalised)
            yield normalised


def _node_from_element(element: ElementTree.Element) -> TaxonNode | None:
    """One `<Taxon>` element as a `TaxonNode`, or None if it is unusable."""
    raw_id = element.findtext("TaxId")
    name = element.findtext("ScientificName")
    if not raw_id or not name:
        return None
    try:
        tax_id = int(raw_id)
    except ValueError:
        return None
    return TaxonNode(
        tax_id=tax_id,
        name=_DISAMBIGUATOR.sub(" ", name).strip(),
        rank=(element.findtext("Rank") or "no rank").strip(),
    )


def _normalise(name: str) -> str:
    """Collapse a raw label into something worth sending to a name search."""
    return _DISAMBIGUATOR.sub(" ", name).strip().strip(".,;:").strip()
