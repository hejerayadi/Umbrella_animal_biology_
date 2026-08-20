"""Real phylogenetic distance, from NCBI's taxonomy rather than from a name.

The name-based heuristic this replaces returned one of three values - same
species, same genus, or a flat 0.3 for everything else. Since almost every
useful reference is outside the target's genus, that flat 0.3 was what the
confidence score actually saw nearly every time: a constant, carrying no
information, weighted as though it did. It could not tell that *Loxodonta* and
*Mammuthus* are close relatives, which is exactly the judgement a reconstruction
depends on.

NCBI publishes the lineage. Two organisms that share `Elephantidae` are closer
than two that share only `Mammalia`, and how far down the shared path runs is a
real measure of proximity. That is what this computes.

Failure is expected and cheap: taxonomy is an enrichment, never a precondition.
Anything unreachable falls back to the caller's heuristic rather than failing a
run, and a name NCBI does not know simply has no lineage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from configuration.logging import get_logger
from domain.exceptions import ReconstructionError
from infrastructure.ncbi.client import NCBIClient

_log = get_logger(__name__)

#: Ranks worth counting, outermost first. Restricted to the major ones because
#: NCBI's intermediate ranks are applied unevenly between clades - counting
#: every `subfamily` and `tribe` would make two equally-related pairs score
#: differently based on how well-studied their branch happens to be.
_RANKS = (
    "superkingdom",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
)

_LINEAGE = re.compile(r"<Lineage>(.*?)</Lineage>", re.DOTALL)
_SCIENTIFIC_NAME = re.compile(r"<ScientificName>(.*?)</ScientificName>", re.DOTALL)


@dataclass(slots=True)
class TaxonomyService:
    """Lineage lookups for organism names, cached for the life of the process.

    Cached because a run asks about the same handful of organisms repeatedly -
    once per reference per round - and NCBI rate-limits anonymous callers to
    three requests a second. The cache is per process rather than in the graph
    state: it is derived data that never changes, so checkpointing it would only
    make the checkpoint bigger.
    """

    client: NCBIClient
    _lineages: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _misses: set[str] = field(default_factory=set)

    async def relatedness(self, target: str, candidates: list[str]) -> dict[str, float]:
        """Proximity of each candidate organism to the target, on 0..1.

        Returns only the organisms taxonomy could actually place. The caller
        keeps its heuristic for the rest, so a partial answer improves what it
        can and leaves the remainder no worse.
        """
        target_lineage = await self.lineage(target)
        if not target_lineage:
            return {}

        scores: dict[str, float] = {}
        for candidate in candidates:
            lineage = await self.lineage(candidate)
            if lineage:
                scores[candidate] = _shared_depth(target_lineage, lineage)

        return scores

    async def lineage(self, organism: str) -> tuple[str, ...]:
        """The taxonomic lineage of one organism, outermost first.

        An empty tuple when the name is unknown or NCBI is unreachable - both
        of which are ordinary, and neither of which is worth an exception.
        """
        name = organism.strip()
        if not name or name in self._misses:
            return ()
        if name in self._lineages:
            return self._lineages[name]

        try:
            identifiers = await self.client.search(name, database="taxonomy", limit=1)
            if not identifiers:
                self._misses.add(name)
                return ()

            raw = await self.client.fetch_taxonomy(identifiers[0])
        except ReconstructionError as error:
            # Not cached as a miss: the name may be perfectly good and the
            # service merely down, and a later slice should try again.
            _log.info("taxonomy_lookup_failed", organism=name, error=str(error))
            return ()

        lineage = _parse_lineage(raw)
        if not lineage:
            self._misses.add(name)
            return ()

        self._lineages[name] = lineage
        return lineage


def _parse_lineage(raw: str) -> tuple[str, ...]:
    """The lineage out of an efetch taxonomy XML payload, plus the organism.

    Parsed with a regex rather than an XML parser: one field is wanted out of a
    large document, the shape is stable, and pulling in a parser to read it
    would be the more fragile choice.
    """
    match = _LINEAGE.search(raw)
    if not match:
        return ()

    names = [part.strip() for part in match.group(1).split(";") if part.strip()]

    # The record's own scientific name is the last step of its lineage and is
    # not repeated inside `<Lineage>`; without it two species in one genus would
    # be indistinguishable from the same species.
    own = _SCIENTIFIC_NAME.search(raw)
    if own:
        leaf = own.group(1).strip()
        if leaf and (not names or names[-1] != leaf):
            names.append(leaf)

    return tuple(names)


def _shared_depth(target: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    """How much of two lineages coincide, on 0..1.

    The fraction of the *target's* lineage the candidate also sits on. Measured
    against the target rather than against the longer of the two, because the
    question is how well the candidate stands in for the target - a
    finely-classified candidate should not be penalised for the extra ranks its
    own branch happens to carry.

    Same organism gives 1.0; sharing only the root gives close to 0.
    """
    if not target:
        return 0.0

    shared = 0
    for left, right in zip(target, candidate, strict=False):
        if left != right:
            break
        shared += 1

    return round(shared / len(target), 4)
