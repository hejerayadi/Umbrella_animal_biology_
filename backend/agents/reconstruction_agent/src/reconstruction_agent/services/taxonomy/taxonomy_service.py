"""How closely related two organisms are, measured rather than guessed.

The question this answers is "how much should a homologue from *this* organism
count towards reconstructing *that* one". A sequence from the same species is
near-conclusive; one from a different order is weak evidence about individual
bases however well it aligns.

It is computed from NCBI lineages, not from names. A previous implementation
scored relatedness from a three-value heuristic over organism strings, which
cannot tell a congener from a distant relative and quietly flattened the one
signal that separates a trustworthy fill from a plausible one.
"""

from __future__ import annotations

from reconstruction_agent.domain.models.taxonomy import TargetProfile, TaxonNode
from reconstruction_agent.integrations.ncbi.taxonomy import TaxonomyClient


class TaxonomyService:
    """Lineage-based relatedness, over the cached taxonomy client."""

    def __init__(self, client: TaxonomyClient) -> None:
        self._client = client

    async def profile_for(
        self,
        scientific_name: str,
        tax_id: int | None,
        molecule_type: object = None,
    ) -> TargetProfile:
        """Build a target profile from an organism identity.

        Carries biology only. What collection to search is a separate decision,
        made later and from this.

        A name without a tax id is resolved here rather than left unresolved.
        That case is not exotic: it is every caller who supplied a sequence
        instead of an accession, because nothing fetched a record that could
        carry the id. Leaving it None costs more than a missing label - the
        lineage stays empty, no taxonomic scope can be derived from it, and the
        homology round returns before dispatching a single search. The species
        the caller named would reach the response as a display string and
        change nothing about what was searched.
        """
        from reconstruction_agent.domain.enums import MoleculeType

        if tax_id is None and scientific_name:
            node = await self._client.resolve_name(scientific_name)
            if node is not None:
                tax_id = node.tax_id

        lineage: tuple[TaxonNode, ...] = ()
        if tax_id is not None:
            lineage = await self._client.lineage(tax_id)

        return TargetProfile(
            scientific_name=scientific_name or "unknown organism",
            tax_id=tax_id,
            taxonomy_lineage=lineage,
            taxonomy_ranks={node.rank: node.name for node in lineage if node.rank != "no rank"},
            molecule_type=(
                molecule_type if isinstance(molecule_type, MoleculeType) else MoleculeType.UNKNOWN
            ),
        )

    async def relatedness(self, profile: TargetProfile, tax_id: int | None) -> float:
        """How close `tax_id` is to the target, 0..1.

        The shared fraction of the two lineages: 1.0 for the same taxon, high
        for a congener, low for a distant relative, 0.0 for an unrelated one.
        Normalising by the longer lineage keeps the scale comparable between
        organisms whose trees are recorded at different depths.

        An unresolved organism scores 0.0 rather than a neutral middle value.
        Evidence whose provenance cannot be established should not be able to
        raise a confidence.
        """
        if tax_id is None or not profile.taxonomy_lineage:
            return 0.0
        if tax_id == profile.tax_id:
            return 1.0

        other = await self._client.lineage(tax_id)
        if not other:
            return 0.0

        shared = _common_prefix_length(profile.taxonomy_lineage, other)
        if shared == 0:
            return 0.0
        return shared / max(len(profile.taxonomy_lineage), len(other))

    async def closest_organism(
        self, profile: TargetProfile, tax_ids: tuple[int | None, ...]
    ) -> float:
        """The relatedness of the closest of several organisms.

        The maximum rather than the mean: one homologue from the same species
        is strong evidence, and averaging it against a handful of distant ones
        would discard exactly the signal worth having.
        """
        scores = [await self.relatedness(profile, tax_id) for tax_id in tax_ids]
        return max(scores, default=0.0)


def _common_prefix_length(left: tuple[TaxonNode, ...], right: tuple[TaxonNode, ...]) -> int:
    """How many leading lineage nodes two organisms share.

    Lineages are ordered root-first, so a shared prefix is a shared ancestry
    and its length is the depth of the lowest common ancestor.
    """
    shared = 0
    for a, b in zip(left, right, strict=False):
        if a.tax_id != b.tax_id:
            break
        shared += 1
    return shared
