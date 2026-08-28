"""Taxonomy as biology - the target's identity, independent of any provider.

`TargetProfile` is the single most important boundary in this agent. It answers
"what organism is this, and what kind of molecule", and nothing else. It holds
no BLAST database code, no ENA division, no provider vocabulary of any kind.

That separation is not tidiness. The previous implementation carried a division
string on the profile, which meant a taxonomic fact and a provider's catalogue
layout were the same field - and when the two disagreed (EMBL's "other
vertebrates" division excludes mammals, so a polar bear search never saw a
single mammal) there was no way to notice, because the profile looked correct.
Provider choice is now a decision made from this profile, in the homology
layer, against a catalogue read at runtime - and a decision can be re-made when
the evidence says it was wrong.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import MoleculeType


class TaxonNode(BaseModel):
    """One node of an NCBI Taxonomy lineage."""

    model_config = ConfigDict(frozen=True)

    tax_id: int
    name: str
    #: NCBI's rank string ("species", "genus", "family", ..., "no rank").
    rank: str = "no rank"

    def __str__(self) -> str:
        return f"{self.name} ({self.rank}, txid{self.tax_id})"


class TargetProfile(BaseModel):
    """What organism the target sequence belongs to, and what molecule it is.

    Everything here comes from NCBI Taxonomy and the sequence record itself, so
    it stays true no matter which homology provider is used - or if EMBL-EBI is
    replaced entirely tomorrow.
    """

    model_config = ConfigDict(frozen=True)

    scientific_name: str
    tax_id: int | None = None
    #: Ordered root -> target. The last element is the target's own taxon, so
    #: position in this tuple *is* specificity: a later match is a closer clade.
    taxonomy_lineage: tuple[TaxonNode, ...] = ()
    #: Convenience view over the lineage, rank -> name, for logs and payloads.
    #: Derived data: `taxonomy_lineage` is the authority.
    taxonomy_ranks: dict[str, str] = Field(default_factory=dict)
    molecule_type: MoleculeType = MoleculeType.UNKNOWN

    @property
    def lineage_tax_ids(self) -> frozenset[int]:
        """Every taxon this organism belongs to, for containment tests."""
        return frozenset(node.tax_id for node in self.taxonomy_lineage)

    def contains(self, tax_id: int) -> bool:
        """Whether `tax_id` is an ancestor of (or equal to) this organism.

        This is the question a reference collection has to answer: "would a
        member of my clade be in your set at all?"
        """
        return tax_id in self.lineage_tax_ids or tax_id == self.tax_id

    def depth_of(self, tax_id: int) -> int | None:
        """How specific `tax_id` is within this lineage, or None if unrelated.

        Larger is more specific. This single number is what lets a mammal-level
        collection outrank a vertebrate-level one without anybody writing down
        that mammals are vertebrates - the lineage already says so.
        """
        for depth, node in enumerate(self.taxonomy_lineage):
            if node.tax_id == tax_id:
                return depth
        return None

    @property
    def lineage_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.taxonomy_lineage)

    def summary(self) -> str:
        """A one-line description safe to put in a log or an API payload."""
        molecule = self.molecule_type.value.lower().replace("_", " ")
        if self.tax_id is None:
            return f"{self.scientific_name} ({molecule}, unresolved taxonomy)"
        return f"{self.scientific_name} (txid{self.tax_id}, {molecule})"
