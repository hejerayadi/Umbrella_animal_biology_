"""What one homology search round produced.

The shape everything downstream reads, so alignment, candidate building and
scoring are written against homology rather than against a search service.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.taxonomy import TaxonNode


class SearchScope(BaseModel):
    """One search scope proposed for a gap, with the reason it was proposed.

    A scope is a taxonomic restriction - "NCBI core_nt limited to Ursidae" -
    not a collection. NCBI accepts a tax id directly, so the clade searched is
    stated rather than inferred from a provider label.
    """

    model_config = ConfigDict(frozen=True)

    #: Provider-side identifier, e.g. `core_nt@txid9632`. Opaque; recorded
    #: for provenance and to avoid repeating a scope, never interpreted.
    code: str
    label: str
    score: float = 0.0
    #: The clade this collection covers, as resolved through NCBI Taxonomy.
    #: None when the label names nothing taxonomy recognises.
    resolved_taxon: TaxonNode | None = None
    #: How deep that clade sits in the target lineage. None when unrelated.
    containment_depth: int | None = None
    #: Machine-readable reason, carried into provenance and the run log so a
    #: wrong choice is visible afterwards instead of being inferred from poor
    #: results.
    rationale: str = ""

    @property
    def is_taxonomically_supported(self) -> bool:
        return self.containment_depth is not None


class HomologyRound(BaseModel):
    """Everything one search round produced, winner and losers alike."""

    model_config = ConfigDict(frozen=True)

    gap_id: str
    outcomes: tuple[HomologySearchOutcome, ...] = ()
    candidates: tuple[SearchScope, ...] = ()
    #: Why these collections were chosen, for provenance and the run log.
    selection_rationale: str = ""

    @property
    def best(self) -> HomologySearchOutcome | None:
        """The outcome with the most hits actually crossing the gap.

        Gap-spanning count, not hit count: a search returning fifty strong hits
        none of which covers both flanks has found nothing that can fill the
        hole, and ranking on raw hits would call that a success.
        """
        usable = [outcome for outcome in self.outcomes if outcome.is_usable]
        if not usable:
            return None
        return max(usable, key=lambda o: (o.gap_spanning_hits, o.total_hits))

    @property
    def hits(self) -> tuple[HomologHit, ...]:
        best = self.best
        return best.hits if best else ()

    @property
    def searched_codes(self) -> tuple[str, ...]:
        return tuple(outcome.database_code for outcome in self.outcomes)

    @property
    def found_support(self) -> bool:
        return self.best is not None

    @property
    def any_search_completed(self) -> bool:
        """Whether at least one collection actually answered.

        The difference between "we looked and there is nothing" and "we never
        got an answer" is the whole point of this agent, and it must not be
        collapsed. A round where every search timed out has established nothing
        about the biology.
        """
        return any(outcome.succeeded for outcome in self.outcomes)

    @property
    def failure_summary(self) -> str:
        """Why the searches did not answer, for the unresolved explanation."""
        errors = [o.error for o in self.outcomes if o.error]
        return "; ".join(errors[:2]) if errors else "no collection was searched"
