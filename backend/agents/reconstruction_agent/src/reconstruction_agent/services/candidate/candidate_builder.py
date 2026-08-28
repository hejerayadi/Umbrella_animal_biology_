"""Turning alignment evidence into competing, scored candidates.

The rule that shapes this module: ambiguity is preserved. When the references
disagree about what belongs in the gap, that disagreement is a finding. Each
distinct proposal becomes its own candidate, carrying its own support and its
own score, and the choice between them is made from numbers in the open rather
than by averaging them into a consensus that no reference actually proposed.

Collapsing early is how a plurality of two weak references beats a single
near-identical one, and how a chimeric sequence that exists nowhere in nature
ends up being returned as the answer.
"""

from __future__ import annotations

from reconstruction_agent.domain.models.alignment import AlignmentSupport, ReferenceFill
from reconstruction_agent.domain.models.candidate import Candidate, CandidateScores
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.services.scoring.confidence_engine import (
    ConfidenceEngine,
    alignment_score,
    homology_score,
)
from reconstruction_agent.services.taxonomy.taxonomy_service import TaxonomyService
from reconstruction_agent.services.validation.validators import (
    ValidationThresholds,
    aggregate_score,
    validate,
)


class CandidateBuilder:
    """Builds and scores the competing reconstructions for one gap."""

    def __init__(
        self,
        *,
        engine: ConfidenceEngine,
        taxonomy: TaxonomyService,
        thresholds: ValidationThresholds | None = None,
    ) -> None:
        self._engine = engine
        self._taxonomy = taxonomy
        self._thresholds = thresholds or ValidationThresholds()

    async def build(
        self,
        support: AlignmentSupport,
        context: GapContext,
        profile: TargetProfile,
        hits: tuple[HomologHit, ...] = (),
    ) -> tuple[Candidate, ...]:
        """Every distinct proposal for `context`, best-scoring first."""
        if not support.has_support:
            return ()

        by_accession = {hit.accession: hit for hit in hits}
        grouped = _group_by_sequence(support.spanning_fills)

        candidates = [
            await self._score(
                index=index,
                sequence=sequence,
                fills=fills,
                support=support,
                context=context,
                profile=profile,
                hits=by_accession,
            )
            for index, (sequence, fills) in enumerate(grouped.items(), start=1)
        ]
        candidates.sort(key=lambda candidate: -candidate.final_confidence)
        return tuple(candidates)

    async def _score(
        self,
        *,
        index: int,
        sequence: str,
        fills: list[ReferenceFill],
        support: AlignmentSupport,
        context: GapContext,
        profile: TargetProfile,
        hits: dict[str, HomologHit],
    ) -> Candidate:
        """One candidate, with every component of its confidence computed."""
        supporting = tuple(fill.accession for fill in fills)
        backing = [hits[acc] for acc in supporting if acc in hits]

        verdicts = validate(sequence, context, self._thresholds)
        scores = CandidateScores(
            homology=homology_score(
                identity=max((hit.identity for hit in backing), default=0.0),
                coverage=max((hit.query_coverage for hit in backing), default=0.0),
                supporting_hits=len(supporting),
            ),
            alignment=alignment_score(len(fills), max(len(support.fills), 1)),
            conservation=support.conservation,
            evolutionary=await self._taxonomy.closest_organism(
                profile, tuple(fill.organism_tax_id for fill in fills)
            ),
            # Left unset: Evo 2 is invoked later, and only when it can break a
            # tie. None means "not consulted", which the engine treats
            # differently from a disagreement.
            evo2=None,
            validation=aggregate_score(verdicts),
        )
        confidence = self._engine.confidence(scores)

        return Candidate(
            candidate_id=f"{context.gap_id}_cand_{index}",
            gap_id=context.gap_id,
            sequence=sequence,
            supporting_hits=supporting,
            supporting_organisms=_distinct_organisms(fills),
            scores=scores,
            validation_results=verdicts,
            final_confidence=confidence,
            confidence_level=self._engine.level(confidence),
            rationale=self._engine.explain(scores, confidence),
        )

    def with_evo2(self, candidate: Candidate, agreement: float) -> Candidate:
        """Re-score `candidate` with an Evo 2 agreement result folded in.

        Rebuilding the confidence here rather than storing the agreement beside
        it is what stops the arbitration from being computed and then lost: the
        number is only ever recorded as part of a recomputed score.
        """
        scores = candidate.scores.model_copy(update={"evo2": agreement})
        confidence = self._engine.confidence(scores)
        return candidate.model_copy(
            update={
                "scores": scores,
                "final_confidence": confidence,
                "confidence_level": self._engine.level(confidence),
                "rationale": self._engine.explain(scores, confidence),
            }
        )


def _group_by_sequence(fills: tuple[ReferenceFill, ...]) -> dict[str, list[ReferenceFill]]:
    """Distinct proposed sequences, each with the references backing it."""
    grouped: dict[str, list[ReferenceFill]] = {}
    for fill in fills:
        if fill.bases:
            grouped.setdefault(fill.bases, []).append(fill)
    return grouped


def _distinct_organisms(fills: list[ReferenceFill]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for fill in fills:
        if fill.organism:
            seen.setdefault(fill.organism, None)
    return tuple(seen)
