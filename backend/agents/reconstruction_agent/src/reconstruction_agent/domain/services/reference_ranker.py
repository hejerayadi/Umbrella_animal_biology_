"""Order candidate reference sequences by how much they should be trusted.

Which references are used, and in what order, drives everything downstream:
the alignment is only as good as what goes into it, and external calls are
rate-limited, so the top of this ordering is often all that gets fetched.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Reference


@dataclass(frozen=True, slots=True)
class ReferenceRanker:
    """Scores references on alignment quality and phylogenetic closeness.

    The weights encode a deliberate judgement: a strong local alignment is the
    better signal (a close relative that does not actually align over this
    region tells us nothing about it), but closeness breaks ties and guards
    against a spuriously good alignment to something distant.

    Weights are constructor arguments so the evaluation suite can sweep them
    rather than having them baked into the ranking logic.
    """

    identity_weight: float = 0.5
    coverage_weight: float = 0.2
    relatedness_weight: float = 0.3

    def score(self, reference: Reference) -> float:
        """A single 0..1 trust score for one reference.

        Missing components score 0 rather than being skipped, so a reference
        with no alignment data cannot outrank one that was actually measured.
        """
        identity = reference.identity or 0.0
        coverage = reference.coverage or 0.0
        relatedness = reference.relatedness or 0.0

        total_weight = self.identity_weight + self.coverage_weight + self.relatedness_weight
        weighted = (
            identity * self.identity_weight
            + coverage * self.coverage_weight
            + relatedness * self.relatedness_weight
        )
        return weighted / total_weight if total_weight else 0.0

    def rank(self, references: list[Reference], *, limit: int | None = None) -> list[Reference]:
        """References best-first, optionally truncated to `limit`.

        The tie-break on accession keeps the ordering deterministic, which the
        evaluation suite depends on to compare runs.
        """
        ordered = sorted(
            references,
            key=lambda reference: (-self.score(reference), reference.accession),
        )
        return ordered[:limit] if limit is not None else ordered

    def filter_usable(
        self, references: list[Reference], *, minimum_identity: float = 0.7
    ) -> list[Reference]:
        """Drop references too divergent to inform a reconstruction.

        Below roughly 70% identity over the flanks, reference bases spanning
        the gap are not evidence about the target - they are a different
        sequence that happens to look similar.
        """
        return [
            reference
            for reference in references
            if reference.identity is None or reference.identity >= minimum_identity
        ]
