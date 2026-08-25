"""Which reference sequences are worth reading a reconstruction out of.

Alignment statistics say how well a sequence matches. They do not say whether
it *should* match: a processed pseudogene aligns beautifully to the gene it came
from and then disagrees about exactly the bases being reconstructed, because it
has been accumulating mutations without selection ever since it was inserted.

This is not hypothetical. On a live run over the human MT-CO1 region, NCBI's
best matches were nuclear mitochondrial insertions - NUMTs - which are the
sequences most likely to be returned for exactly the queries users bring. The
agent abstained, correctly, but it abstained because the evidence was poisoned
rather than absent.

Judgement here is on annotation text and on length plausibility, both of which
are properties of the record rather than of the alignment. Nothing here is
tuned to a fixture: the terms are the ones GenBank actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.models import Reference

#: Records that are homologous but not the thing being reconstructed. A
#: pseudogene's whole nature is that it diverges from the functional copy.
_DISQUALIFYING = (
    "pseudogene",
    "numt",
    "nuclear mitochondrial",
    "unverified",
)

#: Records that may be fine but are less trustworthy: a fragment or a purely
#: computational annotation is weaker evidence than a curated complete one.
_SUSPECT = (
    "partial",
    "predicted",
    "hypothetical",
    "putative",
    "low-quality",
    "low quality",
    "misc_feature",
)

#: A reference far shorter than the region it must span cannot carry the fill,
#: and one far longer is usually a whole chromosome that happens to contain it.
_MIN_LENGTH_RATIO = 0.5
_MAX_LENGTH_RATIO = 50.0


@dataclass(frozen=True, slots=True)
class ReferenceQualityPolicy:
    """Scores and filters references on what the record says about itself.

    Weights are constructor arguments so the evaluation suite can sweep them.
    """

    #: Multiplier applied to a reference whose annotation disqualifies it. Not
    #: zero: a pseudogene still shows where the region sits, and on a gap with
    #: no other evidence a heavily discounted answer beats none. It cannot
    #: outvote a functional homologue, which is the point.
    disqualified_penalty: float = 0.2
    suspect_penalty: float = 0.7
    implausible_length_penalty: float = 0.5

    #: Below this, a reference is not offered to the aligner at all.
    minimum_usable_penalty: float = 0.15

    def penalty(self, reference: Reference, *, expected_length: int | None = None) -> float:
        """How much to discount this reference, on 0..1. 1.0 is no discount."""
        text = " ".join(
            part.lower()
            for part in (reference.description, reference.organism)
            if part
        )

        penalty = 1.0
        if any(term in text for term in _DISQUALIFYING):
            penalty *= self.disqualified_penalty
        elif any(term in text for term in _SUSPECT):
            penalty *= self.suspect_penalty

        if not self._length_plausible(reference, expected_length):
            penalty *= self.implausible_length_penalty

        return penalty

    def _length_plausible(self, reference: Reference, expected_length: int | None) -> bool:
        """Whether the record is a plausible size for what it has to cover.

        Only judged when both numbers are known: an unfetched reference has no
        length, and treating that as implausible would penalise a hit for not
        having been retrieved yet.
        """
        if not expected_length or not reference.residues:
            return True

        ratio = len(reference.residues) / expected_length
        return _MIN_LENGTH_RATIO <= ratio <= _MAX_LENGTH_RATIO

    def is_usable(self, reference: Reference, *, expected_length: int | None = None) -> bool:
        """Whether this reference should be offered to the aligner at all."""
        return self.penalty(reference, expected_length=expected_length) >= (
            self.minimum_usable_penalty
        )

    def describe(self, reference: Reference) -> str | None:
        """Why a reference was discounted, for the run log and the evidence trail."""
        text = " ".join(
            part.lower()
            for part in (reference.description, reference.organism)
            if part
        )
        hit = next((term for term in (*_DISQUALIFYING, *_SUSPECT) if term in text), None)
        return f"annotated {hit!r}" if hit else None
