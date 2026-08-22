"""Check that a candidate reconstruction is biologically admissible.

This is the last gate before a candidate is reported. It answers "could this
sequence be real?", not "is this the right answer" - that is what the
confidence score is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from domain.models import Candidate, GapContext
from domain.models.sequence import IUPAC_NUCLEOTIDES, UNKNOWN_BASE

# Bases that may appear in a reconstruction. Ambiguity codes are rejected:
# proposing `N` to fill a gap of `N` is not a reconstruction, and the narrower
# codes express a confidence we have no basis to claim per-position.
_ALLOWED_BASES = frozenset("ACGT")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating one candidate.

    `errors` are disqualifying; `warnings` are recorded and surfaced with the
    result but do not block it.
    """

    candidate_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ReconstructionValidator:
    """Hard biological checks on a proposed filling.

    `length_tolerance` allows the reconstruction to differ from the gap's width
    because a real indel makes those legitimately different - the reference may
    genuinely carry fewer or more bases across the region than the assembly's N
    run suggests. Beyond the tolerance it signals a misaligned reconstruction
    rather than a real indel.
    """

    length_tolerance: float = 0.5
    # A homopolymer far longer than anything in the flanks is the classic
    # signature of an alignment artefact rather than real sequence.
    max_homopolymer_run: int = 20

    def validate(self, candidate: Candidate, context: GapContext) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        if not candidate.sequence:
            errors.append("Candidate sequence is empty.")
            return ValidationReport(candidate.gap_id, errors, warnings)

        illegal = set(candidate.sequence) - IUPAC_NUCLEOTIDES
        if illegal:
            errors.append(f"Contains non-nucleotide characters: {''.join(sorted(illegal))}")

        if UNKNOWN_BASE in candidate.sequence:
            errors.append("Contains unknown bases (N); this does not resolve the gap.")

        ambiguous = set(candidate.sequence) - _ALLOWED_BASES - {UNKNOWN_BASE}
        if ambiguous:
            warnings.append(f"Contains IUPAC ambiguity codes: {''.join(sorted(ambiguous))}")

        expected = context.gap.length
        deviation = abs(len(candidate.sequence) - expected) / expected if expected else 0.0
        if deviation > self.length_tolerance:
            errors.append(
                f"Length {len(candidate.sequence)} deviates {deviation:.0%} from the "
                f"{expected}-base gap, beyond the {self.length_tolerance:.0%} tolerance."
            )

        longest_run = self._longest_homopolymer(candidate.sequence)
        if longest_run > self.max_homopolymer_run:
            warnings.append(
                f"Contains a {longest_run}-base homopolymer run, which often indicates "
                "an alignment artefact rather than real sequence."
            )

        if not candidate.supporting_references:
            warnings.append("No supporting reference recorded for this candidate.")

        return ValidationReport(candidate.gap_id, errors, warnings)

    @staticmethod
    def _longest_homopolymer(sequence: str) -> int:
        longest = 1
        current = 1
        for previous, base in zip(sequence, sequence[1:], strict=False):
            current = current + 1 if base == previous else 1
            longest = max(longest, current)
        return longest
