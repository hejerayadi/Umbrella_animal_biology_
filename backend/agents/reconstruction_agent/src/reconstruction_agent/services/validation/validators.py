"""Deterministic biological checks on a proposed sequence.

These are the last line before a reconstruction is returned, and none of them
involves a language model. Each asks a question with an objective answer, and
each failure names something a biologist would recognise as wrong.

A check that fails does not necessarily sink the candidate - length and GC
disagreement are warnings that lower the score, because a real indel can
legitimately change a gap length. Alphabet violation is different: a sequence
containing an ambiguity character has not answered the question that was asked,
so it is fatal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from reconstruction_agent.domain.models.candidate import ValidationVerdict
from reconstruction_agent.domain.models.sequence import DNA_ALPHABET, GapContext


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    """Where each check draws its line. Configurable, never hardcoded inline."""

    #: Fractional deviation from the expected gap length that still passes.
    length_tolerance: float = 0.25
    #: Absolute GC-fraction difference from the flanks that still passes.
    gc_tolerance: float = 0.25
    #: Longest single-base run permitted, as a fraction of the sequence.
    max_homopolymer_fraction: float = 0.6
    #: Distinct bases a sequence of reasonable length should contain.
    min_distinct_bases: int = 2


def validate(
    sequence: str, context: GapContext, thresholds: ValidationThresholds | None = None
) -> tuple[ValidationVerdict, ...]:
    """Run every check against `sequence`."""
    limits = thresholds or ValidationThresholds()
    return (
        check_alphabet(sequence),
        check_not_empty(sequence),
        check_length(sequence, context, limits),
        check_gc_consistency(sequence, context, limits),
        check_low_complexity(sequence, limits),
    )


def check_alphabet(sequence: str) -> ValidationVerdict:
    """Every base must be unambiguous.

    Fatal rather than advisory: the request was to resolve an unresolved
    region, so returning a sequence that is itself unresolved answers nothing.
    """
    offenders = sorted(set(sequence.upper()) - DNA_ALPHABET)
    if offenders:
        return ValidationVerdict(
            check="alphabet",
            passed=False,
            reason=f"contains non-ACGT characters: {', '.join(offenders)}",
            score=0.0,
            fatal=True,
        )
    return ValidationVerdict(check="alphabet", passed=True)


def check_not_empty(sequence: str) -> ValidationVerdict:
    if not sequence:
        return ValidationVerdict(
            check="non_empty",
            passed=False,
            reason="no sequence proposed",
            score=0.0,
            fatal=True,
        )
    return ValidationVerdict(check="non_empty", passed=True)


def check_length(
    sequence: str, context: GapContext, limits: ValidationThresholds
) -> ValidationVerdict:
    """The fill should be about as long as the hole it fills.

    Advisory, not fatal. The assembly gap length is itself an estimate, and a
    genuine insertion or deletion in the target relative to its homologues
    changes it - so a mismatch lowers confidence rather than rejecting outright.
    """
    expected = context.length
    if expected <= 0:
        return ValidationVerdict(check="length", passed=True)

    deviation = abs(len(sequence) - expected) / expected
    if deviation > limits.length_tolerance:
        return ValidationVerdict(
            check="length",
            passed=False,
            reason=(
                f"length {len(sequence)} differs from the {expected}-base gap by {deviation:.0%}"
            ),
            score=max(0.0, 1.0 - deviation),
        )
    return ValidationVerdict(check="length", passed=True, score=1.0 - deviation)


def check_gc_consistency(
    sequence: str, context: GapContext, limits: ValidationThresholds
) -> ValidationVerdict:
    """Base composition should resemble the surrounding sequence.

    A fill whose GC content is wildly unlike its flanks is either from the
    wrong region or from a compositionally different genome - both of which are
    reasons to trust it less.
    """
    flanks = context.left_flank + context.right_flank
    if not sequence or not flanks:
        return ValidationVerdict(check="gc_consistency", passed=True)

    difference = abs(_gc_fraction(sequence) - _gc_fraction(flanks))
    if difference > limits.gc_tolerance:
        return ValidationVerdict(
            check="gc_consistency",
            passed=False,
            reason=(f"GC content differs from the flanking sequence by {difference:.0%}"),
            score=max(0.0, 1.0 - difference),
        )
    return ValidationVerdict(check="gc_consistency", passed=True, score=1.0 - difference)


def check_low_complexity(sequence: str, limits: ValidationThresholds) -> ValidationVerdict:
    """A fill that is one base repeated is not a reconstruction.

    This is the shape a degenerate consensus takes when the evidence has
    collapsed, so it is worth catching explicitly rather than letting it through
    with a respectable-looking score.
    """
    if not sequence:
        return ValidationVerdict(check="low_complexity", passed=True)

    counts = Counter(sequence.upper())
    dominant = counts.most_common(1)[0][1] / len(sequence)

    if dominant > limits.max_homopolymer_fraction:
        return ValidationVerdict(
            check="low_complexity",
            passed=False,
            reason=f"{dominant:.0%} of the sequence is a single repeated base",
            score=max(0.0, 1.0 - dominant),
        )
    if len(sequence) > 3 and len(counts) < limits.min_distinct_bases:
        return ValidationVerdict(
            check="low_complexity",
            passed=False,
            reason="sequence uses too few distinct bases to be informative",
            score=0.0,
        )
    return ValidationVerdict(check="low_complexity", passed=True)


def aggregate_score(verdicts: tuple[ValidationVerdict, ...]) -> float:
    """The validation component of a candidate confidence, 0..1.

    A check marked fatal zeroes it outright; otherwise the checks are averaged,
    so several mild disagreements weigh more than one. Fatality is read from
    the verdict rather than inferred from a zero score, because an advisory
    check can reach zero without disqualifying anything.
    """
    if not verdicts:
        return 1.0
    if any(v.fatal and not v.passed for v in verdicts):
        return 0.0
    return sum(v.score for v in verdicts) / len(verdicts)


def _gc_fraction(sequence: str) -> float:
    upper = sequence.upper()
    if not upper:
        return 0.0
    return sum(1 for base in upper if base in "GC") / len(upper)
