"""How closely a reference's organism matches the one being reconstructed.

`ReferenceRanker` already weights `Reference.relatedness` at 0.3, but nothing
was filling it in. The only writer is `evolutionary_context`, and in practice
that tool returns "Completed without producing usable evidence" - so every
reference scored 0 on relatedness, the ranking collapsed to identity and
coverage alone, and the target's own species had no advantage over any other
vertebrate that aligned as well.

That is not a cosmetic gap. Measured on a 21-base gap in human mtDNA COI: with
the reference pool widened from coding-only to all vertebrate sequence, the
consensus drifted off the human sequence onto the mammalian one, returning
`TAATAATCTTCTTTATAGTTA` where the truth is `TAATAATCTTCTTCATAGTAA` - 19 of 21
bases, reported at 0.82 confidence. The two wrong bases are exactly the columns
where non-human references outvoted human ones.

This module supplies the missing signal from something always available: the
organism name the caller already gave. It is a fallback, not a replacement -
a relatedness measured by `evolutionary_context` is a real phylogenetic
distance and always wins over the name comparison here.

Deliberately no common-name table. "human" does not match "Homo sapiens" here,
and inventing a small alias map would make the tests this was written against
pass while leaving every unlisted species exactly as broken. Resolving a common
name to a binomial is the platform's species-resolution job, upstream of this
agent; when the caller supplies a binomial, this works, and when it does not,
the ranking is no worse than it was.
"""
from __future__ import annotations

#: Same species named the same way. The strongest signal available without a
#: phylogeny, and the case that matters: reconstructing human sequence from
#: human references.
CONSPECIFIC = 1.0

#: Same genus, different species - "Homo sapiens" against "Homo neanderthalensis".
#: Well above an unrelated vertebrate, well below the animal's own species.
CONGENERIC = 0.6


def _normalise(name: str | None) -> str | None:
    """Case-folded, whitespace-collapsed, or None when there is nothing to compare."""
    if not name:
        return None
    collapsed = " ".join(name.split()).casefold()
    return collapsed or None


def _genus(name: str) -> str | None:
    """The first token, when the name looks like a binomial rather than one word."""
    parts = name.split(" ")
    return parts[0] if len(parts) >= 2 else None


def organism_affinity(target: str | None, candidate: str | None) -> float | None:
    """0..1 for how close `candidate` is to `target`, or None when unknowable.

    None rather than 0.0 for "cannot tell": the caller uses it to decide whether
    to fill a missing relatedness at all, and writing 0.0 would assert that an
    unidentified organism is definitely unrelated.
    """
    left, right = _normalise(target), _normalise(candidate)
    if left is None or right is None:
        return None

    if left == right:
        return CONSPECIFIC

    # A subspecies or strain suffix still names the same species:
    # "Homo sapiens" against "Homo sapiens neanderthalensis".
    if right.startswith(left + " ") or left.startswith(right + " "):
        return CONSPECIFIC

    left_genus, right_genus = _genus(left), _genus(right)
    if left_genus and right_genus and left_genus == right_genus:
        return CONGENERIC

    return None
