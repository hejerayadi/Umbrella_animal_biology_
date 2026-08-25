"""Which gaps to spend a run's search budget on, when there are far too many.

A finished mitogenome has one or two gaps and this never matters. A real draft
assembly is a different problem entirely: measured on `NW_007907101`, the polar
bear scaffold the Genome Agent hands over, the **first megabase alone holds 34
N-runs of 10 bases or more** - about 500 across its 15.9 Mb.

The budget is eight BLAST calls. Attempting five hundred gaps does not produce a
worse answer, it produces no answer: the planner emits a step per gap, the
selector refuses almost all of them for budget, and the run spends its slices
generating refusal records instead of reconstructing anything.

So the gaps are ranked and the run commits to the few it can actually finish.
The ones it does not reach are reported as not attempted, with the reason - they
are not silently dropped, and the count is part of the result. Being explicit
about having examined 8 of 500 gaps is honest; appearing to have considered all
500 and resolved none is not.
"""
from __future__ import annotations

from domain.models import GapContext


def priority(context: GapContext) -> tuple[int, int, int]:
    """Sort key for one gap: lower is attempted sooner.

    Ordered by what actually predicts a usable reconstruction:

    1. **Both flanks usable.** The query is the two flanks joined, and the
       carrier test asks which references cross the junction between them. A
       gap with only one flank - at the very start or end of a sequence, or
       butted against another N-run - has no junction to cross, so no hit can
       ever be measured as carrying it.
    2. **Shorter gaps first.** A short gap is more likely to sit inside a single
       homologous block, and the consensus over it is read from more agreeing
       columns. A 3 kb hole in a draft scaffold is the least likely thing in the
       file to be recoverable from homology.
    3. **Position**, purely so the order is stable and a rerun attempts the same
       gaps as the run before it.
    """
    return (0 if context.has_both_flanks else 1, context.gap.length, context.gap.start)


def select(
    contexts: list[GapContext], *, limit: int, skipped: dict[str, str] | None = None
) -> tuple[list[GapContext], dict[str, str]]:
    """The gaps worth attempting, and why each of the rest was not.

    `skipped` is what the validation policy already rejected - those keep their
    own, more specific reason rather than being relabelled here.

    Returns the selected contexts in their original order, not in priority
    order: downstream reporting reads more naturally along the sequence, and the
    ranking has already done its work by deciding *which* gaps these are.
    """
    already = dict(skipped or {})
    attemptable = [c for c in contexts if c.identifier not in already]

    if limit <= 0 or len(attemptable) <= limit:
        return attemptable, already

    ranked = sorted(attemptable, key=priority)
    chosen = {context.identifier for context in ranked[:limit]}

    for context in ranked[limit:]:
        already[context.identifier] = (
            f"Not attempted: this sequence has {len(attemptable)} gaps and the run's "
            f"search budget covers {limit}. Gaps were ranked by whether both flanks "
            f"are usable and by length; this one ranked below the cut."
        )

    return [c for c in contexts if c.identifier in chosen], already
