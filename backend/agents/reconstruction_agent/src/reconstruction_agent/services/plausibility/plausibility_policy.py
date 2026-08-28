"""When Evo 2 is worth a call.

Evo 2 plays two roles, and the policy answers a different question for each.

**As a generator it is always called when there is nothing else.** A gap with
no gap-spanning homologue used to be reported unresolved; a model trained on
genomes still has something to say about what belongs between two flanks, and
declining to ask leaves the region empty for want of a question. This is the
case the capability exists for, and no other condition gates it.

**As an arbiter it is called only where there is a tie to break.** Running it
on a fill twelve conspecific mitogenomes already agree on cannot improve the
answer and can only introduce a disagreement with the evidence - so arbitration
is invoked at a margin or in the borderline band, and declined elsewhere.

What both share is a cost: a generation call is tens of seconds against a
300-second deadline shared with every other gap, and the region has to be short
enough for the continuation to stay coherent.

The decision is a pure function of measured numbers, so it is reproducible and
testable without a network.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconstruction_agent.domain.models.candidate import Candidate

#: Two candidates closer than this are a tie worth breaking.
DEFAULT_MARGIN = 0.05

#: A lone candidate scoring inside this band is returnable but not comfortably
#: so, and a second opinion changes what gets reported.
DEFAULT_BORDERLINE_LOW = 0.15
DEFAULT_BORDERLINE_HIGH = 0.55


@dataclass(frozen=True, slots=True)
class ArbitrationPolicy:
    """Whether the candidates for one gap need a second opinion."""

    margin: float = DEFAULT_MARGIN
    borderline_low: float = DEFAULT_BORDERLINE_LOW
    borderline_high: float = DEFAULT_BORDERLINE_HIGH
    #: Beyond this the generation is too long to be a useful tiebreaker.
    max_gap_length: int = 500

    def decide(self, candidates: tuple[Candidate, ...], *, gap_length: int) -> Decision:
        """Whether to call Evo 2 for this gap, and why."""
        # Length gates everything, generation included: past this the
        # continuation stops being coherent and a long fabricated fill is worse
        # than an honest refusal.
        if gap_length > self.max_gap_length:
            return Decision(
                False,
                f"A {gap_length}-base region is longer than Evo 2 can usefully cover.",
            )

        if not candidates:
            # The case this capability exists for. Homology found nothing that
            # spans the gap, so the model's continuation is not a second
            # opinion - it is the only proposal there will be.
            return Decision(
                True,
                "No homologue spans this region, so Evo 2 is asked to propose the fill.",
            )

        if any(candidate.scores.evo2 is not None for candidate in candidates):
            # Asking twice cannot change the answer and spends the budget.
            return Decision(False, "These candidates have already been arbitrated.")

        if len(candidates) > 1:
            gap = candidates[0].final_confidence - candidates[1].final_confidence
            if gap < self.margin:
                return Decision(
                    True,
                    f"The top two candidates are {gap:.3f} apart, inside the "
                    f"{self.margin:.2f} margin.",
                )

        best = candidates[0].final_confidence
        if self.borderline_low <= best <= self.borderline_high:
            return Decision(
                True,
                f"The best candidate sits at {best:.2f}, in the borderline band "
                f"{self.borderline_low:.2f}-{self.borderline_high:.2f}.",
            )

        return Decision(
            False,
            f"The best candidate is settled at {best:.2f}; homology has already decided it.",
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether to arbitrate, with the reason recorded either way.

    The reason is kept for the negative case too: "Evo 2 was not consulted" is
    something a reviewer will want explained, and reconstructing it afterwards
    from the scores is guesswork.
    """

    arbitrate: bool
    reason: str
