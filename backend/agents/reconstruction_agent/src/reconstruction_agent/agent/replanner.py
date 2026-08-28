"""Turning a named deficit into a different thing to measure.

This mapping is Python, not a prompt, and deliberately so. It is the agent's
scientific policy: what to do when the evidence is thin. It must be identical
on every run, reviewable in a diff, and testable deficit by deficit.

Every strategy obeys one rule - **change what gets measured**. A replan that
re-runs the same call with the same arguments burns an iteration and returns
the same answer, which is how an agent loops until its budget is gone. So each
strategy either widens an input, moves to a scope not yet tried, or reaches for
evidence of a different kind. A deficit with nothing new left to try returns no
plan at all, and the graph finalises honestly instead of spinning.

REPLAN is counted here. RETRY - a transient transport failure - is handled by
the HTTP layer's retry policy and never reaches this module: conflating the two
would let a flaky network exhaust the agent's thinking budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reconstruction_agent.agent.state import GapState
from reconstruction_agent.domain.enums import CriticDeficit, ToolName

#: How much more flanking sequence to take when the anchor was too weak.
FLANK_WIDENING_FACTOR = 2
MAX_FLANK_SIZE = 2000

#: How far to relax the e-value when hits were too few. One order of magnitude:
#: enough to admit more distant matches, not enough to admit noise.
EXPECT_RELAXATION_FACTOR = 100.0
MAX_EXPECT = 1e-1

#: More references to align when the vote was too thin to be conclusive.
WIDER_HIT_LIMIT = 25


@dataclass(frozen=True)
class Strategy:
    """A concrete change to what the next round measures."""

    #: The actions to run, in order. Empty means nothing new is worth trying.
    actions: tuple[ToolName, ...] = ()
    #: Argument overrides, keyed by tool name.
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Scopes to stop searching, because they have been measured as weak.
    exhaust_scopes: frozenset[str] = frozenset()
    rationale: str = ""

    @property
    def actionable(self) -> bool:
        return bool(self.actions)


#: Everything downstream of a fresh search. Used by the deficits that change
#: what the search itself measures.
_FROM_SEARCH: tuple[ToolName, ...] = (
    ToolName.SEARCH_HOMOLOGS,
    ToolName.GET_HOMOLOG_SEQUENCES,
    ToolName.ALIGN_HOMOLOGS,
    ToolName.ANALYZE_ALIGNMENT,
    ToolName.GENERATE_CANDIDATES,
    ToolName.FINALIZE_RESULT,
)

#: Arbitration, folded back into the scores, then committed. `score_candidate`
#: is what makes the Evo 2 call worth having: without it the agreement is
#: computed and never read.
_ARBITRATE: tuple[ToolName, ...] = (
    ToolName.EVALUATE_WITH_EVO2,
    ToolName.SCORE_CANDIDATE,
    ToolName.FINALIZE_RESULT,
)

#: Everything downstream of a fresh fetch, for deficits about the alignment.
_FROM_FETCH: tuple[ToolName, ...] = (
    ToolName.GET_HOMOLOG_SEQUENCES,
    ToolName.ALIGN_HOMOLOGS,
    ToolName.ANALYZE_ALIGNMENT,
    ToolName.GENERATE_CANDIDATES,
    ToolName.FINALIZE_RESULT,
)


class Replanner:
    """Chooses the next strategy for a diagnosed deficit."""

    def plan_for(self, deficit: CriticDeficit, state: GapState) -> Strategy:
        """What to do about `deficit`, given what has already been measured."""
        handler = getattr(self, f"_{deficit.value.lower()}", None)
        if handler is None:
            return Strategy(rationale=f"No strategy is defined for {deficit.value}.")
        strategy: Strategy = handler(state)
        return strategy

    # --- Scoping deficits --------------------------------------------------

    def _database_mismatch(self, state: GapState) -> Strategy:
        """Move to a scope not yet tried, never to an invented one.

        The scopes searched are derived from the target's own lineage, so the
        fix is to stop searching the ones already measured as weak and let the
        service offer the next one. Nothing here names a clade.
        """
        weak = _weak_scopes(state)
        already = state.get("exhausted_scopes", frozenset())
        remaining = weak - already
        if not remaining:
            return Strategy(rationale="Every scope derived from the lineage has been measured.")

        return Strategy(
            actions=_FROM_SEARCH,
            exhaust_scopes=already | remaining,
            # A wider scope limit is what makes a *different* scope available:
            # excluding the weak ones without raising the limit would simply
            # return fewer candidates.
            overrides={ToolName.SEARCH_HOMOLOGS.value: {"scope_limit": 5}},
            rationale=(
                f"{len(remaining)} scope(s) returned only distant hits; "
                "searching the next scopes the lineage offers."
            ),
        )

    def _no_homologs(self, state: GapState) -> Strategy:
        """Nothing came back at all: widen the net before concluding absence."""
        expect = _current_expect(state)
        if expect >= MAX_EXPECT:
            return Strategy(rationale="The search is already as permissive as it can usefully be.")
        return Strategy(
            actions=_FROM_SEARCH,
            overrides={
                ToolName.SEARCH_HOMOLOGS.value: {
                    "expect": min(expect * EXPECT_RELAXATION_FACTOR, MAX_EXPECT),
                    "scope_limit": 5,
                }
            },
            rationale="No homologue was returned; relaxing the e-value and widening the scopes.",
        )

    def _insufficient_coverage(self, state: GapState) -> Strategy:
        """Hits exist but are too few or too partial to conclude from."""
        expect = _current_expect(state)
        return Strategy(
            actions=_FROM_SEARCH,
            overrides={
                ToolName.SEARCH_HOMOLOGS.value: {
                    "expect": min(expect * EXPECT_RELAXATION_FACTOR, MAX_EXPECT),
                    "max_hits": 100,
                },
                ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": WIDER_HIT_LIMIT},
            },
            rationale="Too few homologues to be conclusive; admitting more distant matches.",
        )

    def _no_gap_spanning_homolog(self, state: GapState) -> Strategy:
        """Homologues align but stop short of the gap: fetch further down the list."""
        if _already_widened(state, ToolName.GET_HOMOLOG_SEQUENCES, "limit"):
            return Strategy(
                rationale="More homologues were already fetched and still none spans the gap."
            )
        return Strategy(
            actions=_FROM_FETCH,
            overrides={ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": WIDER_HIT_LIMIT}},
            rationale="No fetched homologue crosses the gap; fetching further down the hit list.",
        )

    # --- Alignment and candidate deficits ----------------------------------

    def _insufficient_context(self, state: GapState) -> Strategy:
        """The flanks could not anchor the alignment: take more of them."""
        current = _current_flank(state)
        widened = min(current * FLANK_WIDENING_FACTOR, MAX_FLANK_SIZE)
        if widened <= current:
            return Strategy(rationale="The flanks are already as wide as the record allows.")

        return Strategy(
            actions=(ToolName.GET_SEQUENCE_CONTEXT, *_FROM_SEARCH),
            overrides={ToolName.GET_SEQUENCE_CONTEXT.value: {"flank_size": widened}},
            rationale=(
                f"The flanks did not anchor the alignment; widening them to {widened} bases."
            ),
        )

    def _ambiguous_alignment(self, state: GapState) -> Strategy:
        """References disagree: more of them is the only thing that settles it."""
        if _already_widened(state, ToolName.GET_HOMOLOG_SEQUENCES, "limit"):
            return Strategy(
                rationale=(
                    "The references still disagree with a wider set; "
                    "the region is genuinely ambiguous."
                )
            )
        return Strategy(
            actions=_FROM_FETCH,
            overrides={ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": WIDER_HIT_LIMIT}},
            rationale="References disagree across the gap; aligning more of them to settle it.",
        )

    def _competing_candidates(self, state: GapState) -> Strategy:
        """Two fills sit within the margin.

        More references first, arbitration second. The order matters: another
        observed genome is evidence, whereas Evo 2 is a modelling opinion, and
        spending the opinion before the evidence would let a model settle a
        question that more data would have settled properly.
        """
        if not _already_widened(state, ToolName.GET_HOMOLOG_SEQUENCES, "limit"):
            return Strategy(
                actions=_FROM_FETCH,
                overrides={ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": WIDER_HIT_LIMIT}},
                rationale=(
                    "Two candidates are within the margin; adding references to separate them."
                ),
            )

        if not _already_arbitrated(state):
            return Strategy(
                actions=_ARBITRATE,
                # Forced: the policy would decline on its own criteria, but the
                # critic has named this a tie and nothing else is left to try.
                overrides={ToolName.EVALUATE_WITH_EVO2.value: {"force": True}},
                rationale=(
                    "A wider reference set did not separate the candidates; "
                    "asking Evo 2 to break the tie."
                ),
            )

        return Strategy(
            rationale=(
                "Neither more references nor arbitration separated the candidates; "
                "both are reported."
            )
        )

    def _biological_validation_failed(self, state: GapState) -> Strategy:
        """The leading fill is not plausible sequence.

        Worth one attempt at the runners-up: candidates are ranked by
        confidence, not by validity, so a leader that fails a check can sit
        above an alternative that passes. Beyond that, more homologues do not
        make an implausible sequence plausible.
        """
        if len(state.get("candidates", ())) > 1 and not _already_revalidated(state):
            return Strategy(
                actions=(ToolName.VALIDATE_CANDIDATE, ToolName.FINALIZE_RESULT),
                overrides={ToolName.VALIDATE_CANDIDATE.value: {}},
                rationale=(
                    "The leading candidate failed validation; re-checking every candidate "
                    "in case an alternative passes."
                ),
            )
        return Strategy(
            rationale=(
                "The best candidate is not biologically plausible; no change to the "
                "search would make it so."
            )
        )

    def _evo2_disagreement(self, state: GapState) -> Strategy:
        """Arbitration contradicted the homology evidence.

        Never resolved in Evo 2's favour. A genome model saying it would have
        written something else is not evidence that a dozen sequenced relatives
        are wrong, so the disagreement is recorded and reported - and the
        confidence already carries it as a weighted term.
        """
        return Strategy(
            rationale=(
                "Evo 2 disagreed with the homology evidence; the disagreement is reported "
                "and weighted, not resolved in the model's favour."
            )
        )


def _weak_scopes(state: GapState) -> frozenset[str]:
    """Scopes that returned hits but nothing crossing the gap."""
    return frozenset(
        scope.database_code
        for scope in state.get("measured_scopes", ())
        if scope.gap_spanning_hits == 0
    )


def _current_expect(state: GapState) -> float:
    override = state.get("overrides", {}).get(ToolName.SEARCH_HOMOLOGS.value, {})
    value = override.get("expect", 1e-5)
    return float(value)


def _current_flank(state: GapState) -> int:
    override = state.get("overrides", {}).get(ToolName.GET_SEQUENCE_CONTEXT.value, {})
    context = state.get("context")
    if "flank_size" in override:
        return int(override["flank_size"])
    if context is not None:
        return max(len(context.left_flank), len(context.right_flank))
    return 500


def _already_widened(state: GapState, tool: ToolName, key: str) -> bool:
    """Whether this lever has already been pulled once for this gap."""
    return key in state.get("overrides", {}).get(tool.value, {})


def _already_arbitrated(state: GapState) -> bool:
    """Whether Evo 2 has already been asked about this gap.

    Asking twice cannot change the answer - the generation is near-greedy - and
    it spends a budget slot the run may need elsewhere.
    """
    return bool(state.get("evo2_agreement"))


def _already_revalidated(state: GapState) -> bool:
    return ToolName.VALIDATE_CANDIDATE.value in state.get("overrides", {})
