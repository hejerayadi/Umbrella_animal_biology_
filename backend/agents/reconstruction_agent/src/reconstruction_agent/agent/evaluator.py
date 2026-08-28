"""Deciding whether the evidence gathered so far is enough.

Entirely deterministic, and that is the design rather than a simplification.
"Is this enough evidence" reduces to "did a candidate clear the confidence
floor", which the `ConfidenceEngine` already computes from measured scores. A
model asked the same question would be re-deciding something already decided,
with no new information and no reproducibility.

The evaluator therefore reads state and returns one of a closed set of
decisions. `agent/router.py` turns that decision into an edge.
"""

from __future__ import annotations

from reconstruction_agent.agent.state import GapState
from reconstruction_agent.domain.enums import EvaluationDecision
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.orchestration.deadline import Deadline
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine


class Evaluator:
    """Reads the evidence and says what the graph should do next."""

    def __init__(self, engine: ConfidenceEngine) -> None:
        self._engine = engine

    def evaluate(
        self, state: GapState, *, deadline: Deadline, budget: BudgetLedger
    ) -> EvaluationDecision:
        """What the current evidence supports."""
        # Time and budget are checked first and unconditionally. A candidate
        # that would have cleared the floor after one more round is not a
        # reason to start a round there is no time to finish.
        if deadline.expired():
            return EvaluationDecision.BUDGET_EXHAUSTED
        if budget.exhausted:
            return EvaluationDecision.BUDGET_EXHAUSTED

        candidates = state.get("candidates", ())
        if not candidates:
            return EvaluationDecision.MORE_EVIDENCE_REQUIRED

        best = candidates[0]
        if not self._engine.is_returnable(best.final_confidence):
            return EvaluationDecision.MORE_EVIDENCE_REQUIRED

        # A candidate clearing the floor while a rival sits within the
        # arbitration margin is ready to be *reported*, not settled: the
        # critic gets a chance to name COMPETING_CANDIDATES first.
        return EvaluationDecision.CANDIDATE_READY
