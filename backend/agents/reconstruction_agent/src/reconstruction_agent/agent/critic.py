"""Naming why the evidence is insufficient.

The critic answers one question and produces one value from a closed set. That
narrowness is what makes the replan loop terminate: a precise deficit maps to a
strategy that changes what gets measured, while a vague critique produces a
retry of the search that just failed.

Diagnosis is rule-based first. Every deficit here is decidable from measurements
already in state - how many hits came back, how many crossed the gap, how far
the references agree, how close the top two candidates sit - so a model is
asked only when the rules find nothing, and its answer is accepted only if it
names a deficit the evidence does not contradict.
"""

from __future__ import annotations

from pydantic import BaseModel

from reconstruction_agent.agent.prompts.loader import load_prompt
from reconstruction_agent.agent.state import GapState
from reconstruction_agent.domain.enums import CriticDeficit
from reconstruction_agent.integrations.llm.base import LLMClient
from reconstruction_agent.observability.logger import get_logger

_log = get_logger(__name__)

#: Below this, hits are too distant to fill a gap accurately. Above it, a weak
#: result is a coverage problem rather than a scoping one.
DISTANT_IDENTITY = 0.80

#: Two candidates closer than this are competing rather than ranked.
COMPETING_MARGIN = 0.05

#: Fewer gap-spanning references than this cannot outvote a single error.
THIN_SUPPORT = 3


class Diagnosis(BaseModel):
    """The model's answer, before it is checked against the evidence."""

    deficit: str | None = None
    sufficient: bool = False
    rationale: str = ""


class Critic:
    """Diagnoses the deficit blocking a gap."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def diagnose(self, state: GapState) -> CriticDeficit | None:
        """The deficit to replan against, or None if there is nothing to act on."""
        ruled = diagnose_by_rule(state)
        if ruled is not None:
            return ruled

        if not self._llm.available:
            return None

        prompt = load_prompt("critic")
        answer = await self._llm.structured(
            system=prompt.system,
            user=prompt.render_user(
                gap_id=state.get("gap_id", ""),
                gap_length=state["gap"].length if state.get("gap") else 0,
                evidence=summarise_evidence(state),
                history=summarise_history(state),
            ),
            schema=Diagnosis,
        )
        if answer is None or answer.sufficient or not answer.deficit:
            return None

        try:
            return CriticDeficit(answer.deficit)
        except ValueError:
            _log.info("critic_deficit_unknown", proposed=answer.deficit)
            return None


def diagnose_by_rule(state: GapState) -> CriticDeficit | None:
    """The deficit the measurements already imply, if any.

    Ordered from the earliest failure to the latest: a run with no homologues
    has nothing to say about alignment, so checking in pipeline order names the
    deficit that actually blocks progress rather than a symptom downstream.
    """
    context = state.get("context")
    if context is not None and not context.has_usable_flanks:
        return CriticDeficit.INSUFFICIENT_CONTEXT

    scopes = state.get("measured_scopes", ())
    total_hits = sum(item.total_hits for item in scopes)
    spanning = sum(item.gap_spanning_hits for item in scopes)

    # A search that never completed is not evidence about the biology. Reading
    # it as NO_HOMOLOGS would diagnose an absence that was never measured, and
    # the replan it produces - relaxing the e-value, widening the scopes -
    # spends the remaining budget answering the wrong question. A transport
    # failure is a RETRY concern, handled in the HTTP layer, and it must not
    # reach the scientific replan loop at all.
    if scopes and all(item.error for item in scopes):
        return None

    if scopes and total_hits == 0:
        return CriticDeficit.NO_HOMOLOGS

    if total_hits and not spanning:
        # Hits exist and none crosses the gap. When those hits are also
        # distant, the search was scoped into the wrong clade rather than the
        # homologues being absent - a different deficit with a different fix.
        if _best_identity(state) < DISTANT_IDENTITY:
            return CriticDeficit.DATABASE_MISMATCH
        return CriticDeficit.NO_GAP_SPANNING_HOMOLOG

    support = state.get("support")
    if support is not None:
        if not support.has_support:
            return CriticDeficit.NO_GAP_SPANNING_HOMOLOG
        if support.conflicting_positions:
            return CriticDeficit.AMBIGUOUS_ALIGNMENT
        if len(support.spanning_fills) < THIN_SUPPORT:
            return CriticDeficit.INSUFFICIENT_COVERAGE

    candidates = state.get("candidates", ())
    if len(candidates) > 1:
        margin = candidates[0].final_confidence - candidates[1].final_confidence
        if margin < COMPETING_MARGIN:
            return CriticDeficit.COMPETING_CANDIDATES

    if candidates and not candidates[0].passed_validation:
        return CriticDeficit.BIOLOGICAL_VALIDATION_FAILED

    return None


def _best_identity(state: GapState) -> float:
    hits = state.get("hits", ())
    return max((hit.identity for hit in hits), default=0.0)


def summarise_evidence(state: GapState) -> str:
    """What was measured, in the few lines a prompt can use.

    Never the sequences themselves: a thousand bases of DNA in a prompt buys no
    diagnostic power and crowds out the measurements that do.
    """
    lines: list[str] = []
    for scope in state.get("measured_scopes", ()):
        lines.append(
            f"- scope {scope.database_code}: {scope.total_hits} hits, "
            f"{scope.gap_spanning_hits} crossing the gap"
            + (f", failed: {scope.error}" if scope.error else "")
        )

    support = state.get("support")
    if support is not None:
        lines.append(
            f"- alignment: {len(support.spanning_fills)} references cross the gap, "
            f"conservation {support.conservation:.2f}, "
            f"{len(support.conflicting_positions)} conflicting positions"
        )

    candidates = state.get("candidates", ())
    for index, candidate in enumerate(candidates[:3], start=1):
        lines.append(
            f"- candidate {index}: confidence {candidate.final_confidence:.2f}, "
            f"{len(candidate.supporting_hits)} supporting hits"
        )

    return "\n".join(lines)


def summarise_history(state: GapState) -> str:
    """The actions taken, one line each."""
    return "\n".join(
        f"- {record.tool.value}: {'ok' if record.ok else record.reason}"
        for record in state.get("tool_history", ())
    )
