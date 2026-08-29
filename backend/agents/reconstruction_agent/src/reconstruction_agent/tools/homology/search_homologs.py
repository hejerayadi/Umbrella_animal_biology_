"""Find sequences that cross the gap.

**The planner never names a search scope.** It asks for homologues; which
taxonomic scopes are worth searching is derived from the target's own lineage
by `HomologyService`, and the tool reports back which ones it used and what
each measured. Letting a model choose the scope would put a guessed clade on
the critical path of a scientific claim, and a wrong guess is the one failure
mode this agent was rebuilt around.

**The probe is the search.** The ranked scopes are dispatched concurrently with
real parameters, and the one returning the most hits that actually cross the
gap wins. Probing first and then searching would cost two round trips to learn
what one already tells us, and the deadline does not allow it.

`exclude_scopes` is the replanner's lever: a scope that has already been
measured as weak is passed back here so the next round moves on rather than
re-running the search that just failed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.result import HomologyEvidence
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.orchestration.deadline import Deadline
from reconstruction_agent.services.homology.homology_service import HomologyService
from reconstruction_agent.services.homology.round import HomologyRound
from reconstruction_agent.tools.base import Tool, ToolOutcome


class SearchHomologsInput(BaseModel):
    """Note the absence of a database or scope argument. That is deliberate."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    gap_id: str
    context: GapContext
    profile: TargetProfile
    deadline: Deadline
    budget: BudgetLedger
    #: How many taxonomic scopes to race. Raised by the replanner when the
    #: narrow scopes came back empty and a wider one is worth the time.
    scope_limit: int = Field(default=3, ge=1, le=6)
    max_hits: int = Field(default=50, ge=5, le=500)
    #: Relaxed by the replanner on INSUFFICIENT_COVERAGE.
    expect: float = Field(default=1e-5, gt=0.0)
    #: Scopes already measured as weak. Never re-searched.
    exclude_scopes: frozenset[str] = frozenset()
    time_budget: float | None = None


class SearchHomologsOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: HomologyRound
    hits: tuple[HomologHit, ...] = ()
    #: Every scope measured this round, winner included. Fed back to the next
    #: call as `exclude_scopes` so a weak scope demotes itself.
    measured: tuple[HomologySearchOutcome, ...] = ()


class SearchHomologsTool(Tool[SearchHomologsInput, SearchHomologsOutput]):
    """Homologues crossing the gap, from the best-scoring taxonomic scope."""

    name = ToolName.SEARCH_HOMOLOGS
    description = (
        "Search for homologues crossing the gap. Scopes are derived from the target "
        "lineage - do not name one. Relax expect or raise scope_limit when coverage "
        "is insufficient."
    )
    input_model = SearchHomologsInput

    def __init__(self, homology: HomologyService) -> None:
        self._homology = homology

    async def run(self, request: SearchHomologsInput) -> ToolOutcome[SearchHomologsOutput]:
        try:
            round_ = await self._homology.search_for_gap(
                request.profile,
                request.context,
                deadline=request.deadline,
                budget=request.budget,
                limit=request.scope_limit,
                max_hits=request.max_hits,
                expect=request.expect,
                exclude=request.exclude_scopes,
                time_budget=request.time_budget,
            )
        except ReconstructionError as error:
            return ToolOutcome(tool=self.name, ok=False, reason=str(error), transport_error=True)

        output = SearchHomologsOutput(round=round_, hits=round_.hits, measured=round_.outcomes)

        if not round_.found_support:
            # A search that never completed proves nothing about the biology.
            # Saying "no homologues exist" on that basis would be a false
            # scientific claim, so the two cases are reported separately.
            if not round_.any_search_completed:
                return ToolOutcome(
                    tool=self.name,
                    ok=False,
                    data=output,
                    reason=(
                        "No homology search completed in the time available "
                        f"({round_.failure_summary}), so no evidence was gathered."
                    ),
                    transport_error=True,
                )
            return ToolOutcome(
                tool=self.name,
                ok=False,
                data=output,
                reason="The searched scopes returned no homologue crossing this region.",
            )

        return ToolOutcome(tool=self.name, ok=True, data=output)

    def summarise(self, outcome: ToolOutcome[SearchHomologsOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        round_ = outcome.data.round
        return {
            "scopes": [
                {
                    "scope": item.database_code,
                    "hits": item.total_hits,
                    "spanning": item.gap_spanning_hits,
                    "seconds": round(item.duration_seconds, 1),
                    "error": item.error,
                }
                for item in round_.outcomes
            ],
            "winner": round_.best.database_code if round_.best else None,
            "rationale": round_.selection_rationale,
        }

    def contribute(self, outcome: ToolOutcome[SearchHomologsOutput]) -> EvidenceContribution:
        """The searched scopes, the rationale, and what the best one measured.

        Contributed even when the call reported failure: a search that returned
        hits but none crossing the gap has measured something real, and losing
        it here is what let a run report zero gap-spanning hits beside an exact
        reconstruction.
        """
        if outcome.data is None:
            return EvidenceContribution(providers=("NCBI BLAST",))

        round_ = outcome.data.round
        best = round_.best
        if best is not None:
            top = max(best.hits, key=lambda hit: hit.identity, default=None)
            homology = HomologyEvidence(
                hits_examined=best.total_hits,
                gap_spanning_hits=best.gap_spanning_hits,
                best_identity=top.identity if top else 0.0,
                closest_organism=top.organism if top else None,
            )
        else:
            # No scope produced a gap-spanning hit, so there is no winner to
            # measure from - but the search still examined hits, and leaving
            # this None reports `hits_examined: 0`, which reads as "BLAST found
            # nothing". It found plenty; none of it crossed the gap, and those
            # are different findings. One says the region has no homologous
            # sequence on record, the other says it has homologs that stop at
            # the gap edges. Aggregated across every scope tried.
            examined = sum(item.total_hits for item in round_.outcomes)
            every_hit = [hit for item in round_.outcomes for hit in item.hits]
            top = max(every_hit, key=lambda hit: hit.identity, default=None)
            homology = HomologyEvidence(
                hits_examined=examined,
                gap_spanning_hits=0,
                best_identity=top.identity if top else 0.0,
                closest_organism=top.organism if top else None,
            )

        return EvidenceContribution(
            providers=("NCBI BLAST",),
            databases_searched=round_.searched_codes,
            database_rationale=round_.selection_rationale,
            homology=homology,
        )
