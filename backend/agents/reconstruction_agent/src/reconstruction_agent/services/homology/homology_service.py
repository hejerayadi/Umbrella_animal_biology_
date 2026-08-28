"""Homology search against NCBI BLAST, scoped by tax id.

The scope is *stated*, not inferred. NCBI accepts `ENTREZ_QUERY=txid40674[ORGN]`,
so the clade searched is the tax id the agent already holds from the lineage it
already fetched - nothing is matched against a collection label, nothing can be
mismatched, and the failure this agent was rebuilt around (searching a
collection that excludes the target clade) cannot occur.

What remains is a choice of *how narrow* to be. A family-level scope returns
close relatives and little else; a class-level scope returns far more sequence,
most of it too distant to fill a gap accurately. Those are alternative levels
of specificity, so the configured ranks are searched concurrently and the one
with the most gap-spanning hits wins.

The record being repaired is excluded from its own search. It cannot be
evidence about its own unresolved region, and in a ground-truth measurement -
where bases are withheld artificially while the public record still holds them
- leaving it in would turn the exercise into a lookup of the answer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

from reconstruction_agent.config.settings import HomologySettings, NcbiBlastSettings
from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.domain.models.taxonomy import TargetProfile, TaxonNode
from reconstruction_agent.integrations.ncbi.taxonomy import TaxonomyClient
from reconstruction_agent.integrations.ncbi_blast.client import NcbiBlastClient
from reconstruction_agent.integrations.ncbi_blast.parser import parse_hits
from reconstruction_agent.observability.logger import get_logger
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.orchestration.deadline import Deadline
from reconstruction_agent.services.homology.round import HomologyRound, SearchScope

_log = get_logger(__name__)

#: How many of the winning hits get their organism resolved against taxonomy.
#: Only the top hits are ever fetched and aligned, so resolving the whole result
#: set spends the E-utilities rate limit on evidence no candidate is scored
#: from - and that limit is enforced by silent blocking, which takes the
#: sequence fetches down with it.
RESOLVED_ORGANISM_LIMIT = 12


class HomologyService:
    """Finds homologues for a gap by searching NCBI BLAST at taxonomic scopes."""

    def __init__(
        self,
        *,
        blast: NcbiBlastClient,
        taxonomy: TaxonomyClient,
        settings: NcbiBlastSettings,
        homology: HomologySettings,
    ) -> None:
        self._blast = blast
        self._taxonomy = taxonomy
        self._settings = settings
        self._homology = homology

    @property
    def name(self) -> str:
        return "NCBI"

    async def search_for_gap(
        self,
        profile: TargetProfile,
        context: GapContext,
        *,
        deadline: Deadline,
        budget: BudgetLedger,
        limit: int = 3,
        max_hits: int = 50,
        expect: float = 1e-5,
        evidence: tuple[HomologySearchOutcome, ...] = (),
        exclude: frozenset[str] = frozenset(),
        time_budget: float | None = None,
    ) -> HomologyRound:
        """One round of searches, one per taxonomic scope."""
        scopes = self._scopes(profile, limit=limit, exclude=exclude)
        if not scopes:
            return HomologyRound(
                gap_id=context.gap_id,
                selection_rationale=(
                    "No usable taxonomic scope: the target has no resolved lineage, "
                    "so the search could not be restricted to its relatives."
                ),
            )

        # Reserved before dispatch, never counted after.
        granted = budget.reserve(ToolName.SEARCH_HOMOLOGS, len(scopes))
        if granted <= 0:
            return HomologyRound(
                gap_id=context.gap_id,
                candidates=scopes,
                selection_rationale="Search budget exhausted before this round.",
            )
        scopes = scopes[:granted]

        query = context.query_sequence()
        junction = len(context.left_flank)
        # The homology phase's share, not the whole working window. Taking the
        # window entire is what lets one slow BLAST queue finish the run with
        # hits it never had time to fetch.
        timeout = time_budget if time_budget is not None else deadline.remaining_for_work()
        timeout = min(timeout, deadline.remaining_for_work())
        excluded = self._excluded_accession(context)

        _log.info(
            "ncbi_round_dispatched",
            gap_id=context.gap_id,
            scopes=[scope.code for scope in scopes],
            excluded_accession=excluded,
            query_length=len(query),
            timeout_seconds=round(timeout, 1),
        )

        outcomes = await self._race(
            [
                self._search_one(
                    scope,
                    query=query,
                    junction=junction,
                    max_hits=max_hits,
                    expect=expect,
                    timeout=timeout,
                    excluded=excluded,
                )
                for scope in scopes
            ],
            grace_seconds=self._homology.front_runner_grace_seconds,
        )

        result = HomologyRound(
            gap_id=context.gap_id,
            outcomes=tuple(outcomes),
            candidates=scopes,
            selection_rationale=" | ".join(f"{s.code}: {s.rationale}" for s in scopes),
        )

        if (best := result.best) is not None:
            resolved: list[HomologySearchOutcome] = []
            for outcome in result.outcomes:
                if outcome.database_code == best.database_code:
                    resolved.append(await self._resolve_organisms(outcome))
                else:
                    resolved.append(outcome)
            result = result.model_copy(update={"outcomes": tuple(resolved)})

        _log.info(
            "ncbi_round_completed",
            gap_id=context.gap_id,
            results=[
                {
                    "scope": o.database_code,
                    "hits": o.total_hits,
                    "spanning": o.gap_spanning_hits,
                    "seconds": round(o.duration_seconds, 1),
                    "error": o.error,
                }
                for o in result.outcomes
            ],
            winner=best.database_code if (best := result.best) else None,
        )
        return result

    def _scopes(
        self, profile: TargetProfile, *, limit: int, exclude: frozenset[str]
    ) -> tuple[SearchScope, ...]:
        """The taxonomic restrictions worth searching, narrowest first.

        Narrowest first because a close relative fills a gap accurately and a
        distant one does not. Ranks the target has no taxon at are skipped
        silently - not every lineage records every rank.
        """
        by_rank = {node.rank: node for node in profile.taxonomy_lineage}
        scopes: list[SearchScope] = []

        for rank in self._homology.ncbi_scope_ranks:
            node = by_rank.get(rank)
            if node is None:
                continue
            code = f"{self._settings.database}@txid{node.tax_id}"
            if code in exclude:
                continue
            depth = profile.depth_of(node.tax_id)
            scopes.append(
                SearchScope(
                    code=code,
                    label=f"NCBI {self._settings.database} restricted to {node.name}",
                    # Ranked by specificity, so the narrowest scope leads.
                    score=float(depth if depth is not None else 0),
                    resolved_taxon=node,
                    containment_depth=depth,
                    rationale=f"scoped to {node.name} ({rank}, txid{node.tax_id})",
                )
            )
            if len(scopes) >= limit:
                break

        # No rank matched but the organism itself is known: scope to it. Better
        # than an unrestricted search, which returns the whole nucleotide
        # collection ranked by similarity and mostly wastes the deadline.
        #
        # `exclude` is honoured here exactly as it is for the ranked scopes.
        # Without that check a replan that has already measured this scope as
        # weak is handed it again, and re-runs an identical search for an
        # identical answer - measured at 51 seconds a round on NW_007907101,
        # spent twice.
        if not scopes and profile.tax_id is not None:
            code = f"{self._settings.database}@txid{profile.tax_id}"
            if code in exclude:
                return ()
            scopes.append(
                SearchScope(
                    code=code,
                    label=f"NCBI {self._settings.database} restricted to {profile.scientific_name}",
                    resolved_taxon=None,
                    containment_depth=None,
                    rationale=f"scoped to the organism itself (txid{profile.tax_id})",
                )
            )

        scopes.sort(key=lambda scope: -scope.score)
        return tuple(scopes)

    def _excluded_accession(self, context: GapContext) -> str | None:
        """The record being repaired, if it is to be kept out of its own search."""
        if not self._homology.exclude_target_accession:
            return None
        return context.source_accession

    async def _search_one(
        self,
        scope: SearchScope,
        *,
        query: str,
        junction: int,
        max_hits: int,
        expect: float,
        timeout: float,
        excluded: str | None,
    ) -> HomologySearchOutcome:
        """Search one scope, never raising.

        A failed search is a result about that scope; letting it propagate
        would discard whatever its siblings found.
        """
        started = time.monotonic()
        tax_filter = scope.code.split("@", 1)[-1]
        entrez = f"{tax_filter}[ORGN]"
        if excluded:
            # Strip any version suffix: Entrez matches the versionless form.
            entrez += f" NOT {excluded.split('.')[0]}[ACCN]"

        try:
            payload = await self._blast.search(
                query,
                database=self._settings.database,
                max_hits=max_hits,
                expect=expect,
                entrez_query=entrez,
                timeout=timeout,
            )
            hits = parse_hits(payload, database_code=scope.code, query_length=len(query))
            spanning = sum(1 for hit in hits if hit.spans(junction))
            return HomologySearchOutcome(
                database_code=scope.code,
                database_label=scope.label,
                hits=hits,
                gap_spanning_hits=spanning,
                duration_seconds=time.monotonic() - started,
            )
        except ReconstructionError as error:
            _log.warning(
                "ncbi_search_failed",
                scope=scope.code,
                error=str(error),
                code=error.code.value,
            )
            return HomologySearchOutcome(
                database_code=scope.code,
                database_label=scope.label,
                duration_seconds=time.monotonic() - started,
                error=str(error),
            )

    @staticmethod
    async def _race(
        searches: list[Coroutine[Any, Any, HomologySearchOutcome]],
        *,
        grace_seconds: float,
    ) -> list[HomologySearchOutcome]:
        """Run the scopes together, preferring the narrowest without waiting forever.

        Arriving first is not evidence of being right. The scopes are dispatched
        narrowest first and the narrowest gives the closest relatives, so the
        round prefers it over whichever finished soonest.

        That preference is bounded. NCBI queue times vary enormously between
        identical submissions - the narrowest scope has been measured finishing
        in 50 s and, on the next run, still queued at 270 s - so an unbounded
        wait hands the whole run to the slowest queue and ends it holding hits
        it never had time to fetch. Once usable evidence is in hand the front
        runner gets `grace_seconds` more and no longer than that.
        """
        tasks = [asyncio.ensure_future(search) for search in searches]
        front_runner = tasks[0] if tasks else None
        pending = set(tasks)
        collected: list[HomologySearchOutcome] = []
        #: Set when usable evidence first lands, and only then.
        grace_expires_at: float | None = None

        try:
            while pending:
                timeout = (
                    None
                    if grace_expires_at is None
                    else max(grace_expires_at - time.monotonic(), 0.0)
                )
                done, pending = await asyncio.wait(
                    pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    # The grace ran out with the front runner still queued.
                    break

                collected.extend(task.result() for task in done)
                usable = any(o.is_usable for o in collected)
                if not usable:
                    continue
                if front_runner is None or front_runner.done():
                    break
                if grace_expires_at is None:
                    grace_expires_at = time.monotonic() + grace_seconds
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return collected

    async def _resolve_organisms(self, outcome: HomologySearchOutcome) -> HomologySearchOutcome:
        """Attach tax ids to the top hits, so distance can be computed."""
        resolved: dict[str, int | None] = {}
        hits: list[HomologHit] = []

        for index, hit in enumerate(outcome.hits):
            if not hit.organism or index >= RESOLVED_ORGANISM_LIMIT:
                hits.append(hit)
                continue
            if hit.organism not in resolved:
                node: TaxonNode | None = await self._taxonomy.resolve_name(hit.organism)
                resolved[hit.organism] = node.tax_id if node else None
            hits.append(hit.model_copy(update={"organism_tax_id": resolved[hit.organism]}))

        return outcome.model_copy(update={"hits": tuple(hits)})

    async def aclose(self) -> None:
        await self._blast.aclose()
