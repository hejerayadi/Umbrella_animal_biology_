"""The reconstruction use case: request in, evidence-backed result out.

This is the seam the API talks to. It owns the order of work and the decision
to stop; it owns none of the biology, which lives in the services it calls.

Three rules govern every path through it.

**Refusal is a result.** A gap with no gap-spanning homologue, contradictory
alignments, or a best candidate below the confidence floor comes back
UNRESOLVED with its original coordinates and a stated reason. Nothing here ever
invents a base to avoid an empty answer.

**Partial success is success.** Eight resolved gaps out of ten is reported as
eight resolved gaps, not as a failed run.

**The clock is authoritative.** When the deadline reserve is reached the run
finalises with what it has rather than starting work it cannot finish.
"""

from __future__ import annotations

import time
import uuid

from reconstruction_agent.agent.runner import GapRunner
from reconstruction_agent.config.settings import Settings
from reconstruction_agent.domain.enums import GapStatus, ToolName, UnresolvedReason
from reconstruction_agent.domain.exceptions import InvalidRequestError, ReconstructionError
from reconstruction_agent.domain.models.alignment import AlignmentSupport
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.request import ReconstructionRequest
from reconstruction_agent.domain.models.result import (
    AlignmentEvidence,
    Evo2Evidence,
    GapEvidence,
    GapReconstruction,
    HomologyEvidence,
    Provenance,
    ReconstructionResult,
)
from reconstruction_agent.domain.models.sequence import Gap, GapContext, SequenceRecord
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.integrations.mafft.client import MafftClient
from reconstruction_agent.integrations.mafft.parser import build_fasta, parse_alignment
from reconstruction_agent.observability.logger import get_logger
from reconstruction_agent.orchestration.budget import BudgetLedger, BudgetLimits
from reconstruction_agent.orchestration.deadline import Deadline, PhaseBudget
from reconstruction_agent.services.alignment.gap_analyzer import analyze_gap
from reconstruction_agent.services.candidate.candidate_builder import CandidateBuilder
from reconstruction_agent.services.homology.homology_service import HomologyService
from reconstruction_agent.services.homology.round import HomologyRound
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.services.sequence.sequence_service import (
    SequenceService,
    build_context,
    detect_gaps,
    reconcile_requested_gaps,
    triage,
)
from reconstruction_agent.services.taxonomy.taxonomy_service import TaxonomyService

_log = get_logger(__name__)

#: Bases of context taken either side of a gap. Long enough to anchor an
#: alignment uniquely, short enough that the search stays fast.
FLANK_SIZE = 500


class ReconstructionService:
    """Reconstructs the unresolved regions named by one request."""

    def __init__(
        self,
        *,
        settings: Settings,
        sequences: SequenceService,
        taxonomy: TaxonomyService,
        homology: HomologyService,
        mafft: MafftClient,
        candidates: CandidateBuilder,
        engine: ConfidenceEngine,
        runner: GapRunner | None = None,
    ) -> None:
        self._settings = settings
        self._sequences = sequences
        self._taxonomy = taxonomy
        self._homology = homology
        self._mafft = mafft
        self._candidates = candidates
        self._engine = engine
        #: When present, per-gap work runs through the agent loop - planning,
        #: tool selection, evaluation, criticism and replanning - instead of the
        #: fixed pipeline below. The pipeline remains as the loop's own
        #: deterministic reference and is what the graph's tools call into.
        self._runner = runner
        self._phases = PhaseBudget(
            homology_seconds=settings.reconstruction.homology_budget_seconds,
            alignment_seconds=settings.reconstruction.alignment_budget_seconds,
            arbitration_seconds=settings.reconstruction.arbitration_budget_seconds,
        )

    async def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        """Answer one reconstruction request."""
        config = self._settings.reconstruction
        request_id = f"rec_{uuid.uuid4().hex[:16]}"
        deadline = Deadline(
            total_seconds=config.deadline_seconds,
            reserve_seconds=config.finalization_reserve_seconds,
        )
        budget = BudgetLedger(
            limits=BudgetLimits(
                max_iterations=config.max_iterations,
                max_tool_calls=config.max_tool_calls,
                max_blast_calls=config.max_blast_calls,
                max_mafft_calls=config.max_mafft_calls,
                max_evo2_calls=config.max_evo2_calls,
                max_llm_calls=config.max_llm_calls,
            )
        )

        record = await self._resolve_record(request)
        # The placeholder for an unnamed target belongs to `profile_for`, which
        # now also resolves a name into a tax id. Passing the placeholder from
        # here would send it to a taxonomy search that can only ever miss.
        profile = await self._taxonomy.profile_for(
            request.scientific_name or record.organism or "",
            record.tax_id,
            record.molecule_type,
        )

        # Caller-supplied coordinates are checked against the record before
        # anything is searched: callers differ on the coordinate origin, and
        # the record is the only authority on where the ambiguity really is.
        if request.target_gaps:
            gaps, coordinate_notes = reconcile_requested_gaps(request.target_gaps, record)
        else:
            gaps, coordinate_notes = detect_gaps(record), ()

        for note in coordinate_notes:
            _log.info("gap_coordinates_reconciled", request_id=request_id, detail=note)

        attempted, deferred = triage(
            gaps, max_gaps=config.max_gaps_per_run, max_gap_length=config.max_gap_length
        )

        _log.info(
            "reconstruction_started",
            request_id=request_id,
            accession=record.accession,
            organism=profile.scientific_name,
            gaps_found=len(gaps),
            gaps_attempted=len(attempted),
            gaps_deferred=len(deferred),
            deadline_seconds=config.deadline_seconds,
        )

        results: list[GapReconstruction] = []
        for gap in attempted:
            if deadline.expired():
                results.append(_not_attempted(gap, UnresolvedReason.DEADLINE_EXCEEDED))
                continue
            if self._runner is not None:
                results.append(
                    await self._runner.run(
                        gap,
                        request_id=request_id,
                        deadline=deadline,
                        budget=budget,
                        # The caller's accession, never `record.accession`.
                        # For a pasted sequence that field is a label -
                        # "supplied-sequence", or the assembly id - and
                        # GetSequenceContextTool checks `accession` before
                        # `residues`, so passing it makes the tool try to fetch
                        # the label from NCBI. It fails, and because every later
                        # action needs the record, the whole gap is skipped
                        # without a single BLAST being run.
                        accession=request.sequence_accession,
                        residues=None if request.sequence_accession else record.residues,
                        scientific_name=profile.scientific_name,
                    )
                )
                continue
            results.append(
                await self._reconstruct_gap(
                    build_context(record, gap, flank_size=FLANK_SIZE),
                    profile,
                    deadline=deadline,
                    budget=budget,
                )
            )

        results.extend(
            _not_attempted(gap, _deferral_reason(gap, config.max_gap_length)) for gap in deferred
        )
        results.sort(key=lambda item: item.gap.start)

        result = ReconstructionResult(
            request_id=request_id,
            sequence_accession=record.accession,
            assembly_id=request.assembly_id,
            target_profile=profile,
            reconstructions=tuple(results),
            iteration_count=budget.iterations,
        )
        _log.info(
            "reconstruction_finished",
            request_id=request_id,
            status=result.status.value,
            resolved=result.resolved_gaps,
            unresolved=result.unresolved_gaps,
            elapsed_seconds=round(deadline.elapsed, 1),
            budget=budget.snapshot(),
        )
        return result

    async def _resolve_record(self, request: ReconstructionRequest) -> SequenceRecord:
        """The sequence to repair, fetched or taken from the request."""
        if request.sequence_accession:
            return await self._sequences.fetch(request.sequence_accession)
        if request.residues:
            return SequenceRecord(
                accession=request.assembly_id or "supplied-sequence",
                residues=request.residues,
                organism=request.scientific_name,
            )
        raise InvalidRequestError("No sequence accession or residues were supplied.")

    async def _reconstruct_gap(
        self,
        context: GapContext,
        profile: TargetProfile,
        *,
        deadline: Deadline,
        budget: BudgetLedger,
    ) -> GapReconstruction:
        """Gather evidence for one gap and decide what it supports."""
        budget.start_iteration()
        started = time.monotonic()

        if not context.has_usable_flanks:
            return _unresolved(
                context.gap,
                UnresolvedReason.INSUFFICIENT_GAP_SPANNING_HOMOLOGS,
                "The gap has no usable flanking sequence to anchor an alignment.",
            )

        round_ = await self._homology.search_for_gap(
            profile,
            context,
            deadline=deadline,
            budget=budget,
            # The homology phase's share of the clock. Without this the search
            # takes the whole working window and the hits it found arrive with
            # no time left to fetch or align them.
            time_budget=self._phases.for_homology(deadline),
        )
        provenance = _provenance(round_, budget)

        if not round_.found_support:
            # A search that never returned proves nothing about the biology.
            # Reporting it as an absence of homologues would be a false
            # scientific claim - the honest answer is that the run ran out of
            # time before the evidence arrived.
            if not round_.any_search_completed:
                return _unresolved(
                    context.gap,
                    UnresolvedReason.DEADLINE_EXCEEDED,
                    (
                        "No homology search completed within the run deadline, so "
                        f"no evidence was gathered for this region ({round_.failure_summary})."
                    ),
                    provenance=provenance,
                    evidence=GapEvidence(homology=_homology_evidence(round_)),
                )
            return _unresolved(
                context.gap,
                UnresolvedReason.NO_HOMOLOGS_FOUND,
                "The searched collections returned no homologue crossing this region.",
                provenance=provenance,
                evidence=GapEvidence(homology=_homology_evidence(round_)),
            )

        hits = await self._sequences.fetch_homolog_sequences(round_.hits, deadline=deadline)

        # BLAST found homologues crossing the gap and not one of their sequences
        # could be fetched. That is a retrieval failure, and calling it an
        # absence of gap-spanning homologues would be a false claim about
        # evidence this run had already measured.
        if round_.hits and not hits:
            return _unresolved(
                context.gap,
                UnresolvedReason.EVIDENCE_RETRIEVAL_FAILED,
                (
                    f"The search found {round_.best.gap_spanning_hits if round_.best else 0} "
                    "homologues crossing this region, but none of their sequences could be "
                    "retrieved, so no alignment was possible."
                ),
                provenance=provenance,
                evidence=GapEvidence(homology=_homology_evidence(round_)),
            )

        support = await self._align(context, hits, deadline=deadline, budget=budget)

        if support is None or not support.has_support:
            return _unresolved(
                context.gap,
                UnresolvedReason.INSUFFICIENT_GAP_SPANNING_HOMOLOGS,
                (
                    "Homologues were found, but none of them aligned across the "
                    "missing region, so nothing supports a fill."
                ),
                provenance=provenance,
                evidence=GapEvidence(
                    homology=_homology_evidence(round_),
                    alignment=_alignment_evidence(support, len(hits)),
                ),
            )

        candidates = await self._candidates.build(support, context, profile, hits)
        evidence = GapEvidence(
            homology=_homology_evidence(round_),
            alignment=_alignment_evidence(support, len(hits)),
            evo2=Evo2Evidence(consulted=False),
            validation_checks=tuple(v.check for v in candidates[0].validation_results)
            if candidates
            else (),
            validation_failures=candidates[0].failed_checks() if candidates else (),
        )

        _log.info(
            "gap_evaluated",
            gap_id=context.gap_id,
            candidates=len(candidates),
            best_confidence=round(candidates[0].final_confidence, 3) if candidates else 0.0,
            seconds=round(time.monotonic() - started, 1),
        )

        return self._decide(context.gap, candidates, provenance, evidence)

    def _decide(
        self,
        gap: Gap,
        candidates: tuple[Candidate, ...],
        provenance: Provenance,
        evidence: GapEvidence,
    ) -> GapReconstruction:
        """Accept the best candidate, or refuse and say why."""
        if not candidates:
            return _unresolved(
                gap,
                UnresolvedReason.INSUFFICIENT_GAP_SPANNING_HOMOLOGS,
                "No candidate sequence could be assembled from the alignment.",
                provenance=provenance,
                evidence=evidence,
            )

        best, alternatives = candidates[0], candidates[1:]

        if not best.passed_validation and best.scores.validation == 0.0:
            return _unresolved(
                gap,
                UnresolvedReason.BIOLOGICAL_VALIDATION_FAILED,
                f"The best candidate failed validation: {', '.join(best.failed_checks())}.",
                provenance=provenance,
                evidence=evidence,
                alternatives=alternatives,
            )

        if not self._engine.is_returnable(best.final_confidence):
            return _unresolved(
                gap,
                UnresolvedReason.CONFIDENCE_BELOW_THRESHOLD,
                (
                    f"The best candidate reached only {best.final_confidence:.2f} "
                    f"confidence, below the floor for a reported reconstruction. "
                    f"{best.rationale}"
                ),
                provenance=provenance,
                evidence=evidence,
                alternatives=alternatives,
            )

        return GapReconstruction(
            gap=gap,
            status=GapStatus.RESOLVED,
            selected_candidate=best,
            alternatives=alternatives,
            evidence=evidence,
            provenance=provenance,
            explanation=best.rationale,
            warnings=_warnings(best, alternatives),
        )

    async def _align(
        self,
        context: GapContext,
        hits: tuple[HomologHit, ...],
        *,
        deadline: Deadline,
        budget: BudgetLedger,
    ) -> AlignmentSupport | None:
        """Align the flanks with the homologues and read the gap columns."""
        if not hits or budget.reserve(ToolName.ALIGN_HOMOLOGS) <= 0:
            return None
        if deadline.expired():
            return None

        try:
            aligned = await self._mafft.align(
                build_fasta(context.query_sequence(), hits),
                timeout=deadline.remaining_for_work(),
            )
        except ReconstructionError as error:
            _log.warning("alignment_failed", gap_id=context.gap_id, error=str(error))
            return None

        return analyze_gap(parse_alignment(aligned), context, hits)


def _provenance(round_: HomologyRound, budget: BudgetLedger) -> Provenance:
    # Recorded from what actually ran, not from a fixed list: with two homology
    # providers available, a claim about which services backed a reconstruction
    # has to be true of that reconstruction.
    searched = {
        outcome.database_label.split()[0] for outcome in round_.outcomes if outcome.database_label
    }
    return Provenance(
        providers=("NCBI", *sorted(searched - {"NCBI"})),
        databases_searched=round_.searched_codes,
        database_rationale=round_.selection_rationale,
        tool_calls=budget.tool_calls,
    )


def _homology_evidence(round_: HomologyRound) -> HomologyEvidence:
    best = round_.best
    if best is None:
        # No scope produced a gap-spanning hit. Reporting an empty
        # HomologyEvidence here says `hits_examined: 0`, which reads as "BLAST
        # found nothing" - but the search routinely examines hundreds that
        # simply stop at the gap edges, and the two findings mean different
        # things. Aggregate what was actually seen across every scope tried.
        every_hit = [hit for item in round_.outcomes for hit in item.hits]
        top = max(every_hit, key=lambda hit: hit.identity, default=None)
        return HomologyEvidence(
            hits_examined=sum(item.total_hits for item in round_.outcomes),
            gap_spanning_hits=0,
            best_identity=top.identity if top else 0.0,
            closest_organism=top.organism if top else None,
        )
    top = max(best.hits, key=lambda hit: hit.identity, default=None)
    return HomologyEvidence(
        hits_examined=best.total_hits,
        gap_spanning_hits=best.gap_spanning_hits,
        best_identity=top.identity if top else 0.0,
        closest_organism=top.organism if top else None,
    )


def _alignment_evidence(support: AlignmentSupport | None, aligned: int) -> AlignmentEvidence:
    if support is None:
        return AlignmentEvidence(references_aligned=aligned)
    return AlignmentEvidence(
        references_aligned=aligned,
        references_spanning_gap=support.spanning_count,
        conservation=support.conservation,
        competing_fills=len(support.distinct_fills()),
    )


def _warnings(best: Candidate, alternatives: tuple[Candidate, ...]) -> tuple[str, ...]:
    """Caveats worth reporting alongside an accepted reconstruction."""
    warnings: list[str] = []
    if alternatives and alternatives[0].final_confidence >= best.final_confidence * 0.9:
        warnings.append(
            "A competing candidate scored nearly as highly; the evidence does not "
            "cleanly separate them."
        )
    warnings.extend(
        f"validation warning: {verdict.reason}"
        for verdict in best.validation_results
        if not verdict.passed
    )
    return tuple(warnings)


def _unresolved(
    gap: Gap,
    reason: UnresolvedReason,
    explanation: str,
    *,
    provenance: Provenance | None = None,
    evidence: GapEvidence | None = None,
    alternatives: tuple[Candidate, ...] = (),
) -> GapReconstruction:
    """A gap left as it was found, with the reason stated."""
    return GapReconstruction(
        gap=gap,
        status=GapStatus.UNRESOLVED,
        unresolved_reason=reason,
        explanation=explanation,
        alternatives=alternatives,
        evidence=evidence or GapEvidence(),
        provenance=provenance or Provenance(),
    )


def _not_attempted(gap: Gap, reason: UnresolvedReason) -> GapReconstruction:
    return GapReconstruction(
        gap=gap,
        status=GapStatus.SKIPPED,
        unresolved_reason=reason,
        explanation=_deferral_explanation(reason),
    )


def _deferral_reason(gap: Gap, max_gap_length: int) -> UnresolvedReason:
    if gap.length > max_gap_length:
        return UnresolvedReason.GAP_TOO_LONG
    return UnresolvedReason.NOT_ATTEMPTED


def _deferral_explanation(reason: UnresolvedReason) -> str:
    if reason is UnresolvedReason.GAP_TOO_LONG:
        return "The gap is longer than this agent will attempt to reconstruct."
    if reason is UnresolvedReason.DEADLINE_EXCEEDED:
        return "The run deadline was reached before this region could be examined."
    return (
        "This run committed to the regions it could finish within its budget; "
        "this one was not attempted."
    )
