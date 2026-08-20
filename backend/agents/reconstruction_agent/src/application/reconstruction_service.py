"""The agent's use case, start to finish.

Takes an orchestrator request, resolves the target sequence, runs one slice of
the agent, and reports what happened. This is the seam the API layer talks to;
it knows nothing about HTTP.

The unit of work is a slice, not a run: see `run_agent.py`. A slice that runs
out of wall clock comes back unfinished, and the API turns that into a CONTINUE
the orchestrator will retry.
"""
from __future__ import annotations

from dataclasses import dataclass

from application.result_builder import ResultBuilder
from application.run_agent import AgentRunner
from configuration.logging import bind_run_context, get_logger
from configuration.settings import Settings
from contracts.input import ReconstructionRequest
from contracts.output import ReconstructionResult
from domain.exceptions import InvalidSequenceError
from domain.models import Sequence
from infrastructure.ncbi.client import NCBIClient
from infrastructure.persistence.repository import RunRepository
from observability.events import EventEmitter
from observability.tracing import new_run_id, set_run_id
from tools.ncbi.mapper import to_references
from tools.registry import ToolRegistry

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReconstructionOutcome:
    """One slice's result, plus whether the work is done.

    `finished` False becomes a CONTINUE for the orchestrator; the result is
    still populated, because partial findings are worth reporting even when the
    run is not over.
    """

    result: ReconstructionResult
    finished: bool
    continuation_reason: str | None = None
    needs_agent: str | None = None
    prompt_to_target_agent: str | None = None


class ReconstructionService:
    """Orchestrates one reconstruction slice, end to end."""

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        *,
        events: EventEmitter | None = None,
        ncbi_client: NCBIClient | None = None,
        checkpointer: object | None = None,
        runs: RunRepository | None = None,
    ) -> None:
        self._settings = settings
        self._events = events or EventEmitter()
        self._runner = AgentRunner(
            settings, registry, self._events, checkpointer=checkpointer
        )
        self._builder = ResultBuilder()
        self._runs = runs
        # Only needed when a request names an accession instead of supplying
        # residues, so it is built lazily by `_resolve_target`.
        self._ncbi = ncbi_client

    async def reconstruct(
        self, request: ReconstructionRequest, *, trace_id: str | None = None
    ) -> ReconstructionOutcome:
        """Run one slice for this request.

        Raises `InvalidSequenceError` when no target can be resolved; every
        other outcome, including "nothing could be reconstructed", comes back
        as a result.
        """
        run_id = new_run_id()
        set_run_id(run_id)

        # Without a trace id from the orchestrator - a direct v1 call - the run
        # gets its own, and simply never resumes. That is correct: there is no
        # earlier slice to resume from.
        thread_id = trace_id or run_id

        target = await self._resolve_target(request)
        bind_run_context(run_id=run_id, trace_id=thread_id, sequence_id=target.identifier)
        _log.info(
            "reconstruction_started",
            sequence_id=target.identifier,
            length=len(target),
            completeness=round(target.completeness, 4),
        )

        outcome = await self._runner.run_slice(
            run_id=run_id,
            trace_id=thread_id,
            instruction=request.instruction,
            target=target,
            organism=request.organism or target.organism,
            requested_organisms=request.reference_organisms,
        )

        result = self._builder.build(outcome.state)
        await self._record(thread_id, run_id, outcome, result)

        return ReconstructionOutcome(
            result=result,
            finished=outcome.finished,
            continuation_reason=outcome.continuation_reason,
            needs_agent=outcome.state.get("needs_agent"),
            prompt_to_target_agent=outcome.state.get("prompt_to_target_agent"),
        )

    async def _record(
        self,
        trace_id: str,
        run_id: str,
        outcome: object,
        result: ReconstructionResult,
    ) -> None:
        """Upsert the audit row for this run. Never fails the request."""
        if self._runs is None or not self._runs.enabled:
            return

        state = getattr(outcome, "state", {})
        finished = getattr(outcome, "finished", True)
        status = (
            "needs_agent"
            if state.get("needs_agent")
            else ("completed" if finished else "continue")
        )

        await self._runs.record(
            trace_id=trace_id,
            run_id=run_id,
            sequence_id=result.sequence_id,
            organism=result.organism,
            status=status,
            stop_reason=state.get("stop_reason"),
            slice_count=state.get("slice_index", 0) + 1,
            iterations=result.iterations,
            gaps_total=len(result.gaps),
            gaps_resolved=result.reconstructed_count,
            tool_calls_used=state.get("budget_tool_calls", 0),
            llm_tokens_used=state.get("budget_llm_tokens", 0),
            summary=result.summary,
        )

    async def _resolve_target(self, request: ReconstructionRequest) -> Sequence:
        """The sequence to repair, from the request or fetched by accession."""
        if request.sequence is not None:
            return Sequence.parse(
                request.sequence.identifier,
                request.sequence.residues,
                organism=request.sequence.organism or request.organism,
                description=request.sequence.description,
            )

        if request.accession:
            return await self._fetch_by_accession(request.accession, request.organism)

        raise InvalidSequenceError(
            "No target sequence given. Provide `context.sequence` with residues, or "
            "`context.accession` naming a sequence to fetch."
        )

    async def _fetch_by_accession(self, accession: str, organism: str | None) -> Sequence:
        client = self._ncbi or NCBIClient(
            self._settings.ncbi, timeout=self._settings.http.timeout_seconds
        )
        raw = await client.fetch_fasta([accession])
        references = to_references(raw)

        if not references or not references[0].residues:
            raise InvalidSequenceError(
                f"NCBI returned no sequence for accession {accession!r}."
            )

        record = references[0]
        return Sequence.parse(
            record.accession,
            record.residues or "",
            organism=organism or record.organism,
            description=record.description,
            accession=accession,
        )
