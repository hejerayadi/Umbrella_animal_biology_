"""The agent's use case, start to finish.

Takes an orchestrator request, resolves the target sequence, runs the agent,
and returns a `ReconstructionResult`. This is the seam the API layer talks to;
it knows nothing about HTTP.
"""
from __future__ import annotations

from ..configuration.logging import get_logger
from ..configuration.settings import Settings
from ..contracts.input import ReconstructionRequest
from ..contracts.output import ReconstructionResult
from ..domain.exceptions import InvalidSequenceError
from ..domain.models import Sequence
from ..infrastructure.ncbi.client import NCBIClient
from ..observability.events import EventEmitter
from ..observability.tracing import new_run_id, set_run_id
from ..tools.ncbi.mapper import to_references
from ..tools.registry import ToolRegistry
from .result_builder import ResultBuilder
from .run_agent import AgentRunner

_log = get_logger(__name__)


class ReconstructionService:
    """Orchestrates one reconstruction, end to end."""

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        *,
        events: EventEmitter | None = None,
        ncbi_client: NCBIClient | None = None,
    ) -> None:
        self._settings = settings
        self._events = events or EventEmitter()
        self._runner = AgentRunner(settings, registry, self._events)
        self._builder = ResultBuilder()
        # Only needed when a request names an accession instead of supplying
        # residues, so it is built lazily by `_resolve_target`.
        self._ncbi = ncbi_client

    async def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        """Run the agent for one request.

        Raises `InvalidSequenceError` when no target can be resolved; every
        other outcome, including "nothing could be reconstructed", comes back
        as a result.
        """
        run_id = new_run_id()
        set_run_id(run_id)

        target = await self._resolve_target(request)
        _log.info(
            "Reconstructing %s (%d bases, %.1f%% complete).",
            target.identifier,
            len(target),
            target.completeness * 100,
            extra={"run_id": run_id},
        )

        state = await self._runner.run(
            run_id=run_id,
            instruction=request.instruction,
            target=target,
            organism=request.organism or target.organism,
            requested_organisms=request.reference_organisms,
        )

        return self._builder.build(state)

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
