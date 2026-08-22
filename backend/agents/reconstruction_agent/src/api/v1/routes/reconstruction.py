"""Reconstruction endpoints.

Two doors onto the same service, deliberately:

- `POST /execute` - the orchestrator's. Mounted at the root, and answers with
  the bare repo-wide `AgentResult`, never the `{data, meta, error}` envelope.
  `backend/orchestrator/schema.py` parses that shape directly, so wrapping it
  would break every reconstruction the system performs.

- `POST /api/v1/reconstructions` - for any other client. Typed body, and the
  standard envelope.

Both are communication only. The work happens in `ReconstructionService`.
"""
from __future__ import annotations

import time

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends, Request

from api.dependencies import get_service, get_settings_dependency
from api.v1.envelope import Envelope, ErrorCode
from api.v1.schemas import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    ReconstructionData,
    ReconstructionRequestBody,
)
from application.reconstruction_service import ReconstructionOutcome, ReconstructionService
from application.result_builder import ResultBuilder
from configuration.logging import bind_run_context, clear_run_context, get_logger
from configuration.settings import Settings
from contracts.input import ReconstructionRequest
from domain.exceptions import (
    ExternalServiceError,
    InvalidSequenceError,
    NoGapsFoundError,
    RateLimitError,
)

_log = get_logger(__name__)

# Root-mounted: the orchestrator calls {base_url}/execute.
orchestrator_router = APIRouter(tags=["orchestrator"])

# Mounted under /api/v1.
router = APIRouter(prefix="/reconstructions", tags=["reconstruction"])


def _to_agent_result(outcome: ReconstructionOutcome) -> AgentResult:
    """Map one slice's outcome onto the orchestrator's contract.

    Three of the four statuses are reachable from a successful slice, and the
    distinction matters to the orchestrator's router:

    - NEEDS_AGENT - phylogeny is the blocker; the capability resolver routes it.
    - CONTINUE    - the slice ran out of wall clock. `retryable` must be True,
                    or the worker node converts it straight to FAILED.
    - COMPLETED   - the work is over. That includes abstaining: "these gaps
                    cannot be reconstructed from what is available" is an
                    answer, not an error.
    """
    payload = ResultBuilder.to_output_payload(outcome.result)

    if outcome.needs_agent:
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent=outcome.needs_agent,
            prompt_to_target_agent=outcome.prompt_to_target_agent,
            # Findings so far must travel with an escalation: the orchestrator
            # builds `escalation_signatures` from the context keys, and an
            # escalation that adds nothing new is force-failed as a loop.
            output=payload,
        )

    if not outcome.finished:
        return AgentResult(
            status=AgentStatus.CONTINUE,
            output=payload,
            continuation_reason=outcome.continuation_reason,
            retryable=True,
        )

    return AgentResult(status=AgentStatus.COMPLETED, output=payload)


@orchestrator_router.post("/execute", response_model=AgentResult)
async def execute(
    request: AgentRequest,
    http_request: Request,
    service: ReconstructionService = Depends(get_service),
    settings: Settings = Depends(get_settings_dependency),
) -> AgentResult:
    """Reconstruct unresolved regions of the sequence named in the request.

    Always answers with an `AgentResult`, never an HTTP error: the
    orchestrator's router expects one schema back every time and already knows
    how to handle a FAILED status. A 500 with FastAPI's {"detail": ...} body
    would break that contract.
    """
    # The orchestrator's run-wide id. Stable across CONTINUE retries, unlike
    # X-Request-Id which is unique per HTTP attempt - so this is what the
    # checkpoint is keyed on, and losing it would mean never resuming.
    trace_id = http_request.headers.get(settings.observability.correlation_id_header_trace)
    bind_run_context(endpoint="execute", trace_id=trace_id)

    try:
        parsed = ReconstructionRequest.from_agent_request(request.instruction, request.context)
        outcome = await service.reconstruct(parsed, trace_id=trace_id)
        result = _to_agent_result(outcome)

        _log.info(
            "execute_answered",
            status=result.status.value,
            finished=outcome.finished,
            resolved=outcome.result.reconstructed_count,
        )
        return result

    except NoGapsFoundError as error:
        # Not a failure: the caller's assembly is already complete over the
        # region examined, which is a real answer to their question.
        return AgentResult(status=AgentStatus.COMPLETED, output=str(error))

    except InvalidSequenceError as error:
        # The orchestrator can act on this - it names what was missing - so it
        # is reported as a failure with a usable message rather than a 422.
        _log.warning("request_rejected", reason=str(error))
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Reconstruction Agent error: {error}",
            error=str(error),
        )

    except Exception as error:  # noqa: BLE001 - the boundary must not leak exceptions
        _log.exception("reconstruction_failed")
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Reconstruction Agent error: {error}",
            error=str(error),
        )

    finally:
        clear_run_context()


@router.post("", response_model=Envelope[ReconstructionData])
async def create_reconstruction(
    body: ReconstructionRequestBody,
    service: ReconstructionService = Depends(get_service),
) -> Envelope[ReconstructionData]:
    """Reconstruct a sequence, answering in the `{data, meta, error}` envelope.

    Like `/execute`, this returns 200 with a populated `error` rather than an
    HTTP error status: the envelope is the contract, and a client that has to
    branch on both a status code and an error field has two things to get
    wrong instead of one.
    """
    started = time.perf_counter()
    request_id = correlation_id.get()
    bind_run_context(endpoint="create_reconstruction")

    try:
        parsed = ReconstructionRequest.from_agent_request(
            body.instruction,
            {
                "sequence": (
                    {"identifier": body.sequence_id, "residues": body.sequence}
                    if body.sequence
                    else None
                ),
                "accession": body.accession,
                "organism": body.organism,
                "reference_organisms": body.reference_organisms,
                "max_gap_length": body.max_gap_length,
                "min_confidence": body.min_confidence,
            },
        )
        # No trace id: a direct caller has no earlier slice to resume, so the
        # run gets its own thread and completes in one call.
        outcome = await service.reconstruct(parsed)

        return Envelope.ok(
            ReconstructionData(result=outcome.result, finished=outcome.finished),
            request_id=request_id,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    except InvalidSequenceError as error:
        return Envelope.fail(
            ErrorCode.INVALID_SEQUENCE, str(error), request_id=request_id
        )

    except NoGapsFoundError as error:
        return Envelope.fail(ErrorCode.NO_GAPS_FOUND, str(error), request_id=request_id)

    except RateLimitError as error:
        return Envelope.fail(
            ErrorCode.RATE_LIMITED, str(error), retryable=True, request_id=request_id
        )

    except ExternalServiceError as error:
        return Envelope.fail(
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            str(error),
            retryable=error.retryable,
            request_id=request_id,
        )

    except Exception as error:  # noqa: BLE001 - the boundary must not leak exceptions
        _log.exception("reconstruction_failed")
        return Envelope.fail(
            ErrorCode.INTERNAL_ERROR,
            f"Reconstruction failed: {error}",
            request_id=request_id,
        )

    finally:
        clear_run_context()
