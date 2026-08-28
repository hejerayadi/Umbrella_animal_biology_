"""`POST /execute` - the endpoint the Umbrella orchestrator calls.

Mounted at the root and answering with the bare `AgentResult`, never the
`{data, meta, error}` envelope. The orchestrator parses that shape directly, so
wrapping it would break every reconstruction the platform performs.

It also always answers HTTP 200. The orchestrator's worker node turns any
non-200, and any body it cannot parse, into a local failure with a message
about the agent being unreachable - which would hide a perfectly well-described
error behind a transport-sounding one. So every exception becomes a FAILED
result carrying its own explanation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from reconstruction_agent.api.dependencies import (
    get_reconstruction_service,
    get_settings_dependency,
)
from reconstruction_agent.api.v1.mappers.reconstruction_mapper import to_agent_output
from reconstruction_agent.api.v1.schemas.agent import AgentRequest, AgentResult, AgentStatus
from reconstruction_agent.config.settings import Settings
from reconstruction_agent.domain.exceptions import InvalidRequestError, ReconstructionError
from reconstruction_agent.domain.models.request import ReconstructionRequest
from reconstruction_agent.observability.logger import (
    bind_run_context,
    clear_run_context,
    get_logger,
)
from reconstruction_agent.services.reconstruction.escalation import (
    GENOME_AGENT,
    build_escalation,
    should_escalate,
)
from reconstruction_agent.services.reconstruction.reconstruction_service import (
    ReconstructionService,
)

_log = get_logger(__name__)

router = APIRouter(tags=["orchestrator"])


@router.post("/execute", response_model=AgentResult)
async def execute(
    request: AgentRequest,
    http_request: Request,
    service: ReconstructionService = Depends(get_reconstruction_service),
    settings: Settings = Depends(get_settings_dependency),
) -> AgentResult:
    """Reconstruct the unresolved regions of the sequence named in the request."""
    trace_id = http_request.headers.get(settings.observability.trace_id_header)
    bind_run_context(endpoint="execute", trace_id=trace_id)

    try:
        parsed = ReconstructionRequest.from_agent_request(request.instruction, request.context)
        result = await service.reconstruct(parsed)

        # COMPLETED even when nothing was reconstructed. "These regions cannot
        # be resolved from the available evidence" is an answer to the question
        # that was asked, and reporting it as a failure would make the
        # responder tell the user the agent broke.
        return AgentResult(status=AgentStatus.COMPLETED, output=to_agent_output(result))

    except InvalidRequestError as error:
        # A request naming a species but no sequence is not malformed - it is
        # under-specified, and another agent can supply what is missing. That
        # is a hand-off, not a failure.
        if should_escalate(request.context):
            prompt, findings = build_escalation(request.instruction, request.context, str(error))
            _log.info("escalating_for_target", target_agent=GENOME_AGENT, reason=str(error))
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent=GENOME_AGENT,
                prompt_to_target_agent=prompt,
                # Carries a context key that was not there before. The
                # orchestrator force-fails an escalation whose context
                # signature is unchanged, reading it as a loop.
                output=findings,
            )

        # Nobody available can supply what this request lacks, so it comes back
        # as a failure with a usable message rather than a 422.
        _log.warning("request_rejected", reason=str(error))
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Reconstruction Agent error: {error}",
            error=str(error),
        )

    except ReconstructionError as error:
        _log.warning("reconstruction_failed", code=error.code.value, reason=str(error))
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Reconstruction Agent error: {error}",
            error=str(error),
            retryable=error.retryable,
        )

    except Exception as error:  # noqa: BLE001 - the boundary must not leak exceptions
        _log.exception("reconstruction_crashed")
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Reconstruction Agent error: {error}",
            error=str(error),
        )

    finally:
        clear_run_context()
