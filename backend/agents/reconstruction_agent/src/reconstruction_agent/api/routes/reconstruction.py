"""The agent's single endpoint: POST /execute.

Communication only - no business logic. It validates the orchestrator's
request, hands it to `ReconstructionService`, and wraps whatever comes back in
an `AgentResult`. The orchestrator is the only caller; the frontend never
reaches an agent directly.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...application.reconstruction_service import ReconstructionService
from ...application.result_builder import ResultBuilder
from ...configuration.logging import get_logger
from ...contracts.input import ReconstructionRequest
from ...domain.exceptions import InvalidSequenceError, NoGapsFoundError
from ..dependencies import get_service
from ..schemas import AgentRequest, AgentResult, AgentStatus

_log = get_logger(__name__)

router = APIRouter(tags=["reconstruction"])


@router.post("/execute", response_model=AgentResult)
async def execute(
    request: AgentRequest,
    service: ReconstructionService = Depends(get_service),
) -> AgentResult:
    """Reconstruct unresolved regions of the sequence named in the request.

    Always answers with an `AgentResult`, never an HTTP error: the
    orchestrator's router expects one schema back every time and already knows
    how to handle a FAILED status. A 500 with FastAPI's {"detail": ...} body
    would break that contract.
    """
    try:
        parsed = ReconstructionRequest.from_agent_request(request.instruction, request.context)
        result = await service.reconstruct(parsed)

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=ResultBuilder.to_output_payload(result),
        )

    except NoGapsFoundError as error:
        # Not a failure: the caller's assembly is already complete over the
        # region examined, which is a real answer to their question.
        return AgentResult(status=AgentStatus.COMPLETED, output=str(error))

    except InvalidSequenceError as error:
        # The orchestrator can act on this - it names what was missing - so it
        # is reported as a failure with a usable message rather than a 422.
        _log.warning("Rejected a reconstruction request: %s", error)
        return AgentResult(
            status=AgentStatus.FAILED, output=f"Reconstruction Agent error: {error}"
        )

    except Exception as error:  # noqa: BLE001 - the boundary must not leak exceptions
        _log.exception("Reconstruction failed.")
        return AgentResult(
            status=AgentStatus.FAILED, output=f"Reconstruction Agent error: {error}"
        )
