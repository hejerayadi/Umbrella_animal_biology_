"""The agent's exception hierarchy, rooted at a single class.

One root means the HTTP boundary can catch `ReconstructionError` and produce a
correct, structured response for every failure the agent knows how to describe,
while anything else escaping is by definition an unexpected bug and becomes
`INTERNAL_ERROR`.

Every exception carries a stable `ErrorCode` and a `retryable` flag, because
the caller's two questions are always "what went wrong" and "is it worth trying
again". Deciding that here, where the cause is known, beats re-deriving it from
a message at the boundary.

Note what is *not* modelled as an exception: insufficient evidence. A gap that
cannot be reconstructed is a result carrying an `UnresolvedReason`, and raising
for it would turn a valid scientific answer into an HTTP failure.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.domain.enums import ErrorCode


class ReconstructionError(Exception):
    """Base class for every failure this agent can describe."""

    #: Overridden by subclasses; the code reported when none is passed.
    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    #: Whether the same request could plausibly succeed on a later attempt.
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.retryable = self.default_retryable if retryable is None else retryable
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        return self.message


class InvalidRequestError(ReconstructionError):
    """The request could not be understood well enough to start work."""

    default_code = ErrorCode.INVALID_REQUEST


class InvalidGapCoordinatesError(InvalidRequestError):
    """A gap's coordinates are impossible, or fall outside the sequence."""

    default_code = ErrorCode.INVALID_GAP_COORDINATES


class NotFoundError(ReconstructionError):
    """A named record does not exist at the provider."""

    default_code = ErrorCode.SEQUENCE_NOT_FOUND


class SequenceNotFoundError(NotFoundError):
    """The requested sequence accession could not be retrieved."""

    default_code = ErrorCode.SEQUENCE_NOT_FOUND


class AssemblyNotFoundError(NotFoundError):
    """The requested assembly could not be resolved."""

    default_code = ErrorCode.ASSEMBLY_NOT_FOUND


class ExternalServiceError(ReconstructionError):
    """An upstream biological service failed.

    Carries the service name so a log line or an error payload can say which
    one, without every call site formatting that into the message itself.
    """

    default_code = ErrorCode.UPSTREAM_UNAVAILABLE
    default_retryable = True

    def __init__(
        self,
        service: str,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"{service}: {message}",
            code=code,
            retryable=retryable,
            details={"service": service, **(details or {})},
        )
        self.service = service


class RateLimitError(ExternalServiceError):
    """An upstream service asked us to slow down.

    `retry_after` is the server's own instruction and outranks any backoff
    curve we would otherwise compute.
    """

    default_code = ErrorCode.RATE_LIMITED
    default_retryable = True

    def __init__(
        self,
        service: str,
        message: str = "rate limited",
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(service, message, details=details)
        self.retry_after = retry_after


class ServiceTimeoutError(ExternalServiceError):
    """An upstream service did not answer inside its allotted time."""

    default_retryable = True


class BudgetExhaustedError(ReconstructionError):
    """The run consumed its allowance for a resource before finishing.

    Raised only where a caller must stop; the graph normally *checks* the budget
    and finalises gracefully rather than letting this propagate.
    """

    default_code = ErrorCode.BUDGET_EXHAUSTED


class DeadlineExceededError(ReconstructionError):
    """The run's wall-clock deadline passed.

    Like `BudgetExhaustedError`, this is a last resort: the normal path notices
    the deadline approaching and finalises with the evidence in hand, which is
    how partial results survive.
    """

    default_code = ErrorCode.DEADLINE_EXCEEDED
