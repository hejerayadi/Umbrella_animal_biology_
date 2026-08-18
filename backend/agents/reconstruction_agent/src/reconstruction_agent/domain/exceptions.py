"""The agent's exception hierarchy.

One root (`ReconstructionError`) so the API boundary can catch everything this
package raises without also swallowing genuine bugs like `TypeError`.

The split that matters is `retryable`: infrastructure failures the caller could
usefully try again (a rate limit, a timeout) versus domain failures that will
fail identically forever (a sequence with no gaps in it).
"""
from __future__ import annotations


class ReconstructionError(Exception):
    """Base class for every error this agent raises deliberately."""

    retryable: bool = False


# --- Input / domain errors -------------------------------------------------


class InvalidSequenceError(ReconstructionError):
    """The target sequence is missing, empty, or not nucleotide data."""


class NoGapsFoundError(ReconstructionError):
    """The sequence contains nothing to reconstruct.

    Not a failure of the agent: it is a legitimate answer that the caller's
    assembly is already complete over the region examined.
    """


class GapTooLargeError(ReconstructionError):
    """A gap exceeds the configured maximum length.

    Reconstruction quality collapses over long spans, so the agent refuses
    rather than emitting a confident-looking guess.
    """

    def __init__(self, gap_id: str, length: int, maximum: int) -> None:
        super().__init__(
            f"Gap {gap_id} is {length} bases, above the {maximum}-base limit for reconstruction."
        )
        self.gap_id = gap_id
        self.length = length
        self.maximum = maximum


class NoReferencesFoundError(ReconstructionError):
    """No homologous reference sequence could be found for the flanking context."""


class ValidationFailedError(ReconstructionError):
    """A candidate reconstruction failed a hard biological validity check."""


# --- Infrastructure errors -------------------------------------------------


class ExternalServiceError(ReconstructionError):
    """An external service (NCBI, EMBL-EBI, the LLM) did not answer usefully."""

    def __init__(self, service: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
        self.retryable = retryable


class RateLimitError(ExternalServiceError):
    """We exceeded a published request budget. Always worth retrying, later."""

    def __init__(self, service: str, retry_after: float | None = None) -> None:
        detail = f"rate limited{f', retry after {retry_after}s' if retry_after else ''}"
        super().__init__(service, detail, retryable=True)
        self.retry_after = retry_after


class JobTimeoutError(ExternalServiceError):
    """A submitted BLAST/MAFFT job did not finish inside the poll window."""

    def __init__(self, service: str, job_id: str, waited_seconds: float) -> None:
        super().__init__(
            service, f"job {job_id} still running after {waited_seconds:.0f}s", retryable=True
        )
        self.job_id = job_id


class ToolExecutionError(ReconstructionError):
    """A registered tool raised while the graph was running it."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"Tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name


class LLMUnavailableError(ReconstructionError):
    """Planning needed an LLM but none is configured or reachable."""
