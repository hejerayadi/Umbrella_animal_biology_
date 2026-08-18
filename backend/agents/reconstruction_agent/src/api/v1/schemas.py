"""The HTTP wire shapes for v1.

Two families live here, and the difference matters:

1. `AgentRequest` / `AgentResult` reproduce the repo-wide agent contract that
   `backend/orchestrator/schema.py` parses. They are duplicated here rather
   than imported because this agent runs in its own `.venv` and must not
   depend on the `backend` package - that isolation is what lets it pin its
   own dependencies. Changing them means changing the orchestrator too.

2. The `*Data` models are payloads carried inside the `{data, meta, error}`
   envelope on the v1 resource endpoints. They are ours to evolve.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contracts.output import ReconstructionResult

# --- Orchestrator contract (not enveloped) ---------------------------------


class AgentStatus(str, Enum):
    """Mirrors `backend/orchestrator/schema.py`."""

    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


class AgentRequest(BaseModel):
    """What the orchestrator POSTs to /execute."""

    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """What every /execute call answers with, success or failure.

    Deliberately the only response shape for that endpoint: the orchestrator's
    router expects one schema back every time and already handles a FAILED
    status, so errors are reported in this envelope rather than as an HTTP
    error body.

    The last three fields exist because `backend/orchestrator/schema.py` parses
    them. `retryable` in particular is load-bearing: a CONTINUE without it is
    turned straight into FAILED by the orchestrator's worker node.
    """

    status: AgentStatus
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    output: Any | None = None
    #: Why the agent stopped short, shown to the user if the retries run out.
    continuation_reason: str | None = None
    #: Whether re-calling this agent unchanged could finish the work. Only
    #: meaningful with CONTINUE; the orchestrator retries three times.
    retryable: bool = False
    error: str | None = None


# --- v1 resource payloads (carried inside `data`) --------------------------


class ReconstructionRequestBody(BaseModel):
    """Body of `POST /api/v1/reconstructions`.

    The typed, self-documenting way in - as opposed to `/execute`, whose
    loose `context` dict exists because the orchestrator shares it between
    agents.
    """

    model_config = ConfigDict(frozen=True)

    instruction: str = Field(
        default="Reconstruct the unresolved regions of this sequence.",
        description="What the caller wants done, in natural language.",
    )
    sequence: str | None = Field(
        default=None, description="Nucleotide residues, IUPAC codes permitted."
    )
    sequence_id: str = Field(default="input_sequence")
    accession: str | None = Field(
        default=None, description="NCBI accession to fetch the target from instead."
    )
    organism: str | None = None
    reference_organisms: list[str] = Field(default_factory=list)
    max_gap_length: int | None = Field(default=None, ge=1)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ServiceStatus(BaseModel):
    """Whether one external dependency is usable right now."""

    model_config = ConfigDict(frozen=True)

    configured: bool
    detail: str | None = None


class HealthData(BaseModel):
    """Payload of `GET /api/v1/health`.

    Reports capability, not just liveness: the common failure here is a
    deployment that starts cleanly with no EMBL-EBI contact address and then
    fails every reconstruction.
    """

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    agent: str = "reconstruction_agent"
    version: str = "0.1.0"
    environment: str = "development"
    tools: list[str] = Field(default_factory=list)
    llm_provider: str = "none"
    services: dict[str, ServiceStatus] = Field(default_factory=dict)


class ReconstructionData(BaseModel):
    """Payload of `POST /api/v1/reconstructions`."""

    model_config = ConfigDict(frozen=True)

    result: ReconstructionResult
    #: False when the agent ran out of wall clock mid-run. Direct callers get
    #: the partial result rather than a retry, because only the orchestrator
    #: has the CONTINUE machinery to resume it.
    finished: bool = True
