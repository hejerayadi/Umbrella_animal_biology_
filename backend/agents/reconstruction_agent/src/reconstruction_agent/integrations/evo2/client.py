"""Evo 2 on NVIDIA NIM: generate a continuation of the left flank.

The only capability this endpoint actually offers. `/forward` returns
embeddings, not logits, so an existing candidate cannot be scored - see
`models.py` for the measurements behind that. The agent therefore asks Evo 2 to
write its own continuation and compares, which is an agreement check.

Unavailability is a first-class outcome, not an error. Evo 2 is consulted to
break a tie between candidates that are already supported by real homology, so
a run that cannot reach it loses a tiebreaker and keeps everything else. The
client says so plainly rather than raising, and the confidence engine records
`None` for the score - which it treats differently from a measured zero.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.config.settings import NvidiaSettings
from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import ExternalServiceError
from reconstruction_agent.integrations.evo2.models import Evo2Generation, Evo2GenerationRequest
from reconstruction_agent.integrations.http.client import ServiceClient
from reconstruction_agent.integrations.http.retry import RetryPolicy
from reconstruction_agent.observability.logger import get_logger

SERVICE = "evo2"
_log = get_logger(__name__)

#: Characters a returned continuation may contain. Anything else is stripped:
#: the endpoint occasionally emits whitespace, and a stray character would make
#: a base-by-base comparison silently wrong rather than loudly.
_BASES = frozenset("ACGT")

#: The longest continuation worth asking for. Evo 2 drifts on long generations,
#: and a gap this size is not being settled by a tiebreaker anyway.
MAX_GENERATION_BASES = 500


class Evo2Client:
    """Generates a continuation of a prompt sequence."""

    def __init__(self, settings: NvidiaSettings, *, timeout: float | None = None) -> None:
        self._settings = settings
        self._client: ServiceClient | None = None
        self._timeout = timeout or settings.timeout_seconds

    @property
    def available(self) -> bool:
        """Whether Evo 2 can be reached at all.

        Checked by callers before spending a budget slot, so an unconfigured
        deployment costs nothing rather than costing a failed call.
        """
        return bool(self._settings.configured)

    def _service(self) -> ServiceClient:
        """The HTTP client, built on first use.

        Lazy because most runs never consult Evo 2: arbitration is conditional,
        so a client constructed eagerly would open a connection pool for a
        service the run will not call.
        """
        if self._client is None:
            assert self._settings.api_key is not None  # guarded by `available`
            self._client = ServiceClient(
                service=SERVICE,
                base_url=self._settings.base_url,
                timeout=self._timeout,
                # One request at a time. NIM is rate-limited per key and this
                # is a tiebreaker, not a throughput path.
                requests_per_second=1.0,
                retry_policy=RetryPolicy(max_attempts=2, initial_delay=2.0),
                headers={
                    "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def generate(self, prompt: str, *, num_tokens: int) -> Evo2Generation:
        """Continue `prompt` for `num_tokens` bases."""
        if not self.available:
            raise ExternalServiceError(
                SERVICE,
                "NVIDIA_API_KEY is not configured, so Evo 2 cannot be consulted.",
                code=ErrorCode.EVO2_UNAVAILABLE,
            )
        if num_tokens > MAX_GENERATION_BASES:
            raise ExternalServiceError(
                SERVICE,
                f"A {num_tokens}-base continuation exceeds the {MAX_GENERATION_BASES}-base "
                "limit beyond which Evo 2 output is not usable as a tiebreaker.",
                code=ErrorCode.EVO2_UNAVAILABLE,
            )

        request = Evo2GenerationRequest(sequence=prompt, num_tokens=num_tokens)
        # `json_body`, not `json`: ServiceClient renames the argument so that
        # `json` stays available as a response helper on the same class.
        response = await self._service().post("/generate", json_body=request.to_payload())
        return _parse(response.json(), expected=num_tokens)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def _parse(payload: Any, *, expected: int) -> Evo2Generation:
    """Read a NIM response, tolerating the shapes it actually returns.

    The continuation has appeared under several keys across deployments, and a
    missing one is not worth failing a run over: the caller treats an empty
    generation as "no arbitration available", which is the same outcome as the
    service being down.
    """
    if not isinstance(payload, dict):
        return Evo2Generation()

    raw = ""
    for key in ("sequence", "generated_sequence", "text", "output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            raw = value
            break

    cleaned = "".join(char for char in raw.upper() if char in _BASES)
    if len(cleaned) > expected:
        # NIM sometimes echoes part of the prompt back. Only the requested
        # window is the continuation.
        cleaned = cleaned[:expected]

    probs = payload.get("sampled_probs")
    sampled: tuple[float, ...] = ()
    if isinstance(probs, list):
        sampled = tuple(
            float(value) for value in probs[: len(cleaned)] if isinstance(value, int | float)
        )

    if not cleaned:
        _log.info("evo2_returned_no_sequence", keys=sorted(payload))

    return Evo2Generation(sequence=cleaned, sampled_probs=sampled)
