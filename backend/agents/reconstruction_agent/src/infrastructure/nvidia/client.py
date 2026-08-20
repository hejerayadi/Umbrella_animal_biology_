"""NVIDIA NIM access for Evo 2, the genomic foundation model.

Evo 2 provides a kind of evidence homology cannot: what a model trained on
genomes expects to see in a given context. A consensus that aligns cleanly but
contradicts that expectation is exactly the failure the alignment pipeline
cannot see on its own.

## What this API actually offers

Two endpoints, and the distinction shapes everything below:

- `POST /forward` returns *embeddings* for named layers, base64-encoded `.npy`
  inside a zip. It does not return logits - asking for an `output_layers` of
  `["logits"]` is rejected with "StripedHyena has no attribute `logits`". So
  there is no way to read the likelihood of a sequence you already have.
- `POST /generate` returns a continuation of a prompt, and with
  `enable_sampled_probs` the model's confidence at each generated position.

There is no `/score` or `/likelihood` endpoint. Scoring an arbitrary candidate
directly is therefore not possible, and this client does not pretend otherwise:
it exposes generation, and the tool layer turns that into an agreement check.

Note on the catalog name: NGC lists this as `evo2-40b-forward`, but that is the
`/forward` endpoint of the `evo2-40b` model, not a separate model path.
`https://.../arc/evo2-40b-forward` is a 404.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from configuration.logging import get_logger
from configuration.settings import NvidiaSettings
from domain.exceptions import ExternalServiceError
from infrastructure.http.client import ServiceClient
from infrastructure.http.retry import RetryPolicy

_log = get_logger(__name__)

# The API rejects temperature <= 0, so "greedy" is approximated by a very low
# temperature plus top_k=1. Determinism matters: the evaluation suite compares
# runs, and a sampled continuation would make every run differ.
_NEAR_GREEDY_TEMPERATURE = 0.01
_GREEDY_TOP_K = 1


@dataclass(frozen=True, slots=True)
class Continuation:
    """What Evo 2 expects to follow a prompt."""

    sequence: str
    #: Model confidence at each generated position, 0..1, one per base.
    sampled_probs: list[float]

    @property
    def mean_confidence(self) -> float:
        """How sure the model was, averaged over the continuation."""
        if not self.sampled_probs:
            return 0.0
        return sum(self.sampled_probs) / len(self.sampled_probs)


class Evo2Client:
    """Generates nucleotide continuations with Evo 2 via NVIDIA NIM."""

    def __init__(self, settings: NvidiaSettings) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service="nvidia_evo2",
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            # Self-imposed pacing: NIM endpoints are metered per key and a
            # multi-gap run would otherwise fire these in one burst.
            requests_per_second=1.0,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=2.0),
            headers={
                "Authorization": f"Bearer {settings.api_key or ''}",
                "Content-Type": "application/json",
            },
        )

    def _require_key(self) -> None:
        if not self._settings.configured:
            raise ExternalServiceError(
                "nvidia_evo2",
                "NVIDIA_API_KEY is not set; Evo 2 is unavailable.",
                retryable=False,
            )

    async def generate(
        self,
        prompt: str,
        *,
        num_tokens: int,
        temperature: float = _NEAR_GREEDY_TEMPERATURE,
        top_k: int = _GREEDY_TOP_K,
    ) -> Continuation:
        """What Evo 2 predicts follows `prompt`, with per-position confidence.

        Used to ask "what would a genome model put here?" so a candidate
        reconstruction can be compared against it. The agent never adopts this
        output as sequence: a generated base has no traceable evidence behind
        it, and every base this agent reports must be attributable to a
        reference.
        """
        self._require_key()

        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")

        response = await self._client.request(
            "POST",
            "/generate",
            json={
                "sequence": prompt,
                "num_tokens": num_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "enable_sampled_probs": True,
            },
        )
        payload = self._json(response.text)

        sequence = payload.get("sequence")
        if not isinstance(sequence, str):
            raise ExternalServiceError(
                "nvidia_evo2", f"no sequence in Evo 2 response: {list(payload)}"
            )

        raw_probs = payload.get("sampled_probs") or []
        probs = [float(value) for value in raw_probs if isinstance(value, (int, float))]

        return Continuation(sequence=sequence.upper(), sampled_probs=probs)

    async def embeddings(self, sequence: str, *, layer: str = "blocks.28.mlp.l3") -> str:
        """Raw layer embeddings for `sequence`, base64 `.npy` inside a zip.

        Returned undecoded: decoding needs numpy, which this agent does not
        otherwise depend on, and nothing here consumes embeddings yet. Present
        because `/forward` is the endpoint the NGC catalog advertises, and a
        caller reaching for it should find it rather than reinvent the request.
        """
        self._require_key()

        response = await self._client.request(
            "POST",
            "/forward",
            json={"sequence": sequence, "output_layers": [layer]},
        )
        payload = self._json(response.text)

        data = payload.get("data")
        if not isinstance(data, str):
            raise ExternalServiceError(
                "nvidia_evo2", f"no embedding data in Evo 2 response: {list(payload)}"
            )
        return data

    @staticmethod
    def _json(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ExternalServiceError(
                "nvidia_evo2", f"expected JSON, got {raw[:200]!r}"
            ) from error

        if not isinstance(payload, dict):
            raise ExternalServiceError(
                "nvidia_evo2", f"expected a JSON object, got {type(payload)}"
            )
        return payload

    async def aclose(self) -> None:
        await self._client.aclose()
