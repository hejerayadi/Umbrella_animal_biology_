"""The Evo 2 client's wire layer.

Every other Evo 2 test replaces the whole client with a double, which is what
let a real defect through: the client called `ServiceClient.post(json=...)`
while the method takes `json_body`, and nothing exercised that call until a
live run crashed on it. Faking one level too high hides the seam being tested.

So the double here is the HTTP transport, not the client - the client's own
argument-passing and parsing are what run.
"""

from __future__ import annotations

from typing import Any

import pytest

from reconstruction_agent.config.settings import NvidiaSettings
from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import ExternalServiceError
from reconstruction_agent.integrations.evo2.client import MAX_GENERATION_BASES, Evo2Client, _parse


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _Transport:
    """Records exactly how the client called it."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.path: str | None = None
        self.kwargs: dict[str, Any] = {}

    async def post(self, path: str, **kwargs: Any) -> _Response:
        self.path = path
        self.kwargs = kwargs
        return _Response(self._payload)


def _client(payload: Any) -> tuple[Evo2Client, _Transport]:
    settings = NvidiaSettings(_env_file=None, api_key="nvapi-test")  # type: ignore[arg-type]
    client = Evo2Client(settings)
    transport = _Transport(payload)
    client._client = transport  # type: ignore[assignment]
    return client, transport


class TestTheRequestReachesTheEndpointCorrectly:
    async def test_the_body_is_sent_as_json_body(self) -> None:
        """The defect this file exists for: `json=` is silently wrong, and the
        failure only surfaces at call time."""
        client, transport = _client({"sequence": "ACGT"})
        await client.generate("ACGT" * 50, num_tokens=4)

        assert transport.path == "/generate"
        assert "json_body" in transport.kwargs, "the payload must reach ServiceClient"
        assert "json" not in transport.kwargs

    async def test_the_payload_carries_the_measured_parameters(self) -> None:
        """NIM rejects `temperature <= 0`, so near-greedy is expressed as a
        tiny positive temperature with top_k=1."""
        client, transport = _client({"sequence": "ACGT"})
        await client.generate("ACGT" * 50, num_tokens=4)
        payload = transport.kwargs["json_body"]

        assert payload["num_tokens"] == 4
        assert payload["temperature"] > 0.0
        assert payload["top_k"] == 1
        assert payload["enable_sampled_probs"] is True


class TestRefusalsThatCostNothing:
    async def test_an_unconfigured_key_raises_before_any_call(self) -> None:
        client = Evo2Client(NvidiaSettings(_env_file=None))
        assert not client.available

        with pytest.raises(ExternalServiceError) as caught:
            await client.generate("ACGT", num_tokens=4)
        assert caught.value.code is ErrorCode.EVO2_UNAVAILABLE

    async def test_a_region_beyond_the_generation_limit_is_refused(self) -> None:
        """Evo 2 drifts on long generations; a long fabricated fill is worse
        than an honest refusal."""
        client, transport = _client({"sequence": "ACGT"})

        with pytest.raises(ExternalServiceError):
            await client.generate("ACGT", num_tokens=MAX_GENERATION_BASES + 1)
        assert transport.path is None


class TestParsingWhatNimActuallyReturns:
    def test_the_continuation_is_read_from_any_of_its_key_names(self) -> None:
        for key in ("sequence", "generated_sequence", "text", "output"):
            assert _parse({key: "ACGT"}, expected=4).sequence == "ACGT"

    def test_non_base_characters_are_stripped(self) -> None:
        """A stray newline would make a base-by-base comparison silently wrong."""
        assert _parse({"sequence": "ac gt\n"}, expected=4).sequence == "ACGT"

    def test_an_echoed_prompt_is_trimmed_to_the_requested_window(self) -> None:
        assert _parse({"sequence": "ACGTACGT"}, expected=4).sequence == "ACGT"

    def test_sampling_probabilities_are_kept_when_present(self) -> None:
        parsed = _parse({"sequence": "ACGT", "sampled_probs": [0.9, 0.8, 0.7, 0.6]}, expected=4)

        assert parsed.sampled_probs == (0.9, 0.8, 0.7, 0.6)
        assert parsed.mean_confidence is not None

    def test_a_response_with_no_sequence_parses_to_nothing_usable(self) -> None:
        """Treated by the caller as "no answer", which is the same outcome as
        the service being down - not an error that ends the run."""
        assert _parse({"detail": "model busy"}, expected=4).sequence == ""

    def test_a_non_dict_response_does_not_raise(self) -> None:
        assert _parse(["unexpected"], expected=4).sequence == ""
