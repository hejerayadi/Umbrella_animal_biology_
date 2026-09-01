"""The orchestrator contract.

Every assertion here protects the same invariant: `POST /execute` answers HTTP
200 with a parseable `AgentResult`, whatever happened inside.

That is not defensive style, it is the contract. The orchestrator worker node
turns any non-200, and any body it cannot parse, into a local failure reading
"agent unreachable" - so an honest, well-described error returned as a 422
reaches the user as a transport problem, and the real reason is lost.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.fakes import isolated_settings

from reconstruction_agent.api.dependencies import get_reconstruction_service
from reconstruction_agent.domain.enums import GapStatus, UnresolvedReason
from reconstruction_agent.domain.exceptions import SequenceNotFoundError
from reconstruction_agent.domain.models.request import ReconstructionRequest
from reconstruction_agent.domain.models.result import GapReconstruction, ReconstructionResult
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.main import create_app

CARD_OUTPUT_KEYS = {"reconstruction", "reconstruction_summary", "reconstruction_best_fill"}


class _StubService:
    """Stands in for the reconstruction service, with a scripted outcome."""

    def __init__(self, result: ReconstructionResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _unresolved_result() -> ReconstructionResult:
    """A run that found the gap and honestly could not fill it."""
    return ReconstructionResult(
        request_id="rec_test",
        sequence_accession="NC_003428.1",
        reconstructions=(
            GapReconstruction(
                gap=Gap(gap_id="gap_1", start=10, end=20),
                status=GapStatus.UNRESOLVED,
                unresolved_reason=UnresolvedReason.INSUFFICIENT_GAP_SPANNING_HOMOLOGS,
                explanation="No homologue aligned across the region.",
            ),
        ),
    )


def _client(service: _StubService) -> TestClient:
    app = create_app(isolated_settings())
    app.dependency_overrides[get_reconstruction_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


def _execute(client: TestClient, context: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        "/execute",
        json={"instruction": "Reconstruct the unresolved regions.", "context": context},
        headers={"X-Trace-Id": "trace-1", "X-Request-Id": "req-1"},
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


class TestAlwaysAnswersTheContract:
    def test_a_successful_run_returns_completed(self) -> None:
        client = _client(_StubService(result=_unresolved_result()))
        body = _execute(client, {"sequence_accession": "NC_003428.1"})

        assert body["status"] == "completed"

    def test_an_unfillable_gap_is_completed_not_failed(self) -> None:
        """Refusing to invent a sequence is the agent doing its job.

        FAILED would make the orchestrator report a breakdown to the user for
        what is actually a correct scientific answer.
        """
        client = _client(_StubService(result=_unresolved_result()))
        body = _execute(client, {"sequence_accession": "NC_003428.1"})

        assert body["status"] == "completed"
        gap = body["output"]["reconstruction"]["reconstructions"][0]
        assert gap["status"] == "UNRESOLVED"
        assert gap["unresolved_reason"] == "INSUFFICIENT_GAP_SPANNING_HOMOLOGS"
        assert gap["selected_candidate"] is None

    def test_a_missing_target_fails_without_an_http_error(self) -> None:
        """The context named no sequence, so there is nothing to reconstruct.

        Returned as a FAILED result with a usable message rather than a 4xx,
        which the orchestrator would render as an unreachable agent.
        """
        client = _client(_StubService(result=_unresolved_result()))
        response = client.post("/execute", json={"instruction": "Fix it", "context": {}})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert "accession" in body["error"].lower()

    def test_an_upstream_failure_is_reported_as_failed(self) -> None:
        client = _client(_StubService(error=SequenceNotFoundError("No such record.")))
        response = client.post(
            "/execute", json={"instruction": "x", "context": {"accession": "BAD"}}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    def test_an_unexpected_crash_is_still_a_parseable_result(self) -> None:
        """The boundary must not leak an exception, however unexpected."""
        client = _client(_StubService(error=RuntimeError("something broke")))
        response = client.post(
            "/execute", json={"instruction": "x", "context": {"accession": "NC_1"}}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert body["error"]

    @pytest.mark.parametrize(
        "context",
        [
            {"sequence_accession": "NC_003428.1"},
            {"accession": "NC_003428.1"},
            {"sequence": "ACGTNNNNNACGT"},
            {"sequence": {"residues": "ACGTNNNNNACGT"}},
        ],
    )
    def test_the_target_is_read_from_any_key_other_agents_use(
        self, context: dict[str, Any]
    ) -> None:
        """The context is shared, and the Genome agent seeds several spellings."""
        client = _client(_StubService(result=_unresolved_result()))
        assert _execute(client, context)["status"] == "completed"


class TestOutputContract:
    def test_the_output_carries_the_keys_card_json_declares(self) -> None:
        """Output key names are a cross-agent contract.

        Other agents test for these in the shared context; renaming one breaks
        whoever reads it, silently.
        """
        client = _client(_StubService(result=_unresolved_result()))
        output = _execute(client, {"sequence_accession": "NC_003428.1"})["output"]

        assert set(output) == CARD_OUTPUT_KEYS

    def test_the_output_is_a_dict_so_it_merges_into_shared_context(self) -> None:
        """Only a dict `output` is merged; a string would be dropped."""
        client = _client(_StubService(result=_unresolved_result()))
        output = _execute(client, {"sequence_accession": "NC_003428.1"})["output"]

        assert isinstance(output, dict)

    def test_the_response_is_never_enveloped(self) -> None:
        """`/execute` returns the bare AgentResult, never {data, meta, error}.

        The orchestrator parses this shape directly.
        """
        client = _client(_StubService(result=_unresolved_result()))
        body = _execute(client, {"sequence_accession": "NC_003428.1"})

        assert "data" not in body
        assert "meta" not in body
        assert "status" in body

    def test_the_summary_never_claims_more_than_was_achieved(self) -> None:
        client = _client(_StubService(result=_unresolved_result()))
        output = _execute(client, {"sequence_accession": "NC_003428.1"})["output"]

        assert output["reconstruction_best_fill"] is None
        assert "0 of 1" in output["reconstruction_summary"]

    def test_the_payload_is_json_serialisable(self) -> None:
        """Anything the orchestrator cannot serialise becomes an unusable reply."""
        client = _client(_StubService(result=_unresolved_result()))
        body = _execute(client, {"sequence_accession": "NC_003428.1"})

        json.dumps(body)


class TestInvalidCoordinates:
    def test_malformed_gap_coordinates_are_rejected_with_a_reason(self) -> None:
        """These decide which bases get replaced, so they are checked up front."""
        client = _client(_StubService(result=_unresolved_result()))
        response = client.post(
            "/execute",
            json={
                "instruction": "x",
                "context": {
                    "accession": "NC_1",
                    "target_gaps": [{"start": 100, "end": 50}],
                },
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert "end" in body["error"]
