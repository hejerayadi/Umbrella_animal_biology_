"""The hand-off as the orchestrator actually performs it.

Not a hand-written approximation of the payload: this reproduces the exact call
`backend/orchestrator/langgraph/nodes/worker_node.py` makes - the two headers,
the `{instruction, context}` body, a bare `AgentResult` parsed back - and feeds
it the payload `genome_agent/orchestrator_adapter.py` builds when it detects a
Scaffold assembly.

Everything asserted here is a cross-agent contract. Breaking any of it breaks
the run for reasons that surface three services away from the cause.

The service beneath is stubbed. What is under test is the boundary - parsing,
status mapping, output keys - and running it against real NCBI would make a
contract test depend on a BLAST queue.
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

#: The context the Genome Agent emits on a reconstruction hand-off. Field for
#: field what `orchestrator_adapter.to_result()` puts in `AgentResult.output`,
#: with the coordinates in NCBI's own convention: 1-based, end inclusive.
GENOME_AGENT_CONTEXT: dict[str, Any] = {
    "scientific_name": "Ursus maritimus",
    "assembly_id": "GCF_000687225.1",
    "sequence_accession": "NW_007907101",
    "assembly_level": "Scaffold",
    "target_gaps": [
        {
            "start": 125431,
            "end": 125475,
            "length": 45,
            "left_flank": "ACTGGCATTACGGATCCATTGCAGTTACGGATCCATTGCAGTTAC",
            "right_flank": "GCTAAGCATTACGGATCCATTGCAGTTACGGATCCATTGCAGTTA",
        }
    ],
}

#: The instruction the adapter sends. A constant on their side, so a constant
#: here: the contract lives on the context payload, not on free text.
GENOME_AGENT_INSTRUCTION = "Reconstruct the selected unresolved regions."

#: The whole vocabulary `AgentStatus` accepts. The orchestrator fails the run
#: on anything else.
AGENT_STATUSES = {"completed", "needs_agent", "continue", "failed"}

CARD_OUTPUT_KEYS = {"reconstruction", "reconstruction_summary", "reconstruction_best_fill"}


class _StubService:
    """Stands in for the reconstruction service, with a scripted outcome."""

    def __init__(
        self, result: ReconstructionResult | None = None, error: Exception | None = None
    ) -> None:
        self._result = result
        self._error = error
        self.seen: ReconstructionRequest | None = None

    async def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        self.seen = request
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _resolved_result() -> ReconstructionResult:
    return ReconstructionResult(
        request_id="rec_test",
        sequence_accession="NW_007907101",
        assembly_id="GCF_000687225.1",
        reconstructions=(
            GapReconstruction(
                gap=Gap(gap_id="gap_1", start=125430, end=125475),
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


def _execute(client: TestClient, instruction: str, context: Any) -> Any:
    """One orchestrator hop, headers included."""
    return client.post(
        "/execute",
        headers={"X-Trace-Id": "trace-genome-handoff", "X-Request-Id": "req-1"},
        json={"instruction": instruction, "context": context},
    )


class TestTheGenomeAgentHandoff:
    def test_the_handoff_payload_is_accepted_and_answered(self) -> None:
        response = _execute(
            _client(_StubService(_resolved_result())),
            GENOME_AGENT_INSTRUCTION,
            GENOME_AGENT_CONTEXT,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] in AGENT_STATUSES
        # Never the envelope, and never FastAPI's error shape: the orchestrator
        # parses a bare AgentResult and fails the run on anything else.
        assert "data" not in body
        assert "detail" not in body

    def test_every_field_the_genome_agent_sends_is_read(self) -> None:
        """A key the adapter sets and this agent ignores is a silent hand-off
        failure: the run proceeds with less information than was offered."""
        service = _StubService(_resolved_result())
        _execute(_client(service), GENOME_AGENT_INSTRUCTION, GENOME_AGENT_CONTEXT)

        assert service.seen is not None
        assert service.seen.sequence_accession == "NW_007907101"
        assert service.seen.scientific_name == "Ursus maritimus"
        assert service.seen.assembly_id == "GCF_000687225.1"
        assert service.seen.assembly_level == "Scaffold"
        assert len(service.seen.target_gaps) == 1

    def test_the_caller_s_flanks_and_length_are_kept(self) -> None:
        """Redundant with the coordinates, and therefore worth reading: two
        descriptions of the same region that disagree are how a coordinate
        error announces itself."""
        service = _StubService(_resolved_result())
        _execute(_client(service), GENOME_AGENT_INSTRUCTION, GENOME_AGENT_CONTEXT)

        assert service.seen is not None
        requested = service.seen.target_gaps[0]
        assert requested.left_flank.startswith("ACTGGCATTACG")
        assert requested.right_flank.startswith("GCTAAGCATTACG")

    def test_the_output_keys_the_card_promises_are_present(self) -> None:
        """`card.json` declares these and other agents read them out of the
        shared context. A contract, not a response format."""
        response = _execute(
            _client(_StubService(_resolved_result())),
            GENOME_AGENT_INSTRUCTION,
            GENOME_AGENT_CONTEXT,
        )
        output = response.json().get("output")

        assert isinstance(output, dict)
        assert CARD_OUTPUT_KEYS.issubset(output)

    def test_a_failed_gap_finder_does_not_break_the_handoff(self) -> None:
        """Gap finding is non-fatal upstream: the adapter still escalates, with
        empty gaps and a `warnings` key this agent has never seen before."""
        context = {
            **GENOME_AGENT_CONTEXT,
            "target_gaps": [],
            "warnings": ["gap finding failed: NCBI unreachable"],
        }
        response = _execute(
            _client(_StubService(_resolved_result())), GENOME_AGENT_INSTRUCTION, context
        )

        assert response.status_code == 200
        assert response.json()["status"] in AGENT_STATUSES


class TestEveryFailurePathStillAnswersTheOrchestrator:
    """A non-200, or a body the worker node cannot parse, reaches the user as
    "agent unreachable" - and the real reason is lost."""

    def test_an_unknown_accession(self) -> None:
        service = _StubService(error=SequenceNotFoundError("no such record"))
        response = _execute(_client(service), GENOME_AGENT_INSTRUCTION, GENOME_AGENT_CONTEXT)

        assert response.status_code == 200
        assert response.json()["status"] in AGENT_STATUSES

    def test_a_context_naming_no_target_at_all(self) -> None:
        response = _execute(
            _client(_StubService(_resolved_result())), "Fill in the polar bear genome.", {}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    def test_a_malformed_body(self) -> None:
        client = _client(_StubService(_resolved_result()))
        response = client.post("/execute", json={"instruction": 42, "context": "not a dict"})

        assert response.status_code == 200
        assert response.json()["status"] in AGENT_STATUSES

    def test_inverted_gap_coordinates(self) -> None:
        context = {**GENOME_AGENT_CONTEXT, "target_gaps": [{"start": 500, "end": 100}]}
        response = _execute(
            _client(_StubService(_resolved_result())), GENOME_AGENT_INSTRUCTION, context
        )

        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    def test_an_unexpected_internal_error(self) -> None:
        service = _StubService(error=RuntimeError("undefined behaviour"))
        response = _execute(_client(service), GENOME_AGENT_INSTRUCTION, GENOME_AGENT_CONTEXT)

        assert response.status_code == 200
        assert response.json()["status"] == "failed"


@pytest.mark.parametrize("field", ["scientific_name", "sequence_accession", "assembly_id"])
def test_the_documented_payload_carries_every_field(field: str) -> None:
    """This payload is quoted in the README; drift against the adapter's own
    shape is worth catching here rather than in a live run."""
    encoded = json.loads(
        json.dumps({"instruction": GENOME_AGENT_INSTRUCTION, "context": GENOME_AGENT_CONTEXT})
    )
    assert encoded["context"][field]


class TestTheContractHoldsBeforeTheEndpointRuns:
    """FastAPI validates the request body before dispatching, so a try/except
    inside `execute` cannot see a schema failure. These paths are covered by
    the exception handlers instead, and the distinction is worth a test: the
    malformed-body case returned `{"detail": ...}` with a 422 until it was."""

    def test_a_malformed_body_answers_with_an_agent_result(self) -> None:
        client = _client(_StubService(_resolved_result()))
        response = client.post("/execute", json={"instruction": 42, "context": "not a dict"})
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "failed"
        assert "detail" not in body
        assert body["error"]

    def test_a_body_missing_every_field_answers_with_an_agent_result(self) -> None:
        client = _client(_StubService(_resolved_result()))
        response = client.post("/execute", json={})

        assert response.status_code == 200
        assert response.json()["status"] in AGENT_STATUSES

    def test_the_enveloped_api_still_reports_a_schema_failure_properly(self) -> None:
        """The exception is for `/execute` only: every other client wants the
        envelope and the real HTTP status."""
        client = _client(_StubService(_resolved_result()))
        response = client.post("/api/v1/reconstructions", json={"instruction": 42})

        assert response.status_code in {404, 422}
        if response.status_code == 422:
            assert "error" in response.json()
