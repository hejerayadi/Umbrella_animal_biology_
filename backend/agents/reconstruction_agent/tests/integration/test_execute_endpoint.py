"""The two HTTP contracts.

`POST /execute` belongs to the orchestrator and must keep returning the bare
`AgentResult`. `POST /api/v1/reconstructions` is ours and returns the
`{data, meta, error}` envelope. These are the tests that would catch a change
breaking the rest of the system, so they assert on the shapes as much as on
the science.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import get_service
from api.v1.envelope import ErrorCode
from api.v1.schemas import AgentStatus
from application.reconstruction_service import ReconstructionOutcome
from contracts.output import ReconstructionResult
from domain.exceptions import InvalidSequenceError


class StubService:
    """Stands in for the real service so no external call is made."""

    def __init__(self, outcome: ReconstructionOutcome | Exception) -> None:
        self._outcome = outcome
        self.received_trace_id: str | None = None

    async def reconstruct(
        self, request: object, *, trace_id: str | None = None
    ) -> ReconstructionOutcome:
        self.received_trace_id = trace_id
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


@pytest.fixture
def result() -> ReconstructionResult:
    return ReconstructionResult(
        sequence_id="test_seq",
        organism="Testus organismus",
        original_length=100,
        reconstructed_sequence="ACGT" * 25,
        overall_confidence=0.82,
        summary="Reconstructed 1 region.",
        iterations=2,
        tools_used=["blast_search", "mafft_align"],
    )


@pytest.fixture
def finished(result: ReconstructionResult) -> ReconstructionOutcome:
    return ReconstructionOutcome(result=result, finished=True)


def client_for(service: object) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


class TestOrchestratorExecute:
    """`/execute` must stay exactly as backend/orchestrator/schema.py parses it."""

    def test_returns_completed_with_the_result_payload(
        self, finished: ReconstructionOutcome
    ) -> None:
        response = client_for(StubService(finished)).post(
            "/execute",
            json={"instruction": "Reconstruct the gaps.", "context": {"sequence": "ACGTNNNN"}},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == AgentStatus.COMPLETED.value
        assert body["output"]["reconstruction"]["sequence_id"] == "test_seq"

    def test_is_not_wrapped_in_the_v1_envelope(self, finished: ReconstructionOutcome) -> None:
        """Wrapping this would break every reconstruction the system performs."""
        body = client_for(StubService(finished)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()

        assert set(body) == {
            "status",
            "target_agent",
            "prompt_to_target_agent",
            "output",
            "continuation_reason",
            "retryable",
            "error",
        }
        assert "data" not in body and "meta" not in body

    def test_output_is_namespaced_to_avoid_context_collisions(
        self, finished: ReconstructionOutcome
    ) -> None:
        """The orchestrator merges `output` flat into the context all nine
        agents share, so bare keys like `summary` would collide."""
        output = client_for(StubService(finished)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()["output"]

        assert "reconstruction" in output
        assert output["reconstruction_summary"] == "Reconstructed 1 region."
        # No un-prefixed keys that another agent might also write.
        assert all(key.startswith("reconstruction") for key in output)

    def test_reports_failure_inside_the_envelope_not_as_http_500(self) -> None:
        response = client_for(StubService(RuntimeError("NCBI unreachable"))).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == AgentStatus.FAILED.value
        assert "NCBI unreachable" in body["output"]
        assert body["error"] == "NCBI unreachable"

    def test_accepts_a_request_with_no_context(self, finished: ReconstructionOutcome) -> None:
        response = client_for(StubService(finished)).post(
            "/execute", json={"instruction": "Reconstruct something."}
        )

        assert response.json()["status"] == AgentStatus.COMPLETED.value


class TestContinuation:
    """An unfinished slice must come back as a retryable CONTINUE."""

    def test_unfinished_slice_returns_retryable_continue(
        self, result: ReconstructionResult
    ) -> None:
        outcome = ReconstructionOutcome(
            result=result,
            finished=False,
            continuation_reason="1 of 3 gaps resolved, still working on gap_2.",
        )

        body = client_for(StubService(outcome)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()

        assert body["status"] == AgentStatus.CONTINUE.value
        # Without this the orchestrator converts CONTINUE straight to FAILED.
        assert body["retryable"] is True
        assert "gap_2" in body["continuation_reason"]

    def test_continue_still_carries_the_partial_findings(
        self, result: ReconstructionResult
    ) -> None:
        """Findings gathered before the yield must not be lost if retries run out."""
        outcome = ReconstructionOutcome(result=result, finished=False)

        body = client_for(StubService(outcome)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()

        assert body["output"]["reconstruction"]["sequence_id"] == "test_seq"

    def test_trace_id_header_reaches_the_service(
        self, finished: ReconstructionOutcome
    ) -> None:
        """X-Trace-Id is the checkpoint key; dropping it means never resuming."""
        service = StubService(finished)
        client_for(service).post(
            "/execute",
            json={"instruction": "Reconstruct.", "context": {}},
            headers={"X-Trace-Id": "orchestrator-trace-7"},
        )

        assert service.received_trace_id == "orchestrator-trace-7"

    def test_absent_trace_id_is_not_fatal(self, finished: ReconstructionOutcome) -> None:
        """A direct caller has no earlier slice, so no trace id is correct."""
        service = StubService(finished)
        response = client_for(service).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        )

        assert response.json()["status"] == AgentStatus.COMPLETED.value
        assert service.received_trace_id is None


class TestEscalation:
    """Phylogeny being the blocker becomes a NEEDS_AGENT the resolver can route."""

    def test_needs_agent_names_the_target_and_the_request(
        self, result: ReconstructionResult
    ) -> None:
        outcome = ReconstructionOutcome(
            result=result,
            finished=True,
            needs_agent="Evolution",
            prompt_to_target_agent="Rank these organisms by proximity to Mammuthus.",
        )

        body = client_for(StubService(outcome)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()

        assert body["status"] == AgentStatus.NEEDS_AGENT.value
        assert body["target_agent"] == "Evolution"
        assert "Mammuthus" in body["prompt_to_target_agent"]

    def test_escalation_carries_findings_so_the_loop_guard_sees_progress(
        self, result: ReconstructionResult
    ) -> None:
        """The orchestrator force-fails an escalation that adds no context keys."""
        outcome = ReconstructionOutcome(
            result=result, finished=True, needs_agent="Evolution"
        )

        body = client_for(StubService(outcome)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()

        assert isinstance(body["output"], dict)
        assert body["output"]["reconstruction"]["sequence_id"] == "test_seq"


class TestV1Reconstructions:
    def test_success_populates_data_and_leaves_error_null(
        self, finished: ReconstructionOutcome
    ) -> None:
        response = client_for(StubService(finished)).post(
            "/api/v1/reconstructions",
            json={"sequence": "ACGT" * 30 + "NNNN" + "ACGT" * 30},
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"data", "meta", "error"}
        assert body["error"] is None
        assert body["data"]["result"]["sequence_id"] == "test_seq"
        assert body["data"]["finished"] is True

    def test_meta_carries_version_and_correlation_id(
        self, finished: ReconstructionOutcome
    ) -> None:
        response = client_for(StubService(finished)).post(
            "/api/v1/reconstructions",
            json={"sequence": "ACGTNNNN"},
            headers={"X-Request-ID": "orchestrator-run-42"},
        )

        meta = response.json()["meta"]
        assert meta["api_version"] == "v1"
        assert meta["request_id"] == "orchestrator-run-42"
        assert meta["duration_ms"] is not None

    def test_a_malformed_correlation_id_is_replaced_not_echoed(
        self, finished: ReconstructionOutcome
    ) -> None:
        """An id is echoed into logs and responses, so it must not carry junk."""
        hostile = "bad id\r\nInjected-Header: x"
        response = client_for(StubService(finished)).post(
            "/api/v1/reconstructions",
            json={"sequence": "ACGTNNNN"},
            headers={"X-Request-ID": hostile},
        )

        request_id = response.json()["meta"]["request_id"]
        assert request_id
        assert request_id != hostile
        assert "\n" not in request_id and "\r" not in request_id

    def test_failure_populates_error_and_leaves_data_null(self) -> None:
        response = client_for(StubService(InvalidSequenceError("No sequence given."))).post(
            "/api/v1/reconstructions", json={}
        )

        body = response.json()
        assert body["data"] is None
        assert body["error"]["code"] == ErrorCode.INVALID_SEQUENCE.value
        assert body["error"]["retryable"] is False

    def test_validation_failure_uses_the_envelope(self) -> None:
        """FastAPI's default {"detail": ...} would break the one-shape contract."""
        response = client_for(StubService(RuntimeError("unused"))).post(
            "/api/v1/reconstructions", json={"min_confidence": 5.0}
        )

        assert response.status_code == 422
        body = response.json()
        assert set(body) == {"data", "meta", "error"}
        assert body["error"]["code"] == ErrorCode.VALIDATION_ERROR.value
        assert body["error"]["details"]


class TestHealth:
    def test_reports_the_agent_and_its_tools(self) -> None:
        response = TestClient(create_app()).get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "ok"
        assert data["agent"] == "reconstruction_agent"
        assert "blast_search" in data["tools"]

    def test_reports_whether_external_services_are_configured(self) -> None:
        """A misconfigured deployment should be visible before a query fails."""
        services = TestClient(create_app()).get("/api/v1/health").json()["data"]["services"]

        assert "embl_ebi" in services
        assert "azure" in services
        assert "nvidia_evo2" in services

    def test_reports_whether_checkpoints_are_durable(self) -> None:
        """Without durable checkpoints a CONTINUE can never resume."""
        services = TestClient(create_app()).get("/api/v1/health").json()["data"]["services"]

        assert "checkpoints" in services

    def test_liveness_probe_is_unenveloped_and_cheap(self) -> None:
        response = TestClient(create_app()).get("/api/v1/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
