"""The orchestrator contract: POST /execute always answers with an AgentResult.

These are the tests that would catch a change breaking the rest of the system,
so they assert on the envelope as much as on the science.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reconstruction_agent.api.app import create_app
from reconstruction_agent.api.dependencies import get_service
from reconstruction_agent.api.schemas import AgentStatus
from reconstruction_agent.contracts.output import ReconstructionResult


class StubService:
    """Stands in for the real service so no external call is made."""

    def __init__(self, result: ReconstructionResult | Exception) -> None:
        self._result = result

    async def reconstruct(self, request: object) -> ReconstructionResult:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


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


def client_for(service: object) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


class TestExecute:
    def test_returns_completed_with_the_result_payload(
        self, result: ReconstructionResult
    ) -> None:
        response = client_for(StubService(result)).post(
            "/execute",
            json={"instruction": "Reconstruct the gaps.", "context": {"sequence": "ACGTNNNN"}},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == AgentStatus.COMPLETED.value
        assert body["output"]["sequence_id"] == "test_seq"
        assert body["output"]["overall_confidence"] == 0.82

    def test_reports_failure_inside_the_envelope_not_as_http_500(self) -> None:
        """The orchestrator's router expects one schema back every time."""
        response = client_for(StubService(RuntimeError("NCBI unreachable"))).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == AgentStatus.FAILED.value
        assert "NCBI unreachable" in body["output"]

    def test_accepts_a_request_with_no_context(self, result: ReconstructionResult) -> None:
        response = client_for(StubService(result)).post(
            "/execute", json={"instruction": "Reconstruct something."}
        )

        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.COMPLETED.value

    def test_response_carries_every_contract_field(self, result: ReconstructionResult) -> None:
        body = client_for(StubService(result)).post(
            "/execute", json={"instruction": "Reconstruct.", "context": {}}
        ).json()

        assert set(body) == {"status", "target_agent", "prompt_to_target_agent", "output"}


class TestHealth:
    def test_reports_the_agent_and_its_tools(self) -> None:
        response = TestClient(create_app()).get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["agent"] == "reconstruction_agent"
        assert "blast_search" in body["tools"]

    def test_reports_whether_external_services_are_configured(self) -> None:
        """A misconfigured deployment should be visible before a query fails."""
        body = TestClient(create_app()).get("/health").json()

        assert "embl_ebi" in body["external_services_configured"]
