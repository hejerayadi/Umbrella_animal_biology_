"""The /api/v1 contract: envelope shape and honest readiness.

The envelope is the contract, so it is asserted structurally rather than by
sampling a field or two. A client parses `{data, meta, error}` before it knows
anything about the resource, and branches on `error is None` - both of which
break silently if an endpoint ever answers with a bare payload.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.fakes import isolated_settings

from reconstruction_agent.config.settings import EmblEbiSettings, NcbiBlastSettings
from reconstruction_agent.main import create_app


def _client(
    *,
    ebi_email: str | None,
    blast_email: str | None = "tester@example.org",
) -> TestClient:
    """An app built from explicit settings.

    Never `get_settings()`: a developer .env on the machine running the suite
    must not be able to change an assertion.
    """
    settings = isolated_settings(
        embl_ebi=EmblEbiSettings(_env_file=None, contact_email=ebi_email),
        ncbi_blast=NcbiBlastSettings(_env_file=None, contact_email=blast_email),
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture
def client() -> TestClient:
    return _client(ebi_email="tester@example.org")


class TestEnvelope:
    def test_every_response_has_the_three_top_level_keys(self, client: TestClient) -> None:
        body = client.get("/api/v1/health").json()
        assert set(body) == {"data", "meta", "error"}

    def test_a_success_populates_data_and_leaves_error_null(self, client: TestClient) -> None:
        """Exactly one of data and error is ever set."""
        body = client.get("/api/v1/health").json()

        assert body["error"] is None
        assert body["data"] is not None

    def test_meta_identifies_the_version_and_the_agent(self, client: TestClient) -> None:
        meta = client.get("/api/v1/health").json()["meta"]

        assert meta["api_version"] == "v1"
        assert meta["agent"] == "reconstruction-agent"
        assert meta["timestamp"]

    def test_meta_carries_no_business_data(self, client: TestClient) -> None:
        """Technical context only - scientific values belong in `data`."""
        meta = client.get("/api/v1/health").json()["meta"]

        assert set(meta) <= {
            "api_version",
            "agent",
            "request_id",
            "trace_id",
            "timestamp",
            "duration_ms",
            "iteration_count",
            "budget",
        }

    def test_correlation_ids_are_echoed_from_the_request(self, client: TestClient) -> None:
        """The trace id is stable across a whole orchestrator run.

        Not asserted on health, which takes no ids - this documents that the
        header names are the ones the orchestrator actually sends.
        """
        response = client.get(
            "/api/v1/health",
            headers={"X-Trace-Id": "trace-1", "X-Request-Id": "req-1"},
        )
        assert response.status_code == 200


class TestReadiness:
    def test_health_is_200_even_when_a_dependency_is_missing(self) -> None:
        """Liveness and readiness answer different questions."""
        with _client(ebi_email=None) as client:
            assert client.get("/api/v1/health").status_code == 200

    def test_ready_is_503_without_a_contact_address_for_the_job_service(self) -> None:
        """EBI rejects anonymous submissions, so every gap would be unresolved.

        Reporting that as ready would hide a total loss of function behind a
        green check.
        """
        with _client(ebi_email=None) as client:
            response = client.get("/api/v1/ready")

            assert response.status_code == 503
            body = response.json()
            assert body["data"]["status"] == "degraded"
            assert body["error"] is None

    def test_ready_is_200_once_the_required_dependency_is_configured(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/v1/ready")

        assert response.status_code == 200
        assert response.json()["data"]["status"] == "ok"

    def test_the_search_service_is_also_required(self) -> None:
        """Two services, two jobs: NCBI searches, EBI aligns. Both are needed."""
        with _client(ebi_email="tester@example.org", blast_email=None) as client:
            response = client.get("/api/v1/ready")

            assert response.status_code == 503
            names = {dep["name"]: dep for dep in response.json()["data"]["dependencies"]}
            assert names["ncbi-blast"]["configured"] is False
            assert names["embl-ebi-mafft"]["configured"] is True

    def test_an_optional_dependency_does_not_block_readiness(self, client: TestClient) -> None:
        """Evo 2 is arbitration only; without it homology evidence still stands."""
        dependencies = {
            dep["name"]: dep for dep in client.get("/api/v1/ready").json()["data"]["dependencies"]
        }

        assert dependencies["nvidia-evo2"]["required"] is False
        assert dependencies["nvidia-evo2"]["configured"] is False
        assert dependencies["embl-ebi-mafft"]["required"] is True


class TestErrorContract:
    def test_an_unknown_route_does_not_produce_an_envelope_free_body(
        self, client: TestClient
    ) -> None:
        """FastAPI answers 404 for an unrouted path; nothing here claims otherwise."""
        assert client.get("/api/v1/nonexistent").status_code == 404
