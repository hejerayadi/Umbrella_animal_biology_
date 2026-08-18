from fastapi.testclient import TestClient

from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.main import create_app


def test_health_uses_the_response_envelope() -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"{get_settings().api_prefix}/health")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["status"] == "ok"
    assert body["error"] is None
    assert body["meta"]["api_version"] == "v1"
    assert body["meta"]["request_id"] == response.headers["X-Request-Id"]
    assert body["meta"]["trace_id"] == response.headers["X-Trace-Id"]
    assert body["meta"]["duration_ms"] is not None


def test_trace_id_from_the_caller_is_propagated() -> None:
    trace_id = "7cce77eb-ec91-4353-9f29-2af0c55519b7"
    with TestClient(create_app()) as client:
        response = client.get(f"{get_settings().api_prefix}/health", headers={"X-Trace-Id": trace_id})

    assert response.headers["X-Trace-Id"] == trace_id
    assert response.json()["meta"]["trace_id"] == trace_id


def test_ready_reports_dependencies() -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"{get_settings().api_prefix}/ready")

    assert response.status_code == 200
    names = {item["name"] for item in response.json()["data"]["dependencies"]}
    assert names == {"qdrant", "azure_llm", "persistence", "langsmith"}


def test_metrics_are_wrapped_in_the_envelope() -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"{get_settings().api_prefix}/metrics")

    assert response.status_code == 200
    assert isinstance(response.json()["data"], dict)


def test_frontend_origin_is_allowed_by_cors() -> None:
    with TestClient(create_app()) as client:
        response = client.options(
            f"{get_settings().api_prefix}/ready",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
