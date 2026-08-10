from fastapi.testclient import TestClient

from backend.agents.Protein_visualization.app.api.v1.dependencies import get_knowledge_base, get_orchestrator
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.knowledge_base.embeddings import HashEmbedding
from backend.agents.Protein_visualization.app.knowledge_base.retrieval import KnowledgeBase
from backend.agents.Protein_visualization.app.main import create_app
from backend.agents.Protein_visualization.tests.factories import agent_task
from backend.agents.Protein_visualization.tests.fakes import FakeAlphaFold, FakeRCSB, build_orchestrator

PREFIX = get_settings().api_prefix


def test_analysis_runs_the_workflow_and_returns_the_envelope() -> None:
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator()
    task = agent_task(residue_position=273)

    with TestClient(app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses",
            json=task.model_dump(mode="json"),
            headers={"X-Trace-Id": str(task.trace_id)},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["error"] is None
    assert body["meta"]["task_id"] == str(task.task_id)
    assert body["meta"]["trace_id"] == str(task.trace_id)
    assert body["meta"]["analysis_id"] == body["data"]["output"]["analysis_id"]

    assert body["data"]["status"] == "completed"
    assert body["data"]["target_agent"] is None
    data = body["data"]["output"]
    assert data["status"] == "PARTIAL"
    assert data["validation_status"] == "REVISE"
    assert data["selected_structure"]["external_id"] == "1TUP"
    assert data["molstar_config"]["selections"][0]["residue_number"] == "273"
    assert data["evidence"]


def test_alphafold_fallback_is_reported_as_partial_over_http() -> None:
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator(rcsb=FakeRCSB(ids=[]))

    with TestClient(app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses", json=agent_task().model_dump(mode="json")
        )

    result = response.json()["data"]
    assert result["status"] == "completed"
    data = result["output"]
    assert data["status"] == "PARTIAL"
    assert data["selected_structure"]["structure_type"] == "PREDICTED"
    assert any(warning.startswith("ALPHAFOLD_FALLBACK") for warning in data["warnings"])


def test_no_structure_requests_literature_agent_over_http() -> None:
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator(
        rcsb=FakeRCSB(ids=[]), alphafold=FakeAlphaFold([])
    )

    with TestClient(app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses", json=agent_task().model_dump(mode="json")
        )

    result = response.json()["data"]
    assert result["status"] == "needs_agent"
    assert result["target_agent"] == "literature_agent"
    assert result["prompt_to_target_agent"]
    assert result["output"]["status"] == "NO_STRUCTURE_FOUND"


def test_validation_error_uses_the_envelope() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post(f"{PREFIX}/protein-structure-analyses", json={"input": {}})

    assert response.status_code == 422
    body = response.json()
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["status"] == 422
    assert body["error"]["instance"] == f"{PREFIX}/protein-structure-analyses"
    assert body["error"]["errors"]
    assert body["meta"]["trace_id"] == response.headers["X-Trace-Id"]


def test_unknown_route_uses_the_envelope() -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"{PREFIX}/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_knowledge_search_returns_filtered_hits() -> None:
    app = create_app()
    app.dependency_overrides[get_knowledge_base] = lambda: KnowledgeBase(HashEmbedding())
    with TestClient(app) as client:
        response = client.post(
            f"{PREFIX}/knowledge/search",
            json={"query": "p53 DNA binding domain", "protein_id": "P04637", "taxonomy_id": 9606},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["error"] is None


def test_ingestion_without_api_key_is_rejected(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    monkeypatch.setenv("INTERNAL_INGESTION_API_KEY", "expected-key")
    app = create_app()
    app.dependency_overrides[get_knowledge_base] = lambda: KnowledgeBase(HashEmbedding())
    try:
        with TestClient(app) as client:
            response = client.post(f"{PREFIX}/knowledge/protein/ingestions", json=[])
    finally:
        get_settings.cache_clear()

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
