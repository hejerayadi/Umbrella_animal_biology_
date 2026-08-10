"""The test console and the two things it needs from the API.

The page itself is exercised end to end in a real browser; these cover the
contract it depends on, so a change here fails a test rather than silently
emptying a tab.
"""

from fastapi.testclient import TestClient

from backend.agents.Protein_visualization.api import app as agent_app
from backend.agents.Protein_visualization.app.api.v1.dependencies import (
    get_orchestrator,
    get_taxonomy_capability,
)
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.contracts.envelope import ErrorCode
from backend.agents.Protein_visualization.app.domain.exceptions import (
    ProteinNotFoundError,
    UpstreamServiceError,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import WORKFLOW_SEQUENCE
from backend.agents.Protein_visualization.tests.factories import agent_task
from backend.agents.Protein_visualization.tests.fakes import FakeRCSB, FakeTaxonomy, build_orchestrator

PREFIX = get_settings().api_prefix


def teardown_function() -> None:
    agent_app.dependency_overrides.clear()


def test_the_console_is_served_from_the_agent() -> None:
    """Same-origin with the API it calls, so no CORS entry has to know about it."""
    with TestClient(agent_app) as client:
        response = client.get("/console/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Protein Agent" in body
    # The page drives both endpoints and renders the scene itself.
    assert "/execute" in body
    assert "protein-structure-analyses" in body
    assert "molstar" in body


def test_taxonomy_endpoint_resolves_a_name_to_a_taxon_id() -> None:
    """The console needs this: the full endpoint takes an id, users type a name."""
    agent_app.dependency_overrides[get_taxonomy_capability] = lambda: FakeTaxonomy()

    with TestClient(agent_app) as client:
        response = client.get(f"{PREFIX}/taxonomy", params={"name": "human"})

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["data"] == {"scientific_name": "Homo sapiens", "taxon_id": 9606}


def test_taxonomy_endpoint_reports_an_unresolvable_name() -> None:
    agent_app.dependency_overrides[get_taxonomy_capability] = lambda: FakeTaxonomy(
        error=ProteinNotFoundError("UniProt taxonomy has no species matching 'nope'")
    )

    with TestClient(agent_app) as client:
        response = client.get(f"{PREFIX}/taxonomy", params={"name": "nope"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.protein_not_found


def test_the_response_reports_which_nodes_ran() -> None:
    """Without this the console cannot tell a skipped node from a failed one."""
    agent_app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator()

    with TestClient(agent_app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses",
            json=agent_task(residue_position=273).model_dump(mode="json"),
        )

    analysis = response.json()["data"]["output"]
    executed = analysis["executed_nodes"]

    assert "resolve_protein_identity" in executed
    assert "map_residues_with_sifts" in executed, "a residue was requested, so SIFTS must run"
    assert "search_alphafold" not in executed, "an experimental structure was found"
    assert executed == [node for node in WORKFLOW_SEQUENCE if node in executed], "must follow graph order"
    assert analysis["errors"] == []
    assert analysis["retry_counts"] == {}


def test_a_degraded_provider_is_attributed_to_its_node() -> None:
    """`errors` and `retry_counts` are what colour a node amber in the console.

    `UpstreamServiceError` rather than a bare `TimeoutError`, because that is
    what `JsonHttpClient` raises: every transport failure is wrapped before it
    reaches a node, and only `ProteinAgentError` is degraded instead of raised.
    """
    agent_app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator(
        rcsb=FakeRCSB(error=UpstreamServiceError("RCSB PDB", "request timed out"))
    )

    with TestClient(agent_app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses",
            json=agent_task().model_dump(mode="json"),
        )

    analysis = response.json()["data"]["output"]

    assert "search_experimental_structures" in analysis["executed_nodes"]
    assert any(error.startswith("search_experimental_structures:") for error in analysis["errors"])
    assert analysis["retry_counts"]["search_experimental_structures"] == 1
