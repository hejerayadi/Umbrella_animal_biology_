"""What a response says about the run that produced it.

A `ProteinAnalysisResponse` reports what was produced; these fields report
*how*, and the two differ in ways an operator has to be able to tell apart: an
AlphaFold model because every experimental candidate was rejected reads exactly
like one because RCSB timed out, unless the trace says which.
"""

from fastapi.testclient import TestClient

from backend.agents.Protein_visualization.app.api.v1.dependencies import get_orchestrator
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.domain.exceptions import UpstreamServiceError
from backend.agents.Protein_visualization.app.main import create_app
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import WORKFLOW_SEQUENCE
from backend.agents.Protein_visualization.tests.factories import agent_task
from backend.agents.Protein_visualization.tests.fakes import (
    FakeLanguageModel,
    FakeRCSB,
    build_orchestrator,
)

PREFIX = get_settings().api_prefix


def test_the_response_reports_which_nodes_ran() -> None:
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator()

    with TestClient(app) as client:
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
    """A provider failure that was degraded rather than raised still has to be
    traceable to the node that hit it.

    `UpstreamServiceError` rather than a bare `TimeoutError`, because that is
    what `JsonHttpClient` raises: every transport failure is wrapped before it
    reaches a node, and only `ProteinAgentError` is degraded instead of raised.
    """
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator(
        rcsb=FakeRCSB(error=UpstreamServiceError("RCSB PDB", "request timed out"))
    )

    with TestClient(app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses",
            json=agent_task().model_dump(mode="json"),
        )

    analysis = response.json()["data"]["output"]

    assert "search_experimental_structures" in analysis["executed_nodes"]
    assert any(error.startswith("search_experimental_structures:") for error in analysis["errors"])
    assert analysis["retry_counts"]["search_experimental_structures"] == 1


def test_llm_usage_is_reported_per_call() -> None:
    """Explanation and the critic's audit each make one Azure call; both must show up."""
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: build_orchestrator(llm=FakeLanguageModel())

    with TestClient(app) as client:
        response = client.post(
            f"{PREFIX}/protein-structure-analyses",
            json=agent_task().model_dump(mode="json"),
        )

    usage = response.json()["data"]["output"]["llm_usage"]
    nodes = {entry["node"] for entry in usage}

    assert nodes == {"generate_grounded_explanation", "run_scientific_critic"}
    for entry in usage:
        assert entry["model"] == "fake-gpt"
        assert entry["total_tokens"] == 10
        # No price configured by default, so cost is left unset rather than
        # computed from a guessed rate.
        assert entry["estimated_cost_usd"] is None
