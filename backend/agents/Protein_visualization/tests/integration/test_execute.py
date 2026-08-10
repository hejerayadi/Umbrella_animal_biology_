"""The inter-agent boundary the Grand Orchestrator actually calls.

These exercise `POST /execute` on the same app `backend/run_agents.py` serves,
with the biological providers replaced by the offline fakes. The point is the
translation - shared context in, routing status and shared-context summary out -
not the science, which `tests/e2e/test_workflow.py` already covers.
"""

from fastapi.testclient import TestClient

from backend.agents.Protein_visualization.api import app as agent_app
from backend.agents.Protein_visualization.app.api.v1.dependencies import (
    get_orchestrator,
    get_taxonomy_capability,
)
from backend.agents.Protein_visualization.tests.fakes import (
    FakeRCSB,
    FakeTaxonomy,
    build_orchestrator,
)


def _client(orchestrator=None, taxonomy=None) -> tuple[TestClient, FakeTaxonomy]:  # type: ignore[no-untyped-def]
    fake_taxonomy = taxonomy or FakeTaxonomy()
    agent_app.dependency_overrides[get_orchestrator] = lambda: orchestrator or build_orchestrator()
    agent_app.dependency_overrides[get_taxonomy_capability] = lambda: fake_taxonomy
    return TestClient(agent_app), fake_taxonomy


def teardown_function() -> None:
    agent_app.dependency_overrides.clear()


def test_execute_runs_the_real_workflow_and_summarises_it_for_the_shared_context() -> None:
    client, taxonomy = _client()

    with client:
        response = client.post(
            "/execute",
            json={
                "instruction": "Show me the 3D structure of TP53 in humans",
                "context": {"species": "Homo sapiens", "gene_name": "TP53", "residue_position": 273},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["target_agent"] is None

    structure = body["output"]["protein_structure"]
    assert structure["status"] == "PARTIAL"
    assert structure["validation_status"] == "REVISE"
    assert structure["uniprot_accession"] == "P04637"
    assert structure["gene_symbol"] == "TP53"
    assert structure["structure"]["external_id"] == "1TUP"
    assert structure["structure"]["file_url"].endswith(".cif")

    # The species name from the shared context is what gets resolved.
    assert taxonomy.calls == ["Homo sapiens"]


def test_execute_does_not_leak_the_full_payload_into_the_shared_context() -> None:
    """Whatever comes back under `completed` is merged into every later agent's
    context and rendered into the Responder's prompt, so the Mol* scene and the
    raw evidence records must stay behind the versioned API."""
    client, _ = _client()

    with client:
        response = client.post(
            "/execute",
            json={"instruction": "TP53 structure", "context": {"species": "human", "gene_name": "TP53"}},
        )

    output = response.json()["output"]
    assert set(output) == {"protein_structure"}
    for heavy in ("molstar_config", "evidence", "annotations", "residue_mappings"):
        assert heavy not in output["protein_structure"]


def test_execute_asks_for_a_gene_when_the_context_has_none() -> None:
    client, _ = _client()

    with client:
        response = client.post(
            "/execute",
            json={"instruction": "What does its protein look like?", "context": {"species": "human"}},
        )

    body = response.json()
    assert body["status"] == "needs_agent"
    assert "gene" in body["prompt_to_target_agent"].lower()
    # No workflow ran, so there is nothing to report but the reason.
    assert body["output"] == {"reason": body["prompt_to_target_agent"]}


def test_execute_asks_for_a_species_when_the_context_has_none() -> None:
    client, _ = _client()

    with client:
        response = client.post(
            "/execute",
            json={"instruction": "Show me the TP53 structure", "context": {"gene_name": "TP53"}},
        )

    body = response.json()
    assert body["status"] == "needs_agent"
    assert "species" in body["prompt_to_target_agent"].lower()


def test_execute_accepts_an_accession_without_a_gene_symbol() -> None:
    client, _ = _client()

    with client:
        response = client.post(
            "/execute",
            json={
                "instruction": "Render this protein",
                "context": {"species": "Homo sapiens", "uniprot_accession": "P04637"},
            },
        )

    body = response.json()
    assert body["status"] == "completed"
    assert body["output"]["protein_structure"]["uniprot_accession"] == "P04637"


def test_execute_trusts_a_species_already_resolved_upstream() -> None:
    """An agent that did its own taxonomy work hands over a dict. Re-resolving it
    could disagree with the id it already gave the other agents."""
    taxonomy = FakeTaxonomy()
    client, _ = _client(taxonomy=taxonomy)

    with client:
        response = client.post(
            "/execute",
            json={
                "instruction": "TP53 structure",
                "context": {
                    "species": {"scientific_name": "Homo sapiens", "taxon_id": 9606},
                    "gene_name": "TP53",
                },
            },
        )

    assert response.json()["status"] == "completed"
    assert taxonomy.calls == []


def test_execute_hands_off_when_no_structure_can_be_found() -> None:
    orchestrator = build_orchestrator(rcsb=FakeRCSB(ids=[]), alphafold=_no_predictions())
    client, _ = _client(orchestrator=orchestrator)

    with client:
        response = client.post(
            "/execute",
            json={"instruction": "TP53 structure", "context": {"species": "human", "gene_name": "TP53"}},
        )

    body = response.json()
    assert body["status"] == "needs_agent"
    assert "literature" in body["prompt_to_target_agent"].lower()
    # A hand-off still reports what the workflow did manage to establish.
    assert body["output"]["protein_structure"]["uniprot_accession"] == "P04637"


def test_execute_reports_a_provider_failure_instead_of_raising() -> None:
    taxonomy = FakeTaxonomy(error=RuntimeError("taxonomy index unreachable"))
    client, _ = _client(taxonomy=taxonomy)

    with client:
        response = client.post(
            "/execute",
            json={"instruction": "TP53 structure", "context": {"species": "human", "gene_name": "TP53"}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "taxonomy index unreachable" in body["output"]


def _no_predictions():  # type: ignore[no-untyped-def]
    from backend.agents.Protein_visualization.tests.fakes import FakeAlphaFold

    return FakeAlphaFold(records=[])
