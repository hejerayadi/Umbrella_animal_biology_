"""WORKER-QUALITY AUDIT — are the deterministic stand-ins fit for purpose?

Mocked bioinformatics providers (NCBI, UniProt, ESM-2, MAFFT, IQ-TREE,
ModelFinder, UFBoot, NetworkX) are EXPECTED in this phase and are never
treated as a defect here.  Scientific accuracy of the fixture values is
explicitly out of scope.

What is verified:
  * they are deterministic
  * they receive the right inputs
  * they return schema-conformant results
  * they simulate both success and error paths
  * the HTTP boundary keeps the platform contract

"Called only in the right branch" is verified in test_branch_acceptance.py.

No network. No API key.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    MolecularComparisonResult,
    PhylogeneticResult,
)
from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
    MolecularComparisonMock,
)
from backend.agents.evolution_agent.workers.phylogenetic_tree.worker import (
    PhylogeneticTreeWorker,
)

AGENT_DIR = Path(__file__).resolve().parent.parent
THREE = ["homo sapiens", "pan troglodytes", "mus musculus"]
FIVE = THREE + ["gallus gallus", "danio rerio"]


def rq(species, **kw) -> AgentRequest:
    ctx = kw.pop("context", {})
    if "feature" not in ctx:
        ctx["feature"] = "full_analysis"
    return AgentRequest(instruction="x", context=ctx,
                        species_list=list(species), **kw)


# ===========================================================================
# 1. THE WORKERS ARE DETERMINISTIC
# ===========================================================================

def test_molecular_mock_is_deterministic_across_calls() -> None:
    a = MolecularComparisonMock().run(rq(THREE)).output
    b = MolecularComparisonMock().run(rq(THREE)).output
    assert a == b


def test_phylogenetic_mock_is_deterministic_across_calls() -> None:
    a = PhylogeneticTreeWorker().run(rq(THREE)).output
    b = PhylogeneticTreeWorker().run(rq(THREE)).output
    assert a == b


def test_mocks_are_order_independent_for_the_same_species_set() -> None:
    """Reordering the input must not change the MC scientific payload."""
    a = MolecularComparisonMock().run(rq(THREE)).output
    b = MolecularComparisonMock().run(rq(list(reversed(THREE)))).output
    assert {frozenset({e.species_a, e.species_b}): e.score
            for e in a.similarity_scores} == {
           frozenset({e.species_a, e.species_b}): e.score
           for e in b.similarity_scores}


def test_different_species_sets_give_different_payloads() -> None:
    """Determinism must not mean a constant answer."""
    a = MolecularComparisonMock().run(rq(THREE)).output
    b = MolecularComparisonMock().run(rq(FIVE)).output
    assert a.similarity_scores != b.similarity_scores
    assert (PhylogeneticTreeWorker().run(rq(THREE)).output.newick_tree
            != PhylogeneticTreeWorker().run(rq(FIVE)).output.newick_tree)


# ===========================================================================
# 3. THE WORKERS RECEIVE THE RIGHT INPUTS
# ===========================================================================

def test_species_are_normalised_before_reaching_a_worker() -> None:
    r = MolecularComparisonMock().run(rq(["  Homo Sapiens ", "PAN TROGLODYTES"]))
    assert r.status is AgentStatus.COMPLETED
    assert r.output.species_list == ["homo sapiens", "pan troglodytes"]


def test_workers_read_species_from_request_or_context() -> None:
    from_field = MolecularComparisonMock().run(rq(THREE))
    from_ctx = MolecularComparisonMock().run(
        AgentRequest(instruction="x", context={"species_list": THREE}))
    from_alias = MolecularComparisonMock().run(
        AgentRequest(instruction="x", context={"species": THREE}))
    assert from_field.output == from_ctx.output == from_alias.output


@pytest.mark.asyncio
async def test_orchestrator_hands_workers_canonical_resolved_names(
    make_orchestrator,
) -> None:
    """Common names must be resolved before a worker sees them."""
    seen: list[list[str]] = []

    class _Probe:
        def run(self, request: AgentRequest) -> AgentResult:
            seen.append(list(request.species_list))
            return MolecularComparisonMock().run(request)

    orch = make_orchestrator(mc_worker=_Probe())
    await orch.run(rq(["human", "chimp", "mouse"]))
    assert seen, "worker never called"
    assert seen[0] == ["Homo sapiens", "Pan troglodytes", "Mus musculus"]


# ===========================================================================
# 4. THE WORKERS RETURN SCHEMA-CONFORMANT RESULTS
# ===========================================================================

def _assert_matches_dataclass(obj, klass) -> None:
    assert is_dataclass(obj) and isinstance(obj, klass)
    for f in fields(klass):
        assert hasattr(obj, f.name), f"missing field {f.name}"


def test_molecular_result_matches_its_dataclass() -> None:
    out = MolecularComparisonMock().run(rq(THREE)).output
    _assert_matches_dataclass(out, MolecularComparisonResult)
    assert isinstance(out.alignment, str) and out.alignment.startswith(">")
    assert isinstance(out.similarity_network, dict)
    assert set(out.similarity_network) == set(THREE)


def test_phylogenetic_result_matches_its_dataclass() -> None:
    out = PhylogeneticTreeWorker().run(rq(THREE)).output
    _assert_matches_dataclass(out, PhylogeneticResult)
    assert out.newick_tree.endswith(";")
    assert all(isinstance(v, int) for v in out.bootstrap_support.values())
    assert 0.0 <= out.overall_confidence <= 1.0


def test_pairwise_score_count_is_n_choose_2() -> None:
    for species in (THREE, FIVE):
        n = len(species)
        out = MolecularComparisonMock().run(rq(species)).output
        assert len(out.similarity_scores) == n * (n - 1) // 2


def test_similarity_network_is_symmetric() -> None:
    net = MolecularComparisonMock().run(rq(FIVE)).output.similarity_network
    for node, edges in net.items():
        for edge in edges:
            back = net[edge["neighbour"]]
            assert any(e["neighbour"] == node and e["score"] == edge["score"]
                       for e in back)


@pytest.mark.asyncio
async def test_assembled_output_is_json_serialisable(orchestrator) -> None:
    from backend.agents.evolution_agent.orchestrator_adapter import (
        to_platform_result,
    )
    out = to_platform_result(await orchestrator.run(rq(THREE))).output
    json.dumps(out)


# ===========================================================================
# 5. THE WORKERS SIMULATE SUCCESS *AND* ERRORS
# ===========================================================================

def test_molecular_mock_rejects_fewer_than_two_species() -> None:
    r = MolecularComparisonMock().run(rq(["homo sapiens"]))
    assert r.status is AgentStatus.FAILED
    assert "at least 2" in str(r.output)


def test_molecular_mock_rejects_an_empty_species_list() -> None:
    r = MolecularComparisonMock().run(rq([]))
    assert r.status is AgentStatus.FAILED
    assert "required" in str(r.output).lower()


def test_phylogenetic_mock_rejects_fewer_than_three_species() -> None:
    r = PhylogeneticTreeWorker().run(rq(["homo sapiens", "mus musculus"]))
    assert r.status is AgentStatus.FAILED
    assert "at least 3" in str(r.output)


def test_both_mocks_reject_species_outside_the_catalogue() -> None:
    for worker in (MolecularComparisonMock(), PhylogeneticTreeWorker()):
        r = worker.run(rq(["homo sapiens", "pan troglodytes", "draco magicus"]))
        assert r.status is AgentStatus.FAILED
        assert "draco magicus" in str(r.output).lower()


def test_error_results_still_honour_the_agent_result_contract() -> None:
    r = MolecularComparisonMock().run(rq(["homo sapiens"]))
    assert isinstance(r, AgentResult)
    assert r.status is AgentStatus.FAILED
    assert r.source_agents == ["Molecular Comparison Agent"]
    assert r.confidence is None


def test_mock_error_messages_name_the_failing_sub_agent() -> None:
    mc = MolecularComparisonMock().run(rq(["homo sapiens"]))
    ph = PhylogeneticTreeWorker().run(rq(["homo sapiens", "mus musculus"]))
    assert "molecular comparison" in str(mc.output).lower()
    assert "phylogenetic reconstruction" in str(ph.output).lower()


# ===========================================================================
# 6. HTTP BOUNDARY / PLATFORM CONTRACT
# ===========================================================================

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EVOLUTION_AGENT_IMPL", "orchestrator")
    import backend.agents.evolution_agent.api as api_mod
    api_mod = importlib.reload(api_mod)
    return TestClient(api_mod.app), api_mod


def test_health_reports_the_orchestrator_implementation(client) -> None:
    c, _ = client
    body = c.get("/health").json()
    assert body["is_orchestrator"] is True


def test_default_implementation_is_the_orchestrator(
    monkeypatch,
) -> None:
    """The default implementation is now always the orchestrator."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("EVOLUTION_AGENT_IMPL", raising=False)
    import backend.agents.evolution_agent.api as api_mod
    api_mod = importlib.reload(api_mod)
    c = TestClient(api_mod.app)

    assert c.get("/health").json()["implementation"] == "OrchestratorEvolutionAgent"


def test_execute_returns_the_platform_contract_keys(client) -> None:
    c, api_mod = client
    api_mod._agent._orchestrator._phylo_worker = PhylogeneticTreeWorker()
    body = c.post("/execute", json={
        "instruction": "Compare human, chimp and mouse.",
        "context": {"species_list": THREE, "feature": "full_analysis"},
    }).json()
    assert set(body) >= {"status", "output", "target_agent",
                         "prompt_to_target_agent"}
    assert body["status"] == "completed"
    assert isinstance(body["output"], dict)


def test_execute_never_returns_500_for_a_business_error(client) -> None:
    c, api_mod = client
    api_mod._agent._orchestrator._phylo_worker = PhylogeneticTreeWorker()
    for ctx in ({"species_list": ["homo sapiens"], "feature": "full_analysis"},
                {"species_list": ["draco magicus", "homo sapiens"],
                 "feature": "full_analysis"},
                {"species_list": [], "feature": "full_analysis"},
                {"species_list": THREE, "feature": "banana"}):
        r = c.post("/execute", json={"instruction": "x", "context": ctx})
        assert r.status_code == 200, ctx
        assert r.json()["status"] in {"failed", "continue"}, ctx


def test_execute_contains_no_credential_material(client) -> None:
    import os

    c, _ = client
    text = c.post("/execute", json={
        "instruction": "Compare human, chimp and mouse.",
        "context": {"species_list": THREE, "feature": "full_analysis"},
    }).text
    for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                "GROQ_API_KEY", "GITHUB_TOKEN", "QDRANT_API_KEY"):
        value = os.environ.get(var)
        if value and len(value) > 8:
            assert value not in text
    assert "api_key" not in text.lower()
    assert "bearer " not in text.lower()


def test_worker_crash_is_contained_by_the_http_boundary(client, monkeypatch) -> None:
    """Only api.py stops a worker exception — it must at least stop it."""
    c, api_mod = client

    class _Boom:
        def run(self, request):
            raise RuntimeError("mock MAFFT binary missing")

    monkeypatch.setattr(api_mod._agent._orchestrator, "_mc_worker", _Boom())
    r = c.post("/execute", json={
        "instruction": "compare",
        "context": {"species_list": THREE, "feature": "full_analysis"},
    })
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "Traceback" not in r.text


# ===========================================================================
# 7. KNOWN DEFECTS OUTSIDE THE BRANCH CONTRACT
# ===========================================================================

def test_divergence_time_worker_package_removed() -> None:
    """workers/divergence_time was dead code — removed in the autonomy correction."""
    assert not (AGENT_DIR / "workers/divergence_time").exists()


def test_DEFECT_router_and_aggregator_modules_are_never_imported() -> None:
    """The two modules that would implement branch routing are unused."""
    import re

    importers: list[str] = []
    for py in AGENT_DIR.rglob("*.py"):
        if "__pycache__" in py.parts or py.name in {"router.py", "aggregator.py"}:
            continue
        src = py.read_text(encoding="utf-8")
        if re.search(r"^\s*from\s+\.(router|aggregator)\b", src, re.M):
            importers.append(py.name)
    assert importers == [], f"now wired in: {importers}"


def test_DEFECT_to_platform_result_is_not_idempotent() -> None:
    """Applying the flattener twice stringifies the payload into explanation.

    Any caller that re-normalises an already-normalised result silently
    destroys the structured output.
    """
    from backend.agents.evolution_agent.orchestrator_adapter import (
        to_platform_result,
    )
    from backend.agents.evolution_agent.schema import (
        EvolutionAnalysisResult,
    )

    analysis = EvolutionAnalysisResult(
        species_list=THREE,
        molecular=MolecularComparisonMock().run(rq(THREE)).output,
        phylogenetic=PhylogeneticTreeWorker().run(rq(THREE)).output,
        overall_confidence=0.9,
        source_agents=["Evolution Agent Orchestrator"],
    )
    once = to_platform_result(
        AgentResult(status=AgentStatus.COMPLETED, output=analysis))
    twice = to_platform_result(once)

    assert "newick_tree" in once.output
    assert "newick_tree" not in twice.output
    assert isinstance(twice.output["explanation"], str)


def test_malformed_context_value_is_handled_gracefully() -> None:
    """A non-iterable species_list is now handled gracefully, not a TypeError."""
    from backend.agents.evolution_agent.orchestrator_adapter import (
        resolve_species,
    )
    from backend.agents.evolution_agent.schema import PlannerDecision

    result = resolve_species({"species_list": 12345},
                             PlannerDecision(feature="molecular_comparison"))
    assert result == []
