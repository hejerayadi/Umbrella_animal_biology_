"""ACCEPTANCE SUITE — the branch-selection contract the agent must satisfy.

These tests encode the *target* behaviour agreed for the mock phase:

    User request
        -> LLM intent classification
        -> Evolution Orchestrator
        -> EXACTLY ONE branch

    branch "similarity_network"  -> Molecular Comparison sub-agent only
    branch "phylogenetic_tree"   -> Phylogenetic Reconstruction sub-agent only

A similarity-network request must not build or return a tree.
A phylogenetic-tree request must not build or return a similarity network.
Running both branches is legal only when the user explicitly asks for both.
``full_analysis`` must never be an automatic fallback for an ambiguous intent.

Mocked bioinformatics providers are EXPECTED here and are never treated as a
defect.  What is under test is *which* mock runs, *when*, and *what the caller
gets back*.

Tests that FAIL against the current implementation are the evidence of the
bug — they are written against the contract, not against the code.

No network. No API key. Every LLM and every worker is a deterministic stub.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.evolution_agent.intent import (
    INTENT_SYSTEM_PROMPT,
    RecognizedIntent,
)
from backend.agents.evolution_agent.orchestrator_adapter import (
    OrchestratorEvolutionAgent,
    to_platform_result,
)
from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    MolecularComparisonResult,
    PhylogeneticResult,
    SimilarityEdge,
    SpeciesGroup,
)

AGENT_DIR = Path(__file__).resolve().parent.parent

TWO = ["homo sapiens", "pan troglodytes"]
THREE = ["homo sapiens", "pan troglodytes", "mus musculus"]

# Fields that only ever belong to a phylogenetic answer.
TREE_ONLY_KEYS = ("newick_tree", "tree_url", "bootstrap_support",
                  "confidence_values", "model")
# Fields that only ever belong to a similarity-network answer.
NETWORK_ONLY_KEYS = ("similarity_network", "similarity_scores",
                     "species_groups")


# ===========================================================================
# Deterministic stubs
# ===========================================================================

class WorkerSpy:
    """Records every call so a branch decision can be proven, not inferred."""

    def __init__(self, name: str, result: AgentResult | None = None,
                 raises: BaseException | None = None):
        self.name = name
        self._result = result
        self._raises = raises
        self.calls: list[AgentRequest] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.calls.append(request)
        if self._raises is not None:
            raise self._raises
        return self._result

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def called(self) -> bool:
        return bool(self.calls)


def mc_ok(species=None) -> AgentResult:
    species = species or THREE
    return AgentResult(
        status=AgentStatus.COMPLETED,
        output=MolecularComparisonResult(
            species_list=list(species),
            similarity_scores=[SimilarityEdge(species[0], species[1], 0.98)],
            species_groups=[SpeciesGroup(0, list(species), 0.98)],
            similarity_network={s: [] for s in species},
        ),
        source_agents=["Molecular Comparison Agent"],
    )


def phylo_ok(species=None) -> AgentResult:
    species = species or THREE
    return AgentResult(
        status=AgentStatus.COMPLETED,
        output=PhylogeneticResult(
            newick_tree="(('a':0.6,'b':0.6):8.4,'c':9.0);",
            tree_url="http://mock/tree.svg",
            model="LG+G4",
            bootstrap_support={"node_root": 95},
            confidence_values={s: 0.9 for s in species},
            overall_confidence=0.95,
        ),
        source_agents=["Phylogenetic Tree Agent"],
    )


def make_agent(monkeypatch, make_orchestrator, *, feature, species,
               mc=None, phylo=None):
    """Build the full entry point with a stubbed classifier and spy workers.

    Mirrors production wiring: instruction -> classify_intent -> adapter ->
    orchestrator -> workers.  Only the LLM call is replaced.
    """
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _classify(_prompt, **_kw):
        return RecognizedIntent(feature=feature, species_list=list(species),
                                source="llm")

    monkeypatch.setattr(mod, "classify_intent", _classify)

    mc = mc if mc is not None else WorkerSpy("mc", mc_ok(species))
    phylo = phylo if phylo is not None else WorkerSpy("phylo", phylo_ok(species))
    orch = make_orchestrator(mc_worker=mc, phylo_worker=phylo)
    return OrchestratorEvolutionAgent(orchestrator=orch), mc, phylo


def flat(result: AgentResult) -> dict:
    """The dict the Global Orchestrator would merge into shared context.

    ``OrchestratorEvolutionAgent.run`` already flattens, so a dict output is
    taken as-is.  ``to_platform_result`` is NOT idempotent — re-applying it
    to an already-flat result stringifies the whole payload into
    ``explanation`` — so it is only used on a raw orchestrator result.
    """
    if isinstance(result.output, dict):
        return result.output
    mapped = to_platform_result(result)
    return mapped.output if isinstance(mapped.output, dict) else {}


# ===========================================================================
# 1. SIMILARITY NETWORK BRANCH
# ===========================================================================

@pytest.mark.asyncio
async def test_similarity_request_calls_molecular_comparison_exactly_once(
    monkeypatch, make_orchestrator,
) -> None:
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    await agent.run(AgentRequest(
        instruction="Compare these species and build a similarity network.",
        context={},
    ))
    assert mc.count == 1, f"Molecular Comparison ran {mc.count} times, expected 1"


@pytest.mark.asyncio
async def test_similarity_request_never_calls_phylogenetic_reconstruction(
    monkeypatch, make_orchestrator,
) -> None:
    """A similarity-network request must not touch the tree sub-agent."""
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    await agent.run(AgentRequest(
        instruction="Build a protein similarity network for these species.",
        context={},
    ))
    assert phylo.count == 0, (
        f"Phylogenetic Reconstruction ran {phylo.count} times for a "
        "similarity-network request; it must never run in this branch"
    )


@pytest.mark.asyncio
async def test_similarity_result_contains_the_network(
    monkeypatch, make_orchestrator,
) -> None:
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    out = flat(await agent.run(AgentRequest(
        instruction="Build a similarity network.", context={})))
    assert out.get("similarity_network") is not None
    assert out.get("similarity_scores")


@pytest.mark.asyncio
async def test_similarity_result_carries_no_tree_fields(
    monkeypatch, make_orchestrator,
) -> None:
    """No newick_tree / tree_url / bootstrap_support in a network answer."""
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a similarity network for these species.", context={}))
    out = flat(result)

    leaked = [k for k in TREE_ONLY_KEYS if out.get(k) is not None]
    assert leaked == [], f"tree fields leaked into a network answer: {leaked}"
    assert result.newick_tree is None
    assert result.tree_url is None


@pytest.mark.asyncio
async def test_similarity_network_works_with_only_two_species(
    monkeypatch, make_orchestrator,
) -> None:
    """Two species are enough for a comparison.

    The three-taxa minimum belongs to phylogenetics only and must never
    block a similarity-network request.
    """
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=TWO,
        mc=WorkerSpy("mc", mc_ok(TWO)),
    )
    result = await agent.run(AgentRequest(
        instruction="How similar are humans and chimpanzees?", context={}))

    assert result.status is AgentStatus.COMPLETED, (
        f"a 2-species comparison failed: {result.output!r}"
    )
    assert mc.count == 1
    assert phylo.count == 0


@pytest.mark.asyncio
async def test_two_species_similarity_request_with_the_real_mock_workers(
    monkeypatch, make_orchestrator,
) -> None:
    """End-to-end with the shipped mocks — the demo-critical case.

    "How similar are humans and chimpanzees?" is the canonical Sprint 3
    demo question.  With the real mock workers wired in, the phylogenetic
    mock's 3-taxa rule must not be able to fail a similarity request.
    """
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _classify(_prompt, **_kw):
        return RecognizedIntent(feature="molecular_comparison",
                                species_list=list(TWO), source="llm")

    monkeypatch.setattr(mod, "classify_intent", _classify)
    agent = OrchestratorEvolutionAgent(orchestrator=make_orchestrator())

    result = await agent.run(AgentRequest(
        instruction="How similar are humans and chimpanzees at the molecular "
                    "level?",
        context={}))

    assert result.status is AgentStatus.COMPLETED, (
        f"the canonical 2-species demo question returns "
        f"{result.status.value}: {result.output!r}"
    )


# ===========================================================================
# 2. PHYLOGENETIC TREE BRANCH
# ===========================================================================

@pytest.mark.asyncio
async def test_tree_request_calls_phylogenetic_reconstruction_exactly_once(
    monkeypatch, make_orchestrator,
) -> None:
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE,
    )
    await agent.run(AgentRequest(
        instruction="Build a phylogenetic tree for these species.", context={}))
    assert phylo.count == 1, (
        f"Phylogenetic Reconstruction ran {phylo.count} times, expected 1"
    )


@pytest.mark.asyncio
async def test_tree_request_does_not_build_a_similarity_network(
    monkeypatch, make_orchestrator,
) -> None:
    """The tree branch owns its own mock alignment; MC must not run."""
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE,
    )
    await agent.run(AgentRequest(
        instruction="Build a phylogenetic tree for these species.", context={}))
    assert mc.count == 0, (
        f"Molecular Comparison ran {mc.count} times for a tree-only request"
    )


@pytest.mark.asyncio
async def test_tree_result_contains_the_tree(
    monkeypatch, make_orchestrator,
) -> None:
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a phylogenetic tree.", context={}))
    out = flat(result)
    assert out.get("newick_tree")
    assert out.get("bootstrap_support")
    assert out.get("model")


@pytest.mark.asyncio
async def test_tree_result_carries_no_network_fields(
    monkeypatch, make_orchestrator,
) -> None:
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a phylogenetic tree.", context={}))
    out = flat(result)

    leaked = [k for k in NETWORK_ONLY_KEYS if out.get(k) is not None]
    assert leaked == [], f"network fields leaked into a tree answer: {leaked}"
    assert result.similarity_scores is None
    assert result.alignment_url is None


# ===========================================================================
# 3. TWO DIFFERENT INTENTS -> TWO DIFFERENT EXECUTIONS
# ===========================================================================

@pytest.mark.asyncio
async def test_two_intents_produce_different_worker_call_patterns(
    monkeypatch, make_orchestrator,
) -> None:
    """Proven with spies, not by diffing the payload."""
    agent_n, mc_n, ph_n = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    await agent_n.run(AgentRequest(
        instruction="Compare these species and build a similarity network.",
        context={}))

    agent_t, mc_t, ph_t = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE,
    )
    await agent_t.run(AgentRequest(
        instruction="Build a phylogenetic tree for these species.", context={}))

    assert (mc_n.count, ph_n.count) == (1, 0), (
        f"network branch called (mc={mc_n.count}, phylo={ph_n.count})"
    )
    assert (mc_t.count, ph_t.count) == (0, 1), (
        f"tree branch called (mc={mc_t.count}, phylo={ph_t.count})"
    )


@pytest.mark.asyncio
async def test_two_intents_do_not_produce_identical_json(
    monkeypatch, make_orchestrator,
) -> None:
    agent_n, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    net = flat(await agent_n.run(AgentRequest(
        instruction="Compare these species and build a similarity network.",
        context={})))

    agent_t, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE,
    )
    tree = flat(await agent_t.run(AgentRequest(
        instruction="Build a phylogenetic tree for these species.", context={})))

    assert json.dumps(net, sort_keys=True) != json.dumps(tree, sort_keys=True), (
        "a similarity-network request and a tree request returned identical JSON"
    )


# ===========================================================================
# 4. AMBIGUOUS INTENT
# ===========================================================================

@pytest.mark.asyncio
async def test_ambiguous_request_does_not_run_either_branch(
    monkeypatch, make_orchestrator,
) -> None:
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator, feature=None, species=[],
    )
    await agent.run(AgentRequest(
        instruction="Tell me something about evolution.", context={}))
    assert mc.count == 0 and phylo.count == 0


@pytest.mark.asyncio
async def test_ambiguous_request_returns_a_structured_clarification(
    monkeypatch, make_orchestrator,
) -> None:
    """Ambiguity must surface as machine-readable state, not only prose.

    The Global Orchestrator has to be able to act on it: either a dict
    output carrying a clarification field, or a non-failed status that
    explicitly signals a question back to the user.
    """
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator, feature=None, species=[],
    )
    result = await agent.run(AgentRequest(
        instruction="Tell me something about evolution.", context={}))

    structured = (
        isinstance(result.output, dict)
        and any(k in result.output for k in
                ("clarification_question", "clarification", "question",
                 "needs_clarification", "decision"))
    ) or result.status is AgentStatus.CONTINUE

    assert structured, (
        "ambiguous intent produced only a prose string; the orchestrator "
        f"cannot route on it (status={result.status}, "
        f"output_type={type(result.output).__name__})"
    )


@pytest.mark.asyncio
async def test_ambiguous_request_invents_no_scientific_result(
    monkeypatch, make_orchestrator,
) -> None:
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator, feature=None, species=[],
    )
    result = await agent.run(AgentRequest(
        instruction="Tell me something about evolution.", context={}))
    out = flat(result)
    for key in TREE_ONLY_KEYS + NETWORK_ONLY_KEYS:
        assert out.get(key) is None, f"fabricated {key} for an ambiguous request"
    assert result.newick_tree is None
    assert result.similarity_scores is None


def test_full_analysis_is_not_advertised_as_the_ambiguity_fallback() -> None:
    """The classifier prompt must not tell the model to default to both.

    ``full_analysis`` is only legitimate when the user explicitly asks for
    both results; instructing the model to pick it "when in doubt" turns
    every ambiguous question into a double-branch run.
    """
    lowered = INTENT_SYSTEM_PROMPT.lower()
    assert "when in doubt" not in lowered, (
        "the intent prompt instructs the model to choose full_analysis "
        "when in doubt — ambiguity must trigger clarification instead"
    )


@pytest.mark.asyncio
async def test_missing_feature_does_not_silently_become_full_analysis(
    make_orchestrator,
) -> None:
    """A request that names no feature must not auto-run both branches."""
    mc = WorkerSpy("mc", mc_ok(THREE))
    phylo = WorkerSpy("phylo", phylo_ok(THREE))
    orch = make_orchestrator(mc_worker=mc, phylo_worker=phylo)

    await orch.run(AgentRequest(
        instruction="analyse these species", context={}, species_list=THREE))

    assert not (mc.called and phylo.called), (
        "an unspecified feature silently ran both branches (full_analysis "
        "fallback)"
    )


@pytest.mark.asyncio
async def test_explicit_both_request_is_the_only_double_branch_case(
    monkeypatch, make_orchestrator,
) -> None:
    """When the user explicitly asks for both, both may run."""
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="full_analysis", species=THREE,
    )
    await agent.run(AgentRequest(
        instruction=("Give me both a similarity network and a phylogenetic "
                     "tree for these species."),
        context={}))
    assert mc.count == 1 and phylo.count == 1


# ===========================================================================
# 5. INPUT CONSTRAINTS
# ===========================================================================

@pytest.mark.asyncio
async def test_tree_with_fewer_than_three_species_fails_in_its_own_branch(
    monkeypatch, make_orchestrator,
) -> None:
    """The 3-taxa rule must be enforced by the phylo branch, not globally."""
    phylo = WorkerSpy("phylo", AgentResult(
        status=AgentStatus.FAILED,
        output="Phylogenetic reconstruction requires at least 3 species; got 2.",
        source_agents=["Phylogenetic Tree Agent"],
    ))
    agent, mc, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=TWO, phylo=phylo,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a tree for human and chimp.", context={}))

    assert result.status is AgentStatus.FAILED
    assert "3" in str(result.output)
    assert mc.count == 0, "the unselected MC worker ran anyway"


@pytest.mark.asyncio
async def test_missing_species_fails_before_any_worker(
    monkeypatch, make_orchestrator,
) -> None:
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=[],
    )
    result = await agent.run(AgentRequest(
        instruction="Compare them.", context={}))
    assert result.status is AgentStatus.FAILED
    assert "species" in str(result.output).lower()
    assert mc.count == 0 and phylo.count == 0


@pytest.mark.asyncio
async def test_unknown_species_fails_before_any_worker(
    monkeypatch, make_orchestrator,
) -> None:
    agent, mc, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison",
        species=["homo sapiens", "draco magicus"],
    )
    result = await agent.run(AgentRequest(
        instruction="Compare them.", context={}))
    assert result.status is AgentStatus.FAILED
    assert "draco magicus" in str(result.output).lower()
    assert mc.count == 0 and phylo.count == 0


@pytest.mark.asyncio
async def test_empty_instruction_is_handled_without_crashing(
    monkeypatch, make_orchestrator,
) -> None:
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    result = await agent.run(AgentRequest(instruction="", context={}))
    assert isinstance(result, AgentResult)
    assert result.status in {AgentStatus.COMPLETED, AgentStatus.FAILED,
                             AgentStatus.CONTINUE, AgentStatus.NEEDS_AGENT}


@pytest.mark.asyncio
async def test_malformed_context_is_handled_without_crashing(
    monkeypatch, make_orchestrator,
) -> None:
    """A context whose values are the wrong type must not raise."""
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE,
    )
    for bad in ({"species_list": 12345}, {"species_list": {"a": 1}},
                {"feature": ["not", "a", "string"]}):
        result = await agent.run(
            AgentRequest(instruction="compare", context=bad))
        assert isinstance(result, AgentResult), bad


# ===========================================================================
# 6. MOCKED FAILURES
# ===========================================================================

@pytest.mark.asyncio
async def test_molecular_comparison_failure_is_reported_cleanly(
    monkeypatch, make_orchestrator,
) -> None:
    mc = WorkerSpy("mc", AgentResult(
        status=AgentStatus.FAILED, output="mock NCBI fetch failed",
        source_agents=["Molecular Comparison Agent"]))
    agent, _, phylo = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE, mc=mc,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a similarity network.", context={}))
    assert result.status is AgentStatus.FAILED
    assert "mock ncbi fetch failed" in str(result.output).lower()


@pytest.mark.asyncio
async def test_phylogenetic_failure_is_reported_cleanly(
    monkeypatch, make_orchestrator,
) -> None:
    phylo = WorkerSpy("phylo", AgentResult(
        status=AgentStatus.FAILED, output="mock IQ-TREE failed",
        source_agents=["Phylogenetic Tree Agent"]))
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="phylogenetic_tree", species=THREE, phylo=phylo,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a tree.", context={}))
    assert result.status is AgentStatus.FAILED
    assert "mock iq-tree failed" in str(result.output).lower()


@pytest.mark.asyncio
async def test_failure_of_an_unselected_worker_cannot_affect_the_request(
    monkeypatch, make_orchestrator,
) -> None:
    """A broken phylo mock must be irrelevant to a similarity request."""
    phylo = WorkerSpy("phylo", AgentResult(
        status=AgentStatus.FAILED, output="mock IQ-TREE is down"))
    agent, mc, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE, phylo=phylo,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a similarity network.", context={}))

    assert result.status is AgentStatus.COMPLETED, (
        f"an unselected worker's failure broke the request: {result.output!r}"
    )
    assert phylo.count == 0


@pytest.mark.asyncio
async def test_exception_of_an_unselected_worker_cannot_affect_the_request(
    monkeypatch, make_orchestrator,
) -> None:
    phylo = WorkerSpy("phylo", raises=RuntimeError("mock IQ-TREE segfault"))
    agent, mc, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE, phylo=phylo,
    )
    result = await agent.run(AgentRequest(
        instruction="Build a similarity network.", context={}))
    assert result.status is AgentStatus.COMPLETED
    assert phylo.count == 0


@pytest.mark.asyncio
async def test_worker_exception_becomes_a_failed_result_not_a_crash(
    monkeypatch, make_orchestrator,
) -> None:
    """The orchestrator must convert a worker crash into an AgentResult."""
    mc = WorkerSpy("mc", raises=RuntimeError("mock ESM-2 blew up"))
    agent, _, _ = make_agent(
        monkeypatch, make_orchestrator,
        feature="molecular_comparison", species=THREE, mc=mc,
    )
    try:
        result = await agent.run(AgentRequest(
            instruction="Build a similarity network.", context={}))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"worker exception escaped the orchestrator: "
            f"{type(exc).__name__}: {exc}"
        )
    assert result.status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_llm_timeout_does_not_run_a_branch(monkeypatch,
                                                 make_orchestrator) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _timeout(_p, **_k):
        return RecognizedIntent(feature=None, source="error")

    monkeypatch.setattr(mod, "classify_intent", _timeout)
    mc = WorkerSpy("mc", mc_ok(THREE))
    phylo = WorkerSpy("phylo", phylo_ok(THREE))
    agent = OrchestratorEvolutionAgent(
        orchestrator=make_orchestrator(mc_worker=mc, phylo_worker=phylo))

    result = await agent.run(AgentRequest(
        instruction="Compare human and mouse.", context={}))
    assert result.status is not AgentStatus.COMPLETED
    assert mc.count == 0 and phylo.count == 0


@pytest.mark.asyncio
async def test_invalid_llm_output_does_not_run_a_branch(
    monkeypatch, make_orchestrator,
) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _garbage(_p, **_k):
        return RecognizedIntent(feature=None, source="unparsable")

    monkeypatch.setattr(mod, "classify_intent", _garbage)
    mc = WorkerSpy("mc", mc_ok(THREE))
    phylo = WorkerSpy("phylo", phylo_ok(THREE))
    agent = OrchestratorEvolutionAgent(
        orchestrator=make_orchestrator(mc_worker=mc, phylo_worker=phylo))

    result = await agent.run(AgentRequest(
        instruction="Compare human and mouse.", context={}))
    assert result.status is not AgentStatus.COMPLETED
    assert mc.count == 0 and phylo.count == 0


# ===========================================================================
# 7. SUB-AGENT SEPARATION
# ===========================================================================

def test_exactly_two_sub_agent_worker_packages_are_shipped() -> None:
    """A third worker package must not exist, even dead."""
    packages = sorted(
        p.name for p in (AGENT_DIR / "workers").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    )
    assert packages == ["molecular_comparison", "phylogenetic_tree"], (
        f"unexpected worker packages present: {packages}"
    )


def test_orchestrator_wires_exactly_two_workers() -> None:
    src = (AGENT_DIR / "orchestrator/evolution_orchestrator.py").read_text(
        encoding="utf-8")
    assert "divergence_time" not in src
    assert "molecular_comparison" in src
    assert "phylogenetic_tree" in src


def test_sub_agents_are_distinct_classes_in_distinct_modules() -> None:
    from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
        MolecularComparisonMock,
    )
    from backend.agents.evolution_agent.workers.phylogenetic_tree.worker import (
        PhylogeneticTreeWorker,
    )
    assert MolecularComparisonMock is not PhylogeneticTreeWorker
    assert (MolecularComparisonMock.__module__
            != PhylogeneticTreeWorker.__module__)
    assert (MolecularComparisonMock.run.__code__
            is not PhylogeneticTreeWorker.run.__code__)


def test_sub_agents_have_disjoint_output_schemas() -> None:
    assert set(MolecularComparisonResult.__dataclass_fields__) & set(
        PhylogeneticResult.__dataclass_fields__) == set()


def test_sub_agents_never_import_or_call_each_other() -> None:
    """Neither sub-agent may import the other.

    Shared code (the UniProt fetch both branches need) lives in
    tools.sequences precisely so this stays true: a common dependency is
    fine, a sibling dependency is not.
    """
    import re as _re

    mc_files = ["workers/molecular_comparison/mock.py",
                "workers/molecular_comparison/logic.py"]
    ph_src = (AGENT_DIR / "workers/phylogenetic_tree/worker.py").read_text(
        encoding="utf-8")

    for rel in mc_files:
        src = (AGENT_DIR / rel).read_text(encoding="utf-8")
        assert not _re.search(r"^\s*from\s+\.\.phylogenetic_tree", src, _re.M), rel
        assert "PhylogeneticTreeWorker" not in src, rel

    assert not _re.search(r"^\s*from\s+\.\.molecular_comparison", ph_src, _re.M)
    assert "MolecularComparisonMock" not in ph_src
    assert "MolecularComparisonAgent" not in ph_src



# Directories that are not this agent's source: the vendored virtualenv and
# the unpacked MAFFT / IQ-TREE toolchains. Walking them is both slow and
# wrong -- third-party files are not bound by this agent's invariants, and
# some are not even valid UTF-8.
_VENDORED = {".venv", "venv", "site-packages", "__pycache__", "mafft-win", "iqtree", "node_modules"}


def _agent_sources():
    """Every .py file that is actually part of this agent."""
    for path in AGENT_DIR.rglob("*.py"):
        if _VENDORED & set(path.parts):
            continue
        yield path


def test_only_the_orchestrator_instantiates_the_workers() -> None:
    """No module other than the parent may construct a sub-agent."""
    offenders: list[str] = []
    for py in _agent_sources():
        if py.parent.name == "tests":
            continue
        if py.name in {"evolution_orchestrator.py", "mock.py", "worker.py", "__init__.py"}:
            continue
        src = py.read_text(encoding="utf-8")
        if "MolecularComparisonMock(" in src or "PhylogeneticTreeWorker(" in src:
            offenders.append(py.name)
    assert offenders == [], f"workers constructed outside the parent: {offenders}"


def test_sub_agents_are_independently_testable() -> None:
    from backend.agents.evolution_agent.workers.molecular_comparison.mock import (
        MolecularComparisonMock,
    )
    from backend.agents.evolution_agent.workers.phylogenetic_tree.worker import (
        PhylogeneticTreeWorker,
    )
    r1 = MolecularComparisonMock().run(
        AgentRequest(instruction="x", context={}, species_list=THREE))
    r2 = PhylogeneticTreeWorker().run(
        AgentRequest(instruction="x", context={}, species_list=THREE))
    assert r1.status is AgentStatus.COMPLETED
    assert r2.status is AgentStatus.COMPLETED
