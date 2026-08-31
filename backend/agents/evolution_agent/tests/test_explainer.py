"""Explainer (LLM #2) wiring tests.

Scope: only the Explainer connection —
    Planner -> selected sub-agent(s) -> structured result -> Explainer
    -> AgentResult with `interpretation`.

Everything is injected: the Planner decision, both workers and the
Explainer itself. No Azure, no EBI, no MAFFT, no IQ-TREE, no network.
The real phylogenetic pipeline is never triggered — a fake
``PhylogeneticResult`` is returned by an injected worker instead.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.agents.evolution_agent.explainer import (
    ALLOWED_INPUT_KEYS,
    explain,
    fallback_explanation,
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
    PlannedFeature,
    PlannerDecision,
    SimilarityEdge,
    SpeciesGroup,
)

THREE = ["homo sapiens", "pan troglodytes", "mus musculus"]

# Reference structured payloads — the values that must survive untouched.
MC_SCORES = [
    SimilarityEdge("homo sapiens", "pan troglodytes", 0.98),
    SimilarityEdge("homo sapiens", "mus musculus", 0.85),
]
MC_GROUPS = [SpeciesGroup(0, ["homo sapiens", "pan troglodytes"], 0.98)]
NEWICK = "(('homo sapiens':0.6,'pan troglodytes':0.6)95:8.4,'mus musculus':9.0);"
BOOTSTRAP = {"node_0": 95}
CONFIDENCES = {"node_0": 0.95}


# ===========================================================================
# Doubles
# ===========================================================================

def mc_result(species=None) -> AgentResult:
    species = species or THREE
    return AgentResult(
        status=AgentStatus.COMPLETED,
        output=MolecularComparisonResult(
            species_list=list(species),
            alignment=">a\nMTNI",
            alignment_url="http://x/align.html",
            similarity_scores=list(MC_SCORES),
            species_groups=list(MC_GROUPS),
            similarity_network={s: [] for s in species},
        ),
        source_agents=["Molecular Comparison Agent"],
    )


def phylo_result(species=None) -> AgentResult:
    """A FAKE PhylogeneticResult — MAFFT/IQ-TREE are never invoked."""
    species = species or THREE
    return AgentResult(
        status=AgentStatus.COMPLETED,
        output=PhylogeneticResult(
            newick_tree=NEWICK,
            tree_url="http://x/tree.svg",
            model="LG+G4",
            bootstrap_support=dict(BOOTSTRAP),
            confidence_values=dict(CONFIDENCES),
            overall_confidence=0.95,
        ),
        source_agents=["Phylogenetic Tree Agent"],
    )


class Worker:
    def __init__(self, result=None, raises=None):
        self._result, self._raises = result, raises
        self.calls: list[AgentRequest] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.calls.append(request)
        if self._raises is not None:
            raise self._raises
        return self._result

    @property
    def count(self) -> int:
        return len(self.calls)


class ExplainerSpy:
    """Records every call and what it was given."""

    def __init__(self, text="Interpretation text.", raises=None):
        self._text, self._raises = text, raises
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._text

    @property
    def count(self) -> int:
        return len(self.calls)


class StubLLM:
    """Minimal ainvoke-compatible stub — never touches a network."""

    def __init__(self, content="", raises=None):
        self._content, self._raises = content, raises
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return type("R", (), {"content": self._content})()


def build_agent(make_orchestrator, monkeypatch, *, feature, species=None,
                mc=None, phylo=None, explainer=None):
    """Full entry point with the Planner stubbed and everything injected."""
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    species = species or THREE

    async def _plan(_prompt, **_kw):
        return PlannerDecision(feature=feature, species_list=list(species),
                               source="llm")

    monkeypatch.setattr(mod, "classify_intent", _plan)

    orch = make_orchestrator(
        mc_worker=mc if mc is not None else Worker(mc_result(species)),
        phylo_worker=phylo if phylo is not None else Worker(phylo_result(species)),
        explainer=explainer if explainer is not None else ExplainerSpy(),
    )
    return OrchestratorEvolutionAgent(orchestrator=orch), orch


def ask(instruction="Compare these species."):
    return AgentRequest(instruction=instruction, context={})


# ===========================================================================
# 1-3. The Explainer runs exactly once per successful branch
# ===========================================================================

@pytest.mark.asyncio
async def test_molecular_success_calls_the_explainer_once(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison", explainer=spy)
    result = await agent.run(ask("Build a similarity network."))

    assert result.status is AgentStatus.COMPLETED
    assert spy.count == 1


@pytest.mark.asyncio
async def test_phylogenetic_success_calls_the_explainer_once(
    make_orchestrator, monkeypatch,
) -> None:
    """Uses an injected worker returning a fake PhylogeneticResult."""
    spy = ExplainerSpy()
    phylo = Worker(phylo_result())
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="phylogenetic_tree", phylo=phylo,
                           explainer=spy)
    result = await agent.run(ask("Build a phylogenetic tree."))

    assert result.status is AgentStatus.COMPLETED
    assert phylo.count == 1
    assert spy.count == 1


@pytest.mark.asyncio
async def test_full_analysis_calls_the_explainer_once_for_both(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    mc, phylo = Worker(mc_result()), Worker(phylo_result())
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="full_analysis", mc=mc, phylo=phylo,
                           explainer=spy)
    result = await agent.run(ask("Give me both a network and a tree."))

    assert result.status is AgentStatus.COMPLETED
    assert mc.count == 1 and phylo.count == 1
    assert spy.count == 1, "one combined explanation, not one per worker"
    assert set(spy.calls[0]["results"]) == {"similarity", "phylogeny"}


# ===========================================================================
# 4-6. Non-success paths never spend an Explainer call
# ===========================================================================

@pytest.mark.asyncio
async def test_clarification_never_calls_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    mc, phylo = Worker(mc_result()), Worker(phylo_result())
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="clarification_required", mc=mc,
                           phylo=phylo, explainer=spy)
    result = await agent.run(ask("Tell me something about evolution."))

    assert result.status is AgentStatus.CONTINUE
    assert result.output["decision"] == "clarification_required"
    assert spy.count == 0
    assert mc.count == 0 and phylo.count == 0


@pytest.mark.asyncio
async def test_worker_failure_never_calls_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    failing = Worker(AgentResult(status=AgentStatus.FAILED,
                                 output="worker refused",
                                 source_agents=["Molecular Comparison Agent"]))
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison", mc=failing,
                           explainer=spy)
    result = await agent.run(ask("Build a similarity network."))

    assert result.status is AgentStatus.FAILED
    assert spy.count == 0


@pytest.mark.asyncio
async def test_needs_agent_escalation_never_calls_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    escalating = Worker(AgentResult(
        status=AgentStatus.NEEDS_AGENT,
        target_agent="Genome Agent",
        prompt_to_target_agent="Fetch sequences.",
        source_agents=["Molecular Comparison Agent"]))
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison", mc=escalating,
                           explainer=spy)
    result = await agent.run(ask("Build a similarity network."))

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Genome Agent"
    assert spy.count == 0


@pytest.mark.asyncio
async def test_worker_exception_never_calls_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison",
                           mc=Worker(raises=RuntimeError("boom")),
                           explainer=spy)
    result = await agent.run(ask("Build a similarity network."))

    assert result.status is AgentStatus.FAILED
    assert spy.count == 0


# ===========================================================================
# 7. The Explainer sees only the whitelisted fields
# ===========================================================================

@pytest.mark.asyncio
async def test_explainer_receives_only_allowed_keys(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison", explainer=spy)
    await agent.run(ask("Build a similarity network."))

    given = spy.calls[0]
    assert set(given) <= ALLOWED_INPUT_KEYS, (
        f"unexpected keys passed to the Explainer: "
        f"{sorted(set(given) - ALLOWED_INPUT_KEYS)}"
    )
    assert given["instruction"] == "Build a similarity network."
    assert given["feature"] == "molecular_comparison"
    assert given["species"] == ["Homo sapiens", "Pan troglodytes", "Mus musculus"]
    assert isinstance(given["results"], dict)
    assert isinstance(given["warnings"], list)


@pytest.mark.asyncio
async def test_explainer_never_sees_an_unselected_worker_result(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison", explainer=spy)
    await agent.run(ask("Build a similarity network."))

    results = spy.calls[0]["results"]
    assert "similarity" in results
    assert "phylogeny" not in results, "unselected worker leaked into the prompt"


@pytest.mark.asyncio
async def test_explainer_never_sees_credentials_or_the_orchestrator(
    make_orchestrator, monkeypatch,
) -> None:
    spy = ExplainerSpy()
    agent, orch = build_agent(make_orchestrator, monkeypatch,
                              feature="phylogenetic_tree", explainer=spy)
    await agent.run(ask("Build a tree."))

    blob = repr(spy.calls[0])
    for forbidden in ("API_KEY", "api_key", "AZURE", "EvolutionOrchestrator",
                      "SpeciesResolverService", "Worker object"):
        assert forbidden not in blob, f"{forbidden} leaked to the Explainer"
    assert "orchestrator" not in spy.calls[0]
    assert "context" not in spy.calls[0]


# ===========================================================================
# 8-10. Output shape and immutability of the structured payload
# ===========================================================================

@pytest.mark.asyncio
async def test_output_contains_the_interpretation(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison",
                           explainer=ExplainerSpy("Humans and chimps cluster."))
    result = await agent.run(ask("Build a similarity network."))

    assert result.interpretation == "Humans and chimps cluster."
    assert result.output["interpretation"] == "Humans and chimps cluster."


@pytest.mark.asyncio
async def test_molecular_scores_are_untouched_by_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    """A wildly wrong explanation must not alter a single number."""
    agent, _ = build_agent(
        make_orchestrator, monkeypatch, feature="molecular_comparison",
        explainer=ExplainerSpy("Similarity is 0.11 and there are 9 groups."))
    result = await agent.run(ask("Build a similarity network."))

    assert result.output["similarity_scores"] == [
        {"species_a": "homo sapiens", "species_b": "pan troglodytes",
         "score": 0.98},
        {"species_a": "homo sapiens", "species_b": "mus musculus",
         "score": 0.85},
    ]
    assert result.output["species_groups"] == [
        {"group_id": 0, "species": ["homo sapiens", "pan troglodytes"],
         "mean_score": 0.98},
    ]


@pytest.mark.asyncio
async def test_newick_and_support_are_untouched_by_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(
        make_orchestrator, monkeypatch, feature="phylogenetic_tree",
        explainer=ExplainerSpy("The tree is ((a,b),c); with support 42."))
    result = await agent.run(ask("Build a tree."))

    assert result.output["newick_tree"] == NEWICK
    assert result.output["bootstrap_support"] == BOOTSTRAP
    assert result.output["confidence_values"] == CONFIDENCES
    assert result.output["model"] == "LG+G4"


@pytest.mark.asyncio
async def test_branch_separation_survives_the_explainer(
    make_orchestrator, monkeypatch,
) -> None:
    """Adding the Explainer must not reintroduce cross-branch fields."""
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison")
    net = await agent.run(ask("Build a similarity network."))
    assert net.newick_tree is None
    for key in ("newick_tree", "bootstrap_support", "model", "tree_url"):
        assert net.output.get(key) is None

    agent2, _ = build_agent(make_orchestrator, monkeypatch,
                            feature="phylogenetic_tree")
    tree = await agent2.run(ask("Build a tree."))
    assert tree.similarity_scores is None
    for key in ("similarity_network", "similarity_scores", "species_groups"):
        assert tree.output.get(key) is None


# ===========================================================================
# 11-13. Explainer failure is never fatal
# ===========================================================================

@pytest.mark.asyncio
async def test_explainer_exception_keeps_the_worker_result(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(
        make_orchestrator, monkeypatch, feature="molecular_comparison",
        explainer=ExplainerSpy(raises=RuntimeError("LLM exploded")))
    result = await agent.run(ask("Build a similarity network."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["similarity_scores"][0]["score"] == 0.98
    assert "interpretation_unavailable" in result.output["warnings"]


@pytest.mark.asyncio
async def test_explainer_timeout_keeps_the_worker_result(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(
        make_orchestrator, monkeypatch, feature="phylogenetic_tree",
        explainer=ExplainerSpy(raises=asyncio.TimeoutError("upstream")))
    result = await agent.run(ask("Build a tree."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["newick_tree"] == NEWICK
    assert "interpretation_unavailable" in result.output["warnings"]


@pytest.mark.asyncio
async def test_empty_explanation_produces_a_structured_warning(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="molecular_comparison",
                           explainer=ExplainerSpy(""))
    result = await agent.run(ask("Build a similarity network."))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["warnings"] == ["interpretation_unavailable"]
    # a minimal deterministic sentence still reaches the user
    assert result.output["interpretation"]
    assert "Analysed 3 species" in result.output["interpretation"]


# ===========================================================================
# 14. LLM budget
# ===========================================================================

@pytest.mark.asyncio
async def test_llm_budget_is_two_for_a_successful_branch(
    make_orchestrator, monkeypatch,
) -> None:
    for feature in ("molecular_comparison", "phylogenetic_tree",
                    "full_analysis"):
        agent, _ = build_agent(make_orchestrator, monkeypatch, feature=feature)
        result = await agent.run(ask("Analyse these species."))
        assert result.status is AgentStatus.COMPLETED
        assert result.llm_calls == 2, feature
        assert result.output["llm_calls"] == 2
        assert result.llm_calls <= 2


@pytest.mark.asyncio
async def test_llm_budget_is_one_for_clarification(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(make_orchestrator, monkeypatch,
                           feature="clarification_required")
    result = await agent.run(ask("Tell me something."))
    assert result.llm_calls == 1


@pytest.mark.asyncio
async def test_llm_budget_is_one_when_a_worker_fails(
    make_orchestrator, monkeypatch,
) -> None:
    agent, _ = build_agent(
        make_orchestrator, monkeypatch, feature="molecular_comparison",
        mc=Worker(AgentResult(status=AgentStatus.FAILED, output="nope")))
    result = await agent.run(ask("Build a similarity network."))
    assert result.status is AgentStatus.FAILED
    assert result.llm_calls == 1


@pytest.mark.asyncio
async def test_explainer_is_the_only_call_when_the_feature_is_supplied(
    make_orchestrator, monkeypatch,
) -> None:
    """Feature given by the Global Orchestrator -> Planner skipped."""
    spy = ExplainerSpy()
    orch = make_orchestrator(
        mc_worker=Worker(mc_result()),
        phylo_worker=Worker(phylo_result()),
        explainer=spy,
    )
    agent = OrchestratorEvolutionAgent(orchestrator=orch)
    result = await agent.run(AgentRequest(
        instruction="Build a similarity network.",
        context={"feature": "molecular_comparison", "species_list": THREE}))

    assert result.status is AgentStatus.COMPLETED
    assert spy.count == 1
    assert result.llm_calls == 1


# ===========================================================================
# The explain() module itself — no network, injected LLM
# ===========================================================================

RESULTS = {
    "similarity": {
        "similarity_scores": [
            {"species_a": "homo sapiens", "species_b": "pan troglodytes",
             "score": 0.98},
        ],
        "species_groups": [
            {"group_id": 0, "species": ["homo sapiens", "pan troglodytes"],
             "mean_score": 0.98},
        ],
        "network_nodes": 3,
    }
}


@pytest.mark.asyncio
async def test_explain_returns_the_model_text() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM("Humans and chimps group together."))
    assert got == "Humans and chimps group together."


@pytest.mark.asyncio
async def test_explain_returns_none_without_results() -> None:
    llm = StubLLM("anything")
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results={}, llm=llm)
    assert got is None
    assert llm.calls == 0, "no LLM call for an empty payload"


@pytest.mark.asyncio
async def test_explain_returns_none_on_exception() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM(raises=RuntimeError("down")))
    assert got is None


@pytest.mark.asyncio
async def test_explain_returns_none_on_empty_text() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS, llm=StubLLM("   "))
    assert got is None


@pytest.mark.asyncio
async def test_explain_rejects_a_fabricated_score() -> None:
    """A decimal that is not in the results is a hallucinated measurement."""
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM("Their similarity is 0.42."))
    assert got is None


@pytest.mark.asyncio
async def test_explain_accepts_a_score_present_in_the_results() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM("Similarity reaches 0.98 for the closest pair."))
    assert got == "Similarity reaches 0.98 for the closest pair."


@pytest.mark.asyncio
async def test_explain_rejects_a_species_outside_the_analysed_set() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM("Homo erectus is the closest relative."))
    assert got is None


# ---------------------------------------------------------------------------
# Grounding regression: genus followed by ordinary English prose
# ---------------------------------------------------------------------------

PHYLO_RESULTS = {
    "phylogeny": {
        "newick_tree": ("('Homo sapiens':0.0000022520,'Pan troglodytes':0.0160806457,"
                        "('Mus musculus':0.0000025762,'Gallus gallus':0.2227994490)"
                        "100:0.7755216079);"),
        "model": "MTVER+R2",
        "bootstrap_support": {"node_0": 100},
        "confidence_values": {"node_0": 1.0},
        "overall_confidence": 1.0,
    }
}
FOUR_SPECIES = ["Homo sapiens", "Pan troglodytes", "Mus musculus",
                "Gallus gallus"]


@pytest.mark.asyncio
async def test_explain_accepts_a_genus_followed_by_an_english_word() -> None:
    """Regression: 'Mus+Gallus node' was read as the species 'Gallus node'.

    A real IQ-TREE run produced exactly this sentence and the whole
    interpretation was discarded, leaving interpretation_unavailable.
    """
    text = ("The Mus+Gallus node has bootstrap support 100 and the overall "
            "confidence is 1.0, so that grouping is strongly supported.")
    got = await explain(instruction="Build a tree.",
                        feature="phylogenetic_tree", species=FOUR_SPECIES,
                        results=PHYLO_RESULTS, llm=StubLLM(text))
    assert got == text


@pytest.mark.parametrize("phrase", [
    "The Homo branch is short.",
    "The Gallus lineage is the most distant.",
    "Mus and Gallus cluster together.",
    "The Pan tip sits next to the human one.",
    "The Homo clade shows a similarity of 1.0.",
])
@pytest.mark.asyncio
async def test_explain_accepts_common_prose_after_a_genus(phrase) -> None:
    got = await explain(instruction="Build a tree.",
                        feature="phylogenetic_tree", species=FOUR_SPECIES,
                        results=PHYLO_RESULTS, llm=StubLLM(phrase))
    assert got == phrase, f"false positive on: {phrase!r}"


@pytest.mark.asyncio
async def test_explain_still_rejects_an_unprovided_species() -> None:
    """The guard must keep catching a genuinely hallucinated taxon."""
    got = await explain(instruction="Build a tree.",
                        feature="phylogenetic_tree", species=FOUR_SPECIES,
                        results=PHYLO_RESULTS,
                        llm=StubLLM("Homo erectus branches before Homo sapiens."))
    assert got is None


@pytest.mark.asyncio
async def test_explain_still_rejects_a_fabricated_support_value() -> None:
    """Numerical grounding must survive the false-positive fix."""
    got = await explain(instruction="Build a tree.",
                        feature="phylogenetic_tree", species=FOUR_SPECIES,
                        results=PHYLO_RESULTS,
                        llm=StubLLM("The node has a confidence of 0.73."))
    assert got is None


@pytest.mark.asyncio
async def test_explain_accepts_a_species_pair_that_was_analysed() -> None:
    text = ("Mus musculus and Gallus gallus form a clade with bootstrap "
            "support 100.")
    got = await explain(instruction="Build a tree.",
                        feature="phylogenetic_tree", species=FOUR_SPECIES,
                        results=PHYLO_RESULTS, llm=StubLLM(text))
    assert got == text


@pytest.mark.asyncio
async def test_explain_allows_plain_integers() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM("The 3 species form 1 group."))
    assert got is not None


@pytest.mark.asyncio
async def test_explain_rejects_an_overlong_answer() -> None:
    got = await explain(instruction="compare", feature="molecular_comparison",
                        species=THREE, results=RESULTS,
                        llm=StubLLM("word " * 500))
    assert got is None


def test_fallback_explanation_is_deterministic_and_grounded() -> None:
    a = fallback_explanation("molecular_comparison", THREE, RESULTS)
    b = fallback_explanation("molecular_comparison", THREE, RESULTS)
    assert a == b
    assert "Analysed 3 species" in a
    assert "0.98" in a


def test_explain_signature_is_keyword_only() -> None:
    """Positional passing must be impossible — it is how the wrong object
    ends up in an LLM prompt."""
    import inspect
    sig = inspect.signature(explain)
    positional = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert positional == []
