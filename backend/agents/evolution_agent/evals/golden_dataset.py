"""Golden evaluation dataset for the Evolution Agent (Sprint 4, Task 1).

Every case is a real free-text prompt sent to ``POST /execute`` with an
empty context, so it exercises the actual Planner (LLM #1) end-to-end --
no ``feature`` bypass. This is deliberate: agent/tool selection is one of
the five things Sprint 4 asks us to test, and that can only be tested by
letting the Planner actually choose.

Species names are the offline-catalogue names covered by
``orchestrator/services/species_resolver.py`` -- see that file for the
full list. Cases that use a name outside it (e.g. "dog") are testing the
failure path on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    prompt: str

    # --- Agent/tool selection -------------------------------------------
    # The PlannedFeature value we expect the Planner to route to, or
    # "clarification_required" when the case is deliberately vague / must
    # be refused. Inferred from response *shape* by the runner, not from
    # an internal field, since a black-box caller has no other way to
    # know which worker(s) actually ran.
    expected_feature: str = ""

    # --- Task completion --------------------------------------------------
    expected_status: str = "completed"  # "completed" | "continue" | "failed"
    expected_species: list[str] = field(default_factory=list)  # order-independent

    # --- Correctness (deterministic, domain-grounded) ---------------------
    # For molecular_comparison cases: the pair we expect to be reported as
    # most similar, as a set of two lowercase species names.
    expect_closest_pair: tuple[str, str] | None = None

    # Repeat this case N times and check the Planner's routing decision
    # (and, when completed, the resolved species) stay identical across
    # runs -- the "response consistency" criterion.
    consistency_repeats: int = 1

    notes: str = ""


CASES: list[EvalCase] = [
    EvalCase(
        id="mc_three_species",
        prompt="How similar are human, chimp and mouse at the molecular level?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        expect_closest_pair=("homo sapiens", "pan troglodytes"),
        consistency_repeats=3,
        notes="Baseline case. Human-chimp must be the closest pair regardless of run.",
    ),
    EvalCase(
        id="mc_two_species",
        prompt="Compare the molecular similarity of human and chicken.",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Gallus gallus"],
    ),
    EvalCase(
        id="phylo_five_species",
        prompt="Build a phylogenetic tree for human, chimp, mouse, chicken and zebrafish.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=[
            "Homo sapiens", "Pan troglodytes", "Mus musculus",
            "Gallus gallus", "Danio rerio",
        ],
        notes="5 species so UFBoot actually runs (see worker: needs >=4 to bootstrap in practice).",
    ),
    EvalCase(
        id="phylo_too_few_species",
        prompt="Build a phylogenetic tree for human and chimp.",
        expected_feature="clarification_required",
        expected_status="continue",
        notes=(
            "Deterministic Planner guard: phylogenetic_tree needs >=3 species "
            "(planner.py _apply_guards). Tests that the guard fires regardless "
            "of what the LLM itself proposes."
        ),
    ),
    EvalCase(
        id="full_analysis_explicit_both",
        prompt="Give me both a similarity network and a phylogenetic tree for human, chimp and mouse.",
        expected_feature="full_analysis",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        notes="Must explicitly ask for both outputs to get full_analysis.",
    ),
    EvalCase(
        id="full_analysis_implicit_should_clarify",
        prompt="Tell me everything you can about human, chimp and mouse evolution.",
        expected_feature="clarification_required",
        expected_status="continue",
        notes=(
            "Does NOT explicitly request both a network and a tree. The "
            "planner must ask for clarification rather than guessing "
            "full_analysis -- this is a prompt-adherence test, genuinely "
            "uncertain until run."
        ),
    ),
    EvalCase(
        id="vague_no_species",
        prompt="Tell me about evolution.",
        expected_feature="clarification_required",
        expected_status="continue",
    ),
    EvalCase(
        id="off_topic",
        prompt="What's the weather like today?",
        expected_feature="clarification_required",
        expected_status="continue",
        notes="Off-topic request must not be misrouted into an evolutionary analysis.",
    ),
    EvalCase(
        id="unresolvable_species",
        prompt="Compare the molecular similarity of human and dog.",
        expected_feature="molecular_comparison",
        expected_status="failed",
        notes=(
            "'dog' is outside the offline species catalogue -- must fail "
            "cleanly with a clear message, not crash or hallucinate a result."
        ),
    ),
]
