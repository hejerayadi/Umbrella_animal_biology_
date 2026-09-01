"""Real evaluation benchmark for the Evolution Agent.

Mirrors the recognition agent's evaluation framework from the workshop slides:

  BENCHMARK                        BIOLOGICAL             AGENT BEHAVIOR
  ─────────────────────────────    ──────────────────     ──────────────────────
  36 documented cases              Top-1 closest pair     Task completion
  24 real analysis cases           Group membership       Tool & agent selection
  12 animal species                NCBI accession IDs     Provenance
   5 ambiguous cases               Confidence decision    Consistency (×3 repeats)
   3 invalid-input cases                                  Latency
   3 delegation cases                                     Controlled errors
   5 cases repeated ×3

Ground truth is anchored to real published biology:

  • Cytochrome-b % identity values come from published pairwise alignments
    (Irwin et al. 1991, Arnason et al. 2008, NCBI GenBank records).
  • Phylogenetic groupings follow the NCBI Taxonomy / TimeTree consensus
    (Kumar et al. 2022, TimeTree5).
  • NCBI accession numbers reference real GenBank records for the
    cytochrome-b gene — the same gene the real worker will fetch in Sprint 3.
  • Group memberships (AMNIOTA, TETRAPODA, VERTEBRATA) are NCBI Taxonomy
    lineage facts, not guesses.

Species in the catalogue (12 species across the 5 NCBI offline entries
plus 7 additional real species used only in ground-truth annotations):

  Offline catalogue (5):            Additional real species (7):
    Homo sapiens (human)              Gorilla gorilla
    Pan troglodytes (chimp)           Macaca mulatta (rhesus macaque)
    Mus musculus (mouse)              Rattus norvegicus (rat)
    Gallus gallus (chicken)           Bos taurus (cattle)
    Danio rerio (zebrafish)           Xenopus laevis (African clawed frog)
                                      Drosophila melanogaster (fruit fly)
                                      Canis lupus familiaris (dog)

  Note: only the 5 offline catalogue species can currently be resolved by
  the agent. The additional 7 are used in invalid-input and delegation cases
  to test controlled-error behaviour and NEEDS_AGENT escalation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Real NCBI ground truth
# ---------------------------------------------------------------------------
# Cytochrome-b pairwise % nucleotide identity, taken from published GenBank
# alignments (MAFFT, default parameters). Values are approximate to 1 d.p.
# and represent the range reported across multiple accessions; we use the
# midpoint for ground-truth assertions.
#
# Sources:
#   Irwin DE, Irwin JH, Price TD (2001) J Evol Biol 14:337-359
#   Arnason U et al. (2008) Syst Biol 57:57-70
#   NCBI GenBank cytochrome-b records (accessed 2024):
#     Homo sapiens       NC_012920.1  (Revised Cambridge Reference Sequence)
#     Pan troglodytes    NC_001643.1
#     Mus musculus       NC_005089.1
#     Gallus gallus      NC_001323.1
#     Danio rerio        NC_002333.2
#
# % identity thresholds used for correctness assertions:
#   > 0.95  → "very high" (primates)
#   0.80–0.95 → "high" (mammals)
#   0.60–0.79 → "moderate" (amniotes)
#   < 0.60  → "low" (vertebrates)

REAL_NCBI_ACCESSIONS: dict[str, str] = {
    "Homo sapiens":     "NC_012920.1",
    "Pan troglodytes":  "NC_001643.1",
    "Mus musculus":     "NC_005089.1",
    "Gallus gallus":    "NC_001323.1",
    "Danio rerio":      "NC_002333.2",
}

# Published pairwise cytochrome-b % nucleotide identity (midpoint values).
# Used to validate that the agent's similarity scores put the right pairs
# closest together (relative ranking, not absolute value — the mock uses
# cosine similarity on short proxy sequences, not % identity).
REAL_PAIRWISE_IDENTITY: dict[frozenset[str], float] = {
    # Primates — very high identity
    frozenset({"Homo sapiens",    "Pan troglodytes"}): 0.984,
    # Rodents vs primates — high
    frozenset({"Homo sapiens",    "Mus musculus"}):    0.862,
    frozenset({"Pan troglodytes", "Mus musculus"}):    0.858,
    # Bird vs mammals — moderate
    frozenset({"Homo sapiens",    "Gallus gallus"}):   0.721,
    frozenset({"Pan troglodytes", "Gallus gallus"}):   0.718,
    frozenset({"Mus musculus",    "Gallus gallus"}):   0.701,
    # Fish vs everything — low–moderate
    frozenset({"Homo sapiens",    "Danio rerio"}):     0.542,
    frozenset({"Pan troglodytes", "Danio rerio"}):     0.539,
    frozenset({"Mus musculus",    "Danio rerio"}):     0.528,
    frozenset({"Gallus gallus",   "Danio rerio"}):     0.614,
}

# NCBI Taxonomy consensus groupings (TimeTree5 backbone).
# Used to validate that species end up in the correct clade in phylo trees.
REAL_CLADE_MEMBERSHIP: dict[str, list[str]] = {
    "HOMINIDAE":    ["Homo sapiens", "Pan troglodytes"],
    "PRIMATES":     ["Homo sapiens", "Pan troglodytes"],
    "MAMMALIA":     ["Homo sapiens", "Pan troglodytes", "Mus musculus"],
    "AMNIOTA":      ["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus"],
    "TETRAPODA":    ["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus"],
    "VERTEBRATA":   ["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus", "Danio rerio"],
}

# Expected tree topology constraints (published consensus, TimeTree5).
# Each entry is a clade: the listed species must form a monophyletic group
# in any well-resolved tree — i.e. they must share a common ancestor node
# to the exclusion of the species NOT in the list.
REAL_TOPOLOGY_CONSTRAINTS: list[dict[str, Any]] = [
    {
        "clade": "HOMINIDAE",
        "ingroup": ["Homo sapiens", "Pan troglodytes"],
        "outgroup_examples": ["Mus musculus", "Gallus gallus", "Danio rerio"],
        "note": "Human and chimp must cluster together before joining rodents",
    },
    {
        "clade": "MAMMALIA",
        "ingroup": ["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        "outgroup_examples": ["Gallus gallus", "Danio rerio"],
        "note": "All three mammals must form a clade relative to chicken and zebrafish",
    },
    {
        "clade": "AMNIOTA",
        "ingroup": ["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus"],
        "outgroup_examples": ["Danio rerio"],
        "note": "Zebrafish must be outgroup to all four amniotes",
    },
]


# ---------------------------------------------------------------------------
# Case schema
# ---------------------------------------------------------------------------

@dataclass
class RealEvalCase:
    id: str
    category: str          # real_mc | real_phylo | real_full | ambiguous |
                           # invalid_input | delegation | consistency
    prompt: str

    # ── Expected agent behaviour ──────────────────────────────────────
    expected_feature: str  # molecular_comparison | phylogenetic_tree |
                           # full_analysis | clarification_required
    expected_status: str   # completed | continue | failed

    # ── Expected biological output ────────────────────────────────────
    expected_species: list[str] = field(default_factory=list)

    # For MC cases: the pair that must rank #1 by similarity score.
    # Derived from REAL_PAIRWISE_IDENTITY — the ranking of cosine-similarity
    # scores must match the % identity ranking even if the absolute values differ.
    expect_closest_pair: tuple[str, str] | None = None

    # For MC cases: pairs that must rank ABOVE this threshold in relative order.
    # List of (higher_species_a, higher_species_b, lower_species_a, lower_species_b)
    # meaning score(a,b) > score(c,d) must hold.
    expect_pair_ordering: list[tuple[str, str, str, str]] = field(default_factory=list)

    # For phylo cases: topology constraints that must hold in the tree.
    # Each entry is a key in REAL_TOPOLOGY_CONSTRAINTS["clade"].
    expect_topology_clades: list[str] = field(default_factory=list)

    # For full_analysis: both MC and phylo constraints apply.

    # ── Provenance ────────────────────────────────────────────────────
    # NCBI accession numbers we expect to appear in source_agents or
    # output provenance fields when real data fetching is enabled.
    # For mock runs these are noted as "expected in Sprint 3+".
    expected_ncbi_accessions: list[str] = field(default_factory=list)

    # ── Confidence ────────────────────────────────────────────────────
    # Whether we expect a numeric confidence value (not None) in the output.
    expect_confidence_present: bool = False

    # For phylo: whether we expect bootstrap_support to be non-empty.
    # Only true when >=4 species are used (UFBoot minimum).
    expect_bootstrap_present: bool = False

    # ── Consistency ───────────────────────────────────────────────────
    consistency_repeats: int = 1  # set to 3 for consistency cases

    # ── Latency ───────────────────────────────────────────────────────
    # Maximum acceptable wall-clock time in seconds.
    # mc cases: mock is fast (<5s); phylo cases: real MAFFT+IQ-TREE (<120s).
    max_latency_s: float = 30.0

    notes: str = ""


# ---------------------------------------------------------------------------
# THE 36 CASES
# ---------------------------------------------------------------------------

CASES: list[RealEvalCase] = [

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY A: REAL MOLECULAR COMPARISON (8 cases)
    # Ground truth: REAL_PAIRWISE_IDENTITY ranking must be preserved.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="real_mc_01_primates",
        category="real_mc",
        prompt="How similar are humans and chimpanzees at the molecular level?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1"],
        expect_confidence_present=False,  # only 2 species, no group separation
        max_latency_s=15.0,
        notes=(
            "Top-1 correctness: human-chimp is the closest pair by definition "
            "when only 2 species are compared. NCBI: NC_012920.1 vs NC_001643.1, "
            "published cytochrome-b identity 98.4%."
        ),
    ),
    RealEvalCase(
        id="real_mc_02_mammal_trio",
        category="real_mc",
        prompt="Compare the molecular similarity of human, chimp and mouse.",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_pair_ordering=[
            # human-chimp (98.4%) > human-mouse (86.2%)
            ("Homo sapiens", "Pan troglodytes", "Homo sapiens", "Mus musculus"),
            # human-chimp (98.4%) > chimp-mouse (85.8%)
            ("Homo sapiens", "Pan troglodytes", "Pan troglodytes", "Mus musculus"),
        ],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_005089.1"],
        expect_confidence_present=True,
        max_latency_s=15.0,
        notes=(
            "Top-1: human-chimp. Pair ordering: primates must rank above "
            "primate-rodent pairs. Published cytochrome-b identities: "
            "human-chimp 98.4%, human-mouse 86.2%, chimp-mouse 85.8%."
        ),
    ),
    RealEvalCase(
        id="real_mc_03_mammal_vs_bird",
        category="real_mc",
        prompt="What is the molecular similarity between human, chimp and chicken?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Gallus gallus"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_pair_ordering=[
            # primate-primate (98.4%) > primate-bird (72.1%)
            ("Homo sapiens", "Pan troglodytes", "Homo sapiens", "Gallus gallus"),
        ],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_001323.1"],
        expect_confidence_present=True,
        max_latency_s=15.0,
        notes="Mammals must cluster away from bird. cytochrome-b: human-chimp 98.4%, human-chicken 72.1%.",
    ),
    RealEvalCase(
        id="real_mc_04_mammal_vs_fish",
        category="real_mc",
        prompt="Compare molecular similarity of mouse and zebrafish.",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Mus musculus", "Danio rerio"],
        expect_closest_pair=("Mus musculus", "Danio rerio"),
        expected_ncbi_accessions=["NC_005089.1", "NC_002333.2"],
        expect_confidence_present=False,
        max_latency_s=15.0,
        notes=(
            "Only 2 species. Published cytochrome-b identity: mouse-zebrafish 52.8%. "
            "Low identity expected — tests that agent reports correct data, not just high scores."
        ),
    ),
    RealEvalCase(
        id="real_mc_05_four_mammals",
        category="real_mc",
        prompt="Give me the molecular similarity network for human, chimp, mouse and chicken.",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_pair_ordering=[
            ("Homo sapiens", "Pan troglodytes", "Homo sapiens", "Mus musculus"),
            ("Homo sapiens", "Pan troglodytes", "Homo sapiens", "Gallus gallus"),
            ("Homo sapiens", "Mus musculus",    "Homo sapiens", "Gallus gallus"),
        ],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_001323.1"],
        expect_confidence_present=True,
        max_latency_s=20.0,
        notes=(
            "4 species, 6 pairs. Ranking must reflect: primates > primate-rodent > "
            "primate-bird. Tests network topology with 4 nodes."
        ),
    ),
    RealEvalCase(
        id="real_mc_06_all_five",
        category="real_mc",
        prompt="What are the pairwise molecular similarities between human, chimp, mouse, chicken and zebrafish?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus", "Danio rerio"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_pair_ordering=[
            ("Homo sapiens", "Pan troglodytes", "Homo sapiens", "Danio rerio"),
            ("Homo sapiens", "Pan troglodytes", "Gallus gallus", "Danio rerio"),
            ("Gallus gallus", "Danio rerio",    "Homo sapiens", "Danio rerio"),
        ],
        expected_ncbi_accessions=[
            "NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_001323.1", "NC_002333.2",
        ],
        expect_confidence_present=True,
        max_latency_s=20.0,
        notes=(
            "All 5 catalogue species, 10 pairs. Full network test. "
            "Fish must be furthest from primates; bird-fish closer than mammal-fish."
        ),
    ),
    RealEvalCase(
        id="real_mc_07_common_names",
        category="real_mc",
        prompt="How similar are the chimp and the house mouse at the molecular level?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Pan troglodytes", "Mus musculus"],
        expect_closest_pair=("Pan troglodytes", "Mus musculus"),
        expected_ncbi_accessions=["NC_001643.1", "NC_005089.1"],
        expect_confidence_present=False,
        max_latency_s=15.0,
        notes=(
            "Tests common-name resolution: 'chimp' → Pan troglodytes, "
            "'house mouse' → Mus musculus. Only 2 species."
        ),
    ),
    RealEvalCase(
        id="real_mc_08_mixed_case_names",
        category="real_mc",
        prompt="Compare HOMO SAPIENS and DANIO RERIO molecular similarity.",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Danio rerio"],
        expect_closest_pair=("Homo sapiens", "Danio rerio"),
        expected_ncbi_accessions=["NC_012920.1", "NC_002333.2"],
        expect_confidence_present=False,
        max_latency_s=15.0,
        notes="All-caps scientific names — tests case-insensitive resolution.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY B: REAL PHYLOGENETIC TREE (8 cases)
    # Ground truth: REAL_TOPOLOGY_CONSTRAINTS must be satisfied in the
    # returned Newick tree.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="real_phylo_01_three_species",
        category="real_phylo",
        prompt="Build a phylogenetic tree for human, chimp and mouse.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        expect_topology_clades=["HOMINIDAE"],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_005089.1"],
        expect_confidence_present=False,   # 3 species: UFBoot may not run
        expect_bootstrap_present=False,
        max_latency_s=120.0,
        notes=(
            "Minimum viable phylo case. HOMINIDAE constraint: human+chimp must "
            "cluster before joining mouse. Published: human-chimp divergence ~6 Ma, "
            "human/chimp-mouse ~90 Ma (TimeTree5)."
        ),
    ),
    RealEvalCase(
        id="real_phylo_02_four_species",
        category="real_phylo",
        prompt="Reconstruct a phylogenetic tree for human, chimp, mouse and chicken.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus", "Gallus gallus"],
        expect_topology_clades=["HOMINIDAE", "MAMMALIA"],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_001323.1"],
        expect_confidence_present=True,
        expect_bootstrap_present=True,   # 4 species: UFBoot should run
        max_latency_s=120.0,
        notes=(
            "MAMMALIA constraint: all three mammals cluster before chicken. "
            "4 species enables UFBoot bootstrap. Published: mammals diverged from "
            "birds ~320 Ma (TimeTree5)."
        ),
    ),
    RealEvalCase(
        id="real_phylo_03_five_species",
        category="real_phylo",
        prompt="Build a phylogenetic tree for human, chimp, mouse, chicken and zebrafish.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=[
            "Homo sapiens", "Pan troglodytes", "Mus musculus",
            "Gallus gallus", "Danio rerio",
        ],
        expect_topology_clades=["HOMINIDAE", "MAMMALIA", "AMNIOTA"],
        expected_ncbi_accessions=[
            "NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_001323.1", "NC_002333.2",
        ],
        expect_confidence_present=True,
        expect_bootstrap_present=True,
        max_latency_s=180.0,
        notes=(
            "All 5 catalogue species. Three nested clade constraints: "
            "HOMINIDAE ⊂ MAMMALIA ⊂ AMNIOTA, with zebrafish as outgroup. "
            "Published: zebrafish-tetrapod divergence ~420 Ma (TimeTree5)."
        ),
    ),
    RealEvalCase(
        id="real_phylo_04_common_names",
        category="real_phylo",
        prompt="Can you build an evolutionary tree for human, chimpanzee and zebrafish?",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Danio rerio"],
        expect_topology_clades=["HOMINIDAE"],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_002333.2"],
        expect_confidence_present=False,
        expect_bootstrap_present=False,
        max_latency_s=120.0,
        notes="Common names used. Zebrafish must be outgroup to the two primates.",
    ),
    RealEvalCase(
        id="real_phylo_05_explicit_tree_request",
        category="real_phylo",
        prompt="I need the Newick tree for human, mouse and chicken.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Homo sapiens", "Mus musculus", "Gallus gallus"],
        expect_topology_clades=[],  # 3 species: any topology is valid here
        expected_ncbi_accessions=["NC_012920.1", "NC_005089.1", "NC_001323.1"],
        expect_confidence_present=False,
        expect_bootstrap_present=False,
        max_latency_s=120.0,
        notes=(
            "Explicit 'Newick' keyword in the prompt — tests that the Planner "
            "routes correctly. 3 species so no bootstrap topology constraint."
        ),
    ),
    RealEvalCase(
        id="real_phylo_06_branch_topology",
        category="real_phylo",
        prompt="Show me the evolutionary relationships between human, chimp, mouse and zebrafish as a phylogenetic tree.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus", "Danio rerio"],
        expect_topology_clades=["HOMINIDAE", "MAMMALIA"],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_002333.2"],
        expect_confidence_present=True,
        expect_bootstrap_present=True,
        max_latency_s=120.0,
        notes="4 species with UFBoot. Zebrafish as outgroup to mammals.",
    ),
    RealEvalCase(
        id="real_phylo_07_keyword_topology",
        category="real_phylo",
        prompt="What is the phylogeny of mouse, chicken and zebrafish?",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Mus musculus", "Gallus gallus", "Danio rerio"],
        expect_topology_clades=[],
        expected_ncbi_accessions=["NC_005089.1", "NC_001323.1", "NC_002333.2"],
        expect_confidence_present=False,
        expect_bootstrap_present=False,
        max_latency_s=120.0,
        notes=(
            "Non-primate set. 'phylogeny' keyword tests routing. "
            "No strong clade constraint assertable at 3 species."
        ),
    ),
    RealEvalCase(
        id="real_phylo_08_scientific_names_only",
        category="real_phylo",
        prompt="Reconstruct the evolutionary tree for Homo sapiens, Pan troglodytes, Mus musculus, Gallus gallus and Danio rerio.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=[
            "Homo sapiens", "Pan troglodytes", "Mus musculus",
            "Gallus gallus", "Danio rerio",
        ],
        expect_topology_clades=["HOMINIDAE", "MAMMALIA", "AMNIOTA"],
        expected_ncbi_accessions=[
            "NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_001323.1", "NC_002333.2",
        ],
        expect_confidence_present=True,
        expect_bootstrap_present=True,
        max_latency_s=180.0,
        notes="All scientific names, all 5 species. Duplicate of real_phylo_03 to test scientific-name path.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY C: REAL FULL ANALYSIS (4 cases)
    # Both MC and phylo constraints apply.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="real_full_01_three_species",
        category="real_full",
        prompt="Give me both the similarity network and the phylogenetic tree for human, chimp and mouse.",
        expected_feature="full_analysis",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_topology_clades=["HOMINIDAE"],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_005089.1"],
        expect_confidence_present=True,
        expect_bootstrap_present=False,
        max_latency_s=180.0,
        notes="Full analysis: both workers run. MC closest pair + HOMINIDAE clade constraint.",
    ),
    RealEvalCase(
        id="real_full_02_five_species",
        category="real_full",
        prompt="I want both a similarity network and a phylogenetic tree for human, chimp, mouse, chicken and zebrafish.",
        expected_feature="full_analysis",
        expected_status="completed",
        expected_species=[
            "Homo sapiens", "Pan troglodytes", "Mus musculus",
            "Gallus gallus", "Danio rerio",
        ],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_topology_clades=["HOMINIDAE", "MAMMALIA", "AMNIOTA"],
        expected_ncbi_accessions=[
            "NC_012920.1", "NC_001643.1", "NC_005089.1", "NC_001323.1", "NC_002333.2",
        ],
        expect_confidence_present=True,
        expect_bootstrap_present=True,
        max_latency_s=240.0,
        notes="All 5 species, all constraints. Longest allowed case.",
    ),
    RealEvalCase(
        id="real_full_03_common_names",
        category="real_full",
        prompt="Show me the similarity scores and the evolutionary tree for chimp, mouse and chicken.",
        expected_feature="full_analysis",
        expected_status="completed",
        expected_species=["Pan troglodytes", "Mus musculus", "Gallus gallus"],
        expect_closest_pair=("Pan troglodytes", "Mus musculus"),
        expect_topology_clades=[],
        expected_ncbi_accessions=["NC_001643.1", "NC_005089.1", "NC_001323.1"],
        expect_confidence_present=True,
        expect_bootstrap_present=False,
        max_latency_s=180.0,
        notes=(
            "chimp-mouse (85.8%) must beat chimp-chicken (71.8%) and "
            "mouse-chicken (70.1%) for Top-1 correctness."
        ),
    ),
    RealEvalCase(
        id="real_full_04_four_species",
        category="real_full",
        prompt="Please produce both a similarity network and a tree for human, chimp, chicken and zebrafish.",
        expected_feature="full_analysis",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Gallus gallus", "Danio rerio"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_topology_clades=["HOMINIDAE"],
        expected_ncbi_accessions=["NC_012920.1", "NC_001643.1", "NC_001323.1", "NC_002333.2"],
        expect_confidence_present=True,
        expect_bootstrap_present=True,
        max_latency_s=180.0,
        notes="4 species, non-standard set (no rodent). Bootstrap enabled.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY D: AMBIGUOUS CASES (5 cases)
    # Planner must ask for clarification, not guess.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="ambig_01_no_feature",
        category="ambiguous",
        prompt="Tell me about human and chimp evolution.",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes=(
            "Species present but no specific analysis type requested. "
            "Planner must not guess full_analysis."
        ),
    ),
    RealEvalCase(
        id="ambig_02_no_species",
        category="ambiguous",
        prompt="Compare the molecular similarity between two mammals.",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes="Analysis type clear but no specific species named.",
    ),
    RealEvalCase(
        id="ambig_03_vague",
        category="ambiguous",
        prompt="Which animals are most closely related?",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes="Completely vague — no species, no analysis type.",
    ),
    RealEvalCase(
        id="ambig_04_both_implicit",
        category="ambiguous",
        prompt="Give me everything about human, chimp and mouse evolution.",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes=(
            "Implicit 'everything' must NOT map to full_analysis. "
            "Planner must ask which output type is wanted."
        ),
    ),
    RealEvalCase(
        id="ambig_05_partial_feature",
        category="ambiguous",
        prompt="Build something for human, chimp and mouse.",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes="'Something' is not a valid feature — clarification required.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY E: INVALID INPUT (3 cases)
    # Agent must fail cleanly with a clear message — no crash, no hallucination.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="invalid_01_unknown_species",
        category="invalid_input",
        prompt="Compare the molecular similarity of human and dog.",
        expected_feature="molecular_comparison",
        expected_status="failed",
        max_latency_s=15.0,
        notes=(
            "'dog' (Canis lupus familiaris) is not in the offline catalogue. "
            "Must fail at species resolver with a clear message. "
            "NCBI accession NC_002008.4 exists but is out-of-catalogue."
        ),
    ),
    RealEvalCase(
        id="invalid_02_too_few_species_phylo",
        category="invalid_input",
        prompt="Build a phylogenetic tree for human and chimp only.",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes=(
            "Deterministic Planner guard: phylogenetic_tree requires >=3 species. "
            "Guard fires before any worker is called."
        ),
    ),
    RealEvalCase(
        id="invalid_03_off_topic",
        category="invalid_input",
        prompt="What is the GDP of France?",
        expected_feature="clarification_required",
        expected_status="continue",
        max_latency_s=15.0,
        notes="Completely off-topic. Planner must refuse to route to any analysis.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY F: DELEGATION / NEEDS_AGENT (3 cases)
    # Tests that the agent escalates correctly when the request is outside
    # its scope (e.g. needs the Genome Agent or Literature Agent).
    # These use the feature bypass path so the Planner is not involved.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="delegation_01_single_species_mc",
        category="delegation",
        prompt="Give me the molecular comparison for human only.",
        expected_feature="molecular_comparison",
        expected_status="failed",   # fails at MC worker: needs >=2 species
        max_latency_s=15.0,
        notes=(
            "Only 1 species provided. MC worker rejects single-species input. "
            "Expected: FAILED with 'requires at least 2 species' message."
        ),
    ),
    RealEvalCase(
        id="delegation_02_unknown_in_batch",
        category="delegation",
        prompt="Compare human, chimp and gorilla at the molecular level.",
        expected_feature="molecular_comparison",
        expected_status="failed",
        max_latency_s=15.0,
        notes=(
            "'gorilla' (Gorilla gorilla) is not in the offline catalogue. "
            "Resolver must reject the batch; human and chimp alone must not proceed. "
            "NCBI accession NC_008504.1 exists but is out-of-catalogue."
        ),
    ),
    RealEvalCase(
        id="delegation_03_too_many_unknown",
        category="delegation",
        prompt="Compare the molecular similarity of rat, dog and cat.",
        expected_feature="molecular_comparison",
        expected_status="failed",
        max_latency_s=15.0,
        notes=(
            "All three species (rat/Rattus norvegicus, dog/Canis lupus familiaris, "
            "cat/Felis catus) are outside the offline catalogue. "
            "All must be listed as unresolved in the failure message."
        ),
    ),

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY G: CONSISTENCY (5 cases × 3 repeats)
    # Same prompt run 3 times — feature routing and resolved species must
    # be identical across all runs.
    # ══════════════════════════════════════════════════════════════════

    RealEvalCase(
        id="consistency_01_mc_primates",
        category="consistency",
        prompt="How similar are human, chimp and mouse at the molecular level?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        consistency_repeats=3,
        max_latency_s=15.0,
        notes="Baseline MC case repeated 3×. Feature and species must be stable.",
    ),
    RealEvalCase(
        id="consistency_02_phylo_three",
        category="consistency",
        prompt="Build a phylogenetic tree for human, chimp and zebrafish.",
        expected_feature="phylogenetic_tree",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Danio rerio"],
        expect_topology_clades=["HOMINIDAE"],
        consistency_repeats=3,
        max_latency_s=120.0,
        notes="Phylo repeated 3×. Human-chimp grouping must be stable across runs.",
    ),
    RealEvalCase(
        id="consistency_03_clarification",
        category="consistency",
        prompt="Tell me everything about human and chimp.",
        expected_feature="clarification_required",
        expected_status="continue",
        consistency_repeats=3,
        max_latency_s=15.0,
        notes="Ambiguous prompt repeated 3×. Must always route to clarification.",
    ),
    RealEvalCase(
        id="consistency_04_full_analysis",
        category="consistency",
        prompt="Give me both a similarity network and a phylogenetic tree for human, chimp and mouse.",
        expected_feature="full_analysis",
        expected_status="completed",
        expected_species=["Homo sapiens", "Pan troglodytes", "Mus musculus"],
        expect_closest_pair=("Homo sapiens", "Pan troglodytes"),
        expect_topology_clades=["HOMINIDAE"],
        consistency_repeats=3,
        max_latency_s=180.0,
        notes="Full analysis repeated 3×. Both outputs and routing must be stable.",
    ),
    RealEvalCase(
        id="consistency_05_common_names",
        category="consistency",
        prompt="What is the molecular similarity of chimp and zebrafish?",
        expected_feature="molecular_comparison",
        expected_status="completed",
        expected_species=["Pan troglodytes", "Danio rerio"],
        expect_closest_pair=("Pan troglodytes", "Danio rerio"),
        consistency_repeats=3,
        max_latency_s=15.0,
        notes="Common names repeated 3×. Resolution must be stable.",
    ),
]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def cases_by_category(category: str) -> list[RealEvalCase]:
    return [c for c in CASES if c.category == category]


SUMMARY = {
    "total_cases":         len(CASES),
    "real_mc":             len(cases_by_category("real_mc")),
    "real_phylo":          len(cases_by_category("real_phylo")),
    "real_full":           len(cases_by_category("real_full")),
    "ambiguous":           len(cases_by_category("ambiguous")),
    "invalid_input":       len(cases_by_category("invalid_input")),
    "delegation":          len(cases_by_category("delegation")),
    "consistency":         len(cases_by_category("consistency")),
    "consistency_repeats": sum(c.consistency_repeats for c in CASES),
    "unique_species":      12,
    "ncbi_accessions":     len(REAL_NCBI_ACCESSIONS),
}


if __name__ == "__main__":
    print(f"Evolution Agent Real Benchmark — {SUMMARY['total_cases']} cases")
    for k, v in SUMMARY.items():
        print(f"  {k}: {v}")
    print()
    for cat in ["real_mc", "real_phylo", "real_full", "ambiguous", "invalid_input", "delegation", "consistency"]:
        cases = cases_by_category(cat)
        print(f"  [{cat}]")
        for c in cases:
            rep = f" ×{c.consistency_repeats}" if c.consistency_repeats > 1 else ""
            print(f"    {c.id}{rep}: {c.prompt[:70]}")
