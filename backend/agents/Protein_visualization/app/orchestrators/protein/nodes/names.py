"""Canonical node names, used by the graph, the routers, and the log context."""

VALIDATE_INPUT = "validate_input"
RESOLVE_IDENTITY = "resolve_protein_identity"
SEARCH_EXPERIMENTAL = "search_experimental_structures"
EVALUATE_PDB = "evaluate_pdb_results"
SEARCH_ALPHAFOLD = "search_alphafold"
SELECT_STRUCTURE = "select_structure"
FETCH_ANNOTATIONS = "fetch_annotations"
RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
JOIN_RESULTS = "join_results"
MAP_RESIDUES = "map_residues_with_sifts"
BUILD_EVIDENCE = "build_evidence_pack"
BUILD_SCENE = "build_molstar_scene"
GENERATE_EXPLANATION = "generate_grounded_explanation"
RUN_CRITIC = "run_scientific_critic"
COMPLETE = "complete"
RETURN_PARTIAL = "return_partial"
NEEDS_CLARIFICATION = "return_needs_clarification"
SCIENTIFIC_ABSTAIN = "scientific_abstain"

# Topological order of the graph in `graph.py`. `executed_nodes` is a set, so a
# reader has no way to tell what ran before what; sorting a run against this
# gives back the order the workflow actually moves in. Branches that run in
# parallel are listed in the order `routers.PARALLEL_RETRIEVAL` declares them.
WORKFLOW_SEQUENCE: tuple[str, ...] = (
    VALIDATE_INPUT,
    RESOLVE_IDENTITY,
    SEARCH_EXPERIMENTAL,
    EVALUATE_PDB,
    SEARCH_ALPHAFOLD,
    SELECT_STRUCTURE,
    FETCH_ANNOTATIONS,
    RETRIEVE_KNOWLEDGE,
    JOIN_RESULTS,
    MAP_RESIDUES,
    BUILD_EVIDENCE,
    BUILD_SCENE,
    GENERATE_EXPLANATION,
    RUN_CRITIC,
    COMPLETE,
    RETURN_PARTIAL,
    NEEDS_CLARIFICATION,
    SCIENTIFIC_ABSTAIN,
)
