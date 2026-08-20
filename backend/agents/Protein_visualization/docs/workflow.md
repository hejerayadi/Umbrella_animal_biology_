# Protein analysis workflow

The sub-orchestrator runs a LangGraph `StateGraph`
([app/orchestrators/protein/graph.py](../app/orchestrators/protein/graph.py)). The
next node is chosen from the state, not from a fixed list.

```mermaid
flowchart TD
    START([START]) --> VALIDATE[validate_input]
    VALIDATE -->|invalid| CLARIFY[return_needs_clarification]
    VALIDATE -->|valid| IDENTITY[resolve_protein_identity]

    IDENTITY -->|unresolved| ABSTAIN[scientific_abstain]
    IDENTITY --> PDB[search_experimental_structures]
    IDENTITY --> INTERPRO[fetch_annotations]
    IDENTITY --> QDRANT[retrieve_knowledge]

    PDB --> EVAL[evaluate_pdb_results]
    EVAL -->|valid candidate| SELECT[select_structure]
    EVAL -->|none valid| AF[search_alphafold]
    AF --> SELECT

    SELECT --> JOIN[join_results]
    INTERPRO --> JOIN
    QDRANT --> JOIN

    JOIN -->|residue, mutation or region requested| SIFTS[map_residues_with_sifts]
    JOIN -->|otherwise| EVIDENCE[build_evidence_pack]
    SIFTS --> EVIDENCE

    EVIDENCE --> SCENE[build_molstar_scene]
    SCENE -->|include_explanation| LLM[generate_grounded_explanation]
    SCENE -->|otherwise| CRITIC[run_scientific_critic]
    LLM --> CRITIC

    CRITIC -->|ACCEPT| COMPLETE[complete]
    CRITIC -->|REVISE or warnings| PARTIAL[return_partial]
    CRITIC -->|ABSTAIN| ABSTAIN

    CLARIFY --> END([END])
    COMPLETE --> END
    PARTIAL --> END
    ABSTAIN --> END
```

## Where the decisions are made

| Decision | Location |
|---|---|
| Input is answerable | `validate_input` — otherwise `NEEDS_CLARIFICATION` |
| Identity is the anchor | `route_after_identity` — no identity, no structure |
| Parallel evidence gathering | `route_after_identity` returns three nodes at once |
| A PDB entry is usable | `StructureNodes.evaluate_pdb` (chain, coverage floor, metadata) |
| AlphaFold is needed | `route_after_pdb_evaluation` — fallback only |
| Which structure wins | `StructureCapability.select` — deterministic scores, never the LLM |
| SIFTS is needed | `route_mapping_required` — residue, mutation or region requested |
| The result is publishable | `run_scientific_critic` → `route_after_critic` |

## Failure behaviour

`join_results` is deferred, so it runs once all branches have settled. Every
optional node converts a provider failure into a coded warning
(`PDB_TIMEOUT`, `ANNOTATIONS_UNAVAILABLE`, `RETRIEVAL_UNAVAILABLE`,
`SIFTS_UNAVAILABLE`, `LLM_UNAVAILABLE`, …) and the workflow continues with the
evidence it has, ending in `PARTIAL`. Only an unresolved identity stops it.

A residue is highlighted only when SIFTS reports it observed; otherwise the scene
omits it, a warning is raised, and the critic downgrades the verdict to `REVISE`
or `ABSTAIN`. An AlphaFold model is always reported as `PREDICTED`.
