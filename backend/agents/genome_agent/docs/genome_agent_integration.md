# Genome Agent — Orchestrator Integration Plan

> Genome Agent has **no sub-orchestrator layer**. The Genome Agent Orchestrator
> talks directly to its four subagents (Species Resolver, Genome Metadata,
> Gene Annotation, Visualization). So everywhere the general playbook says
> "Sub-Orchestrator ↔ Agent," read it here as "Genome Agent Orchestrator ↔ Subagent."

```
Genome Agent Orchestrator
        │
        ├──> Species Resolver        (order 1)
        ├──> Genome Metadata         (order 2, parallel)
        ├──> Gene Annotation         (order 2, parallel)
        └──> Visualization           (order 3)
```

---

## 1. Connect each agent to the orchestrator

Each subagent is registered with the orchestrator as a callable with a fixed
input/output contract (dataclasses already defined in the spec):

| Subagent | Input dataclass | Output dataclass |
|---|---|---|
| Species Resolver | `SpeciesResolverInput` | `SpeciesResolverOutput` |
| Genome Metadata | `GenomeMetadataInput` | `GenomeMetadataOutput` |
| Gene Annotation | `GeneAnnotationInput` | `GeneAnnotationOutput` |
| Visualization | `VisualizationInput` | `VisualizationOutput` |

The orchestrator holds references to all four; no discovery step is needed
since the roster is static.

---

## 2. Implement task delegation

The orchestrator validates the incoming `user_question` (species name
parseable?), then dispatches work strictly per the routing table already
defined:

- Genome size / chromosome count / karyotype / assembly level → **Genome Metadata**
- Genes / gene function / feature locations → **Gene Annotation**
- "Show me" / "visualize" / chromosome map / comparison → **Visualization**
- Species Resolver always runs first, regardless of the above

If the question asks for more than one thing, the orchestrator fans out to
multiple subagents and merges the outputs into a single answer.

---

## 3. Pass required context to each agent

Context flows strictly downstream, never sideways between parallel agents:

```
user_question ──> Species Resolver ──> assembly_id ──┬──> Genome Metadata
                                                       └──> Gene Annotation

genome_size_bp + gene_table ──> Visualization
```

- Species Resolver only needs `species_name` extracted from the question.
- Genome Metadata and Gene Annotation each only need `assembly_id` —
  nothing else is threaded through.
- Visualization needs whatever the earlier steps produced (`genome_size_bp`,
  `gene_table`, and `assembly_accessions` for size comparisons).

---

## 4. Handle agent responses

Each subagent returns a typed `AgentResult`-style output. The orchestrator:

1. Reads the typed output directly (no re-parsing/re-validating fields that
   are already validated inside the subagent).
2. Attaches source + accession metadata to every fact used downstream, since
   the final answer must be cited (source database + accession ID).
3. Passes structured outputs (not prose) between steps — e.g. Visualization
   receives `genome_size_bp` and `gene_table` as data, not as a summary.

---

## 5. Handle dependencies between agents

| Step | Depends on | Provides |
|---|---|---|
| Species Resolver | user question | `assembly_id`, `scientific_name` |
| Genome Metadata | `assembly_id` | `genome_size_bp`, `chromosome_count` |
| Gene Annotation | `assembly_id` | `gene_table`, `gene_list` |
| Visualization | `genome_size_bp`, `gene_table` | `chart_data` |

Genome Metadata and Gene Annotation are independent of each other — neither
reads the other's output — so they are dependency-free with respect to one
another and only gated on Step 1.

---

## 6. Implement sequential execution where required

Strict ordering is required in two places:

- **Species Resolver must complete before anything else runs.** Nothing else
  has a valid `assembly_id` until it returns.
- **Visualization runs last.** It needs `genome_size_bp` and/or `gene_table`,
  which only exist once step 2 has finished (fully or partially).

---

## 7. Implement parallel execution where appropriate

Genome Metadata and Gene Annotation run **concurrently** once `assembly_id`
is available — they hit different NCBI endpoints (Assembly vs. Gene) and
share no state. Implementation-wise this is a simple `asyncio.gather` (or
equivalent) over the two calls, both seeded with the same `assembly_id`.

---

## 8. Aggregate agent results

The orchestrator merges whatever subset of `{genome_metadata, gene_annotation,
chart_data}` succeeded into one `GenomeAgentOutput`, then runs the
reasoning/LLM explanation step over the merged data to produce the final
cited, plain-language answer. Aggregation must tolerate partial results (see
next section) rather than requiring all three to be present.

---

## 9. Handle agent failures

Failure handling is asymmetric by design — not every failure is fatal:

| Failure | Orchestrator behavior |
|---|---|
| Species Resolver fails | **Stop immediately.** `status=FAILED`, return `"No genome assembly found for [species]."` No downstream calls are attempted. |
| Genome Metadata fails | `status` stays `COMPLETED`. Return whatever else succeeded; log the gap as a `SourceError`. |
| Gene Annotation fails | Same as above — degrade gracefully, don't block the rest of the answer. |
| Visualization (`chromosome_map` / `size_comparison`) fails | Return the text answer, omit the chart, and note in the response that it couldn't be generated. |
| Visualization (`protein_structure` requested) | Not a failure — it's a **handoff**. `status=NEEDS_AGENT` with `target_agent="protein_structure_visualization_agent"` and `prompt_to_target_agent` set, bubbled up unchanged to the platform orchestrator. Genome Agent never calls that agent itself. |

This means the only genuinely fatal path is species resolution failing;
everything downstream of that is best-effort and additive.

---

## 9a. Reconstruction handoff (assembly-quality gate, runs before Visualization)

This is not in the original routing table above, but it is real, implemented,
and tested (`tests/test_reconstruction_path.py`) — the plan is updated here
to document it.

**Genome Metadata does double duty.** Whenever `get_genome_metadata_node`
(`workflows/nodes/genome_data_nodes.py`) gets a result back from NCBI, it
checks `assembly_level` regardless of whether the caller actually asked for
metadata (`needs_metadata` only gates whether the metadata dict is *surfaced*
to the user — the gap check always runs):

| `assembly_level` | Orchestrator behavior |
|---|---|
| `Chromosome` / `Complete Genome` | No gap. `reconstruction_need` stays `None`. Graph proceeds to Visualization as normal. |
| `Scaffold` / `Contig` | Gap detected. `reconstruction_need = {status: NEEDS_AGENT, target_agent: None, prompt_to_target_agent: "..."}` is written to state. |

**Routing consequence.** `_route_after_join_parallel` checks
`reconstruction_need.status` *before* checking `visualization_scope`. If a
gap was detected, the graph goes straight to `reconstruction_resolver` and
**Visualization never runs at all** for that turn — even if the user asked
for a chart. Reconstruction takes priority because a chart built on an
incomplete/gapped assembly would be misleading.

**Reconstruction resolver.** `reconstruction_resolver_node` fills in the
`target_agent` (via the same `resolve_capability` / `resolve_capability_fallback`
LLM-with-deterministic-fallback pattern used by the protein-structure handoff)
and finalizes `prompt_to_target_agent`, then hands off to `explanation_writer`.

**Adapter precedence.** `orchestrator_adapter.to_result()` checks
`reconstruction_need` before the visualization `NEEDS_AGENT` check, for the
same reason the graph does — the two are mutually exclusive in practice
(the graph only reaches `generate_visualization` when `reconstruction_need`
was *not* triggered), but checking in the same order keeps the adapter's
logic legible on its own.

This makes Genome Metadata's role broader than "genome size / chromosome
count / karyotype" (section 2's routing table) — it is also the sole
gatekeeper for assembly-quality, on every request, independent of what the
user asked for.

---

## Summary: mapping to the standard playbook

| Standard step | Genome Agent equivalent |
|---|---|
| Connect agents to sub-orchestrator | Orchestrator holds Species Resolver, Genome Metadata, Gene Annotation, Visualization |
| Task delegation | Routing table (genome size → Metadata, genes → Annotation, "show me" → Visualization) |
| Context passing | `assembly_id` fans out from Species Resolver; `genome_size_bp`/`gene_table` feed Visualization |
| Response handling | Typed outputs + citation metadata attached before merge |
| Dependencies | Metadata & Annotation depend only on Resolver, not each other |
| Sequential execution | Resolver first, Visualization last |
| Parallel execution | Metadata + Annotation run concurrently |
| Aggregation | Merge partial results into one `GenomeAgentOutput`, then explain |
| Failure handling | Resolver failure = hard stop; Metadata/Annotation/local-chart failure = degrade gracefully; protein-structure request = delegate via `NEEDS_AGENT` |
| Reconstruction handoff (new) | Genome Metadata flags `Scaffold`/`Contig` assemblies on every request; graph reroutes to `reconstruction_resolver` before Visualization and hands off via `NEEDS_AGENT` to the Reconstruction Agent |

---

## Appendix: implementation status (verified against the repo)

All nine sections above are implemented and passing in
`LAKHAL-Farah/testing_a_theory` → `genome_agent/`:

- `orchestrator.py` — LangGraph wiring matches the shape in this doc exactly
  (`query_router → species_resolver → parallel_kickoff → {get_genome_metadata,
  get_gene_annotation} → join_parallel → {reconstruction_resolver |
  generate_visualization | explanation_writer} → END`).
- `orchestrator_adapter.py` — maps `GenomeAgentState` onto the platform's
  `AgentRequest`/`AgentResult` contract; handles the FAILED / COMPLETED /
  NEEDS_AGENT (reconstruction) / NEEDS_AGENT (visualization) branches.
- `schemas/inputs.py` / `schemas/outputs.py` — the four dataclass contracts
  from section 1, unchanged from the plan.
- Test suite: **115 passed, 0 failed, 12 skipped** (skips are all live-network
  NCBI eutils / live-LLM NVIDIA API tests gated behind env vars, not code
  issues).

**Cleanup done as part of this integration pass:**
- Fixed `tests/test_reconstruction_path.py`: three calls to
  `asyncio.get_event_loop().run_until_complete(...)` broke under Python 3.12
  (no implicit event-loop creation outside async context); replaced with
  `asyncio.run(...)`, matching the pattern used elsewhere in the suite.
  This alone took the file from 0/31 passing to 31/31.
- Removed `workflows/nodes/genome_metadata_node.py`: an orphaned duplicate
  of `get_genome_metadata_node` (in `genome_data_nodes.py`) that was never
  imported by the orchestrator or any test, and — critically — did **not**
  implement the reconstruction-gap check, so if anything had ever been wired
  to it instead of the real node, Scaffold/Contig assemblies would have
  silently skipped the reconstruction handoff.
- Tidied the precedence comments in `orchestrator_adapter.to_result()` so the
  reconstruction-vs-visualization `NEEDS_AGENT` ordering is explained inline
  rather than marked `# NEW`.

**Known limitation:** live-network verification (real NCBI eutils calls via
`scripts/run_full_integration_demo.py` / `scripts/run_orchestrator_scenarios.py`)
could not be executed in this sandbox — its network allowlist does not
include `eutils.ncbi.nlm.nih.gov`. Run those scripts in an environment with
NCBI access (or CI) to confirm live data flow end to end; the mocked pytest
suite already confirms the routing/aggregation/failure-handling logic itself.
