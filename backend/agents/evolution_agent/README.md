# Evolution Agent — Sprint 4

Part of the **Umbrella BioHub** platform. Analyses evolutionary relationships between species using an autonomous planner, branch-specific workers, and real bioinformatics tools.

---

## What it does

The Evolution Agent is a **sub-orchestrator** — it classifies the user's intent, selects the right worker(s), and returns only the relevant results.

| Request type | Workers called | Output fields |
|---|---|---|
| Molecular comparison only | MC Agent | `similarity_network`, `similarity_scores`, `species_groups` |
| Phylogenetic tree only | Phylo Agent | `newick_tree`, `model`, `bootstrap_support` |
| Both (full analysis) | MC Agent + Phylo Agent | All of the above |
| Ambiguous / no feature | None | Returns `continue` with a clarification question |

---

## Autonomy correction (9 steps)

| Step | What changed |
|---|---|
| 1. PlannedFeature enum | `molecular_comparison`, `phylogenetic_tree`, `full_analysis`, `clarification_required` |
| 2. Planner (LLM + guards) | GPT-5-mini classifies intent; deterministic guards prevent bad dispatch |
| 3. Branch-specific dispatch | Orchestrator only runs the worker(s) the planner selected |
| 4. Clarification node | Ambiguous requests return a structured follow-up question |
| 5. No full_analysis fallback | Orchestrator/adapter reject missing or invalid features |
| 6. Branch-specific output | Tree-only requests don't include network fields; MC-only don't include tree fields |
| 7. Explainer (LLM) | Second LLM call writes a human-readable explanation of the results |
| 8. Safe error handling | `try/except` in dispatch; safe species resolver for all input types |
| 9. Dead code removed | `divergence_time` worker deleted; unused `router.py`/`aggregator.py` kept for reference |

---

## Pipeline

```
User question (free text or structured)
        ↓
Planner (GPT-5-mini)  — intent classification + deterministic guards
        ↓
Species Resolver  — "human" → "Homo sapiens"
        ↓
Dispatch (feature-dependent)
   ├── molecular_comparison  →  MC Worker (mock scores + network)
   ├── phylogenetic_tree     →  MAFFT (EBI API / local) + IQ-TREE (local)
   ├── full_analysis         →  both workers
   └── clarification         →  returns follow-up question
        ↓
Explainer (GPT-5-mini)  — human-readable summary
        ↓
AgentResult (flat, branch-specific)
```

---

## LangSmith Tracing (Sprint 4, Task 3)

Every request is traced end-to-end in [LangSmith](https://smith.langchain.com) with no code changes required — just set three environment variables in your `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=evolution-agent
```

When tracing is enabled, each request produces a trace tree like this:

```
Evolution Agent request                ← OrchestratorEvolutionAgent.run()
  └── Planner (LLM #1)                 ← planner.plan()  [LLM call]
  └── EvolutionOrchestrator graph      ← LangGraph graph.ainvoke()
        ├── plan_node                  ← reads planner decision from state
        ├── species_resolver_node      ← "human" → "Homo sapiens"
        ├── dispatch_node              ← calls MC worker and/or Phylo worker
        ├── assemble_node              ← combines worker outputs
        └── explain_node
              └── Explainer (LLM #2)   ← explainer.explain()  [LLM call]
```

**What each span tells you:**

| Span | Useful for |
|---|---|
| `Evolution Agent request` | End-to-end latency, total LLM calls (max 2), final status |
| `Planner (LLM #1)` | What feature the LLM chose, which species it extracted, prompt tokens |
| `EvolutionOrchestrator graph` | Which nodes ran, state at each transition |
| `dispatch_node` | Worker call latency, which workers were invoked |
| `Explainer (LLM #2)` | Whether interpretation was grounded, grounding violation reason if discarded |

**Debugging with traces:**

- **Wrong feature dispatched** → check the `Planner` span input/output; look at the raw LLM response before `_apply_guards` ran.
- **Species not resolving** → check `species_resolver_node` in the graph span; the state diff shows `unresolved_species`.
- **Interpretation discarded** → check the `Explainer` span; the grounding check logs the exact violation reason.
- **High latency** → `dispatch_node` shows worker time; MAFFT+IQ-TREE latency appears here for phylo requests.
- **Unexpected failures** → `fail_node` in the graph span captures the exact error message and which worker triggered it.

**Monitoring in LangSmith:**

Once traces start appearing, set up these monitors in the LangSmith UI:

1. **Failure rate** — filter by `status=failed` on the root `Evolution Agent request` span.
2. **Planner consistency** — tag by `feature` output; watch for drift if the same prompt routes to different features across runs.
3. **LLM call count** — the `llm_calls` field in the response must be ≤ 2 per request; alert if it exceeds this.
4. **Latency by feature** — `molecular_comparison` should be < 5s (mock); `phylogenetic_tree` varies with MAFFT+IQ-TREE.

---

## Evaluation (Sprint 4, Tasks 1 & 2)

### Agent Evaluation

9 golden cases covering all five Sprint 4 criteria (agent/tool selection, task completion, correctness, relevance, response consistency). Results are in `evals/report.md`.

```powershell
# Server must be running on port 8002 first
python -m backend.agents.evolution_agent.evals.run_eval
```

| Criterion | Result |
|---|---|
| Agent/tool selection | 8/8 passed |
| Task completion | 9/9 passed |
| Correctness (deterministic) | 3/3 passed |
| Response consistency | 1/1 passed |
| Relevance (LLM judge, mean) | 4.2/5 |

**Weakness fixed:** Explainer relevance was 2.0/5 on phylo and full-analysis cases because the original prompt was feature-agnostic. Rewritten in `explainer.py` with feature-specific field guidance — scores rose to 4.2/5 mean.

### RAG Evaluation

The Species Resolver (`orchestrator/services/species_resolver.py`) is the agent's retrieval component. It is evaluated with RAGAS-style metrics (no LLM required).

```powershell
python backend\agents\evolution_agent\evals\rag_eval.py
```

| RAGAS Metric | Score |
|---|---|
| Retrieval Relevance | 1.000 |
| Context Quality | 1.000 |
| Answer Faithfulness | 1.000 |
| Answer Relevance | 1.000 |

36/36 cases passed across 8 categories: exact scientific names, case normalisation, common names, secondary aliases, fuzzy substring fallback, edge cases, unknown species, and batch resolution. Results in `evals/rag_report.md`.

---

## Real bioinformatics tools

| Tool | Role | Source |
|---|---|---|
| **MAFFT** | Multiple sequence alignment | EBI REST API (primary), local binary fallback (`mafft/mafft-win/mafft.bat`) |
| **IQ-TREE** | Phylogenetic tree building + ModelFinder | Local binary (`iqtree/iqtree-2.3.6-Windows/bin/iqtree2.exe`) |

**How it works:**
1. Species sequences are fetched from the catalogue
2. MAFFT aligns the sequences (EBI API or local)
3. IQ-TREE runs ModelFinder to select the best substitution model, builds the tree with UFBoot bootstrap support
4. Returns a Newick tree with bootstrap values and confidence scores

**Guards:**
- Phylogenetic tree requires **3+ species** (IQ-TREE rejects bootstrap with fewer)
- IQ-TREE auto-selects the best model (mtMAM, mtVer, GTR, etc.)

---

## Input / Output

**Input (`AgentRequest`)**

```json
{
  "instruction": "Build a phylogenetic tree for human, chimp and mouse.",
  "context": {
    "species_list": ["homo sapiens", "pan troglodytes", "mus musculus"],
    "feature": "phylogenetic_tree"
  }
}
```

| Field | Type | Required |
|---|---|---|
| `instruction` | str | Yes — free-text question |
| `context.species_list` | list[str] | Yes — min 2 for MC, min 3 for phylo |
| `context.feature` | str | No — planner auto-detects if missing |

**Output (`AgentResult`)**

```json
{
  "status": "completed",
  "output": {
    "status": "completed",
    "decision": "analysis_complete",
    "explanation": "Phylogenetic tree built for 3 species using mtMAM model with 95% overall confidence.",
    "overall_confidence": 0.95,
    "newick_tree": "(homo:0.0000019625,pan:0.0178874652,mus:0.2409514745);",
    "model": "mtMAM",
    "bootstrap_support": {},
    "similarity_network": null,
    "source_agents": ["Evolution Agent Orchestrator", "Phylogenetic Tree Agent"]
  }
}
```

Output is **flat** — tree-only requests don't include `similarity_network`; MC-only don't include `newick_tree`.

---

## Quick start

### 1. Setup

```powershell
cd backend\agents\evolution_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure (optional — enables LLM planner)

Create `backend/agents/evolution_agent/.env`:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview
EBI_MAFFT_EMAIL=your-email@example.com
```

### 3. Start the server

```powershell
$env:EVOLUTION_AGENT_IMPL = "orchestrator"
python -m uvicorn backend.agents.evolution_agent.api:app --port 8002 --reload
```

### 4. Open Swagger

**http://localhost:8002/docs**

### 5. Test it

```powershell
# Molecular comparison only
curl -X POST http://localhost:8002/execute -H "Content-Type: application/json" -d '{"instruction":"How similar are humans and chimpanzees?","context":{"species_list":["homo sapiens","pan troglodytes"],"feature":"molecular_comparison"}}'

# Phylogenetic tree only
curl -X POST http://localhost:8002/execute -H "Content-Type: application/json" -d '{"instruction":"Build a phylogenetic tree for human, chimp and mouse.","context":{"species_list":["homo sapiens","pan troglodytes","mus musculus"],"feature":"phylogenetic_tree"}}'

# Full analysis
curl -X POST http://localhost:8002/execute -H "Content-Type: application/json" -d '{"instruction":"Give me both a similarity network and a phylogenetic tree.","context":{"species_list":["homo sapiens","pan troglodytes","mus musculus"],"feature":"full_analysis"}}'

# Clarification (ambiguous)
curl -X POST http://localhost:8002/execute -H "Content-Type: application/json" -d '{"instruction":"Tell me about evolution.","context":{}}'
```

---

## Test cases

| Case | Input | Expected |
|---|---|---|
| MC only (2 species) | `feature: molecular_comparison` | completed, `similarity_network` present, `newick_tree: null` |
| Phylo only (3 species) | `feature: phylogenetic_tree` | completed, `newick_tree` present, `model`, `similarity_network: null` |
| Full analysis | `feature: full_analysis` | completed, both present |
| Clarification | no feature, empty context | `continue` with clarification question |
| Guard: 2-species tree | phylo + 2 species | `failed` — needs 3+ |
| Guard: invalid feature | `feature: banana` | `continue` with "Could you rephrase?" |
| 5-species tree | phylo + 5 species | completed, real MAFFT + IQ-TREE, bootstrap values |

---

## Running tests

```powershell
python -m pytest backend\agents\evolution_agent\tests -v
```

**145 tests, all passing.** No API key required for unit tests.

| Test file | Tests | Covers |
|---|---|---|
| `test_orchestrator_pipeline.py` | 20 | LangGraph pipeline end-to-end |
| `test_adapter.py` | 25 | OrchestratorAdapter branch-specific output |
| `test_mock_quality_audit.py` | 42 | Deterministic MC worker quality |
| `test_branch_acceptance.py` | 38 | Guards, clarification, dispatch |

---

## Folder structure

```
evolution_agent/
├── api.py                        HTTP boundary (AgentRequest → AgentResult)
├── schema.py                     PlannedFeature, PlannerDecision, all result types
├── planner.py                    LLM planner + deterministic guards
├── intent.py                     Backward-compat re-exports from planner
├── explainer.py                  LLM explainer — feature-aware, grounded summaries
├── orchestrator_adapter.py       Branch-specific output mapping, legacy path
├── orchestrator/
│   ├── evolution_orchestrator.py LangGraph pipeline, feature-dependent dispatch
│   └── services/
│       └── species_resolver.py   Offline dict + Qdrant backend (Sprint 3+)
├── framework/
│   └── llm_client.py             Azure → Groq → GitHub Models priority chain
├── tools/
│   ├── __init__.py               Exports mafft_align, iqtree_build
│   ├── mafft.py                  EBI REST API (primary) + local binary fallback
│   └── iqtree.py                 Local IQ-TREE binary (ModelFinder + UFBoot)
├── workers/
│   ├── molecular_comparison/
│   │   └── mock.py               Deterministic MC mock (5 species, calibrated scores)
│   └── phylogenetic_tree/
│       └── worker.py             Real phylo worker: MAFFT → IQ-TREE → Newick
├── evals/
│   ├── golden_dataset.py         9 agent eval cases (all 5 Sprint 4 criteria)
│   ├── run_eval.py               Agent eval runner → report.json + report.md
│   ├── report.md                 Agent eval results (4.2/5 relevance after fix)
│   ├── rag_eval.py               RAG eval for Species Resolver (36 cases, 4 metrics)
│   └── rag_report.md             RAG eval results (1.000 across all RAGAS metrics)
├── tests/
│   ├── conftest.py               Shared fixtures
│   ├── test_orchestrator_pipeline.py
│   ├── test_adapter.py
│   ├── test_mock_quality_audit.py
│   └── test_branch_acceptance.py
├── .env.example                  Credentials template (LLM + LangSmith + Qdrant)
├── .env                          Git-ignored: Azure + EBI + LangSmith credentials
├── card.json                     Agent card (platform registry)
├── README.md
└── requirements.txt
```

---

## Branch

`group_d_evolution_agent_phylogenetic_agent` on [hejerayadi/Umbrella_animal_biology_](https://github.com/hejerayadi/Umbrella_animal_biology_)
