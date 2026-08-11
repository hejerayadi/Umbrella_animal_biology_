# Evolution Agent — Sprint 2

Part of the **Umbrella BioHub** platform. Analyses evolutionary relationships between species using a two-step sequential pipeline.

---

## What it does

Given a list of species, the Evolution Agent:

1. **Molecular Comparison** — fetches protein sequences, aligns them (MAFFT), computes similarity scores (ESM-C embeddings), groups close species, builds a similarity network
2. **Phylogenetic Reconstruction** — takes the alignment from step 1, builds an evolutionary tree (IQ-TREE), runs bootstrap validation (UFBoot), returns a Newick tree with confidence values

All tools are **mocked in Sprint 2** (`score_is_mock: true`). Real tools (NCBI, UniProt, MAFFT, ESM-C, IQ-TREE) replace the mocks in Sprint 3.

---

## Pipeline

```
User question
    ↓
GPT-5-mini (intent classification)
    ↓
Species Resolver  ("human" → "Homo sapiens")
    ↓
Subagent 1 — Molecular Comparison
    ↓  alignment
Subagent 2 — Phylogenetic Reconstruction
    ↓
EvolutionAnalysisResult
```

---

## Input / Output

**Input (`EvolutionInput`)**

```python
@dataclass
class EvolutionInput:
    species: list[str]                          # min 2, common names accepted
    question: str                               # free-text question
    target_gene_or_protein: Optional[str] = None
    protein_inputs: Optional[list[str]] = None  # pre-fetched FASTA sequences
    outgroup: Optional[str] = None              # tree root species
```

**Output (`EvolutionOutput`)**

```python
@dataclass
class EvolutionOutput:
    status: str                                  # "completed" | "failed"
    species: list[str]                           # canonical scientific names
    molecular_comparison: Optional[str] = None  # plain-language MC summary
    closest_species: Optional[list[str]] = None # most similar pair
    species_groups: Optional[list[list[str]]] = None
    similarity_network: Optional[str] = None    # JSON adjacency list
    evolutionary_tree: Optional[str] = None     # Newick format
    explanation: Optional[str] = None           # one-sentence summary
```

---

## Quick start

### No setup needed (mock mode)

```bash
python -m uvicorn backend.agents.evolution_agent.api:app --port 8002 --reload
```

Open Swagger: **http://localhost:8002/docs**

Send a request:

```json
{
  "instruction": "Compare homo sapiens, pan troglodytes and mus musculus evolutionarily.",
  "context": {"species_list": ["homo sapiens", "pan troglodytes", "mus musculus"]}
}
```

### With Azure GPT-5-mini (real intent classification)

1. Create `backend/agents/evolution_agent/.env`:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

2. Start the orchestrator:

```powershell
$env:EVOLUTION_AGENT_IMPL = "orchestrator"
python -m uvicorn backend.agents.evolution_agent.api:app --port 8002 --reload
```

3. Ask a free-text question — no `species_list` needed:

```json
{
  "instruction": "How are humans and chimpanzees related at the molecular level?",
  "context": {}
}
```

GPT-5-mini extracts the species and feature automatically.

### Run tests

```bash
python -m pytest backend/agents/evolution_agent/tests/ -v
```

75 tests, all passing. No API key required.

### Run manual tests

```bash
# Pipeline only (no LLM)
python -m backend.agents.evolution_agent.test_manual

# With Azure GPT-5-mini
$env:EVOLUTION_AGENT_IMPL="orchestrator"
python -m backend.agents.evolution_agent.test_manual
```

---

## Test cases (Swagger)

| Case | species_list | Expected |
|---|---|---|
| 3 mammals | `["homo sapiens", "pan troglodytes", "mus musculus"]` | completed, confidence 0.93 |
| 5 species | `["homo sapiens", "pan troglodytes", "mus musculus", "gallus gallus", "danio rerio"]` | completed, confidence 0.80 |
| Common names | `["human", "chimp", "mouse"]` | completed (resolver maps them) |
| 1 species | `["homo sapiens"]` | failed — needs at least 2 |
| Unknown species | `["homo sapiens", "draco magicus"]` | failed — cannot resolve |
| Non-biology question | `"What is the capital of France?"` | failed — no feature |

---

## Folder structure

```
evolution_agent/
├── api.py                        HTTP boundary (AgentRequest → AgentResult)
├── schema.py                     EvolutionInput, EvolutionOutput, all types
├── intent.py                     GPT-5-mini intent classifier
├── orchestrator_adapter.py       Intent → pipeline → flat output
├── mock.py                       Simple mock (no pipeline, no LLM)
├── test_manual.py                Manual test runner (terminal output)
├── pytest.ini
├── requirements.txt
├── card.json                     Agent card (platform registry)
├── .env.example                  Credentials template
├── README.md
├── framework/
│   └── llm_client.py             Azure / Groq / GitHub Models priority chain
├── orchestrator/
│   ├── evolution_orchestrator.py LangGraph 6-node state machine
│   ├── router.py                 Feature → worker mapping
│   ├── aggregator.py             MC + phylo result assembly
│   └── services/
│       └── species_resolver.py   Common name → scientific name
├── workers/
│   ├── molecular_comparison/     Subagent 1 mock (FASTA + scores + network)
│   ├── phylogenetic_tree/        Subagent 2 mock (Newick + bootstrap)
│   └── divergence_time/          Reserved for Sprint 3
├── tests/
│   ├── conftest.py
│   ├── test_orchestrator_pipeline.py  22 end-to-end pipeline tests
│   ├── test_workers.py                35 worker unit tests
│   └── test_adapter.py                18 adapter + intent tests
└── docs/
    └── live_capture.json         7 captured test cases with full output
```

---

## Agent Framework

**Why LangGraph** — the pipeline is a fixed sequence with a data dependency (alignment from step 1 → step 2). LangGraph's `StateGraph` maps directly to this: nodes are steps, edges are dependencies, conditional edges handle failures. No role-based abstraction needed (CrewAI), no manual state management needed (raw LangChain).

**LLM — Azure GPT-5-mini** — used only for intent classification (one call per request, ~200 tokens). Extracts the feature type and species list from free-text. $0.25/M input tokens.

**Mock species catalogue** — 5 species with biologically calibrated fixture values:

| Species | Common name | Role |
|---|---|---|
| Homo sapiens | Human | Reference |
| Pan troglodytes | Chimpanzee | Closest to human (0.98) |
| Mus musculus | Mouse | Distant mammal (0.85) |
| Gallus gallus | Chicken | Non-mammal amniote (0.72) |
| Danio rerio | Zebrafish | Distant outgroup (0.54) |

---

## Branch

`group_d_evolution_agent` on [hejerayadi/Umbrella_animal_biology_](https://github.com/hejerayadi/Umbrella_animal_biology_)
