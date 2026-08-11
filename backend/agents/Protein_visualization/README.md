# Umbrella Protein Agent

A FastAPI service that resolves gene/protein identifiers, discovers experimental and predicted structures, maps functional annotations, and returns a viewer-ready Mol* specification. External evidence is kept distinct by source and optional failures produce a partial response.

**There is no mock implementation.** Every request runs the LangGraph workflow
against the real UniProt, RCSB PDB, AlphaFold, InterPro and SIFTS endpoints.
Deterministic fakes exist only inside `tests/` and cannot be selected through
application configuration.

## Two entry points, one process

`api.py` is the service. `backend/run_agents.py` starts it on port 8008 and it
answers on both:

| Route | Caller | Body |
|---|---|---|
| `POST /execute` | Grand Orchestrator | `{instruction, context}` in, `{status, target_agent, prompt_to_target_agent, output}` out — the contract every agent in this repo speaks |
| `GET /health` | platform liveness check | which implementation is serving |
| `POST /api/v1/protein-structure-analyses` | frontend viewer, scripts | the full `AgentTask`, answering with the complete analysis: Mol* scene, every annotation, every evidence record |
| `GET /api/v1/taxonomy?name=…` | anyone building an `AgentTask` | species name → `{scientific_name, taxon_id}` |
| `GET /api/v1/ready` | operators | whether Qdrant, the LLM and persistence are reachable |

`/execute` holds no business logic: it hands the request to
[orchestrator_adapter.py](orchestrator_adapter.py), which resolves identity
against UniProt before running the workflow. See that module's docstring for
why the gene and species are resolved *together* rather than the species alone.

The shared context also carries optional protein-analysis intent extracted from
the chat message: `mutation`, `residue_position`, `requested_regions`,
`preferred_source` (`auto`, `pdb`, or `alphafold`) and `include_explanation`.
The adapter validates these fields before contacting a provider and forwards
them to the same `ProteinTaskInput` used by the versioned scientific endpoint.

### What `/execute` puts in the shared context

Deliberately a summary, not the whole `ProteinAnalysisResponse`: everything a
completed agent returns is merged into the context every later agent sees and
rendered into the Responder's prompt, so the residue mappings, alternative
structures and evidence list would be paid for in tokens on every subsequent
turn while answering nothing the user asked.

| Key | Type | Consumer |
|---|---|---|
| `protein_structure` | prose sentence | the Responder writes the answer from it; other agents test for it before escalating |
| `uniprot_accession` | string | later agents |
| `protein_explanation` | string | the Responder |
| `protein_warnings` | list | the Responder, via `_quality_constraints` |
| `protein_viewer` | Mol* scene | **the frontend** — `orchestrator-client.ts` reads it to render the 3D viewer. Skipped by the Responder's prompt builder (`_RENDER_ONLY_KEYS`) because it is coordinates, not prose |

Changing these key names or types breaks the frontend viewer and the other
agents, not just this one.

```powershell
curl.exe -X POST http://localhost:8008/execute -H "Content-Type: application/json" `
  -d '{\"instruction\":\"3D structure of TP53 in humans\",\"context\":{\"species\":\"Homo sapiens\",\"gene_name\":\"TP53\",\"residue_position\":273}}'
```

## What a response says about its own run

Alongside the analysis, every response carries the trace of how it was
produced. A result says what was found; these say how, and the two differ in
ways an operator has to be able to tell apart — an AlphaFold model because
every experimental candidate was rejected reads exactly like one because RCSB
timed out, unless the trace says which.

| Field | Meaning |
|---|---|
| `executed_nodes` | Which of the 18 graph nodes ran, in `WORKFLOW_SEQUENCE` order. A node absent here was skipped, not failed |
| `errors` | `"<node>: <ExceptionType>"` per provider failure that was degraded into a warning rather than raised |
| `retry_counts` | How many times each node retried |
| `llm_usage` | One entry per Azure OpenAI call (explanation, critic audit): node, model, latency and token counts, taken from Azure's own response rather than estimated locally |

`llm_usage[].estimated_cost_usd` is filled in only when
`AZURE_INPUT_PRICE_PER_1K_USD` and `AZURE_OUTPUT_PRICE_PER_1K_USD` are both set
(see `.env.example`). Azure pricing depends on region, contract and model
version, none of which the API exposes, so it is left unset rather than guessed.

## Infrastructure in this sprint

- **Qdrant** — a managed cluster; set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. There is no `docker-compose.yml`: nothing is provisioned locally. When `QDRANT_URL` is absent, retrieval reports `RETRIEVAL_UNAVAILABLE` before loading BGE-M3; it never presents an empty in-memory store as a successful production search.
- **PostgreSQL** — out of scope for this sprint. `PERSISTENCE_ENABLED=false` keeps the repository layer dormant; the API answers without a database.
- **Azure** — the LLM provider for explanation and critic generation (`LLM_PROVIDER=azure`).
- **BGE-M3** — the embedding model for ingestion and retrieval, and the only one `get_embedding_provider` will build. It ships in the `embeddings` extra (`uv sync --extra embeddings`), because it pulls torch; without that extra, the retrieval node reports `RETRIEVAL_UNAVAILABLE` in the response warnings rather than falling back to anything. `HashEmbedding` carries no semantics and is reachable only from `tests/`.

## Run locally

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; both are
managed with [uv](https://docs.astral.sh/uv/). There is no `requirements.txt`.

```powershell
# creates backend\agents\Protein_visualization\.venv from the lock file
uv sync --project backend\agents\Protein_visualization

Copy-Item backend\agents\Protein_visualization\.env.example backend\agents\Protein_visualization\.env
# then fill in QDRANT_* and AZURE_*

# every command runs from the REPOSITORY ROOT: this agent imports itself as
# backend.agents.Protein_visualization.*, and that package is rooted there
uv run --project backend\agents\Protein_visualization python -m backend.agents.Protein_visualization.scripts.serve
```

Optional extras, installed only when you want them:

```powershell
# BAAI/bge-m3 retrieval through sentence-transformers (pulls torch, ~2 GB).
# Required for knowledge retrieval to run at all: without it the retrieval node
# degrades to RETRIEVAL_UNAVAILABLE, which is reported in the response warnings.
# Identity, structures, annotations and residue mapping do not need it.
uv sync --project backend\agents\Protein_visualization --extra embeddings

# the psycopg driver, needed only when DATABASE_URL points at PostgreSQL
uv sync --project backend\agents\Protein_visualization --extra postgres
```

Adding or changing a dependency goes through uv so that `uv.lock` stays in step
— CI runs `uv sync --locked` and fails if the two have drifted apart:

```powershell
uv add --project backend\agents\Protein_visualization "some-package>=1,<2"
uv add --project backend\agents\Protein_visualization --dev "some-dev-tool>=1,<2"
```

`python -m backend.run_agents` starts the service for you, along with every
other agent.

The standalone service's host and port are read from `.env`. Port 8000 belongs
to the Global Orchestrator's API and 8008 to the agent above, so it uses a
third:

```env
APP_HOST=127.0.0.1
APP_PORT=8010
APP_RELOAD=true
```

> The first analysis on a fresh machine downloads the BGE-M3 embedding model
> (~2.3 GB) before it can answer, which takes far longer than any client
> timeout. Warm the cache once, in advance:
>
> ```powershell
> .venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
> ```

Open `http://127.0.0.1:${APP_PORT}/docs`, then drive the agent:

```powershell
# shorthand used below; still run everything from the repository root
$agent = "backend\agents\Protein_visualization"

# in-process: real providers -> LangGraph -> response
uv run --project $agent python -m backend.agents.Protein_visualization.scripts.run_orchestrator --accession P04637 --gene TP53 --residue 273

# over HTTP against the running server
uv run --project $agent python -m backend.agents.Protein_visualization.scripts.smoke_test --accession P04637 --gene TP53

# index knowledge into the managed Qdrant collection
uv run --project $agent python -m backend.agents.Protein_visualization.scripts.seed_qdrant documents.json

# or fetch real UniProt/InterPro evidence and index it directly
uv run --project $agent python -m backend.agents.Protein_visualization.scripts.seed_protein_knowledge --accession P04637
```

The runtime always calls the configured real UniProt, RCSB PDB, AlphaFold,
InterPro and SIFTS providers and always uses BGE-M3 for retrieval. Deterministic
fakes exist only inside the automated test suite and cannot be selected through
application configuration.

Every response uses the `{data, meta, error}` envelope described in the [API contract](docs/api-contract.md).
The analysis endpoint places an inter-agent `AgentResult` in `data`; the detailed
scientific response is nested under `data.output`.

## The workflow

`POST /api/v1/protein-structure-analyses` runs a LangGraph `StateGraph`: identity is
resolved first, then structures, annotations and knowledge are fetched in parallel;
AlphaFold is a fallback, SIFTS runs only when a residue, mutation or region is
requested, and a scientific critic returns `ACCEPT`, `REVISE` or `ABSTAIN`.
See [workflow](docs/workflow.md) for the graph and the decision table.

## Observability

Use `LOG_FORMAT=pretty` for compact, colorized local logs (`text` is an alias), and
`LOG_FORMAT=json` for JSON Lines in deployments. Both renderers carry the same
structured fields: `timestamp`, `level`, `logger`, `event`, and — when in scope —
`request_id`, `trace_id`, `task_id`, `analysis_id`, `node`, `capability`, `status`,
`duration_ms`, `error_code`.

```env
LOG_LEVEL=INFO
LOG_FORMAT=pretty
```

Pretty output shortens correlation IDs on compact success lines. Warning and
error panels, as well as JSON logs, retain the complete IDs and exception details.

```json
{"timestamp":"2026-08-06T10:00:00Z","level":"INFO","logger":"app.analyses","event":"protein_analysis.completed","request_id":"…","trace_id":"…","task_id":"…","status":"completed","duration_ms":412}
```

Correlation ids are taken from the `X-Trace-Id` / `X-Request-Id` request headers when the Grand Orchestrator supplies them, generated otherwise, and echoed back on the response.

In code:

```python
with log_context(task_id=str(task.task_id)):  # fields inherited by every log line
    with log_stage(logger, "search_pdb", node="search_pdb") as outcome:
        outcome["candidates"] = len(candidates)  # attached to search_pdb.completed
```

`GET /api/v1/ready` reports whether Qdrant, the Azure LLM, and persistence are configured and reachable.

## Verify

Ruff, mypy and pytest all read their configuration from `pyproject.toml`; there
is no separate `ruff.toml`, `mypy.ini` or `pytest.ini`. Run them from the
repository root, exactly as CI does:

```powershell
$agent = "backend\agents\Protein_visualization"

uv run --project $agent ruff check $agent
uv run --project $agent ruff format --check $agent
uv run --project $agent mypy --config-file $agent\pyproject.toml $agent\app
uv run --project $agent pytest $agent\tests -q -m "not live and not bge and not qdrant"
```

That command is the deterministic offline gate. The `live`, `bge`, and
`qdrant` markers are opt-in because they contact external services, may incur
Azure cost, download the embedding model, or write to a managed collection;
passing the offline gate does not claim those operational checks were run.

`mypy` has to be started from the repository root: that is what makes it resolve
these files as `backend.agents.Protein_visualization.*` rather than as a second,
unrelated `app.*` package.

See [API contract](docs/api-contract.md), [workflow](docs/workflow.md), and [agent card](docs/agent-card.md).
