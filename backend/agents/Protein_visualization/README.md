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
| `POST /api/v1/protein-structure-analyses` | frontend viewer | the full `AgentTask`, answering with the complete analysis: Mol* scene, every annotation, every evidence record |
| `GET /api/v1/taxonomy?name=…` | anyone building an `AgentTask` | species name → `{scientific_name, taxon_id}`. The full endpoint takes an id; users type a name |
| `GET /console/` | you | the test console, below |

`/execute` is a translation layer only (`app/api/execute.py`); the scientific
decisions stay in the workflow and in `result_policy.to_agent_result`. It reads
`species` plus a gene symbol or accession out of the shared context, resolves
the species name to an NCBI taxonomy id against UniProt, and returns a
**summary** rather than the full payload — whatever an agent returns under
`completed` is merged into every later agent's context and rendered into the
Responder's prompt, so the Mol* scene would cost thousands of tokens a turn.

Without a species, or without any of gene / accession / sequence, it answers
`needs_agent` with a prompt describing what it is missing. That is a real
dependency on upstream agents, read from the context it was actually given.

```powershell
curl.exe -X POST http://localhost:8008/execute -H "Content-Type: application/json" `
  -d '{\"instruction\":\"3D structure of TP53 in humans\",\"context\":{\"species\":\"Homo sapiens\",\"gene_name\":\"TP53\",\"residue_position\":273}}'
```

## Test console

**<http://localhost:8008/console/>** — one self-contained Bootstrap page in
[`console/`](console/index.html), served by the agent itself so it is same-origin
with the API and no CORS entry has to know about it. It is a developer tool: it
runs real analyses against the real providers, so it is not mounted when
`APP_ENV=production`.

A form drives either endpoint — species, gene or accession, residue, mutation,
regions, preferred source, explanation on/off — with presets for the cases worth
re-running (human TP53 with a residue, a common-name species, an accession that
falls back to AlphaFold, a request with no gene that ends in a hand-off). Then
six tabs:

- **Workflow** — all 18 graph nodes, each marked *ran*, *degraded* (with its
  retry count and error) or *skipped*, so you can see that AlphaFold was never
  reached because an experimental structure survived, or that SIFTS ran because
  you asked for a residue. Fed by `executed_nodes` / `errors` / `retry_counts`
  on the response. On `/execute` it instead shows the routing status and the
  exact summary the orchestrator merges into the shared context.
- **Structure** — the Mol* viewer loading the scene the workflow built, with the
  SIFTS-mapped residues selected and focused.
- **JSON** — the request as sent and the response as received, side by side.
- **Evidence** — evidence records, annotations, residue mappings and the
  structure candidates that lost.
- **Explanation & usage** — the grounded summary and its stated limitations,
  plus a table of every Azure OpenAI call this run made: node, model, latency,
  input/output/total tokens, and an estimated cost if `AZURE_INPUT_PRICE_PER_1K_USD`
  / `AZURE_OUTPUT_PRICE_PER_1K_USD` are set (see `.env.example`) — tokens and
  latency always come from Azure's own response either way.
- **Knowledge search** — queries the RAG store directly (`POST
  /api/v1/knowledge/search`), independent of any analysis run. Protein and
  taxon id prefill from the last run, or set them by hand to explore what is
  indexed for a protein you haven't analyzed yet.

The page is exercised end to end in headless Chromium; `tests/integration/test_console.py`
covers the API contract it depends on.

## Infrastructure in this sprint

- **Qdrant** — a managed cluster; set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. There is no `docker-compose.yml`: nothing is provisioned locally.
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

The backend host and port are read from `.env`:

```env
APP_HOST=127.0.0.1
APP_PORT=8000
APP_RELOAD=true
```

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

`mypy` has to be started from the repository root: that is what makes it resolve
these files as `backend.agents.Protein_visualization.*` rather than as a second,
unrelated `app.*` package.

See [API contract](docs/api-contract.md), [workflow](docs/workflow.md), and [agent card](docs/agent-card.md).
