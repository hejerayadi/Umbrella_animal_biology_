# Umbrella Protein Agent

A FastAPI service that resolves gene/protein identifiers, discovers experimental and predicted structures, maps functional annotations, and returns a viewer-ready Mol* specification. External evidence is kept distinct by source and optional failures produce a partial response.

## Infrastructure in this sprint

- **Qdrant** — a managed cluster; set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. There is no `docker-compose.yml`: nothing is provisioned locally.
- **PostgreSQL** — out of scope for this sprint. `PERSISTENCE_ENABLED=false` keeps the repository layer dormant; the API answers without a database.
- **Azure** — the LLM provider for explanation and critic generation (`LLM_PROVIDER=azure`).
- **BGE-M3** — the embedding model for ingestion and retrieval.

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
# Without it, app.knowledge_base.embeddings falls back to HashEmbedding.
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
