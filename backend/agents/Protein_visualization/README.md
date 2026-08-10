# Umbrella Protein Agent

A FastAPI service that resolves gene/protein identifiers, discovers experimental and predicted structures, maps functional annotations, and returns a viewer-ready Mol* specification. External evidence is kept distinct by source and optional failures produce a partial response.

## Infrastructure in this sprint

- **Qdrant** — a managed cluster; set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. There is no `docker-compose.yml`: nothing is provisioned locally.
- **PostgreSQL** — out of scope for this sprint. `PERSISTENCE_ENABLED=false` keeps the repository layer dormant; the API answers without a database.
- **Azure** — the LLM provider for explanation and critic generation (`LLM_PROVIDER=azure`).
- **BGE-M3** — the embedding model for ingestion and retrieval.

## Two ways this agent runs

| | Port | Serves | Callers |
|---|---|---|---|
| `api.py` | 8008 | `POST /execute` | the Global Orchestrator only |
| `app/main.py` | 8010 | the full scientific API + Swagger | scripts, and you |

`api.py` is the platform boundary: it takes `{instruction, context}` from the
orchestrator, resolves the gene and species named in the chat message against
UniProt, runs the same workflow, and returns an `AgentResult`. See
[orchestrator_adapter.py](orchestrator_adapter.py) for why identity is resolved
there rather than trusted from the context.

`app/main.py` is the same workflow with its scientific contract exposed
directly, which is what the scripts below drive.

## Run locally

```powershell
cd backend/agents/Protein_visualization
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
Copy-Item .env.example .env   # then fill in QDRANT_* and AZURE_*
```

Both services are started from the **repository root**, so that the
`backend.agents.Protein_visualization.*` package path resolves:

```powershell
# the orchestrator-facing agent (what the chat UI reaches)
.\backend\agents\Protein_visualization\.venv\Scripts\python.exe -m uvicorn `
    backend.agents.Protein_visualization.api:app --port 8008

# the standalone scientific service
.\backend\agents\Protein_visualization\.venv\Scripts\python.exe -m scripts.serve
```

`python -m backend.run_agents` starts the first one for you, along with every
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
# in-process: real providers -> LangGraph -> response
python -m scripts.run_orchestrator --accession P04637 --gene TP53 --residue 273

# over HTTP against the running server
python -m scripts.smoke_test --accession P04637 --gene TP53

# index knowledge into the managed Qdrant collection
python -m scripts.seed_qdrant documents.json

# or fetch real UniProt/InterPro evidence and index it directly
python -m scripts.seed_protein_knowledge --accession P04637
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

```powershell
ruff check .
ruff format --check .
mypy --config-file mypy.ini app
pytest -q -m "not live and not bge and not qdrant"
```

See [API contract](docs/api-contract.md), [workflow](docs/workflow.md), and [agent card](docs/agent-card.md).
