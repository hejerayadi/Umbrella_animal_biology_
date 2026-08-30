# Umbrella

AI-powered multi-agent platform for animal genomics and biodiversity research. Nine specialist agents sit behind one conversational interface: ask a question in plain English, and a central orchestrator decides which agents to run, feeds them each other's findings, and writes the answer.

- [Architecture](#architecture)
- [The agents](#the-agents)
- [Inside the orchestrator](#inside-the-orchestrator)
- [How the orchestrator processes a request](#how-the-orchestrator-processes-a-request)
- [The agent contract](#the-agent-contract)
- [Running it](#running-it)
- [Working on an agent](#working-on-an-agent)

## Architecture

Every agent is an independent HTTP service that talks **only** to the orchestrator. The frontend never calls an agent, and **agents never call each other**.

```
Frontend  ->  Backend API  ->  Global Orchestrator  ->  Agent service (api.py)  ->  agent logic
   React      POST /api/v1/chat   LangGraph state machine     POST :8001-8009            |
Frontend  <---------------------  final answer  <------------  AgentResult  <------------+
```

Agents are reached over HTTP, never imported. [backend/registry.py](backend/registry.py) holds URLs and `card.json` metadata — no agent Python code — so a broken agent package cannot stop the orchestrator from starting, and any agent can be moved to a container or a remote host by setting `<AGENTNAME>_AGENT_URL`.

```
backend/
  app/                     FastAPI application: auth, sessions, workspace, chat endpoint
    api/v1/workspace.py    POST /api/v1/chat - where a user message enters the system
  registry.py              Agent names -> card.json + service URLs
  run_agents.py            Dev helper: starts all nine agent services, each in its own venv
  orchestrator/            SHARED - planner, extractor, router, resolver, responder, langgraph/
  agents/<agent>/          One folder per agent: api.py, card.json, logic, tests, own venv
frontend/                  React / TanStack Start web UI
```

## The agents

Each agent owns one scientific domain and one port. What it *claims* to do lives in its `card.json` — that file, not its code, is what the orchestrator routes on.

| Agent | Port | What it does |
|---|---|---|
| **Genome** | 8001 | Resolves a species to its NCBI assembly, then returns genome size, chromosome count, assembly level, the annotated gene table and gene-symbol list, plus genome-size comparisons against reference species. |
| **Evolution** | 8002 | Compares molecular sequences to reconstruct phylogeny (alignment, maximum-likelihood trees, branch support), and builds a functional similarity network from protein embeddings. |
| **Biodiversity** | 8003 | A sub-orchestrator of its own: species distribution mapping, habitat visualization, biodiversity hotspot detection and migration analysis over GBIF/IUCN/WorldClim/Movebank, returning a report and a rendered map. |
| **Literature** | 8004 | Searches and summarizes published evidence (PubMed, Semantic Scholar) with citations and DOIs, and supports scientific writing — drafting abstracts, introductions and discussions, and recommending publication venues. |
| **Multimodal Recognition** | 8005 | Identifies the species in a supplied photograph via BioCLIP-2, returning ranked taxonomic candidates annotated with GBIF and NCBI Taxonomy ids. Needs both an image and a text instruction; "uncertain" is a real answer, not a failure. |
| **Reconstruction** | 8006 | Reconstructs incomplete genomes — fills gaps in fragmented or partial sequence data. Works on a specific sequence or accession, so the Genome agent normally resolves the target and hands off to it. |
| **Trait Discovery** | 8007 | Identifies a species' traits and the genes behind them via Gene Ontology and UniProt — morphology, behaviour, life history, and the gene-trait associations and pathway evidence explaining an adaptation. |
| **3D Protein Structure** | 8008 | Resolves a gene or accession to a protein identity, finds the best experimental or predicted structure (RCSB PDB, AlphaFold), maps domains and binding sites onto it, and returns a viewer-ready molecular scene. |
| **Image Generation** | 8009 | Renders 2D scientific illustrations with FLUX.2-pro from the traits other agents gathered. Escalates to Trait Discovery when the traits it needs aren't in context yet. |

Port 8000 is the backend API.

## Inside the orchestrator

| Component | File | LLM? | Job |
|---|---|---|---|
| **Planner** | [planner.py](backend/orchestrator/planner.py) | Yes | Decides whether the message needs an agent at all, which agent starts, and whether a follow-up agent is scheduled |
| **Extractor** | [extractor.py](backend/orchestrator/extractor.py) | Yes | Pulls the subject of the question (species, gene, trait) out of the sentence and seeds it into `context` |
| **Router** | [router.py](backend/orchestrator/router.py) | **No** | Pure if/else traffic cop — reads `status`, picks the next node |
| **Capability Resolver** | [capability_resolver.py](backend/orchestrator/capability_resolver.py) | Yes | When an agent says "I need help", picks who can help |
| **Responder** | [responder.py](backend/orchestrator/responder.py) | Yes | Writes the final prose answer from the collected findings and failures |
| **Execution engine** | [langgraph/](backend/orchestrator/langgraph/) | — | LangGraph state machine wiring all of the above together |

The planner and resolver choose agents by reading each agent's `card.json` — **never** its Python code.

## How the orchestrator processes a request

A message arrives at `POST /api/v1/chat` with a query, an optional starting context, and optionally an uploaded image id. The endpoint hands it to `GlobalOrchestrator.run()`, which builds a [WorkflowState](backend/orchestrator/state.py) — the "clipboard" every graph node reads and updates — and invokes the compiled graph:

```
START -> planner -+-> direct_answer -------------------------------------------> END
                  |
                  +-> extractor -> <agent> -+-> capability_resolver -> <agent> -+
                                            +-> same agent (retry)             |
                                            +-> waiting agent (resume)         |
                                            +-> responder ---------------------+-> END
```

Take *"What traits does the woolly mammoth have?"*:

1. **Planner** reads every `card.json` and picks `Trait`. (A greeting or a question about the platform gets `needs_agent: false` and goes straight to `direct_answer`.)
2. **Extractor** seeds `context["species"] = "woolly mammoth"`, so the first agent finds the subject already in context instead of failing on "species not specified".
3. **Trait** is POSTed at `:8007/execute` and returns `needs_agent` — "I need the genome first".
4. **Router** sees `needs_agent`, parks Trait on the waiting stack and routes to the **Capability Resolver**, which reads the cards and picks `Genome`.
5. **Genome** runs at `:8001/execute` and returns `completed` with `{"genome": ...}`. Its output is merged into the shared context.
6. **Router** sees a waiting agent and resumes **Trait**, which now finds `context["genome"]` and returns `completed`.
7. **Responder** writes the final answer from everything collected; the endpoint returns it with the execution history and context.

Four mechanisms carry the run:

**Shared context.** A `completed` (or escalating) agent's `output` dict is merged into one context dict passed to every later agent. **Output key names are a cross-agent contract** — rename a key you emit and you break whoever reads it.

**Escalation.** An agent that cannot proceed returns `needs_agent` with a plain-English `prompt_to_target_agent`. It does not choose or call the helper — the resolver does. The orchestrator remembers who is waiting (a stack, so nested A→B→C dependencies resume in order), sends the helper the request that was actually made rather than the user's original sentence, and replays the same instruction when it resumes the paused agent.

**Loop breaking.** Each escalation snapshots the context keys that existed at the time. If an agent escalates again and nothing new has arrived, the dependency is unsatisfiable, so the run is converted to `failed` instead of spinning. A retryable `continue` is retried on a bounded 1/2/4-second backoff, then fails.

**Follow-ups and honest failure.** The planner can schedule one agent to run *after* the main line of work — used for rendering, which nothing escalates to because only the user's sentence asks for a picture. A failure does not cancel that follow-up: the two halves of "what traits let the Arctic fox survive the cold, and draw it" are independent. Every failure is recorded in `state.failures`, so the responder reports the half that fell over even when a later agent succeeded.

An unreachable agent is not a crash: the worker node turns any transport error or unparseable reply into `status: "failed"` and the graph routes on to the responder as usual. Every agent call carries `X-Trace-Id` (stable for the whole run) and `X-Request-Id` (unique per attempt).

## The agent contract

Each agent exposes exactly one endpoint, `POST /execute`, taking an `AgentRequest` and always returning an `AgentResult` — including on failure, so the orchestrator only ever parses one schema.

**Input** — `instruction` (`str`, what this agent is being asked) and `context` (`dict`, everything produced so far this run).

**Output** — `status`, `output`, plus `target_agent` / `prompt_to_target_agent` when escalating:

| Status | Return it when | What the orchestrator does |
|---|---|---|
| `completed` | You did the work | Merges your `output` dict into context, resumes whoever was waiting |
| `needs_agent` | You need another agent's data first | Asks the resolver, runs that agent, calls you again with a fuller context |
| `failed` | You cannot proceed at all | Records the failure, runs any scheduled follow-up, responder explains honestly |
| `continue` | You need another turn | Calls you again, bounded retries |

## Running it

**Prerequisites:** Python 3.11+, Node.js 18+, Git.

Every `.env` has a committed `.env.example` next to it — copy it and fill it in. `.env.example` never holds a real secret; `.env` is git-ignored and never committed. The two that matter first: `backend/.env` (database, Redis, SMTP) and `backend/orchestrator/.env` (Azure OpenAI — planner, extractor, resolver and responder all fail without it). Anything prefixed `VITE_` in the frontend is compiled into browser code and is **public**.

One-time setup, from the **repository root**:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\orchestrator\requirements.txt
python -m backend.run_agents --setup   # a .venv per agent; slow, but only once
```

`--setup` picks the right tool per agent: `uv sync` where there's a `pyproject.toml` (currently `Protein_visualization`), `pip install -r requirements.txt` otherwise.

Then three terminals, all from the repository root:

```powershell
python -m backend.run_agents                              # 1 - all nine agents, 8001-8009
python -m uvicorn backend.api:app --reload --port 8000    # 2 - backend API
cd frontend; npm install; npm run dev                     # 3 - frontend
```

Terminal 1 is required: the orchestrator reaches agents over HTTP, so without it every agent returns `failed`. Ctrl+C stops all nine — killing the terminal instead leaves the uvicorn children holding ports 8001-8009:

```powershell
8001..8009 | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -EA SilentlyContinue } |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

> Always use the `-m` form from the repository root. `python backend\api.py`, or `cd backend` first, fails with `ImportError: attempted relative import with no known parent package`.

To run or test one agent on its own — `/docs` on its port gives you a live request form:

```powershell
python -m uvicorn backend.agents.<agent>.api:app --port <port> --reload
python -m pytest backend\agents\<agent>\tests -v
```

## Working on an agent

**You own exactly one directory: `backend/agents/<your_agent>/`.** Everything in it is yours; nothing outside it is. `backend/orchestrator/`, `backend/registry.py`, `backend/app/` and `frontend/` are shared — if you believe one has to change, raise it with the group rather than editing it in your branch. Each agent keeps its own `.venv` and its own pinned `requirements.txt`, so one agent's dependencies can never break another's.

**Never call another agent.** Don't import one, don't HTTP-call one. When you need data you don't have, return `needs_agent` and say what you need in plain English — the orchestrator routes it, merges the result into `context`, and calls you again:

```python
if "genome" not in request.context:
    return AgentResult(
        status=AgentStatus.NEEDS_AGENT,
        target_agent="Genome",          # a hint; the resolver may pick differently
        prompt_to_target_agent="Retrieve the genome of the requested species.",
    )
```

**Keep `card.json` honest.** This is the step people forget, and it silently breaks routing. The planner and resolver never read your Python — they read your `description` and `capabilities`, so write them as the questions you want routed to you, and keep `output` matching the keys you actually emit. If the planner never picks you, this file is almost always why.

**Keep the contract.** `POST /execute` must always accept an `AgentRequest` and always return an `AgentResult`, whatever happens inside — `api.py` already guarantees that, including turning a crash into `failed`, so leave its try/except alone. Cover every status your agent can return with tests in your own `tests/` folder; [trait_discovery_agent](backend/agents/trait_discovery_agent/) is the reference for a large agent, with its own workflows, subagents and suite inside one folder.

Branch as `feature/<your-agent>/<what-you-are-building>`, and stage your folder (`git add backend/agents/<your_agent>`) rather than `.` — if `git status` shows anything outside it, ask before changing it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: attempted relative import with no known parent package` | Ran a file directly instead of as a module | Use `python -m ...` from the repository root |
| `ModuleNotFoundError: No module named 'fastapi'` | Wrong venv active, or deps not installed | Activate your agent's `.venv`, then `pip install -r ...\requirements.txt` |
| Every agent returns `failed` with "unreachable" | Agent services aren't running | `python -m backend.run_agents` in another terminal |
| `KeyError: 'azure_endpoint'` | Orchestrator `.env` missing | Copy `backend\orchestrator\.env.example` and fill it in |
| The planner never routes to your agent | `card.json` doesn't describe what you do | Rewrite `description` and `capabilities` as the questions you want to receive |
| `[WinError 10048] address already in use` | Port already taken | Stop the old process, or use a different `--port` |
