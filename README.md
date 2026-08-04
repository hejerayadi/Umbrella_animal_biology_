# Umbrella_animal_biology_

AI-powered multi-agent platform for animal genomics and biodiversity research. It integrates genomic databases, LLMs, and bioinformatics to analyze genomes, reconstruct missing DNA regions, discover gene-trait relationships, compare species, and explore biological knowledge through a conversational interface.

## Architecture

Every agent is independent and talks **only** to the Global Orchestrator. The frontend never calls an agent directly.

```
Frontend  ->  Global Orchestrator  ->  Agent API (api.py)  ->  Agent Logic
                  :8000                   HTTP :8001-8009           |
Frontend  <-  Global Orchestrator  <-------  AgentResult  <---------+
```

Each agent exposes exactly one endpoint, `POST /execute`, which takes an `AgentRequest` and always returns an `AgentResult` — including on failure, so the orchestrator only ever parses one schema.

Agents are reached over HTTP, never imported. [backend/registry.py](backend/registry.py) holds URLs, not instances, so the orchestrator depends on no agent's Python code and one broken agent package cannot stop it from starting.

## Project Structure

```
backend/
  api.py                   HTTP entry point for the frontend (POST /api/chat)
  registry.py              The only place the orchestrator meets the agents
  orchestrator/            Planner, Capability Resolver, Responder, Router
    langgraph/             LangGraph execution engine (nodes/ + graph wiring)
    requirements.txt       Orchestrator dependencies
    .env.example           Azure OpenAI credentials template
  agents/<agent>/
    api.py                 Communication boundary - no business logic
    mock.py                Agent logic (mock for now)
    schema.py              AgentRequest / AgentResult / AgentStatus
    card.json              Capabilities, read by the orchestrator's planner
    requirements.txt       This agent's dependencies only
    .env.example           This agent's environment template
frontend/                  React / TanStack Start web UI
```

## Prerequisites

- Python 3.11+
- Node.js 18+

## Environment files

Every `.env` in this repo has a committed `.env.example` next to it. **Copy the example, then fill it in** — `.env.example` is committed and must never contain a real secret; `.env` is git-ignored and must never be committed.

```powershell
copy backend\orchestrator\.env.example backend\orchestrator\.env
```

| Copy this | → to | Holds | Required? |
|---|---|---|---|
| `backend/orchestrator/.env.example` | `backend/orchestrator/.env` | `azure_endpoint`, `openai_key_azure` | **Yes** — planner/resolver/responder fail without it |
| `frontend/.env.example` | `frontend/.env` | `VITE_ORCHESTRATOR_API_URL` | No — defaults to `http://localhost:8000` |
| `backend/agents/<agent>/.env.example` | `backend/agents/<agent>/.env` | that agent's own keys | No — the mocks need no credentials yet |

The orchestrator's `.env` is loaded by [backend/orchestrator/llm.py](backend/orchestrator/llm.py) relative to its own location, so it works no matter which folder you run from.

Anything prefixed `VITE_` in the frontend is compiled into browser code and is **public** — never put a secret there.

## Running the orchestrator

From the **repository root** (a shared venv for the orchestrator; agents get their own, see below):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\orchestrator\requirements.txt
```

Then either serve the API the frontend talks to:

```powershell
python -m uvicorn backend.api:app --reload --port 8000
```

…or run the console demo, which exercises the full Planner → Worker → Capability Resolver loop against the mocks:

```powershell
python -m backend.test_orchestrator
```

> Always use the `-m` form from the repository root. Running `python backend\api.py`, or `cd backend` first, fails with `ImportError: attempted relative import with no known parent package`.

## Running an agent

Each agent is developed and run independently, in **its own virtual environment**, so one agent's dependencies can never break another's. Replace `<agent>` with a folder name from `backend/agents/`:

```powershell
python -m venv backend\agents\<agent>\.venv
backend\agents\<agent>\.venv\Scripts\Activate.ps1
pip install -r backend\agents\<agent>\requirements.txt
```

Serve it — still **from the repository root**, so Python can resolve the `backend.agents...` package path:

```powershell
python -m uvicorn backend.agents.<agent>.api:app --port <port>
```

| Agent folder | Port | Interactive docs |
|---|---|---|
| `genome_agent` | 8001 | http://localhost:8001/docs |
| `evolution_agent` | 8002 | http://localhost:8002/docs |
| `biodiversity_agent` | 8003 | http://localhost:8003/docs |
| `Literature_Agent` | 8004 | http://localhost:8004/docs |
| `multimodal_recognition_agent` | 8005 | http://localhost:8005/docs |
| `reconstruction_agent` | 8006 | http://localhost:8006/docs |
| `trait_discovery_agent` | 8007 | http://localhost:8007/docs |
| `Protein_visualization` | 8008 | http://localhost:8008/docs |
| `image_generation_agent` | 8009 | http://localhost:8009/docs |

Port 8000 is reserved for the orchestrator API.

Call any agent directly to check it works:

```powershell
curl.exe -X POST http://localhost:8001/execute -H "Content-Type: application/json" -d "{\"instruction\":\"Get the genome\",\"context\":{\"species\":\"woolly mammoth\"}}"
```

Every agent answers with the same shape:

```json
{
  "status": "completed",
  "target_agent": null,
  "prompt_to_target_agent": null,
  "output": { "genome": "Genome sequence of woolly mammoth" }
}
```

`status` is one of `completed`, `needs_agent`, `continue`, `failed`. A crash inside agent logic still returns HTTP 200 with `status: "failed"` — the schema never changes.

### Starting all agents at once

The orchestrator calls agents over HTTP, so **they must be running** before a query can get past the first worker. To start all nine in one terminal:

```powershell
python -m backend.run_agents
```

Ctrl+C stops them all. Logs are interleaved and prefixed with the agent name.

Each agent's URL can be overridden with an environment variable, so the same code runs against local processes, containers, or deployed services:

```powershell
$env:GENOME_AGENT_URL = "http://genome-agent.internal:8001"
```

The pattern is `<AGENTNAME>_AGENT_URL` — `GENOME_AGENT_URL`, `TRAIT_AGENT_URL`, `IMAGEGENERATION_AGENT_URL`, and so on, matching the keys in [backend/registry.py](backend/registry.py).

If an agent is unreachable the orchestrator does not crash: that worker returns `status: "failed"` with the connection error in `output`, and the graph routes on to the responder as usual.

## Running the frontend

```powershell
cd frontend
npm install
npm run dev
```

Then open the local URL printed in the terminal.

## Running everything together

Three terminals, all started from the repository root:

```powershell
# Terminal 1 - all nine agent services (8001-8009)
.\venv\Scripts\Activate.ps1
python -m backend.run_agents

# Terminal 2 - orchestrator API
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.api:app --reload --port 8000

# Terminal 3 - frontend
cd frontend
npm run dev
```

Terminal 1 is required: the orchestrator reaches agents over HTTP, so without it every worker returns `failed`.

Sending a chat message in the UI runs the real orchestrator: the Agent Thinking panel reflects the actual `Planner -> Worker -> Capability Resolver` steps returned by the backend. If the backend isn't reachable, the chat shows an error instead of hanging.

## Tests

The Trait Discovery Agent's LangGraph workflow has its own suite:

```powershell
python -m pytest backend\agents\trait_discovery_agent\tests -v
```
