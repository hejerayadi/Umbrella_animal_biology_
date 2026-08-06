# Umbrella_animal_biology_

AI-powered multi-agent platform for animal genomics and biodiversity research. It integrates genomic databases, LLMs, and bioinformatics to analyze genomes, reconstruct missing DNA regions, discover gene-trait relationships, compare species, and explore biological knowledge through a conversational interface.

---

## Table of contents

- [Architecture](#architecture)
- [How a request flows](#how-a-request-flows)
- [The agent contract](#the-agent-contract)
- [Roadmap](#roadmap)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment files](#environment-files)
- [Running the orchestrator](#running-the-orchestrator)
- [Running an agent](#running-an-agent)
- [**Working on your own agent**](#working-on-your-own-agent) ← start here if you own an agent
- [Running the frontend](#running-the-frontend)
- [Running everything together](#running-everything-together)
- [Tests](#tests)

---

## Architecture

Every agent is independent and talks **only** to the Global Orchestrator. The frontend never calls an agent directly, and **agents never call each other**.

```
Frontend  ->  Global Orchestrator  ->  Agent API (api.py)  ->  Agent Logic
                  :8000                   HTTP :8001-8009           |
Frontend  <-  Global Orchestrator  <-------  AgentResult  <---------+
```

Each agent exposes exactly one endpoint, `POST /execute`, which takes an `AgentRequest` and always returns an `AgentResult` — including on failure, so the orchestrator only ever parses one schema.

Agents are reached over HTTP, never imported. [backend/registry.py](backend/registry.py) holds URLs, not instances, so the orchestrator depends on no agent's Python code and one broken agent package cannot stop it from starting.

### Inside the orchestrator

| Component | File | Uses an LLM? | Job |
|---|---|---|---|
| **Planner** | [planner.py](backend/orchestrator/planner.py) | Yes | Decides whether a message needs an agent at all, and which one starts |
| **Router** | [router.py](backend/orchestrator/router.py) | **No** | Pure if/else traffic cop — reads `status`, picks the next node |
| **Capability Resolver** | [capability_resolver.py](backend/orchestrator/capability_resolver.py) | Yes | When an agent says "I need help", picks who can help |
| **Responder** | [responder.py](backend/orchestrator/responder.py) | Yes | Writes the final prose answer from the collected findings |
| **Execution engine** | [langgraph/](backend/orchestrator/langgraph/) | — | LangGraph state machine wiring all of the above together |

The planner and resolver choose agents by reading each agent's `card.json` — **never** its Python code.

## How a request flows

Take *"What traits does the woolly mammoth have?"*:

```
1. Frontend POSTs to  :8000/api/chat
2. Planner            reads every card.json, picks "Trait"
3. Trait agent        POST :8007/execute  ->  needs_agent ("I need the genome first")
4. Router             sees needs_agent    ->  go to Capability Resolver
5. Capability Resolver reads the cards    ->  picks "Genome"
6. Genome agent       POST :8001/execute  ->  completed {"genome": "..."}
                      output merged into the shared context; Trait resumes
7. Trait agent        POST :8007/execute  ->  completed {"traits": [...]}
8. Responder          writes the final answer from everything collected
9. Frontend           renders it
```

Two mechanisms make this work, and both matter when you build an agent:

**The shared context.** When an agent returns `completed`, its `output` dict is merged into a context dict that every later agent receives. The Genome agent emits `{"genome": ...}`; the Trait agent checks `if "genome" not in request.context`. **Output key names are a cross-agent contract** — if you rename a key you emit, you break whoever reads it. Agree on key names before changing them.

**Escalation.** An agent that cannot proceed returns `needs_agent` with a plain-English `prompt_to_target_agent`. It does **not** choose or call the helper — the Capability Resolver does that. The orchestrator remembers who is waiting and resumes them automatically once the dependency completes.

## The agent contract

This is the whole interface. Everything else about your agent is yours.

**Input** — `AgentRequest`:

| Field | Type | Meaning |
|---|---|---|
| `instruction` | `str` | The user's original question, verbatim |
| `context` | `dict` | Everything agents have produced so far this run |

**Output** — `AgentResult`:

| Field | Type | Meaning |
|---|---|---|
| `status` | `AgentStatus` | `completed` / `needs_agent` / `continue` / `failed` |
| `output` | `Any` | On `completed`, a **dict** — it gets merged into the shared context |
| `target_agent` | `str \| None` | Hint for who could help (the resolver may override) |
| `prompt_to_target_agent` | `str \| None` | Plain English: what you need and why |

**The four statuses:**

| Status | Return it when | What the orchestrator does |
|---|---|---|
| `completed` | You did the work | Merges `output` into context, resumes whoever was waiting |
| `needs_agent` | You need another agent's data first | Asks the resolver, runs that agent, then calls you again |
| `failed` | You cannot proceed at all | Stops running agents, responder explains honestly to the user |
| `continue` | You need another turn | Calls you again (rarely used today) |

## Roadmap

| Phase | Status | What it covers |
|---|---|---|
| **1. Contract & mocks** | ✅ Done | 9 agent folders, each with `schema.py`, `mock.py`, `card.json`; orchestrator with planner, router, capability resolver, responder |
| **2. Framework selection** | ✅ Done | LangGraph vs CrewAI evaluated — see [framework_benchmark_results.md](backend/agents/trait_discovery_agent/docs/framework_benchmark_results.md). **LangGraph chosen**, because conditional branching is a first-class primitive and `needs_agent` routing depends on it |
| **3. Service boundary** | ✅ Done | `api.py` per agent (`POST /execute`); orchestrator calls agents over HTTP; `registry.py` holds URLs, imports no agent code |
| **4. Real agent logic** | 🔄 **In progress — this is where the team is now** | Each owner replaces their `mock.py` with a real implementation. Furthest along: `trait_discovery_agent`, which has a full LangGraph workflow in [workflows/](backend/agents/trait_discovery_agent/workflows/) (its own sub-agents are still mocked). Everyone else is still at `mock.py` |
| **5. Real data sources** | ⬜ Not started | Swap in-agent fixtures for real external APIs and databases, per agent |
| **6. Deployment** | ⬜ Not started | One container per agent (`trait_discovery_agent` already has a [Dockerfile](backend/agents/trait_discovery_agent/Dockerfile)), plus a compose file for the whole stack |

Phase 4 is deliberately parallel: nine people can work at once **because** phase 3 fixed the boundary. As long as your `POST /execute` keeps honouring the contract above, nothing you do inside your folder can break anyone else.

## Project structure

```
backend/
  api.py                   HTTP entry point for the frontend (POST /api/chat)
  registry.py              Agent names -> card.json + service URLs
  run_agents.py            Dev helper: starts all nine agent services at once
  agent_card.py            The AgentCard dataclass
  orchestrator/            SHARED - do not edit unless you own the orchestrator
    planner.py             Picks the starting agent (LLM)
    capability_resolver.py Picks the helping agent (LLM)
    responder.py           Writes the final answer (LLM)
    router.py              Pure routing logic (no LLM)
    state.py               WorkflowState passed between graph nodes
    schema.py              Orchestrator-side AgentResult / AgentStatus
    llm.py                 Shared Azure OpenAI client
    langgraph/             Execution engine
      graph.py             Graph wiring
      orchestrator.py      GlobalOrchestrator - public entry point
      nodes/               One module per node type
    requirements.txt       Orchestrator dependencies
    .env.example           Azure OpenAI credentials template
  agents/<agent>/          YOURS, if you own this agent
    api.py                 Communication boundary - no business logic
    mock.py                Agent logic (replace this with the real thing)
    schema.py              AgentRequest / AgentResult / AgentStatus
    card.json              Capabilities - THIS is what the planner routes on
    description.md         Human-readable objective and example questions
    requirements.txt       This agent's dependencies only
    .env.example           This agent's environment template
frontend/                  React / TanStack Start web UI
```

## Prerequisites

- Python 3.11+
- Node.js 18+ (frontend only)
- Git

## Environment files

Every `.env` in this repo has a committed `.env.example` next to it. **Copy the example, then fill it in** — `.env.example` is committed and must never contain a real secret; `.env` is git-ignored and must never be committed.

```powershell
copy backend\orchestrator\.env.example backend\orchestrator\.env
```

| Copy this | → to | Holds | Required? |
|---|---|---|---|
| `backend/orchestrator/.env.example` | `backend/orchestrator/.env` | `azure_endpoint`, `openai_key_azure` | **Yes** — planner/resolver/responder fail without it |
| `frontend/.env.example` | `frontend/.env` | `VITE_ORCHESTRATOR_API_URL` | No — defaults to `http://localhost:8000` |
| `backend/agents/<agent>/.env.example` | `backend/agents/<agent>/.env` | that agent's own keys | Only once your agent needs credentials |

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

…or run the console demo, which exercises the full Planner → Worker → Capability Resolver loop:

```powershell
python -m backend.test_orchestrator
```

> Always use the `-m` form from the repository root. Running `python backend\api.py`, or `cd backend` first, fails with `ImportError: attempted relative import with no known parent package`.

> Both of these make **real Azure OpenAI calls** (planner + resolver + responder on every query) and need all nine agents running.

## Running an agent

Each agent runs in **its own virtual environment**, so one agent's dependencies can never break another's. Replace `<agent>` with a folder name from `backend/agents/`:

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

### Starting all agents at once

The orchestrator calls agents over HTTP, so **they must be running** before a query can get past the first worker.

**One-time setup** — creates a `.venv` for every agent and installs each one's `requirements.txt`:

```powershell
python -m backend.run_agents --setup
```

This is slow (some agents pin large packages like `torch`), but you only do it once. Re-running it reuses any venv that already exists.

**Then start all nine**, each in its own virtual environment:

```powershell
python -m backend.run_agents
```

```
starting Genome           :8001  [own venv]
starting Evolution        :8002  [own venv]
...
9 agents starting. Ctrl+C to stop them all.
```

Every agent is launched with the interpreter from **its own** `.venv`, not the one running the launcher — that is what keeps the environments isolated. Any agent without a `.venv` yet falls back to the current interpreter and is flagged `[SHARED venv]` with a warning, so a missing setup step is visible rather than silent.

Ctrl+C stops them all. Logs are interleaved and prefixed with the agent name.

> If you kill the launcher some other way (closing the terminal, killing the PID), the nine uvicorn children can be left running and will hold ports 8001-8009. Use Ctrl+C. To clear stragglers:
> ```powershell
> 8001..8009 | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -EA SilentlyContinue } |
>   ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
> ```

Each agent's URL can be overridden with an environment variable, so the same code runs against local processes, containers, or deployed services:

```powershell
$env:GENOME_AGENT_URL = "http://genome-agent.internal:8001"
```

The pattern is `<AGENTNAME>_AGENT_URL` — `GENOME_AGENT_URL`, `TRAIT_AGENT_URL`, `IMAGEGENERATION_AGENT_URL`, and so on, matching the keys in [backend/registry.py](backend/registry.py).

If an agent is unreachable the orchestrator does not crash: that worker returns `status: "failed"` with the connection error in `output`, and the graph routes on to the responder as usual.

---

# Working on your own agent

## Rule 1: stay in your folder

**You own exactly one directory: `backend/agents/<your_agent>/`. Everything in it is yours. Nothing outside it is.**

| | Path | Why |
|---|---|---|
| ✅ **Edit freely** | `backend/agents/<your_agent>/**` | Your agent, your call |
| ❌ **Do not touch** | `backend/agents/<someone_else>/**` | Their logic and their tests — changing it breaks their work with no warning |
| ❌ **Do not touch** | `backend/orchestrator/**` | Shared by all nine agents; one edit affects everyone |
| ❌ **Do not touch** | `backend/registry.py`, `backend/api.py` | Shared wiring |
| ❌ **Do not touch** | `frontend/**` | Different owner entirely |

If you believe something shared has to change, **open an issue or ask in the group first** — don't edit it in your branch. A change to `orchestrator/` or `registry.py` in an agent PR will be sent back.

## Rule 2: never call another agent

Your agent has no idea the other eight exist, and it must stay that way. Do **not** import another agent, and do **not** HTTP-call one.

When you need data you don't have, return `needs_agent` and say what you need in plain English:

```python
if "genome" not in request.context:
    return AgentResult(
        status=AgentStatus.NEEDS_AGENT,
        target_agent="Genome",          # a hint; the resolver may pick differently
        prompt_to_target_agent="Retrieve the genome of the requested species.",
    )
```

The orchestrator routes it, runs the other agent, merges its output into `context`, and calls you again — this time with `context["genome"]` populated. You just check for what you need at the top of `run()` every time.

## Rule 3: keep the contract

Your `POST /execute` must always accept an `AgentRequest` and always return an `AgentResult`, whatever happens inside. `api.py` already guarantees this — including turning a crash into `status: "failed"` — so **leave the try/except in `api.py` alone**.

---

## Step-by-step: building your agent

### Step 1 — Get the repo and branch

```powershell
git clone <repo-url>
cd Umbrella_animal_biology_
git checkout -b feature/<your-agent>/<what-you-are-building>
```

Branch naming follows what's already in the history, e.g. `feature/trait-discovery-agent/sub-orchestrator-routing`.

### Step 2 — Create your own virtual environment

Inside your agent folder, so it never collides with anyone else's:

```powershell
python -m venv backend\agents\<your_agent>\.venv
backend\agents\<your_agent>\.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)`. Confirm you're in the right one:

```powershell
(Get-Command python).Source
```

It must point inside your agent folder. All `.venv` directories are git-ignored — never commit one.

### Step 3 — Install your dependencies

```powershell
pip install -r backend\agents\<your_agent>\requirements.txt
```

Your `requirements.txt` already contains `fastapi` and `uvicorn` under a header that says *"required, do not remove"* — the API layer needs them. Add your own libraries **below** that block.

### Step 4 — Confirm the mock works before changing anything

Always from the repository root:

```powershell
python -m uvicorn backend.agents.<your_agent>.api:app --port <your-port> --reload
```

Open `http://localhost:<your-port>/docs` — FastAPI generates a live page where you can fire test requests without writing any client code. Send one:

```powershell
curl.exe -X POST http://localhost:<your-port>/execute -H "Content-Type: application/json" -d "{\"instruction\":\"test\",\"context\":{}}"
```

If that returns JSON, your environment is correct and everything from here is your own logic.

### Step 5 — Write your real logic

Open `mock.py`. Replace the body of `run()` with the real implementation, keeping the signature:

```python
def run(self, request: AgentRequest) -> AgentResult:
```

Structure it in the same order every time:

```python
def run(self, request: AgentRequest) -> AgentResult:
    # 1. What do I need that I might not have?
    species = request.context.get("species")
    if species is None:
        return AgentResult(status=AgentStatus.FAILED, output="No species provided.")

    # 2. Do I depend on another agent's output?
    if "genome" not in request.context:
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent="Genome",
            prompt_to_target_agent="Retrieve the genome of the requested species.",
        )

    # 3. Do the actual work.
    result = my_real_analysis(species, request.context["genome"])

    # 4. Return a dict - it gets merged into the shared context.
    return AgentResult(status=AgentStatus.COMPLETED, output={"my_finding": result})
```

Grow beyond one file whenever you want — add `logic.py`, `clients/`, `tools/`, whatever suits. Only two things are fixed: `api.py` imports your entry class, and that class exposes `run(request) -> AgentResult`. If you rename the class or move it, update the import in **your own** `api.py`.

`trait_discovery_agent` is the reference for a large agent: [workflows/](backend/agents/trait_discovery_agent/workflows/), [subagents/](backend/agents/trait_discovery_agent/subagents/), [schemas/](backend/agents/trait_discovery_agent/schemas/) and its own test suite, all inside one folder.

### Step 6 — Pin your dependencies

Every library you import goes in your `requirements.txt`, pinned:

```
# --- Add this agent's own dependencies below ---
biopython==1.84
requests==2.32.3
```

If it's not in there, your agent works on your machine and nowhere else.

### Step 7 — Declare your environment variables

Need an API key? Add a **placeholder** to `.env.example`, then create your real `.env` next to it:

```powershell
copy backend\agents\<your_agent>\.env.example backend\agents\<your_agent>\.env
```

`.env.example` is committed and must only ever contain placeholders. `.env` is git-ignored and must never be committed. Load it in your code with `python-dotenv`, reading the file next to your module so it works from any working directory.

### Step 8 — Update `card.json` — do not skip this

**This is the step people forget, and it silently breaks routing.** The Planner and Capability Resolver never read your Python — they read `card.json`. If it doesn't describe what you actually do, the orchestrator will never send you the queries you can handle.

Keep every field that's already there — this is the real `genome_agent/card.json`:

```json
{
  "name": "Genome Agent",
  "type": "worker",
  "description": "Retrieves, analyzes and interprets genomic information.",
  "reports_to": ["Global Scientific Orchestrator"],
  "may_need": ["Literature Agent"],
  "managed_services": ["NCBI", "Ensembl"],
  "capabilities": ["Genome retrieval", "Gene retrieval", "Genome analysis"],
  "input":  { "instruction": "string", "context": "dict" },
  "output": { "genome": "string" }
}
```

| Field | Who reads it | Why it matters |
|---|---|---|
| `description`, `capabilities` | Planner + Capability Resolver (LLM) | Compared against the user's question. **Write them as the questions you want routed to you** |
| `output` | Other agents, via the shared context | Keys must match what you actually put in your `output` dict |
| `name` | Humans | Display only |
| `may_need`, `managed_services`, `reports_to`, `type` | Documentation | Not read by the router today, but keep them accurate |

Only `description`, `capabilities`, `input` and `output` are loaded into the orchestrator ([registry.py:`_load_card`](backend/registry.py)) — the rest is documentation. Don't delete them.

### Step 9 — Write tests

Add a `tests/` folder inside your agent. Test your logic directly — no HTTP, no orchestrator:

```python
def test_missing_species_fails():
    result = GenomeMock().run(AgentRequest(instruction="test", context={}))
    assert result.status == AgentStatus.FAILED
```

Cover every status your agent can return. Run them with your own venv active:

```powershell
python -m pytest backend\agents\<your_agent>\tests -v
```

See [trait_discovery_agent/tests/](backend/agents/trait_discovery_agent/tests/) for a worked example.

### Step 10 — Test your agent standalone

Restart your service and exercise each branch through `/docs` or curl. Check that:

- a normal request returns `completed` with your `output` dict
- a request missing a dependency returns `needs_agent`
- an impossible request returns `failed`
- nothing ever returns a 500

### Step 11 — Test with the orchestrator

Three terminals from the repository root:

```powershell
# Terminal 1 - all nine agents
python -m backend.run_agents

# Terminal 2 - orchestrator demo
python -m backend.test_orchestrator

# Terminal 3 (optional) - the UI
cd frontend; npm run dev
```

Watch the console: every agent call is logged in order, so you can see the planner pick you, your escalations resolve, and your output land in the final context. If the planner never picks you, go back to **step 8** — that's almost always `card.json`.

Terminal 2 needs `backend/orchestrator/.env` filled in and makes real Azure OpenAI calls.

### Step 12 — Commit and open a PR

```powershell
git add backend/agents/<your_agent>
git commit -m "feat(<your-agent>): replace mock with real implementation"
git push -u origin feature/<your-agent>/<what-you-are-building>
```

Note the `git add` path: **stage your folder, not `.`** That's the simplest way to guarantee you haven't accidentally included a shared file. Before pushing, check:

```powershell
git status
```

If anything outside `backend/agents/<your_agent>/` is listed, unstage it and ask the group before changing it.

**PR checklist**

- [ ] Only files inside `backend/agents/<your_agent>/` changed
- [ ] `requirements.txt` lists every library you import, pinned
- [ ] `.env.example` updated with placeholders; no real `.env` committed
- [ ] `card.json` matches what your agent actually does and outputs
- [ ] Tests exist and pass
- [ ] Agent starts and answers `POST /execute` for every status it can return
- [ ] No `.venv/`, `__pycache__/`, or `.env` in the diff

---

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

Each agent owns its tests. To run one agent's suite:

```powershell
python -m pytest backend\agents\<agent>\tests -v
```

Currently implemented — the Trait Discovery Agent's LangGraph workflow:

```powershell
python -m pytest backend\agents\trait_discovery_agent\tests -v
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: attempted relative import with no known parent package` | Ran a file directly instead of as a module | Use `python -m ...` from the repository root |
| `ModuleNotFoundError: No module named 'fastapi'` | Wrong venv active, or deps not installed | Activate your agent's `.venv`, then `pip install -r ...\requirements.txt` |
| Every agent returns `failed` with "unreachable" | Agent services aren't running | `python -m backend.run_agents` in another terminal |
| `KeyError: 'azure_endpoint'` | Orchestrator `.env` missing | `copy backend\orchestrator\.env.example backend\orchestrator\.env` and fill it in |
| The planner never routes to your agent | `card.json` doesn't describe what you do | Rewrite `description` and `capabilities` as the questions you want to receive |
| `[WinError 10048] address already in use` | Port already taken | Stop the old process, or use a different `--port` |
