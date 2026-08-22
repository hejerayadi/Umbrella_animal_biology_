# Biodiversity Agent — Integration Guide

For frontend teams and other domain agents who want to consume the Biodiversity
Agent's four skills (species distribution, habitat, hotspots, migration).

**TL;DR** — start the HTTP boundary, POST an `AgentRequest` to `/execute`, read
the `AgentResult`. That's it. The LangGraph orchestrator, LLM intent
classification and worker dispatch happen server-side and are transparent.

---

## 1. What this agent does

Answers free-text biodiversity questions by routing them to one of four skills:

| Skill | Owner | Backing |
|---|---|---|
| `species_distribution_map` — where is species X observed? | Aziz | Real GBIF Occurrence Search |
| `biodiversity_hotspots` — where does biodiversity peak in region R? | Ouissale | GBIF + grid + Shannon/Simpson/Chao1 + DBSCAN (haversine, silhouette-tuned) |
| `migration_analysis` — where will species X go next? | Miriam Kouki | Qdrant vector search + Random Forest + SHAP + LLM explanation |
| `habitat_visualization` — habitat & conservation status of species X | Sprint 4 | Mock (kept for the next sprint) |

All four expose the same `run(AgentRequest) -> AgentResult` contract via the
same HTTP endpoint. Which one answers is decided by an LLM intent classifier;
the frontend never has to route.

---

## 2. Two ways to consume the agent

### Option A — HTTP (recommended for a web frontend, non-Python backends)

Run the FastAPI boundary once, then any client can talk to it. This is the
path the Global Orchestrator itself uses.

### Option B — Direct Python import (recommended when your code lives in the
same repo and same venv)

`from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator`
and call it. Skips the HTTP hop entirely. See section 6 for a snippet.

---

## 3. Starting the HTTP boundary

From the repository root, with this agent's venv active:

```bash
# Real orchestrator (LangGraph, real GBIF, real Random Forest, real M3 pipeline)
set BIODIVERSITY_AGENT_IMPL=orchestrator
python -m uvicorn backend.agents.biodiversity_agent.api:app --port 8003

# Or the deterministic mock (no external calls, safe for CI, safe for demo)
set BIODIVERSITY_AGENT_IMPL=mock
python -m uvicorn backend.agents.biodiversity_agent.api:app --port 8003
```

The default is `mock` — the port opens even when your environment has no LLM
key set. Set the env var to switch.

Launching the whole platform with `python -m backend.run_agents` sets
`BIODIVERSITY_AGENT_IMPL=orchestrator` for you (see `_AGENT_ENV` there), so
the stack serves the real orchestrator without any extra step. Exporting the
variable yourself still wins if you want the mock back for an offline demo.

**Health check** — confirm which implementation is running:

```bash
curl http://localhost:8003/health
```

Returns:

```json
{
  "agent": "Biodiversity",
  "implementation": "OrchestratorBiodiversityAgent",
  "is_orchestrator": true
}
```

---

## 4. HTTP API reference

### POST `/execute`

**Request** — `AgentRequest`:

```json
{
  "instruction": "Where do African elephants live?",
  "context": {},
  "feature": null,
  "species_name": null,
  "region": "global",
  "time_period": null,
  "include_climate_data": false,
  "session_id": null
}
```

Only `instruction` is required in practice. `feature`, `species_name` and
`region` are optional hints — if omitted, the LLM intent classifier reads
them from `instruction`. `context["features"]` accepts a list of feature
names to run several skills in parallel.

**Response** — `AgentResult`:

```json
{
  "status": "completed",
  "target_agent": null,
  "prompt_to_target_agent": null,
  "output": {
    "biodiversity_report": {
      "findings": { ... skill-specific payload ... },
      "hotspots": [ ... ],
      "migration_route": [ [lat, lon], ... ],
      "observation_count": 1234,
      "confidence": 0.85,
      "source_agents": ["Biodiversity Agent Orchestrator", "Species Distribution Agent"]
    },
    "map_url": "http://localhost:8003/maps/loxodonta_africana.html"
  },
  "map_url": "http://localhost:8003/maps/loxodonta_africana.html",
  "hotspots": null,
  "migration_route": null,
  "observation_count": 1234,
  "confidence": 0.85,
  "source_agents": ["Biodiversity Agent Orchestrator", "Species Distribution Agent"]
}
```

**Status values**: `completed` · `partial` · `needs_agent` (escalation to
another domain agent) · `failed`.

The `output.biodiversity_report` shape is the platform-wide contract every
domain agent respects. Skill-specific fields (`hotspots`, `migration_route`,
`observation_count`, `confidence`) are also mirrored at the top level for
convenience — read whichever suits your frontend.

`map_url` is an **HTTP URL served by this agent**, pointing at a self-contained
folium map: `http://localhost:8003/maps/<name>.html`. Drop it straight into an
`<iframe src=...>`.

It used to be a `file://` path, which no browser will load from a page served
over http - that is why maps never appeared in the frontend. The workers still
write files to `outputs/maps/` (the Streamlit dashboards read them off disk);
`api.py` rewrites the URL on the way out and serves the file at
`GET /maps/{name}`.

Behind a proxy or in a container the agent cannot know its own public address,
so set `BIODIVERSITY_PUBLIC_URL` (e.g. `https://api.example.com/biodiversity`)
and the rewritten URLs follow it. Only files in `outputs/maps/` are reachable.

### GET `/maps/{name}`

The rendered map named by `map_url`, as `text/html`. Only files this agent
wrote are reachable; anything else is a 404.

### GET `/health`

Cheap check of the running implementation. Use it in your frontend to warn
when the agent is serving the mock instead of the real orchestrator.

---

## 5. Frontend snippets

Both snippets below use this helper. `output` carries a plain string when the
agent could not route the question at all, and a `{skill: message}` object when
a worker was reached but failed - passing the object straight to `new Error()`
gives you `[object Object]`.

```js
function describeFailure(output) {
  if (typeof output === "string") return output;
  if (output && typeof output === "object") return Object.values(output).join(" ");
  return "Biodiversity agent failed";
}
```

### Vanilla `fetch`

```js
async function askBiodiversity(question) {
  const r = await fetch("http://localhost:8003/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction: question, context: {} }),
  });
  const result = await r.json();
  if (result.status !== "completed") {
    // `output` is a string for a routing failure but an object for a worker
    // failure (e.g. {migration_analysis: "..."}), so normalise before throwing.
    throw new Error(describeFailure(result.output));
  }
  return result;
}

// Usage
const answer = await askBiodiversity("Where do African elephants live?");
document.querySelector("#map-frame").src = answer.map_url;
document.querySelector("#count").textContent =
  `${answer.observation_count.toLocaleString()} observations`;
```

### React (hook)

```jsx
import { useState } from "react";

export function useBiodiversity() {
  const [state, setState] = useState({ loading: false, data: null, error: null });

  async function ask(instruction, context = {}) {
    setState({ loading: true, data: null, error: null });
    try {
      const r = await fetch("http://localhost:8003/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction, context }),
      });
      const data = await r.json();
      if (data.status !== "completed") throw new Error(describeFailure(data.output));
      setState({ loading: false, data, error: null });
    } catch (e) {
      setState({ loading: false, data: null, error: e.message });
    }
  }

  return { ...state, ask };
}
```

### curl (for testing)

```bash
curl -X POST http://localhost:8003/execute \
  -H "Content-Type: application/json" \
  -d '{"instruction": "biodiversity hotspots in Madagascar"}'
```

---

## 6. Direct Python import (Option B)

Skips HTTP. Useful when your code lives in the same repo and same venv.

```python
import asyncio
from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator
from backend.agents.biodiversity_agent.intent import classify_intent
from backend.agents.biodiversity_agent.schema import AgentRequest

async def ask(question: str):
    intent = await classify_intent(question)
    orch = BiodiversityOrchestrator()
    request = AgentRequest(
        instruction=question,
        feature=intent.feature,
        species_name=intent.species_name,
        region=intent.region,
        context={},
    )
    return await orch.run(request)

result = asyncio.run(ask("Where do African elephants live?"))
print(result.status, result.map_url)
```

The orchestrator is safe to instantiate once and reuse — building it compiles
the LangGraph state machine and opens the Qdrant taxonomy backend.

---

## 7. Environment setup

### Python & dependencies

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate            # Linux/Mac
pip install -r requirements.txt
```

That now covers every worker, migration included - `sentence-transformers`
and `shap` used to be a manual extra step, and skipping it made the migration
skill fail with a message that reads like a credentials problem. Note that
`sentence-transformers` pulls `torch` (~2.5GB), so the first install is slow.

The Streamlit dashboards additionally want `streamlit-folium`:

```bash
pip install streamlit-folium
```

### Environment variables

Create `.env` in **this agent's directory**
(`backend/agents/biodiversity_agent/.env`). `framework/llm_client.py`
walks up from the package and loads the first `.env` it finds, so the
agent-local file wins over one at the repository root:

```
# Azure OpenAI (intent classification + migration explanation)
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini

# Qdrant Cloud (species normalization + migration observations)
QDRANT_URL=https://<cluster>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-key>

# Which agent implementation the HTTP boundary serves
BIODIVERSITY_AGENT_IMPL=orchestrator
```

Everything gracefully degrades:

- No `AZURE_OPENAI_*` → intent classifier returns `feature=None` → the adapter
  answers with a "please rephrase" message instead of a 500.
- No `QDRANT_*` → an in-memory dict fallback resolves the five species we ship
  with (`Loxodonta africana`, `Ursus maritimus`, `Panthera tigris`,
  `Canis lupus`, `Sterna paradisaea` and their common names in FR/EN/ES).
- No `shap` installed → migration prediction still works, only the SHAP table
  disappears from the response.

Note that resolving a species name is not the same as being able to predict its
migration: the Random Forest is trained on **three** species only (`Ciconia
ciconia`, `Megaptera novaeangliae`, `Danaus plexippus`). Any other species gets
a clear "not trained on this species" answer naming those three.

---

## 8. Interactive dashboards (for testing and demo)

Two Streamlit dashboards ship with the agent, both point at the same
orchestrator. Neither is required in production — they are here so you can
sanity-check the agent without writing frontend code.

```bash
# The unified chat dashboard (all four skills, LLM routing, 4 badges)
streamlit run backend/agents/biodiversity_agent/dashboard_modules.py

# The Sprint 2 dashboard (side-by-side badges + map)
streamlit run backend/agents/biodiversity_agent/dashboard.py
```

---

## 9. Where to find things

```
backend/agents/biodiversity_agent/
├── api.py                     ← FastAPI HTTP boundary (this is what you call)
├── orchestrator_adapter.py    ← Bridges HTTP contract ↔ orchestrator
├── intent.py                  ← LLM intent classifier
├── schema.py                  ← AgentRequest / AgentResult / BiodiversityFeature
├── mock.py                    ← Deterministic fallback (safe for CI)
├── dashboard.py               ← Sprint 2 dashboard (side-by-side)
├── dashboard_modules.py       ← Sprint 3 unified chat dashboard
│
├── orchestrator/
│   ├── biodiversity_orchestrator.py   ← LangGraph state machine
│   ├── router.py                       ← Feature → worker mapping
│   ├── aggregator.py                   ← Combines parallel worker results
│   └── services/
│       ├── gbif_client.py              ← pygbif wrapper
│       ├── qdrant_client.py            ← Species taxonomy service
│       └── map_renderer.py             ← Folium map builders
│
├── workers/
│   ├── species_distribution/
│   │   ├── worker.py       ← Real GBIF worker (Aziz)
│   │   ├── mock.py         ← Deterministic fixture worker
│   │   └── schema.py       ← SpeciesDistributionOutput
│   ├── hotspots/
│   │   ├── worker.py       ← Real M3 pipeline (Ouissale)
│   │   ├── mock.py         ← Deterministic fixture worker
│   │   ├── pipeline/       ← 7-module pipeline (grid, indices, DBSCAN, ...)
│   │   ├── followup.py     ← NEEDS_CLARIFICATION helpers
│   │   └── status.py       ← M3 outcome codes
│   ├── migration/
│   │   ├── worker.py       ← Adapter around Miriam's code
│   │   ├── mock.py         ← Deterministic fixture worker
│   │   └── miriam/         ← Miriam's Random Forest + Qdrant + SHAP
│   ├── habitat/
│   │   └── mock.py         ← Sprint 4 placeholder
│   └── common/             ← Shared GBIF / geocode / config helpers
│
└── tests/                  ← 145 offline tests via pytest
```

---

## 10. Running the tests

```bash
python -m pytest backend/agents/biodiversity_agent/tests/ -q
```

All 145 tests are offline: `conftest.py` injects mock workers explicitly, so
the suite does not hit GBIF, Qdrant or any LLM. Green tests do not confirm
that Azure/Qdrant credentials work — the two Streamlit dashboards are the
live smoke test.

---

## 11. Questions?

- Species Distribution / orchestrator / dashboard integration: **Aziz** (Chouaia Mohamed Aziz)
- Biodiversity Hotspots pipeline: **Ouissale**
- Migration Analysis / Random Forest / SHAP: **Miriam Kouki**
