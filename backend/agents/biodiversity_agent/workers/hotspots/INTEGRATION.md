# M3 — how to plug it in

Everything in this folder is **new**. No file that already existed in the project
was edited: `schema.py`, `api.py`, `mock.py`, `card.json`, `requirements.txt`,
`orchestrator/` and every other agent are byte-for-byte as they were.

That means M3 is fully working and fully tested, but **nothing calls it yet** —
the sub-orchestrator still builds `HotspotsMock` by default, exactly as before.
Below is what to run today, and the two optional edits that hand M3 the wheel
when the team is ready.

M3 needs no database, no vector store, no model file and no LLM. It needs the
GBIF API, and nothing else — no API key, no `.env`, no server.

---

## Works right now, with zero edits

```powershell
$py = "backend\agents\biodiversity_agent\.venv\Scripts\python.exe"
```

**The tests** — all offline, no API keys, no network:

```powershell
& $py -m pytest backend\agents\biodiversity_agent\tests -v
```

**The dashboard** — a live GBIF query, a real map, ranked hotspots:

```powershell
& $py -m streamlit run backend\agents\biodiversity_agent\dashboard_modules.py --server.address 127.0.0.1
```

**The worker, from Python** — the same object the orchestrator would call:

```python
from backend.agents.biodiversity_agent.schema import AgentRequest
from backend.agents.biodiversity_agent.workers.hotspots.worker import HotspotsWorker

result = HotspotsWorker().run(AgentRequest(
    instruction="Where are the biodiversity hotspots in the Congo Basin?",
    context={}))
```

**Through the sub-orchestrator**, without touching it — inject the worker, which
its constructor already supports:

```python
from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator
from backend.agents.biodiversity_agent.schema import BiodiversityFeature
from backend.agents.biodiversity_agent.workers.hotspots.worker import HotspotsWorker
from backend.agents.biodiversity_agent.workers.habitat.mock import HabitatMock
from backend.agents.biodiversity_agent.workers.migration.mock import MigrationMock
from backend.agents.biodiversity_agent.workers.species_distribution.mock import (
    SpeciesDistributionMock,
)

orchestrator = BiodiversityOrchestrator(workers={
    BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsWorker(),   # the real one
    BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock(),
    BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
    BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
})
```

`tests/test_m3_integration.py` does exactly this, so the chain is already proven
end to end: Sub-Orchestrator → M3 → Result → Sub-Orchestrator.

---

## The project's two interfaces

The project ships two, and M3 now reaches both.

```
                 Streamlit dashboard                React frontend (Vite)
                 dashboard_modules.py                frontend/src/routes/chat.tsx
                         |                                     |
                         |                            POST /api/chat  (port 8000)
                         |                                     |
                         |                            backend/api.py -> GlobalOrchestrator
                         |                                     |
                         |                            POST /execute   (port 8003)
                         |                                     |
                         +---------> HotspotsWorker <---- workers/hotspots/api_m3.py
                                            |
                                     GBIF -> grid -> DBSCAN -> folium map
```

**The dashboard** (`dashboard_modules.py`) calls the worker in-process. It is the
one that shows the map inline, the live step timeline and the follow-up question.

**The backend chain** needs an HTTP boundary, and the agent's own `api.py` builds
`BiodiversityMock` - a file we may not edit. So [`api_m3.py`](api_m3.py) is a
second app with the same contract, serving the real sub-orchestrator with M3
injected:

```powershell
$py = "backend\agents\biodiversity_agent\.venv\Scripts\python.exe"
& $py -m uvicorn backend.agents.biodiversity_agent.workers.hotspots.api_m3:app --port 8003
```

Port 8003 is the Biodiversity agent's own port, so the orchestrator finds it with
no configuration at all - just start this instead of `api.py`. To run both, put
this one on a free port and redirect the orchestrator through the environment
variable `backend/registry.py` already reads:

```powershell
& $py -m uvicorn backend.agents.biodiversity_agent.workers.hotspots.api_m3:app --port 8013
$env:BIODIVERSITY_AGENT_URL = "http://localhost:8013"
$env:M3_PUBLIC_URL = "http://localhost:8013"
```

### Starting it without losing the terminal

`uvicorn` in the foreground owns the window it was launched from, so testing it
from the same prompt races the server - or kills it. [`serve.ps1`](serve.ps1)
starts it detached, frees the port first (no more `[Errno 10048]`), waits until
`/health` actually answers, and prints the URLs:

```powershell
.\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1            # port 8003
.\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1 -Port 8013
.\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1 -Stop
```

It uses the agent's own venv, and says so - a bare `python` on PATH may be an
interpreter the dependencies were never installed for.

In PowerShell, `curl` is an alias for `Invoke-WebRequest`, which takes `-Headers`
as a dictionary: `curl -H "Content-Type: ..."` fails with a parameter-binding
error. Use `Invoke-RestMethod`, or `curl.exe` for the real thing.

Verified against a running instance:

| Request the worker node sends | Answer |
|---|---|
| `{"instruction": "Where are the biodiversity hotspots in Madagascar?"}` | `completed`, 3 ranked hotspots, `map_url` an absolute URL |
| `{"instruction": "Which part of Africa has the most species?"}` | asks, with the six options |
| `{"instruction": "Compare ... Amazon and the Congo Basin"}` | asks which one, options `[amazon, congo basin]` |
| `{"instruction": "Where do African elephants live?", "species_name": ...}` | routed to the Species Distribution mock, untouched |
| `GET /map/hotspots_madagascar.html` | 200, 17,641 bytes; `..%2f..%2f` traversal 404s |

Two things this boundary has to do that the mock does not:

1. **Infer the feature.** The Global Orchestrator's worker node posts only
   `{instruction, context}`; the sub-orchestrator refuses to guess a feature, and
   the legacy mock defaults to species distribution and answers *"Species not
   specified."* - so without this, a hotspots question cannot reach M3 through the
   project's own chain at all.
2. **Publish the map.** `render_folium` writes a file path, which is useless to a
   browser on another origin. `GET /map/{name}` serves it and `map_url` becomes an
   absolute URL.

### The React frontend, with the map

`backend/api.py` returns `answer`, `execution_history` and `context`; `chat.tsx`
rendered only the first two, so a map an agent produced was unreachable. Three
edits in the working copy fix that - they are the only changes outside this
folder, and they are listed here so the delivery copy can stay byte-for-byte:

**`src/lib/umbrella-types.ts`** - `Message` gains `mapUrl?: string`.

**`src/lib/umbrella-store.tsx`** - a `readMapUrl(context)` helper, and
`mapUrl: readMapUrl(response.context)` on the assistant message. The helper looks
one level down as well as at the top level, because the orchestrator nests each
agent's payload under its own key when several ran.

**`src/components/umbrella/chat-message.tsx`** - after the markdown:

```tsx
{message.mapUrl && done && (
  <iframe
    src={message.mapUrl}
    title="Map"
    loading="lazy"
    sandbox="allow-scripts"
    className="mt-3 h-[420px] w-full rounded-lg border border-border bg-card"
  />
)}
```

`done` gates it so the iframe does not load while the answer is still typing.
`sandbox="allow-scripts"` is what folium needs to draw, and withholds same-origin
access. `tsc --noEmit` passes.

The map URL is absolute because `api_m3.py` rewrites it - a bare file path would
be useless to a browser. Nothing on the Python side needed changing.

### What is still not wired



The gateway still needs Azure OpenAI credentials in `backend/orchestrator/.env`
before any question reaches an agent: with the placeholder values it answers
`openai.APIConnectionError`. That file is not ours to fill in.

## The two optional edits

Apply these when the team wants M3 live by default instead of injected.

### 1. Register the worker in the sub-orchestrator

`orchestrator/biodiversity_orchestrator.py`, in `__init__`, replace the hotspots
entry of the default worker map:

```python
# from
BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsMock(),

# to  (import lazily: it pulls in scikit-learn)
BiodiversityFeature.BIODIVERSITY_HOTSPOTS: _real_hotspots_worker(),
```

```python
def _real_hotspots_worker():
    from ..workers.hotspots.worker import HotspotsWorker
    return HotspotsWorker()
```

Keep the mock for the existing Sprint 2 tests — `tests/conftest.py` builds the
orchestrator with no arguments, and `test_orchestrator_sequential.py` asks for
`region="global"`, which the real M3 correctly refuses as too large to grid. Gate
it on an environment variable if both must coexist:

```python
import os
use_real = os.environ.get("BIODIVERSITY_REAL_MODELS") == "1"
```

### 2. Serve it over HTTP

`api.py` currently instantiates `BiodiversityMock`. To expose M3 on `POST
/execute`, route the hotspots feature through the sub-orchestrator instead of the
legacy mock, or add a branch:

```python
if request.feature == BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value:
    from .workers.hotspots.worker import HotspotsWorker
    return HotspotsWorker().run(request)
```

Nothing else about the HTTP contract changes: M3 already returns one of the four
platform statuses. See *Statuses* below.

### Optionally, `card.json`

The existing card describes the Sprint 2 mock ("Clusters GBIF occurrences with
DBSCAN…"), which is still broadly true, so this is cosmetic. If the planner should
route on what the module now does, the capability list is:

```json
"capabilities": [
  "Read a free-text question into study area and analysis parameters",
  "Estimate the record volume of a region before retrieving it",
  "Bin coordinates onto an equal-area grid",
  "Compute species richness, Shannon, Simpson and Chao1 per cell",
  "Correct richness for sampling effort and report how",
  "Discover hotspot clusters with DBSCAN using a haversine metric",
  "Tune eps and min_samples by an explicit silhouette scan",
  "Rank the clusters and render a richness heatmap"
]
```

---

## Statuses — why `schema.py` did not need touching

`Biodiversity Hotspots.pdf` Table 14 gives M3 six outcomes; the platform's
`AgentStatus` has four, and the Global Orchestrator parses every reply into its
own copy of that enum. Adding members would send it a string it cannot parse.

So M3 keeps its own vocabulary in [`status.py`](status.py) and narrows at the
boundary:

| M3 outcome | `AgentResult.status` | Why |
|---|---|---|
| COMPLETED | `completed` | — |
| PARTIAL | `completed` | a grid and a map were produced; what is missing is in `warnings` |
| NEEDS_CLARIFICATION | `failed` | M3 cannot answer as asked; the reply explains and lists the known regions |
| FAILED | `failed` | — |

Nothing is lost: the original outcome travels in the payload as `status_detail`,
and `outcome_of(result)` recovers it. The dashboard badges on that, which is why
it still shows PARTIAL and NEEDS CLARIFICATION.

---

## What was added

```
workers/hotspots/
├── worker.py             the module: the 8 documented pipeline steps
├── api_m3.py             POST /execute + GET /map, for the backend chain
├── serve.ps1             starts api_m3 detached, on a freed port
├── status.py             M3's outcomes, narrowed to the platform's four
├── schema.py             the doc §7.2 dataclasses
├── description.md        purpose, workflow, method, limitations
├── requirements-m3.txt   the extra dependencies
├── INTEGRATION.md        this file
└── pipeline/             cleaning · grid · indices · effort · clustering · ranking · render

workers/common/           config.py · gbif.py            (shared by future workers)
tests/                    test_m3_pipeline.py · test_m3_integration.py
dashboard_modules.py      the M3 chat dashboard
```

Untouched: `api.py`, `schema.py`, `mock.py`, `card.json`, `requirements.txt`,
`dashboard.py`, `description.md`, `docs/{workflow,qdrant_setup,framework_benchmark_results}.md`,
`orchestrator/**`, `framework/**`, `workers/{habitat,migration,species_distribution}/**`,
the other eight agents, and the frontend.
