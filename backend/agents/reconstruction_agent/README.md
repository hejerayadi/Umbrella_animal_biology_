# Reconstruction Agent

Reconstructs unresolved regions of incomplete animal genomes from homologous
reference sequence, and explains the evidence behind every proposed base.

The scientific objective is in [description.md](description.md); the
orchestrator-facing metadata is in [card.json](card.json). This file is about
how the code is arranged and how to run it.

## What it actually does

Given a nucleotide sequence containing runs of `N`, the agent runs a genuine
agentic loop — not a fixed pipeline. It can reject its own answer and go round
again:

```
load_or_init → detect_gaps → (work?) ──no──> finalize
                                │yes
                                ▼
        ┌──────────► plan ──► select_tools ──► (runnable?) ──no──> critique
        │                            │yes                             ▲
        │                            ▼                                │
        │                   execute_tools ──► observe ──► reason ──► validate
        │                                                              │
        │                                                              ▼
        └──── REVISE ◄──────────────────────────────────────────── decide
                                     │
                  ACCEPT / ABSTAIN / budget spent / out of time
                                     ▼
                                 finalize
```

1. **Finds the gaps** and the known sequence flanking each one.
2. **Plans** which tools to run — with an LLM when configured, otherwise a
   deterministic pipeline that is the correct plan for the common case.
3. **Selects** concrete invocations, refusing any the budget cannot pay for.
4. **Acts**, then records an **observation** per attempt — including the ones
   that found nothing, because "BLAST returned no hits" is evidence about the
   gap.
5. **Reasons** a consensus out of the alignment and **validates** it.
6. **Critiques** its own answer and **decides**: ACCEPT, REVISE (re-plan with
   the objection attached), or ABSTAIN.

**ABSTAIN is a result, not a failure.** "These gaps cannot be reconstructed
from the references available" is a real scientific answer, and it comes back
as `completed` with the gaps marked unresolved.

Every reconstructed base is an inference from homologous sequence, not an
observation. The output says so, carries the accessions behind it, and reports
a confidence that falls as the evidence weakens.

## Talking to the orchestrator

The agent runs inside someone else's HTTP request, and two numbers in
`backend/orchestrator/langgraph/nodes/worker_node.py` shape everything:

| Constraint | Consequence |
| --- | --- |
| **600 s** per call (`AGENT_READ_TIMEOUT_SECONDS` in `worker_node.py`) | One EMBL-EBI BLAST job takes ~193 s, so a round of searches fits in a single call — but a many-gap scaffold still does not. |
| **3 CONTINUE retries** (`CONTINUE_RETRY_DELAYS` in `worker_node.py`) | The agent gets **four HTTP slices**; a fifth is force-failed and every finding discarded. |

So the agent **slices** its work. After 480 s (`AGENT_YIELD_AFTER_SECONDS`) it
checkpoints and returns `CONTINUE` with `retryable=true`; the orchestrator
retries, and the next slice resumes from the checkpoint. On the **last** slice
it returns partial results as `completed` rather than asking for another.

> **Keep these two numbers in step.** The read timeout was once 120 s and the
> yield was sized for it. When the timeout was raised to 600 s for another
> agent, nothing updated the yield: it stayed at 75 s, which is less than half
> one BLAST job, so no run could complete a single search inside its whole
> four-slice allowance and **every reconstruction came back empty**. Nothing
> failed and no test caught it, because every test mocks the network edge and
> so returns from BLAST instantly. See [WHY_NO_OUTPUT.md](WHY_NO_OUTPUT.md).

The checkpoint is keyed on **`X-Trace-Id`** — the orchestrator's run-wide id
(`orchestrator/state.py:35`), which is the only identifier stable across
retries. `X-Request-Id` is unique per attempt and would never resume.

`output` is namespaced under `reconstruction`, because the orchestrator merges
it flat into the context all nine agents share; bare keys like `summary` would
collide.

When phylogeny is the blocker the agent returns `NEEDS_AGENT` for the Evolution
Agent, carrying its findings so far — the orchestrator force-fails an
escalation that adds no new context keys.

## Budgets

Guard rails, not cost control: an unbounded loop gets the whole run
force-failed.

| Budget | Default | Setting |
| --- | --- | --- |
| Tool calls | 24 | `RECONSTRUCTION_MAX_TOOL_CALLS` |
| BLAST / MAFFT calls | 8 / 8 | `RECONSTRUCTION_MAX_BLAST_CALLS`, `..._MAFFT_CALLS` |
| LLM tokens | 60 000 | `RECONSTRUCTION_MAX_LLM_TOKENS` |
| Wall clock per slice | 480 s | `AGENT_YIELD_AFTER_SECONDS` |

The call counts bound **how many gaps** a run works on, not how long it takes:
`execute_tools` dispatches a round with `asyncio.gather`, so eight concurrent
BLAST searches cost about what one costs. The wall clock is the real limit.

Consumption is reported in the result (`budget`, `slices`, `stop_reason`,
`observations`), so a partial answer explains itself rather than looking
truncated. A region left unresolved says **which** kind of unresolved it is —
"no usable reference evidence" is a finding about the sequence, while "the run
ran out of time before the search returned" is a fact about the run and invites
another attempt. Collapsing the two is what let a starved run read as a
scientific abstention.

## Persistence

The agent shares **one database and one schema** with the rest of Umbrella.
The URL is declared **once, in `backend/.env` as `DATABASE_URL`** — this agent
reads that file directly, so there is no second copy to drift when the password
or port changes. Its tables are told apart by name, not by namespace.

Reading a config file is not importing a package: the agent still has its own
`.venv` and never imports `backend`. Set `RECONSTRUCTION_DATABASE_URL` only to
point it somewhere else (a container, a CI job); it wins when present.

- **Checkpoints** — `AsyncPostgresSaver` (`checkpoints`, `checkpoint_writes`,
  `checkpoint_blobs`, `checkpoint_migrations`), created and managed by
  LangGraph itself. Without a database URL the agent falls back to in-memory
  and **CONTINUE cannot resume**; `/api/v1/health` reports which you have.
- **Audit** — `reconstruction_runs`, one row per logical run (keyed by trace
  id) recording slices, budgets and why it stopped. Owned by this agent's
  Alembic history.

```bash
uv run alembic upgrade head
```

**The agent keeps its own Alembic version table**, `alembic_version_reconstruction`.
The backend runs a separate history in the default `alembic_version`; sharing
one row would make each treat the other's revision as unknown and try to repair
it. Autogenerate is scoped to an allow-list drawn from this agent's own
metadata, so it never proposes touching `users`, `invitations` or anything else
the backend adds later.

## Layout

```
reconstruction_agent/
├── pyproject.toml / uv.lock     Dependencies, pinned.
├── api.py                       Launcher shim (see below).
├── card.json                    Read by backend/registry.py. Do not move.
└── src/
    ├── api/                     HTTP boundary. No business logic.
    │   └── v1/                  Versioned surface + {data, meta, error}.
    ├── agent/                   The agentic loop.
    │   ├── graph/               LangGraph nodes, edges, conditions, builder.
    │   ├── planning/            What to do next, and when to stop.
    │   ├── reasoning/           Building reconstructions, and criticising them.
    │   ├── state/               The state object and its merge reducers.
    │   └── prompts/             Prompt *loading*. The text is in src/prompts/.
    ├── prompts/                 Every LLM prompt, as markdown. See its README.
    ├── domain/                  The science. No I/O, no framework.
    │   ├── models/              Sequence, Gap, Reference, Alignment, Candidate.
    │   ├── services/            Gap detection, ranking, consensus, validation.
    │   └── policies/            Confidence scoring and admission rules.
    ├── tools/                   Capabilities the planner can select.
    │   ├── ncbi/ blast/ mafft/  Reference retrieval, homology, alignment.
    │   └── evo/                 Phylogeny heuristic + Evo 2 plausibility.
    ├── infrastructure/          Everything that talks to the outside world.
    │   ├── http/ ncbi/ embl_ebi/
    │   ├── llm/                 Azure OpenAI / Foundry, and the null client.
    │   ├── nvidia/              Evo 2 via NVIDIA NIM.
    │   └── persistence/
    ├── application/             Use cases, expressed without reference to HTTP.
    ├── contracts/               What the agent accepts, returns, and reports.
    ├── configuration/           Settings and structlog setup.
    └── observability/           Events, metrics, correlation ids, tracing.
```

The dependency rule is one-way: `domain` depends on nothing, `tools` and
`application` depend on `domain`, and `api` depends on `application`. Nothing
depends on `api`. That is what keeps the science testable without a network.

`src/` is a flat package root — the importable names are `api`, `domain`,
`agent`, … with no wrapping package. `pythonpath = ["src"]` in `pyproject.toml`
is what makes that work for the tests.

## The two HTTP surfaces

| | `POST /execute` | `POST /api/v1/reconstructions` |
| --- | --- | --- |
| Caller | The orchestrator | Anything else |
| Body | `{instruction, context}` | Typed, self-documenting |
| Response | `{status, target_agent, prompt_to_target_agent, output}` | `{data, meta, error}` |

**`/execute` is deliberately not enveloped.** `backend/orchestrator/schema.py`
parses that exact shape, and wrapping it would break every reconstruction the
system performs. Both endpoints return HTTP 200 with the failure described in
the body, because a client that has to branch on both a status code and an
error field has two things to get wrong instead of one.

`GET /api/v1/health` reports capability, not just liveness — which external
services are actually configured. `GET /api/v1/health/live` is the cheap
container probe.

## Running it

The agent has its own `.venv` and pins its own dependencies, isolated from the
rest of the backend.

```bash
cd backend/agents/reconstruction_agent
cp .env.example .env          # then fill in the credentials
uv sync --group dev           # creates .venv and installs from uv.lock
```

Then either start this agent alone:

```bash
uv run reconstruction-agent   # binds APP_HOST:APP_PORT from .env
```

(Not `python -m api`: the launcher shim `api.py` at the agent root shadows the
installed `api` package when the agent root is the working directory. The
console script is resolved from the venv, so it is unambiguous.)

or start the whole system from the repository root, which is the normal path:

```bash
python -m backend.run_agents --setup   # once, to build every agent's venv
python -m backend.run_agents
```

`backend/run_agents.py` launches this agent as
`backend.agents.reconstruction_agent.api:app`. That module is a shim
([api.py](api.py)) that re-exports the real app from `src/`; the launcher
convention is shared with eight other agents, so the shim is cheaper than
special-casing it.

### Configuration

`.env.example` documents every setting. The ones that decide what works:

| Setting | Effect if unset |
| --- | --- |
| `EMBL_EBI_CONTACT_EMAIL` | **BLAST and MAFFT refuse to run**, so every gap comes back unresolved. EMBL-EBI rejects anonymous job submissions. |
| `AZURE_OPENAI_*` | Planning falls back to the deterministic pipeline. Not a degraded stub — it is the correct plan for the common case, and it is what the tests exercise. |
| `NVIDIA_API_KEY` | The Evo 2 plausibility tool is not registered at all. An unusable tool in the catalogue is worse than an absent one. |
| `NCBI_API_KEY` | Optional; raises the rate budget from 3/s to 10/s. |
| `DATABASE_URL` (in `backend/.env`) | Checkpoints fall back to in-memory, so **CONTINUE cannot resume** and a slow run fails after three retries. |

`APP_ENV` drives the defaults that should differ between a laptop and a
server: `LOG_FORMAT` (pretty vs JSON) and whether `/docs` is exposed.

### Logging

structlog, rendered by rich in development and as JSON lines elsewhere. Every
record carries the correlation id from `X-Request-ID`, so one request — and the
whole reconstruction it triggers — can be followed across the graph, the tools
and the HTTP clients.

Log with an event name and key/value pairs, not an f-string:

```python
log.info("gap_detected", gap_id=gap.identifier, length=gap.length)
```

That is what makes the field queryable once the logs are shipped.

## Testing

```bash
uv run pytest                         # everything offline, with coverage
uv run pytest --no-cov -q             # faster, no coverage
uv run pytest -m "not evaluation"     # skip the slower accuracy suite
RUN_EXTERNAL_TESTS=1 uv run pytest    # also run tests that hit real services
```

- `tests/unit` — domain logic, settings, and prompt loading.
- `tests/tools` — parsing real payload shapes from saved fixtures.
- `tests/agent` — consensus, reasoning, and the stop policy.
- `tests/integration` — both HTTP contracts.
- `tests/evaluation` — punches a hole in a known sequence and checks the agent
  puts back what was there. Accuracy is the measure that matters, and it cannot
  be read off unit tests of the parts.

Two scripts complement the suite:

```bash
uv run python scripts/smoke_test.py              # end to end, no network
uv run python scripts/test_external_services.py  # NCBI/BLAST/MAFFT/Azure/Evo2
uv run python scripts/tune_settings.py --offline # measure the confidence threshold
```

The second is how you diagnose a deployment that starts cleanly but fails
every reconstruction — a wrong Azure deployment name surfaces as a 404 that
reads like a missing model.

### Tuning

`scripts/tune_settings.py` is where the settings come from. It punches holes in
360 synthetic sequences spanning gap length, reference count, reference
agreement, flank availability and divergence, reconstructs each one, and
compares against the known answer.

It reports precision and recall by threshold, and recommends the lowest
threshold that accepts **no** incorrect reconstruction — not the F1 optimum. F1
treats a wrong base and a missing base as equally bad; in an assembly they are
not.

`--offline` runs only that sweep and costs nothing. Without the flag it also
sweeps `LLM_TEMPERATURE` against Azure, measuring whether the planner's JSON
parses, whether it names real tools, and whether the same question gives the
same plan twice. Temperature cannot affect reconstruction accuracy — the bases
come from alignment consensus, never from the model — so reproducibility is the
only thing being optimised there.

Current settings and what produced them:

| Setting | Value | Evidence |
| --- | --- | --- |
| `RECONSTRUCTION_MIN_CONFIDENCE` | 0.15 | Lowest threshold with precision 1.000 over the 360-scenario grid (recall 0.93). Was 0.65 against the old score, which needed a high threshold to compensate for ranking correct above incorrect at only 0.575. |
| `LLM_TEMPERATURE` | 0.0 | 100% parse rate at every temperature tried; 0.0–0.3 gave 0.92 plan repeatability vs 0.83 at 0.7–1.0. |

## Prompts

No prompt text is hardcoded. Everything the agent sends to an LLM is a
markdown file in [src/prompts/](src/prompts/) — see
[its README](src/prompts/README.md) for the naming and the one editing rule
(`.user.md` files go through `str.format`, so literal braces must be doubled).

## The Evolution Agent

`card.json` lists **Evolution Agent** under `may_need`. Ranking references by
phylogenetic proximity is a real part of this agent's job, and the local
heuristic in `tools/evo/` is genus-level only — it cannot tell that *Loxodonta*
and *Mammuthus* are close relatives. When it cannot separate candidates it says
so, and the question is better routed to the Evolution Agent.
