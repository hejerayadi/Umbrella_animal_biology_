# Reconstruction Agent

Reconstructs unresolved regions of incomplete animal genomes from homologous
reference sequence, and explains the evidence behind every proposed base.

The scientific objective is in [description.md](description.md); the
orchestrator-facing metadata is in [card.json](card.json). This file is about
how the code is arranged and how to run it.

## What it actually does

Given a nucleotide sequence containing runs of `N`, the agent:

1. **Finds the gaps** and the known sequence flanking each one.
2. **Decides which are worth attempting** — a gap with no usable flank, or one
   longer than the configured cap, is reported as skipped rather than guessed at.
3. **Searches for homologous references** (BLAST over the flanks, or NCBI
   directly when the useful relatives are already known).
4. **Aligns** those references against the flanks with MAFFT, and locates the
   alignment columns that span the gap.
5. **Reads a consensus** out of those columns, scores it, and validates it.
6. **Critiques its own answer** and loops back for more evidence when the
   support is thin.

Every reconstructed base is an inference from homologous sequence, not an
observation. The output says so, carries the accessions behind it, and reports
a confidence that falls as the evidence weakens.

## Layout

```
src/reconstruction_agent/
├── api/              HTTP boundary. POST /execute, GET /health. No logic.
├── agent/            The agentic loop.
│   ├── graph/        LangGraph nodes, edges, conditions, builder.
│   ├── planning/     What to do next, and when to stop.
│   ├── reasoning/    Building reconstructions, and criticising them.
│   ├── state/        The state object and its merge reducers.
│   └── prompts/      LLM prompts, as reviewable artefacts.
├── domain/           The science. No I/O, no framework, fully unit-testable.
│   ├── models/       Sequence, Gap, Reference, Alignment, Candidate.
│   ├── services/     Gap detection, ranking, consensus, validation.
│   └── policies/     Confidence scoring and admission rules.
├── tools/            Capabilities the planner can select, behind one contract.
│   ├── ncbi/         Reference retrieval.
│   ├── blast/        Homology search.
│   ├── mafft/        Multiple alignment.
│   └── evo/          Phylogenetic context for ranking.
├── infrastructure/   Everything that talks to the outside world.
├── application/      Use cases, expressed without reference to HTTP.
├── contracts/        What the agent accepts, returns, and reports.
├── configuration/    Settings and logging. The only reader of the environment.
└── observability/    Events, metrics, correlation ids, LangSmith tracing.
```

The dependency rule is one-way: `domain` depends on nothing, `tools` and
`application` depend on `domain`, and `api` depends on `application`. Nothing
depends on `api`. That is what keeps the science testable without a network.

## Running it

The agent has its own `.venv` and pins its own dependencies, isolated from the
rest of the backend.

```bash
cd backend/agents/reconstruction_agent
cp .env.example .env          # then fill in the contact addresses
uv sync                       # creates .venv and installs from uv.lock
```

Then either start this agent alone:

```bash
uv run uvicorn reconstruction_agent.api.app:app --port 8006
```

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

### Configuration that matters

Everything is optional except one thing:

- **`EMBL_EBI_CONTACT_EMAIL`** — EMBL-EBI rejects anonymous job submissions.
  Without it the agent still starts and still detects gaps, but BLAST and MAFFT
  refuse to run and every gap comes back unresolved. `/health` reports this.
- `NCBI_API_KEY` — optional; raises the rate budget from 3/s to 10/s.
- `LLM_PROVIDER` — defaults to `none`, which runs the deterministic planner.
  That path is not a degraded stub; it is the correct plan for the common case
  and it is what the tests exercise. An LLM adds adaptive planning and critique.

## Testing

```bash
uv run pytest                         # everything offline, fast
uv run pytest -m "not evaluation"     # skip the slower accuracy suite
RUN_EXTERNAL_TESTS=1 uv run pytest    # also run tests that hit real services
```

- `tests/unit` — domain logic in isolation.
- `tests/tools` — parsing real payload shapes from saved fixtures.
- `tests/agent` — consensus, reasoning, and the stop policy.
- `tests/integration` — the `/execute` contract.
- `tests/evaluation` — punches a hole in a known sequence and checks the agent
  puts back what was there. Accuracy is the measure that matters, and it cannot
  be read off unit tests of the parts.

Two scripts complement the suite:

```bash
uv run python scripts/smoke_test.py             # runs end to end, no network
uv run python scripts/test_external_services.py # checks NCBI/BLAST/MAFFT are reachable
```

## The orchestrator contract

`POST /execute` takes `{instruction, context}` and **always** answers with
`{status, target_agent, prompt_to_target_agent, output}` — including on
failure. The orchestrator's router expects one schema back every time and
already handles a `failed` status; a 500 with FastAPI's `{"detail": ...}` body
would break it.

`context` is read permissively. Useful keys:

| Key | Meaning |
| --- | --- |
| `sequence` | `{identifier, residues, organism}`, or a bare residue string |
| `accession` | An NCBI accession to fetch the target from instead |
| `organism` | Scientific name of the target |
| `reference_organisms` | Restrict the reference search to these |
| `max_gap_length`, `min_confidence` | Per-request overrides |

Unknown keys are kept and ignored — the orchestrator's context is a shared
scratchpad other agents also write to.

`card.json` lists **Evolution Agent** under `may_need`. Ranking references by
phylogenetic proximity is a real part of this agent's job, and the local
heuristic in `tools/evo/` is genus-level only — it cannot tell that *Loxodonta*
and *Mammuthus* are close relatives. When it cannot separate candidates it says
so, and the question is better routed to the Evolution Agent.
