# Reconstruction Agent

Fills unresolved regions (`N...N`) in incomplete animal genome assemblies, from
homology evidence, or refuses and says why. Umbrella slot **8006**.

The agent never generates DNA. Every base it returns came from a real sequenced
organism, arrived through an alignment that demonstrably spans the gap, and
carries a confidence computed from measurable evidence by deterministic code.

- [What it does](#what-it-does)
- [The two contracts](#the-two-contracts)
- [How the search is scoped](#how-the-search-is-scoped)
- [What it refuses to do](#what-it-refuses-to-do)
- [Running and testing](#running-and-testing)
- [Configuration](#configuration)
- [Layout](#layout)
- [Known limits](#known-limits)

## What it does

```
accession ─► NCBI record ─► gap detection ─► triage
                  │
                  ▼
           NCBI Taxonomy ─► TargetProfile
                                  │
                       taxonomic scopes (family, order, class)
                                  │
                   concurrent NCBI BLAST round, target excluded
                                  │
                        fetch homologues ─► MAFFT (EMBL-EBI)
                                  │
                          gap-column analysis
                                  │
                      competing candidates ─► scoring
                                  │
                        RESOLVED  or  UNRESOLVED + reason
```

Measured end to end on a withheld 45-base region of the polar bear mitogenome
`NC_003428.1`: all 45 bases recovered exactly, confidence 0.96, in 78 s.

Two services are used, for two different jobs. **NCBI** provides the sequences,
the taxonomy and the homology search. **EMBL-EBI** provides MAFFT and nothing
else - NCBI publishes no alignment service, and no MAFFT binary is installed
anywhere in this repository.

## The two contracts

**`POST /execute`** - the Umbrella orchestrator's. Root-mounted, returns the bare
repo-wide `AgentResult`, never the envelope, and **always HTTP 200** even on
failure. The orchestrator parses one schema and turns any non-200 into "agent
unreachable", which would hide a well-described error behind a transport one.

```jsonc
// in
{"instruction": "...", "context": {"sequence_accession": "NC_003428.1"}}
// out
{"status": "completed", "output": {
  "reconstruction": { /* full detail */ },
  "reconstruction_summary": "Reconstructed 1 of 1 unresolved regions in Ursus maritimus.",
  "reconstruction_best_fill": {
    "sequence": "AAGCTATCGGG...", "length_bp": 11,
    "gap_id": "gap-1", "start": 337487, "end": 337491,
    "confidence": 0.82, "is_model_generated": false
  }
}}
```

Those three `output` keys are declared in `card.json` and read by other agents
from the shared context. Renaming one breaks them silently.

`reconstruction_best_fill` is one gap's fill, not the repaired record - it
carries `start`/`end`/`length_bp` so that cannot be misread. It was called
`reconstruction_sequence` and was a bare string, which could and did read as
the whole reconstructed sequence when it was seven bases of a 1.15 Mb
scaffold.

**`/api/v1/...`** - everyone else. `{data, meta, error}` on every response, and
HTTP status codes used properly (404 not found, 422 bad domain values, 502
upstream broke, 503 dependency down, 504 deadline). Currently `GET
/api/v1/health` and `GET /api/v1/ready`.

`/ready` answers 503 when a *required* dependency is missing. Without
`EMBL_EBI_CONTACT_EMAIL` the agent starts and detects gaps but resolves none,
because EBI rejects anonymous job submissions - reporting that as healthy would
hide a total loss of function behind a green check.

## How the search is scoped

No database code and no clade name appears anywhere in this repository. The
scope of a search is derived at runtime from the target lineage.

NCBI accepts a taxonomic restriction directly - `ENTREZ_QUERY=txid9632[ORGN]` -
so the clade searched is *stated*, using the tax id the agent already holds
from the lineage it already fetched. Nothing is matched against a collection
label, and the failure this agent was rebuilt around cannot occur: searching a
collection that excludes the target clade is not expressible.

What remains is choosing how narrow to be. Family returns close relatives and
little else; class returns far more sequence, most of it too distant to fill a
gap accurately. The configured ranks (`HOMOLOGY_NCBI_SCOPE_RANKS`) are searched
concurrently, narrowest first, and the scope with the most gap-spanning hits
wins. For *Ursus maritimus* that is Ursidae, then Carnivora, then Mammalia.

**The record being repaired is excluded from its own search** (`NOT
NC_003428[ACCN]`). It cannot be evidence about its own unresolved region, and
in a ground-truth measurement - where bases are withheld from the agent while
the public record still holds them - leaving it in turns the exercise into a
lookup of the answer.

Two wire details are load-bearing and both were measured, not assumed:

- **`blastn`, not megablast.** megablast is a local aligner: on a 1 kb flank
  query it returned twenty hits at identity 1.000 with *none* spanning the gap,
  because it split at the 45-base indel into two half-coverage HSPs. Gapped
  blastn bridges the indel in one HSP, which is the alignment a reconstruction
  needs. `PROGRAM=megablast` is also rejected outright - it is `blastn` with a
  flag.
- **No API key.** NCBI issues keys for E-utilities only; the BLAST URL API
  neither accepts nor is rate-limited by one. What it asks for instead is a
  `tool` and `email` on every call, one request per ten seconds, and one poll
  per minute per search. All three are honoured, and the cost is that a fast
  search is noticed late.

## What it refuses to do

Refusal is a first-class result, returned as `status: completed` with the gap
`UNRESOLVED`, its original coordinates intact, and a stated reason:

| reason | meaning |
|---|---|
| `NO_HOMOLOGS_FOUND` | nothing searchable returned a homologue |
| `INSUFFICIENT_GAP_SPANNING_HOMOLOGS` | homologues exist, none covers both flanks |
| `CONFIDENCE_BELOW_THRESHOLD` | the best candidate scored under the floor |
| `BIOLOGICAL_VALIDATION_FAILED` | the fill failed a deterministic check |
| `DEADLINE_EXCEEDED` / `NOT_ATTEMPTED` | the run committed to what it could finish |

A homologue matching one flank beautifully proves the flank is conserved and
says nothing about the bases between. It is never counted as support.

## Running and testing

```powershell
cd backend\agents\reconstruction_agent
uv sync
uv run ruff check src tests ; uv run ruff format --check src tests
uv run mypy src
uv run pytest tests -q                       # offline, ~1.5 s
```

With the whole platform, from the repository root:

```powershell
python -m backend.run_agents                 # all nine agents
curl http://localhost:8006/api/v1/health
```

**The test that matters.** Everything in `tests/` mocks the network edge, which
proves the code runs but cannot distinguish a working evidence path from a
plausible-looking one. Only ground truth can - a known region is withheld and
the answer compared against what was actually there:

```powershell
uv run python scripts\ground_truth.py                  # the whole battery
uv run python scripts\ground_truth.py --only "polar"   # one case
$env:RUN_EXTERNAL_TESTS=1; uv run pytest tests -q      # same case, as a test
```

It costs a NCBI BLAST search and an EMBL-EBI alignment per case - minutes of
queue time on somebody else's servers - which is why it is marked `external`
and skipped by default.

## Configuration

Copy `.env.example` to `.env`. Everything is documented there; the values that
decide whether the agent works at all:

| variable | why |
|---|---|
| `NCBI_BLAST_CONTACT_EMAIL` | NCBI asks to be able to contact whoever is submitting searches |
| `NCBI_API_KEY` | E-utilities only. Lifts the sequence and taxonomy rate from 3/s to 10/s; the agent paces at 6/s under it, because sitting at the ceiling was measured earning a silent block |
| `EMBL_EBI_CONTACT_EMAIL` | **required for alignment** - EBI rejects anonymous jobs, so without it every gap that reaches MAFFT is unresolved |
| `RECONSTRUCTION_DEADLINE_SECONDS` | 300 by default, well inside the orchestrator's 600 s |
| `NVIDIA_API_KEY` | optional - without it Evo 2 arbitration is simply not registered |

No database code and no clade is configurable, because none is named.

## Layout

```
api/            HTTP only. v1/ is enveloped; orchestrator.py is not.
agent/          Planner, reasoner, critic - see Known limits.
domain/         Frozen models, enums, exceptions. Imports no integration.
services/       The biology: taxonomy, homology, alignment, candidate, scoring, validation.
integrations/   NCBI E-utilities, NCBI BLAST URL API, EMBL-EBI MAFFT, Evo 2, HTTP.
orchestration/  Budget ledger and run deadline.
observability/  Structured logging.
config/         Settings - the only place that reads the environment.
```

`domain/` never imports `integrations/` or `services/`. Provider vocabulary
lives in `integrations/blast/` and `services/homology/` and nowhere else -
`TargetProfile` carries `scientific_name`, `tax_id`, `taxonomy_lineage`,
`taxonomy_ranks`, `molecule_type`, and nothing about any provider.

## Known limits

Stated plainly, because a reader should not have to discover these by reading
the source.

- **The loop is a pipeline, not yet an agent.** `agent/` is scaffolded but the
  run currently executes a fixed sequence. There is no LLM planning, no critic
  verdict, and no replanning: a weak result is reported, not retried with a
  different strategy. The evidence feedback the selector needs for a replan is
  already plumbed and unused.
- **Evo 2 is not wired.** `CandidateScores.evo2` is always `None`, the engine
  redistributes its weight correctly, and nothing calls NVIDIA. NIM exposes no
  scoring endpoint, so when it is wired it will be an agreement check against
  the model's own continuation, not a likelihood.
- **`GET /api/v1/reconstructions/{id}` does not exist.** Only `/health` and
  `/ready` are served under v1; reconstruction is reachable through `/execute`.
- **Nuclear scaffolds will usually time out.** A nuclear BLAST was measured at
  ~550 s against a 300 s deadline. The agent abstains cleanly rather than
  returning a guess, but it does abstain.
- **One round per gap.** No second attempt with widened flanks or a relaxed
  e-value, because that is the replanning that does not exist yet.
- **Only the best HSP per hit is read.** A homologue whose two HSPs straddle
  the junction does span the gap, and the subject coordinates between them give
  the fill directly - but it is currently counted as not spanning. This is what
  made the megablast failure total rather than partial; using gapped blastn
  avoids it without fixing it.
- **The result format is undocumented.** `FORMAT_TYPE=XML` is still served but
  no longer appears in NCBI's parameter list (which offers XML2, XML2_S, JSON2,
  JSON2_S, SAM). It is configurable, so the day it stops being served is a
  config change and a new parser, not a rewrite.
