# Part B — LangSmith Tracing Setup

## Step 1 — Create the project

1. Sign up / log in at https://smith.langchain.com
2. Click **New Project** → name it exactly `genome-agent-sprint4`
3. Go to **Settings → API Keys** → create a key → copy it

## Step 2 — Set environment variables

Add to `backend/agents/genome_agent/.env`  
(copy from `.env.example`, fill in real values):

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=genome-agent-sprint4
```

> **Windows CMD:**  `set LANGCHAIN_TRACING_V2=true`  
> **PowerShell:**   `$env:LANGCHAIN_TRACING_V2="true"`

## Step 3 — Verify zero-code-change tracing works

Run one query through the orchestrator — LangGraph traces automatically
once the vars are exported into the same process:

```bash
# from repo root
python -m backend.agents.genome_agent.evaluation.run_eval --case happy_human_genome_size
```

Open https://smith.langchain.com → your project → you should see one trace
with a tree of spans. If nothing appears, check that the vars are exported
into the Python process that runs the command (not just a different shell).

## Step 4 — Confirm the trace tree shape

In LangSmith, expand the trace. You should see this shape:

```
run: happy_human_genome_size
└─ query_router
└─ species_resolver
   └─ ncbi_eutils_get  ← NCBI taxonomy search span
   └─ ncbi_eutils_get  ← NCBI assembly lookup span
├─ get_genome_metadata  ← should visually OVERLAP with gene_annotation
│  └─ ncbi_eutils_get
└─ get_gene_annotation  ← should visually OVERLAP with genome_metadata
   └─ ncbi_eutils_get
└─ join_parallel
└─ explanation_writer (or reconstruction_resolver → explanation_writer)
```

**What to check (in this order):**

1. Is every expected node present as a span?  
   Missing = function isn't going through the traced entry point.

2. Do `get_genome_metadata` and `get_gene_annotation` overlap on the
   timeline?  
   If they run sequentially instead of in parallel — real bug, flag it.

3. Under `join_parallel`, is exactly ONE tail present?  
   Two tails present = routing bug. Wrong tail = routing logic bug.

4. Under `species_resolver` and the data nodes, are `ncbi_eutils_get`
   spans present?  
   These are the NCBI calls instrumented with `@traceable` in
   `subagents/_ncbi_client.py`. If they're missing, tracing isn't
   reaching the network layer.

## Step 5 — Run the full eval dataset with tracing on

```bash
# runs all 26 cases, each producing a LangSmith trace
python -m backend.agents.genome_agent.evaluation.run_eval
```

Then use the trace review script to check the three things from the guide:

```bash
python -m backend.agents.genome_agent.evaluation.trace_review
```

## Step 6 — Three things to check per trace (from the guide)

For each of 5–10 traces (mix of happy-path and edge cases):

| Check | What to look for |
|---|---|
| **Orchestration decisions** | Does the child span under `join_parallel` match the input? Complete assembly → `explanation_writer` directly. Scaffold → `reconstruction_resolver` first. |
| **Hallucination signals** | Open `explanation_writer` span next to `get_genome_metadata` span. Does every number in the explanation appear in the metadata? Note any mismatch as a labeled failure. |
| **Latency / duplicate calls** | Sort spans by duration — `ncbi_eutils_get` spans are usually the slowest. Scan for duplicate spans with identical arguments in the same run. |

## Step 7 — Package for the review

Pick 2–3 traces that show a real bug (not just clean happy-path):
- Screenshot the timeline showing sequential parallel nodes
- Screenshot a hallucinated number not in the upstream data
- Screenshot a duplicate NCBI call

Pair each screenshot with the corresponding test case ID from
`test_cases.yaml` and the `evaluators.py` metric that catches it.
