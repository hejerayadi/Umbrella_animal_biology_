# Sprint 4 — Task 3: Agent Traceability with LangSmith (Evolution Agent)

**This task's goal (from the sprint brief):** Integrate LangSmith, trace the complete execution flow (user query → orchestrator decision → agent selection → tool calls → RAG retrieval → LLM reasoning/output → final response), use traces to debug and analyze behavior, monitor failures/latency/unnecessary calls.

This document follows that outline section by section.

---

## 1. Integration approach

LangSmith is wired in two ways, matched to what's actually running in this agent:

1. **Automatic** — the orchestrator is a LangGraph `StateGraph`, and every LLM call goes through `langchain-openai`'s `AzureChatOpenAI`. Both integrate with LangSmith natively: once `LANGCHAIN_TRACING_V2=true` is set, every graph node and every LLM call traces itself with zero extra code.
2. **Explicit (`@traceable`)** — everything LangChain *doesn't* auto-instrument: the raw external tool calls (MAFFT, IQ-TREE, UniProt, ESM-2), which are plain Python/subprocess/HTTP calls, not LangChain runnables. Each is decorated with `@traceable(name=..., run_type="tool")` from the `langsmith` SDK.

Critically, `@traceable` is a **no-op** when tracing isn't enabled — so this instrumentation carries zero cost or risk when `LANGCHAIN_TRACING_V2` is unset, and doesn't require every developer on the team to have a LangSmith key to run the agent.

### What's instrumented, file by file

| File | What's tagged |
|---|---|
| `orchestrator_adapter.py` | `OrchestratorEvolutionAgent.run()` — the root span: user query in, final `AgentResult` out |
| `orchestrator/evolution_orchestrator.py` | LangGraph invocation given a clean `run_name` (`"EvolutionOrchestrator graph"`) instead of the generic default |
| `planner.py` | `plan()` — the Planner, LLM call #1 |
| `explainer.py` | `explain()` — the Explainer, LLM call #2 |
| `workers/molecular_comparison/logic.py` | `MolecularComparisonAgent.run()` (agent-level span), `fetch_uniprot_sequence()`, `embed_sequences()` (tool-level spans) |
| `workers/phylogenetic_tree/worker.py` | `PhylogeneticTreeWorker.run()` (agent-level span), `_build_tree()` (sub-pipeline span) |
| `tools/mafft.py` | `align()`, `_align_ebi()`, `_align_local()` — separately tagged so the EBI-vs-local fallback path is visible in the trace, not hidden inside one opaque call |
| `tools/iqtree.py` | `build_tree()`, `_run_local_iqtree()` |

---

## 2. Setup

Three environment variables in `backend/agents/evolution_agent/.env` (documented in `.env.example`, gitignored — never committed):

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your langsmith key>
LANGCHAIN_PROJECT=evolution-agent
```

Get a key free at `smith.langchain.com` → Settings → API Keys. `.env` loads unconditionally at server startup (a separate bug fix earlier this sprint made sure of that — it used to only load on a code path the Planner could skip).

---

## 3. Verified live — the actual execution flow, traced

Rather than describe this abstractly, here is the **real trace tree** pulled directly from the LangSmith API (`POST /runs/query`) for one real request — `"Give me both a similarity network and a phylogenetic tree for human, chimp and mouse."` — run against the live agent with the real Planner, real Azure LLM, real MAFFT/IQ-TREE binaries. 32 spans, zero mocking:

```
[chain] Evolution Agent request                       ← 1. User query
  [chain] Planner (LLM #1)                             ← 2. Orchestrator decision
    [llm]  AzureChatOpenAI
  [chain] EvolutionOrchestrator graph                  ← 2. Orchestrator decision
    [chain] plan → species_resolver → dispatch → assemble → explain
    [chain] dispatch
      [chain] Molecular Comparison Agent                ← 3. Agent selection
        [tool] UniProt sequence fetch   x3               ← 4. Tool calls
        [tool] ESM-2 embedding                           ← 4. Tool calls
      [chain] Phylogenetic Tree Agent                   ← 3. Agent selection
        [tool] MAFFT via EBI REST API   — status: error  ← 4. Tool calls (failure, see below)
        [chain] MAFFT -> IQ-TREE pipeline
          [tool] MAFFT alignment → MAFFT via local binary  — success
          [tool] IQ-TREE build → IQ-TREE local binary run  — success
    [chain] explain
      [chain] Explainer (LLM #2)                        ← 5. LLM reasoning/output
        [llm]  AzureChatOpenAI
                                                          ← 6. Final response (root span's output)
```

Mapped against the brief's required flow:

| Required step | Where it shows up |
|---|---|
| User query | Root span input (`Evolution Agent request`) |
| Orchestrator decision | `Planner (LLM #1)` + the `EvolutionOrchestrator graph` node sequence |
| Agent selection | `Molecular Comparison Agent` / `Phylogenetic Tree Agent` spans — both present here because this was a `full_analysis` request |
| Tool calls | UniProt fetch, ESM-2 embedding, MAFFT (both paths), IQ-TREE — each a separate span with its own timing |
| RAG retrieval | **N/A** — Evolution Agent has no retrieval/RAG component |
| LLM reasoning/output | `AzureChatOpenAI` spans nested under both the Planner and the Explainer |
| Final response | Root span's recorded output — the same `AgentResult` returned to the HTTP caller |

---

## 4. Using traces to debug and monitor — demonstrated, not hypothetical

This same trace already shows the exact monitoring the brief asks for, without needing to stage a failure:

- **Failures**: `MAFFT via EBI REST API` has `status: error` — the EBI web service rejected or failed the request. Directly beneath it, `MAFFT via local binary` shows `success` — the automatic fallback working. Anyone looking at this trace can see, at a glance, that the EBI path is currently unreliable and every phylogenetic request is quietly paying its timeout cost before falling back.
- **Latency**: every span carries `start_time`/`end_time`, so the slow step in any given request (usually IQ-TREE with bootstrap, or the EBI timeout above) is immediately visible without adding custom timing code.
- **Unnecessary calls**: the tree structure makes it obvious if, say, both workers ran when only one was asked for, or if a tool got called more times than the input warranted (e.g. the 3x `UniProt sequence fetch` here is correct — one per species — but would be a visible red flag if a bug caused duplicate fetches for the same species).

---

## Slide-ready summary

- LangSmith integrated two ways: automatic (LangGraph + LangChain LLM calls) + explicit `@traceable` on every raw external tool call
- Zero cost when disabled — `@traceable` no-ops without a key, so this never blocks teammates without one
- **Verified live**, not just wired: pulled the real 32-span trace tree via the LangSmith API and confirmed every required flow step is present
- RAG retrieval correctly N/A for this agent
- The one real request traced happened to capture a live failure-and-fallback (EBI API error → local MAFFT success) — proof the "monitor failures" requirement works, for free, on real traffic
