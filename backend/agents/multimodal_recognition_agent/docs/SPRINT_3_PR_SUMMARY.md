# Pull Request Summary — Sprint 3

> **Not opened.** This document is prepared for review; no PR exists, and nothing has been
> committed or pushed.

---

## Title

```
Sprint 3: productionize the Multimodal Species Recognition Agent (real BioCLIP-2, live GBIF/NCBI, Azure GPT-5 mini)
```

---

## Summary

Sprint 3 takes the Recognition Agent from a mock-only Sprint 2 implementation to one that
runs on real providers end to end, without changing its scientific decision architecture.

BioCLIP-2 classification now executes remotely on the official public Space; GBIF and NCBI
are queried live per candidate; Azure GPT-5 mini acts as a bounded planner and grounded
explainer. Every deterministic safeguard is preserved and test-enforced: BioCLIP-2 remains
the only source of species candidates, the confidence gate closes before taxonomy runs,
taxonomy annotates and nothing more, user text can downgrade but never promote, and GPT
decides nothing scientific.

The agent remains **standalone**. It emits `needs_agent` capability hints; it never calls a
peer. **Global Orchestrator integration is not included in this PR** — see below.

**Scope of change:** 27 files, +10 995 / −126, every path inside
`backend/agents/multimodal_recognition_agent/`. No other agent, orchestrator, registry or
frontend file is touched. The shared schemas and the seven-key output contract are
unchanged.

---

## Major implementation changes

| Area | Change |
| --- | --- |
| **Configuration** | Provider modes (`mock`\|`remote`, `mock`\|`real`, `disabled`\|`fake`\|`azure`). Factories **refuse to start** rather than degrade to a mock — asking for real inference and silently getting fixture data is the failure they exist to prevent. |
| **Classification** | `RemoteBioCLIP2Provider` on `imageomics/bioclip-2-demo` (model target `imageomics/bioclip-2`). One deadline spans construction, upload, queue wait and retrieval; an overrunning job is cancelled, not abandoned. |
| **Taxonomy** | Live GBIF Species API and NCBI Entrez, per candidate. Paced at 2 req/s unkeyed against a published 3 req/s — the margin is measured, not arbitrary. Pacing, never retrying. |
| **Reasoning** | Azure GPT-5 mini, max 2 calls per request, `store=False`, SDK `max_retries=0`. A plan outside the four-step vocabulary is rejected whole; an ungrounded explanation is discarded; a failed planner forfeits the explanation call. |
| **Provenance** | Derived from the provider that actually ran, never from configuration. Top-level and nested taxonomy modes cannot contradict each other by construction. |
| **Disclosure** | Every safety/disclosure string is mode-aware or provider-neutral. A real run no longer describes itself as a Sprint 2 mock — including the `not_identified` clarification question, which now describes the evidence rather than the component. |
| **Error boundary** | The unexpected-exception path returns a fixed safe message and the structured `{error_code, error}` dict; nothing derived from the exception reaches the caller. |
| **Dependency logging** | `httpx`, `httpcore`, `urllib3`, `openai` pinned to `WARNING` at service startup, so a root-level DEBUG run cannot write endpoints or an email address to disk. |
| **Demo** | `demo_sprint3.py` — opt-in guarded, dynamic image path, safe output only. |

---

## Test plan

**Complete offline suite: 1196 passed, 0 failed, 0 skipped, 0 xfailed.**

| Suite | Count |
| --- | --- |
| Focused confidence tests | 13 |
| Architecture and no-leak gates | 63 |
| Phase 3 — remote BioCLIP-2 | 141 |
| Phase 4 — real GBIF/NCBI | 172 |
| Phase 5 — full-flow certification | 113 |
| Phase 7 — resilience and security | 217 |
| Phase 8 — demo and closure | 56 |

```bash
python -m pytest backend/agents/multimodal_recognition_agent/tests -q
```

Every external client is injectable, so the entire suite runs with **no network, no
credentials, no model download and no sleeping** — the NCBI rate limiter takes an injected
clock. **Zero xfails.**

**Opt-in live checks** (each refuses to run without `RECOGNITION_LIVE_SMOKE=1`):

```bash
RECOGNITION_LIVE_SMOKE=1 python -m backend.agents.multimodal_recognition_agent.smoke_test_azure
RECOGNITION_LIVE_SMOKE=1 python backend/agents/multimodal_recognition_agent/smoke_test_taxonomy.py
RECOGNITION_LIVE_SMOKE=1 python -m backend.agents.multimodal_recognition_agent.demo_sprint3 --image <path>
```

---

## Real-service evidence

| Check | Result |
| --- | --- |
| Azure GPT-5 mini | 2 logical calls, 2 HTTP attempts, no retry, `store=False` |
| Remote BioCLIP-2 | 5 ranked labels, correctly ordered, all scores in [0, 1] |
| GBIF / NCBI | *Panthera leo* 5219404/9689; *Panthera tigris* 5219416/9694; a nonexistent binomial matched nothing and invented nothing |
| Complete standalone request | *Loxodonta africana*, `identified`, GBIF 2435350, NCBI 9785, 21.0 s |
| Demo script | *Iguana iguana*, GBIF 2459658, NCBI 8517, `verified`, 21.2 s |
| Real `not_identified` smoke | Coyote photograph — `not_identified`, provider-neutral question, scores identical to Phase 5 |

**Seven real photographs** (Wikimedia Commons, CC BY / CC BY-SA, none in the mock fixture,
deleted after use): Top-1 correct 5/7, expected species in Top-K 6/7 — 4 `identified`,
1 `uncertain`, 2 `not_identified`.

Worth noting for review: `arthropod_bee` was the **correct** species at Top-1 and was still
reported `uncertain` (0.4601 against a 0.75 threshold), and `ambiguous_coyote` returned
`not_identified` with two congeners 0.0355 apart. The agent declines to claim what it
cannot support. **Thresholds were never changed during Sprint 3.**

Reported honestly rather than worked around: the Azure deployment rate-limits under
back-to-back load (the agent falls back deterministically), and GBIF was transiently
unavailable for one candidate during evaluation (the agent degraded correctly). **No
request was retried to force a green result.**

---

## Security evidence

- **No retry anywhere** — one logical provider call is one external attempt.
- **All external waits bounded**; an overrunning BioCLIP job is cancelled.
- **Never an HTTP 500** — the boundary always returns the shared schema.
- **No secret, prompt or image data leaks** — canaries pushed through logs, exceptions,
  serialized state, API responses, provenance, failure outputs and demo output. The image
  never reaches the reasoning model (asserted with a PNG `tEXt` canary); provider failures
  log the exception *type* only.
- **Prompt injection cannot alter the scientific evidence** — 10 instructions, including one
  where the model obeyed the injection and the result was discarded as ungrounded.
- **Concurrency** — 8 simultaneous requests share no state; images, instructions, taxonomy
  reports, context and provenance never cross. LLM budgets reset per request.
- **Nothing sensitive is tracked** — no `.env`, key, personal email, image, weight, cache,
  dataset, log or virtual environment. `.env.example` ships secrets empty (gate-asserted).

---

## Limitations reviewers should know

1. Seven evaluation images certify integration, **not** general accuracy. No threshold
   was tuned on them.
2. Per-stage latency is not instrumented; the equivalent fields are on the response.
3. Text alignment is inert in real mode by design (no offline name catalogue).
4. The dependency logging policy applies at the HTTP entry point only.
5. `INTERNAL_ERROR` is a string literal, not an `ErrorCode` member.
6. The NCBI rate limiter is per-provider — relevant before scaling to multiple replicas.
7. External Azure and GBIF availability is outside our control; the agent degrades
   correctly and reports it honestly rather than retrying to force a green result.

---

## Global Orchestrator integration is **not** included

This PR delivers the **standalone** Recognition Agent only.

**Not implemented and not claimed:** orchestrator routing, shared-context aggregation,
specialist invocation, delegation resume end to end. No orchestrator E2E test was run,
because no such integration exists — Sprint 3's Phase 6 is recorded as
`N/A — The phase was defined as revalidation of an existing integration, but that
prerequisite does not exist.`

**What is ready for that future work:** `POST /execute` on port 8005 registered as
`Multimodal`; the unchanged shared `AgentRequest`/`AgentResult`/`AgentStatus` contract; the
seven output keys validated on every branch; `needs_agent` capability hints and
resume-on-context, both verified live standalone; and a corrected, schema-checked agent
card.

Integrating it is a separate task owned by the orchestrator/integration team.
