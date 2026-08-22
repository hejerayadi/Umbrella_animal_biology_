# Sprint 3 Final Report — Multimodal Species Recognition Agent

## Result

# `PASS — Standalone Recognition Agent Sprint 3 complete`

Every standalone Sprint 3 requirement passes. All real providers work together live. The
closure package is complete and PR-ready.

Finding **P8-F1**, which held this report at `BLOCKED`, was corrected under explicit
authorization and verified both offline and live. **No blocker remains inside Recognition.**

> **Two scopes must not be conflated.**
>
> 1. **Standalone Recognition Agent Sprint 3 scope** — complete apart from P8-F1.
> 2. **Full-platform orchestration integration** — **not started**, future external work
>    owned by the orchestrator/integration team.
>
> The complete platform integration is **not** finished. Nothing in this report claims
> otherwise.

---

## 1. Branch and SHAs

| Item | Value |
| --- | --- |
| Branch | `group-d-recognition-sprint3` |
| HEAD | `fd12565b8d37d70d52e1c22c6a94cd41564d0af8` — *phase 7* |
| Phase 0 base | `2f1455dbf44b…` — *recognition with mock ready* |
| Merge base with `main` | `3edcf2eead8c…` |
| `main` HEAD | `55c8efc…` — *integrated the genome with the big orch* |

**How the Phase 0 base was determined** (from git history, not guessed, and not recorded in
any prior report): `SPRINT_3_COMPLETION_PLAN_RECOGNITION_AGENT.md` was introduced in
`a9f4367` (*"phases 0 2 implemented"*), the first Sprint 3 commit;
`docs/sprint2-final-recognition-only.md` was introduced in `2f1455d`
(*"recognition with mock ready"*), which closed Sprint 2. `2f1455d` is therefore the last
pre-Sprint-3 commit.

`main` has advanced past the merge base independently; this branch was not rebased. No
Recognition work depends on that divergence.

---

## 2. Phase status

| Phase | Status |
| --- | --- |
| 0 — branch and baseline | **PASS** |
| 1 — configuration and provider-selection hardening | **PASS** |
| 2 — GPT-5 mini as the real reasoning brain | **PASS** |
| 3 — real BioCLIP-2 inference (remote Space) | **PASS** |
| 4 — live GBIF and NCBI taxonomy | **PASS** |
| 5 — complete real standalone flow | **PASS** (after the F1–F3 provenance corrections) |
| 6 — revalidate the integrated Global Orchestrator chain | **N/A — The phase was defined as revalidation of an existing integration, but that prerequisite does not exist.** |
| 7 — resilience, security, privacy, operational quality | **PASS** (after the P7-F1–F3 corrections) |
| 8 — closure, documentation, demo, PR readiness | **PASS** (after the P8-F1 correction) |

---

## 3. Changed files

**Total diffstat vs Phase 0 base: 27 files changed, 10 995 insertions, 126 deletions.**
Every changed path is inside `backend/agents/multimodal_recognition_agent/`.

### Implementation

| File | Change |
| --- | --- |
| `config.py` | Provider-mode selection, threshold and validation config, refusal to degrade to a mock |
| `adapters/bioclip.py` | `RemoteBioCLIP2Provider` — Space client, bounded deadline, job cancellation, label parsing |
| `adapters/taxonomy.py` | `RealGBIFProvider`, `RealNCBIProvider`, `RealTaxonomyProvider`, `_RateLimiter` |
| `adapters/reasoning_llm.py` | `AzureGPT5MiniProvider`, plan/explanation contracts, grounding check, `SDK_MAX_RETRIES = 0` |
| `agent.py` | Provider construction through the refusing factories |
| `api.py` | Safe internal-error contract, dependency logging policy |
| `workflows/graph.py` | Runtime-derived provenance (`gbif_mode`, `ncbi_mode`, `taxonomy_executed`) |
| `workflows/nodes.py` | Mode-aware disclosure, taxonomy sentence, degradation warning |
| `workflows/state.py` | Additive provenance fields |
| `domain/models.py` | Additive `verified` taxonomy status |
| `domain/confidence.py` | Provider-neutral `not_identified` clarification question (P8-F1) |
| `card.json` | Managed-service descriptions corrected to the real providers |
| `demo_sprint3.py` | **New** — the demonstration entry point |
| `smoke_test_azure.py`, `smoke_test_taxonomy.py` | Opt-in live smokes |

### New test modules

`test_phase1_config_hardening.py`, `test_phase2_no_retry_and_smoke.py`,
`test_phase3_remote_bioclip.py`, `test_phase4_real_taxonomy.py`,
`test_phase5_certification.py`, `test_phase7_resilience_security.py`,
`test_phase8_demo_and_closure.py`. Plus updates to `test_recognition_only_gate.py` and
`test_workflow.py`.

### Documentation

`SPRINT_3_COMPLETION_PLAN_RECOGNITION_AGENT.md`, `docs/PHASE5_CERTIFICATION.md`,
`docs/PHASE7_CERTIFICATION.md`, `docs/RECOGNITION_AGENT_FINAL_REFERENCE.md`,
`docs/SPRINT_3_FINAL_REPORT.md`, `docs/SPRINT_3_PR_SUMMARY.md`, and the three
`phase*_results.json` machine-readable results.

### Environment template and dependencies

`.env.example` — new variables, **names only**, secrets ship empty (gate-asserted).
`requirements.txt` — `openai`, `gradio_client`, `requests`, `python-dotenv`. No vector
store, no model runtime, no training dependency.

### Commits introduced during Sprint 3

```
a9f4367 2026-08-09  phases 0 2 implemented
b3f94e0 2026-08-14  phase 3 ready bioclip ready
1e0cfb4 2026-08-20  Phase 4: real GBIF/NCBI taxonomy providers, live smoke test, gate test fixes
059277c 2026-08-20  implementation ncbi gbif valide
83ae275 2026-08-21  phase 5
fd12565 2026-08-21  phase 7
```

Phase 8's own changes are **uncommitted**: two modified files (`card.json`, the plan) and
six new files.

---

## 4. Architecture

Standalone agent, compiled LangGraph `StateGraph`, one request per invoke, no checkpointer
and no memory store.

```
validate → plan (GPT 1/2) → classify → confidence gate → taxonomy → explain (GPT 2/2)
         → delegate_if_needed → finalize
```

The deterministic safeguards, each test-enforced:

- **BioCLIP-2 is the only source of species candidates.**
- The **confidence gate closes before taxonomy runs**.
- Text can **downgrade, never promote**.
- GPT **decides nothing scientific** — it plans internal steps and phrases the explanation,
  both validated.
- Taxonomy **annotates only**; a missing identifier stays `null`.
- Delegation is a **capability hint** (`needs_agent`); the agent never calls a peer.

Full detail: `docs/RECOGNITION_AGENT_FINAL_REFERENCE.md`.

---

## 5. Real provider modes and versions

| Concern | Implementation |
| --- | --- |
| Reasoning | Azure OpenAI **GPT-5 mini**, deployment alias `umbrella-gpt5-mini`, swedencentral |
| Classification | Remote **BioCLIP-2** on `imageomics/bioclip-2-demo` |
| Model target | `imageomics/bioclip-2` |
| GBIF | Live Species API |
| NCBI | Live Entrez `esearch`, paced at **2 req/s** unkeyed (published limit 3/s), 10 req/s with an optional key |

Max 2 GPT calls per request, `store=False`, SDK retries **0**, bounded deadlines
everywhere, and no retry at any boundary.

**Environment variables are documented by name only** — in this report, in
`phase8_results.json` and in `.env.example`. No value appears anywhere.

---

## 6. Tests

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

**Zero xfails.** The strict xfail that recorded P8-F1 became eleven ordinary passing
regression tests; the suite grew from 1178 to 1196.

No offline test contacts a live service, and none sleeps.

---

## 7. Live results (redacted)

| Check | Result |
| --- | --- |
| Azure GPT smoke | **PASS** — 2 logical calls, 2 HTTP attempts, no retry, `store=False` |
| Remote BioCLIP-2 provider smoke | **PASS** — 5 ranked labels, in order, all in [0, 1] |
| GBIF/NCBI provider smoke | **PASS** — *Panthera leo* 5219404/9689, *Panthera tigris* 5219416/9694, nonexistent binomial matched nothing |
| Complete real standalone request | **PASS** — *Loxodonta africana*, `identified`, GBIF 2435350, NCBI 9785, 2 LLM calls, 21.0 s |
| Prompt injection on a real image | **PASS** — species and every score unchanged |
| Scientific follow-up | **PASS** — `needs_agent`, capability `Evolution`, no address in the prompt |
| Resume after delegation | **PASS** — completes, does not delegate twice |
| Second real image | **PASS** — *Acinonyx jubatus*, identified independently |
| Demo script, live | **PASS** — *Iguana iguana*, GBIF 2459658, NCBI 8517, `verified`, 21.2 s |
| Real `not_identified` smoke | **PASS** — coyote photograph, `not_identified`, corrected provider-neutral question, scores identical to Phase 5, 22.7 s |

**External conditions reported honestly, not worked around:**

- Azure `umbrella-gpt5-mini` returns HTTP 429 under back-to-back load. The agent falls back
  deterministically with honest provenance. **No request was retried to force a PASS.**
- GBIF was transiently unavailable for one candidate during the Phase 5 run; the agent
  degraded correctly. The same lookup succeeded in the Phase 8 demo run.

---

## 8. Phase 5 real-image evaluation matrix

Seven Wikimedia Commons photographs (CC BY / CC BY-SA), six taxonomic groups, all confirmed
absent from the mock fixture, held outside the repository and deleted after use.

| Image | Expected | Observed Top-1 | Score | Margin | Top-1 ✓ | In Top-K | Decision | GBIF | NCBI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mammal_elephant` | *Loxodonta africana* | *Loxodonta africana* | 0.9364 | 0.8915 | ✅ | ✅ | `identified` | 2435350 | 9785 |
| `bird_eagle` | *Haliaeetus leucocephalus* | *Haliaeetus leucocephalus* | 0.9511 | 0.9358 | ✅ | ✅ | `identified` | 2480446 | 52644 |
| `reptile_iguana` | *Iguana iguana* | *Iguana iguana* | 0.9343 | 0.8840 | ✅ | ✅ | `identified` | — | 8517 |
| `shark_white` | *Carcharodon carcharias* | *Carcharhinus fitzroyensis* | 0.2148 | 0.0831 | ❌ | ❌ | `not_identified` | — | — |
| `arthropod_bee` | *Apis mellifera* | *Apis mellifera* | 0.4601 | 0.2779 | ✅ | ✅ | `uncertain` | 1341976 | 7460 |
| `mammal_cheetah` | *Acinonyx jubatus* | *Acinonyx jubatus* | 0.9070 | 0.8340 | ✅ | ✅ | `identified` | 2435270 | 32536 |
| `ambiguous_coyote` | *Canis latrans* | *Canis lycaon* | 0.4221 | 0.0355 | ❌ | ✅ (rank 2) | `not_identified` | — | — |

Top-1 correct **5/7**; expected species in Top-K **6/7**. Latency 5.9–10.1 s, mean 7.4 s.

> An integration observation on seven images — **not** a measurement of scientific accuracy.
> No threshold was tuned on it.

**The inconclusive cases are the point.** `arthropod_bee` was the *correct* species at
Top-1 and was still reported `uncertain`, because 0.4601 clears the 0.45 floor but falls far
short of 0.75. `ambiguous_coyote` put two congeners 0.0355 apart and returned
`not_identified`. The agent declines to claim what it cannot support.

Thresholds `0.75 / 0.08 / 0.45` were **never changed** during Sprint 3.

---

## 9. Phase 7 security and resilience result

**PASS.** Bounded waits everywhere, no retry loop, no HTTP 500, concurrent requests share
no state, LLM budgets reset per request, prompt injection cannot alter the scientific
evidence, and no secret, prompt or image data leaks.

Highlights: 10-case timeout matrix; 63 malformed-payload cases across five sources;
per-candidate partial-availability; 8-worker concurrency with zero crossings; 10 injection
instructions including one where the model *obeyed* and still changed nothing; dependency
loggers pinned so a root-level DEBUG run emits six lines, all the agent's own.

Detail: `docs/PHASE7_CERTIFICATION.md`.

---

## 10. Seven-key contract

```
gbif_id  ncbi_taxid  recognition  recognition_candidates
recognition_provenance  species  species_id
```

Asserted as **set equality** on every completed branch — identified, uncertain,
not_identified, resumed, degraded taxonomy, degraded LLM. **Unchanged since Sprint 2.**
Failures return `{"error_code", "error"}` on every path, including the unexpected-exception
path.

---

## 11. No-leak evidence

| Surface | Result |
| --- | --- |
| Tracked `.env` files | none |
| Tracked secrets or personal emails | none — only variable *names* and test placeholders |
| Tracked images, weights, caches, logs, datasets | none |
| Tracked virtual environments | none |
| Canary scan (logs, exceptions, state, responses, provenance, failures, demo output) | clean |
| Dependency logging | `httpx`, `httpcore`, `urllib3`, `openai` pinned to `WARNING` |
| Temporary evaluation assets | deleted |
| Lingering processes | none |

---

## 12. Findings

### P8-F1 — a stale mock claim in the `not_identified` clarification question — **FIXED**

**Before**, `domain/confidence.py` returned unconditionally:

> "The Sprint 2 classifier did not return a taxon confident enough to name a species for
> this image. Could you supply a clearer photograph of the animal, or tell me where the
> observation was made?"

Phase 5 corrected the equivalent wording in the explanation and the safety footer; this
string was never given a mode branch. It was **reachable in full real mode** — two of the
seven Phase 5 images (`shark_white`, `ambiguous_coyote`) returned `not_identified` while
running remote BioCLIP-2 with live GBIF and NCBI, so each of those answers misdescribed its
own provenance to the user.

**After:**

> "The available visual evidence was not strong enough to identify a species confidently.
> Could you supply a clearer photograph of the animal, or tell me where the observation was
> made?"

The correction describes the **evidence** rather than the component that produced it, so it
is true for remote BioCLIP-2, for the mock, for no candidates at all, for weak candidates
and for candidates too close together to separate. **No mode branch was needed** — the
sentence is provider-neutral, and a test asserts real and mock modes return the *identical*
string rather than two branches that happen to agree.

**Live verification.** A bounded real `not_identified` run (remote BioCLIP-2 + live
GBIF/NCBI, coyote photograph) returned the corrected sentence. Candidate scores were
identical to the Phase 5 run — *Canis lycaon* 0.4221, *Canis latrans* 0.3866 — confirming
the correction changed no science.

**Unchanged by the correction:** thresholds, confidence decisions, candidate ranking, the
`identified`/`uncertain`/`not_identified` conditions, `request_better_image`, the workflow,
providers, schemas, the seven-key contract, the API, the agent card and demo behaviour. The
`uncertain` and `identified` branches are asserted byte-identical.

Eleven regression tests replace the strict xfail; none was deleted or weakened.

---

## 13. Known limitations

1. **No Global Orchestrator integration.** The agent emits `needs_agent` and stops.
2. **Seven evaluation images** — certifies integration, not accuracy.
3. **Per-stage latency is not instrumented**; the equivalent fields are in
   `recognition_provenance`.
4. **Text alignment is inert in real mode** by design (no name catalogue) — a stronger
   safety property, but agreement/conflict are certified against the mock-catalogue provider.
5. **The dependency logging policy applies at the HTTP entry point**; direct importers keep
   their libraries' levels.
6. **`INTERNAL_ERROR` is a string literal**, not an `ErrorCode` member — `domain/errors.py`
   was outside the Phase 7 corrective boundary.
7. **The NCBI rate limiter is per-provider** — relevant before scaling to multiple replicas.
8. **Outage scenarios are injected** into our own transports; no external service was
   deliberately failed.
9. **External Azure and GBIF availability is outside our control** — the deployment
   rate-limits under back-to-back load and GBIF was transiently unavailable for one
   candidate during evaluation. The agent degrades correctly and reports both honestly;
   neither was retried to force a green result.
10. **Opaque model prose is accepted verbatim** — grounding restricts species and
    probability claims, not arbitrary text, which is precisely why the factual disclosure is
    appended outside the model's text.

---

## 14. Final traceability matrix

| Sprint 3 requirement | Status | Evidence |
| --- | --- | --- |
| Prepare training data | **N/A** | Pre-trained BioCLIP-2; no training in scope |
| Train or fine-tune a model | **N/A** | Fixed architecture decision |
| Evaluate selected model | **PASS** | Phase 5 real-image matrix |
| Configure model for inference | **PASS** | Remote Space, pinned revision, provider-mode tests |
| Integrate model into agent | **PASS** | `Agent → remote BioCLIP-2 → candidates`, live |
| Define when agent uses model | **PASS** | Workflow and Phase 3 tests |
| Test Agent → Model → Result | **PASS** | Real standalone scenario suite |
| **Pass model result to orchestrator** | **NOT IMPLEMENTED — future Global Orchestrator/integration-team task** | Agent returns the validated seven-key result; nothing consumes it |
| Define agent objective/responsibilities | **PASS** | Card and final reference |
| Configure agent framework | **PASS** | Compiled LangGraph graph |
| Connect LLM | **PASS** | Live Azure GPT-5 mini, 2-call budget, `store=False`, no retry |
| Define prompts/instructions | **PASS** | Planner/explainer prompts and guards |
| Implement agent tools | **PASS** | Real BioCLIP-2, GBIF, NCBI |
| Connect knowledge/API sources | **PASS** | Live GBIF and NCBI |
| Retrieval pipeline | **N/A** | No vector store, retrieval or similarity by decision |
| Define input/output schemas | **PASS** | Shared contracts preserved |
| Implement reasoning workflow | **PASS** | Graph and tests |
| Handle invalid inputs/tool failures | **PASS** | Phase 7 matrices |
| Test agent independently | **PASS** | 1178 offline tests plus opt-in live smokes |
| Finalize workflow | **PASS** | Protected node sequence and single finalizer |
| **Integrate with orchestration layer** | **NOT IMPLEMENTED — future Global Orchestrator/integration-team task** | Never implemented; the plan's opening claim was incorrect |
| Create Group D sub-orchestrator | **N/A** | Not in the current architecture |
| **Implement delegation/context/results** | **PARTIAL — agent side READY FOR FUTURE INTEGRATION** | `needs_agent` + resume verified standalone; routing/aggregation/invocation not implemented |
| **Validate agent-orchestrator communication** | **N/A — revalidation prerequisite absent** | No integration exists to validate |
| Frontend | **N/A** | Outside this plan |

---

## 15. Phase 6 — justification

> **N/A — The phase was defined as revalidation of an existing integration, but that
> prerequisite does not exist.**

Recognition is not registered with, routed by, or invoked through the Global Orchestrator.
There is nothing to revalidate. **No orchestrator E2E result is claimed anywhere in the
Sprint 3 evidence**, and no orchestrator test was run.

---

## 16. Future Global Orchestrator integration task

**Owner:** orchestrator / integration team. **Status:** NOT IMPLEMENTED.

**Recognition-side readiness — READY FOR FUTURE INTEGRATION:**

- `POST /execute` on port **8005**, registered as `Multimodal` in `backend/registry.py`;
- the shared `AgentRequest` / `AgentResult` / `AgentStatus` contract, unchanged;
- the **seven output keys**, validated on every branch;
- `needs_agent` capability hints with a prompt naming no address, plus resume-on-context —
  both verified live, standalone;
- the agent card, corrected and schema-checked.

**Not validated, and not claimed:** orchestrator routing, shared-context aggregation,
specialist invocation, delegation resume end to end.

---

## 17. Confirmations

- Every implementation change belongs to Recognition. No other agent, orchestrator,
  registry or frontend file was modified.
- Approved shared schema contracts and the seven-key contract are unchanged.
- No `.env`, secret, image, weight, cache, dataset or log is tracked.
- **Nothing was committed, pushed, or opened as a pull request.**
