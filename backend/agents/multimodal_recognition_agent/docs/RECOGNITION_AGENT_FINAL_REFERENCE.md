# Recognition Agent — Final Reference

**Multimodal Species Recognition Agent, end of Sprint 3.**

This document describes the code that exists today. It supersedes
`sprint2-final-recognition-only.md`, which described a mock-only agent and is retained
only as a historical record — where the two disagree, this document is correct.

> **Scope.** This is a **standalone** agent. It is **not** currently integrated with the
> Global Orchestrator. Nothing in this document should be read as a claim that
> orchestrator routing, context aggregation, specialist invocation or resume has been
> validated end to end — none of that exists yet.

---

## 1. What the agent does

One animal photograph plus one non-empty text instruction, in the same request, produce
one ranked species identification.

Both inputs are mandatory. It cannot identify a species from text alone, and it will not
accept an image with no instruction. It returns `identified`, `uncertain` or
`not_identified` — and the latter two are **completed scientific outcomes**, not failures.

It does **not** provide visual similarity search, reference-image retrieval, or any
nearest-neighbour capability, and it owns no reference-image corpus.

---

## 2. Architecture

A real LangGraph `StateGraph`, compiled once, one request per invoke. No checkpointer and
no memory store: nothing survives a request.

```
START
  → validate_image_and_text
  → plan_or_analyze_text                 (GPT call 1 of 2, optional)
  → classify_with_mock_bioclip2          (node name is historical; the provider is chosen by mode)
  → evaluate_confidence                  (deterministic gate — closes here)
  → validate_taxonomy_with_mock_gbif_and_ncbi
  → explain                              (GPT call 2 of 2, optional)
  → delegate_if_needed
  → finalize → END
```

Every stage routes conditionally: the moment a node writes `error_code`, the graph jumps
straight to `finalize`. `finalize` is the **only** node that builds an `AgentResult`, so
success, uncertain, no-match, delegation and failure all converge on one contract.

### The deterministic safeguards

These are the properties the whole design exists to protect, and each is enforced by tests:

| Safeguard | Where |
| --- | --- |
| **BioCLIP-2 is the only source of species candidates** | `classify` is the sole producer; no other node may add one |
| **The confidence gate is deterministic and closes before taxonomy runs** | `evaluate_confidence` precedes `validate_taxonomy` |
| **Text can downgrade, never promote** | a conflicting instruction yields `uncertain`; it cannot introduce a taxon |
| **GPT decides nothing scientific** | it plans internal steps and phrases the explanation; both are validated |
| **Taxonomy annotates only** | it may not select, reorder, rescore or change a decision |
| **A missing identifier stays `null`** | never inferred to fill a gap |
| **Delegation is a capability hint** | emitted as `needs_agent`; the agent never calls a peer |

### GPT-5 mini's exact role

- **Planner (call 1)** — chooses which optional steps run, from a fixed four-step
  vocabulary (`classify_image`, `score_confidence`, `validate_taxonomy`, `explain`), of
  which `classify_image` and `score_confidence` are mandatory. A plan naming anything
  outside that vocabulary, or missing a mandatory step, is rejected **whole** and the
  deterministic plan is used.
- **Explainer (call 2)** — phrases the finding. Its text is checked before acceptance: a
  species name may appear only if the classifier returned it, and the score may not be
  described as a probability. An ungrounded explanation is discarded.
- **A failed planner forfeits the explanation call.** One failure means the provider is
  not answering to contract on this request; spending the second call would be a retry
  wearing a different hat.

---

## 3. Real providers

| Concern | Implementation |
| --- | --- |
| Reasoning | **Azure OpenAI GPT-5 mini** (`AzureGPT5MiniProvider`) |
| Classification | **Remote BioCLIP-2** on the official public Space `imageomics/bioclip-2-demo` |
| Model target | `imageomics/bioclip-2` |
| GBIF | **Live** GBIF Species API (`https://api.gbif.org/v1/species/match`) |
| NCBI | **Live** Entrez E-utilities `esearch` |

Selected by `BIOCLIP_PROVIDER_MODE` (`mock` \| `remote`), `TAXONOMY_PROVIDER_MODE`
(`mock` \| `real`) and `RECOGNITION_LLM_PROVIDER_MODE` (`disabled` \| `fake` \| `azure`).
Each factory **refuses to start** rather than degrade to a mock: asking for real inference
and silently getting fixture data is the specific failure these factories exist to prevent.

The mock providers still exist and are fully supported for offline development. When one
runs, the response says so — provenance and disclosure are derived from the provider that
actually executed, never from configuration.

### NCBI request rate

NCBI's Entrez guidelines permit 3 requests/second unkeyed. The agent paces at **2/second**
instead. The margin is deliberate: at a measured 2.86/second, 1 request in 8 was still
rejected, because the rate NCBI observes is not the rate we send at — transit bunches
requests at the far end and the accounting windows do not align. With an API key
(`NCBI_API_KEY`, optional) the published 10/second is used.

This is **pacing, not retrying**: the limiter waits *before* sending. Four lookups produce
exactly four requests.

### Timeouts and the no-retry rule

| Provider | Bound |
| --- | --- |
| Azure | `RECOGNITION_LLM_TIMEOUT_SECONDS`, reaching the client constructor |
| Remote BioCLIP-2 | one deadline spanning client construction, upload, queue wait and result retrieval |
| GBIF / NCBI | per-request timeout, `RECOGNITION_TAXONOMY_TIMEOUT_SECONDS` |

**No retry exists anywhere.** The OpenAI SDK is constructed with `max_retries=0` — the
default of 2 would have retried underneath the agent's own no-retry rule, making one
logical call up to three HTTP attempts and silently tripling the effective timeout. One
logical provider call is exactly one external attempt, at every boundary.

An overrunning BioCLIP job is **cancelled**, not abandoned, and the temporary upload file
is removed in a `finally` block.

---

## 4. The output contract — exactly seven keys

Every completed response carries exactly these, and no others:

```
gbif_id  ncbi_taxid  recognition  recognition_candidates
recognition_provenance  species  species_id
```

Asserted as **set equality**, so an added key fails as loudly as a missing one — across
`identified`, `uncertain`, `not_identified`, resumed, degraded-taxonomy and degraded-LLM
branches alike.

A failure instead returns `{"error_code": ..., "error": ...}` — a fixed, data-free message
derived from the code alone.

`recognition_provenance` reports what actually ran: the provider names and modes,
`gbif_mode`/`ncbi_mode` derived from the per-species taxonomy report, `taxonomy_executed`,
`taxonomy_degraded`, `reasoning_llm_calls`, `plan_source`, `explanation_source`, and
`score_is_probability: false`.

---

## 5. Security and resilience

| Property | State |
| --- | --- |
| GPT calls per request | **Maximum 2**, budget fresh per request |
| `store=False` on every Azure request | Yes, asserted on every recorded call |
| SDK retries | **0** |
| BioCLIP deadline and job cancellation | Yes |
| Structured failures | Every path, including the unexpected-exception path |
| Unexpected internal errors | Fixed safe message, `INTERNAL_ERROR`, correlation id logged |
| HTTP 500 | Never — the boundary always returns the shared schema |
| Dependency logging | `httpx`, `httpcore`, `urllib3`, `openai` pinned to `WARNING` at service startup |
| Secret / image / raw-error leakage | None — verified with canaries across logs, exceptions, serialized state, API responses, provenance and failure outputs |
| Concurrency | 8 concurrent requests share no state; images, instructions, taxonomy reports, context and provenance never cross |
| Prompt injection | Cannot add, promote, reorder, rename or rescore a candidate, obtain a third GPT call, or extract a secret |

The image never reaches the reasoning model — asserted with a PNG `tEXt` canary. Provider
failures log the **exception type only**; a message could echo the request.

---

## 6. Tests and evidence

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

No offline test contacts Azure, the BioCLIP Space, GBIF or NCBI, and none sleeps — the
rate limiter's clock is injected.

### Phase 5 real-image evaluation

Seven Wikimedia Commons photographs (CC BY / CC BY-SA), six taxonomic groups, all
confirmed absent from the mock fixture. Images were held outside the repository and
deleted after use.

| Image | Expected | Observed Top-1 | Score | Decision |
| --- | --- | --- | --- | --- |
| `mammal_elephant` | *Loxodonta africana* | *Loxodonta africana* | 0.9364 | `identified` |
| `bird_eagle` | *Haliaeetus leucocephalus* | *Haliaeetus leucocephalus* | 0.9511 | `identified` |
| `reptile_iguana` | *Iguana iguana* | *Iguana iguana* | 0.9343 | `identified` |
| `shark_white` | *Carcharodon carcharias* | *Carcharhinus fitzroyensis* | 0.2148 | `not_identified` |
| `arthropod_bee` | *Apis mellifera* | *Apis mellifera* | 0.4601 | `uncertain` |
| `mammal_cheetah` | *Acinonyx jubatus* | *Acinonyx jubatus* | 0.9070 | `identified` |
| `ambiguous_coyote` | *Canis latrans* | *Canis lycaon* | 0.4221 | `not_identified` |

Top-1 correct 5/7; expected species in Top-K 6/7. Latency 5.9–10.1 s, mean 7.4 s.

> **This is an integration observation on seven images, not a measurement of scientific
> accuracy, and no threshold was tuned on it.**

**The negative and inconclusive cases matter as much as the positives:**

- `arthropod_bee` — the **correct** species at Top-1, reported `uncertain` because 0.4601
  clears the 0.45 floor but falls far short of the 0.75 identification threshold. The agent
  declines to claim what it cannot support.
- `ambiguous_coyote` — *Canis lycaon* 0.4221 against *Canis latrans* 0.3866, a margin of
  0.0355 between two congeners. `not_identified` is the honest outcome; separating those
  two is exactly what the margin rule exists to refuse.
- `shark_white` — the expected species was absent from Top-K entirely. A classifier
  limitation on this image, not a threshold problem.

### Thresholds

`IDENTIFIED_MIN_SCORE = 0.75`, `IDENTIFIED_MIN_MARGIN = 0.08`,
`UNCERTAIN_MIN_SCORE = 0.45` — **unchanged throughout Sprint 3**. The observed remote
BioCLIP-2 distribution is strongly bimodal (0.91–0.95 for unambiguous subjects, 0.21–0.46
otherwise), which this pair separates cleanly. Any future calibration must come from a
properly sized, independently sourced evaluation set.

---

## 7. Known limitations

1. **No Global Orchestrator integration.** The agent emits `needs_agent`; nothing routes it
   yet.
2. **Seven evaluation images.** Enough to certify integration, far too few for an accuracy
   claim.
3. **Per-stage latency is not instrumented** — only end-to-end wall clock. The equivalent
   fields are on the response in `recognition_provenance`.
4. **Azure deployment rate limiting.** `umbrella-gpt5-mini` (swedencentral) returns HTTP
   429 under back-to-back load; the agent falls back deterministically and reports it
   honestly. A capacity note for production planning, not a code defect.
5. **Text alignment is inert in real mode.** `RealTaxonomyProvider` ships no name catalogue
   by design, so user text carries no name signal — a stronger safety property, but it
   means agreement/conflict are certified against the mock-catalogue provider.
6. **The dependency logging policy applies at the HTTP entry point.** Code importing
   `RecognitionAgent` directly keeps its libraries' own levels.
7. **The NCBI rate limiter is per-provider**, so multiple agent instances would each hold
   their own budget. Not a defect at one replica; relevant before scaling out.
8. **Outage scenarios are injected** into our own transports. No external service was
   deliberately failed.

---

## 8. Running it

```
python -m uvicorn backend.agents.multimodal_recognition_agent.api:app --port 8005
```

Single endpoint `POST /execute`, registered as `Multimodal` on port 8005 in
`backend/registry.py`. It always answers with an `AgentResult` — never an `HTTPException`,
because the orchestrator's router expects one schema every time.

**Demonstration:**

```
RECOGNITION_LIVE_SMOKE=1 python -m backend.agents.multimodal_recognition_agent.demo_sprint3 \
    --image path/to/animal.jpg --instruction "Identify the animal in this image."
```

**Opt-in smoke scripts:** `smoke_test_azure.py`, `smoke_test_taxonomy.py`. All three refuse
to run without `RECOGNITION_LIVE_SMOKE=1`, so no import or stray invocation reaches a live
service.

Configuration lives in the agent's git-ignored `.env`; `.env.example` documents every
variable **by name only**.

---

## 9. Superseded claims

The following statements appeared in Sprint 2 documentation and are **no longer true**:

| Obsolete claim | Reality |
| --- | --- |
| "BioCLIP-2 execution is mocked" | Real remote inference on `imageomics/bioclip-2-demo` |
| "GBIF and NCBI are mocked" | Live GBIF Species API and NCBI Entrez |
| "Non-mock modes are unimplemented" | `remote` and `real` are implemented and are the live configuration |
| "Recognition matches images by exact SHA-256 fixture lookup" | That is the **mock** oracle only; the remote classifier accepts arbitrary photographs |
| "Global Orchestrator integration exists" | It does not |
| "The Sprint 2 classifier did not return a taxon…" (clarification question) | Replaced with provider-neutral wording describing the evidence (P8-F1) |
