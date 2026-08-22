# Sprint 3 Completion Plan — Multimodal Species Recognition Agent

> ## FINAL STATUS — recorded at Phase 8 closure
>
> | Phase | Status |
> |---|---|
> | Phase 0 — branch and baseline | **PASS** |
> | Phase 1 — configuration and provider-selection hardening | **PASS** |
> | Phase 2 — GPT-5 mini as the real reasoning brain | **PASS** |
> | Phase 3 — real BioCLIP-2 inference (remote Space) | **PASS** |
> | Phase 4 — live GBIF and NCBI taxonomy | **PASS** |
> | Phase 5 — complete real standalone flow | **PASS** (after the F1–F3 provenance corrections) |
> | Phase 6 — revalidate the integrated Global Orchestrator chain | **N/A — The phase was defined as revalidation of an existing integration, but that prerequisite does not exist.** |
> | Phase 7 — resilience, security, privacy, operational quality | **PASS** (after the P7-F1–F3 corrections) |
> | Phase 8 — closure, documentation, demo, PR readiness | **PASS — Standalone Recognition Agent Sprint 3 complete** (after the P8-F1 correction) — see `docs/SPRINT_3_FINAL_REPORT.md` |
>
> **Correction to Section 1 below.** This plan opens by stating the agent is
> "already implemented, tested, integrated with the Global Orchestrator, and
> validated". The first three are true. **The integration claim is not.**
> Repository reality is authoritative: the Recognition Agent is standalone, it
> emits `needs_agent` capability hints, and nothing routes them. Every statement
> in this plan that assumes an existing orchestrator integration is corrected in
> the traceability matrix in Section 8.
>
> **Two distinct scopes must not be conflated:**
>
> 1. **Standalone Recognition Agent Sprint 3 scope** — **complete**.
> 2. **Full-platform orchestration integration** — *not started*, and owned by
>    the orchestrator/integration team. It is future external work.
>
> The complete platform integration is **not** finished.

## 1. Purpose

This document is the execution plan for completing Sprint 3 for the Multimodal Species Recognition Agent owned by Group D.

The agent workflow is already implemented, tested, integrated with the Global Orchestrator, and validated. Sprint 3 work must therefore **productionize the existing implementation**, not redesign it.

The required final state is:

> A real Recognition Agent that receives one animal image plus a non-empty instruction, uses GPT-5 mini as its reasoning brain, obtains real visual candidates from BioCLIP-2, enriches those candidates through live GBIF and NCBI taxonomy services, preserves all deterministic safety gates, and returns the validated `AgentResult` contract.
>
> *Corrected at Phase 8 closure: this sentence originally ended "…through the already-integrated Global Orchestrator". That integration does not exist. The agent returns the validated contract from its own `POST /execute` endpoint; consuming it is future external work.*

Sprint 3 presentation deadline: **23 August 2026**.

---

## 2. Authority and conflict resolution

Use the following order of authority throughout the work:

1. **The current code on the latest `main` after the completed orchestrator integration and validation.**
2. **`RECOGNITION_AGENT_FINAL_REFERENCE.md`**, for the implemented Recognition architecture and its protected contracts.
3. **`sprint 3(1).docx`**, only for Sprint 3 objectives and acceptance expectations.
4. Older documents, including `Agent details.docx`, are historical and must not override the current code or the final Markdown reference.

If an older document conflicts with the current code, **the current code wins**. Do not merge old and new architectures.

---

## 3. Fixed architectural decisions — do not reopen

The following decisions are final for this Sprint:

- GPT-5 mini is the Recognition Agent's reasoning brain.
- GPT-5 mini performs planning and grounded explanation, with **a maximum of two calls per request**.
- GPT-5 mini does not classify the image, invent species, alter scores, reorder candidates, change confidence decisions, or invent biological identifiers.
- BioCLIP-2 is the selected pre-trained visual-recognition model.
- The team will **not train or fine-tune a model**.
- The implemented workflow order remains:
  `validate → plan → classify → confidence → taxonomy → explain → delegate → finalize`.
- Confidence is evaluated before taxonomy. GBIF and NCBI annotate candidates; they never create, reorder, promote, or score them.
- The existing LangGraph workflow, validation logic, state model, ranking logic, confidence gate, text-image alignment, grounding checks, delegation mechanism, and finalization path remain authoritative.
- There is **no Qdrant**.
- There is **no vector database, embedding retrieval, nearest-neighbour search, reference collection, or similarity-search pipeline**.
- No Qdrant, vector, retrieval, embedding, collection, similarity, or nearest-neighbour component may be introduced.
- The agent continues to require **one image plus one non-empty text instruction**.
- The agent continues to communicate with other capabilities only through `needs_agent`; it never calls another agent directly.
- The Recognition Agent is already connected directly to the Global Orchestrator. Do not create a Group D sub-orchestrator.
- No frontend work is part of this plan.
- Port `8005`, the `Multimodal` registration, `POST /execute`, the fixed `recognition_image` input key, and the shared `AgentRequest` / `AgentResult` / `AgentStatus` contract remain unchanged.
- A `completed` result must preserve exactly these seven top-level output keys:
  `recognition`, `species`, `species_id`, `gbif_id`, `ncbi_taxid`, `recognition_candidates`, `recognition_provenance`.
- `uncertain` and `not_identified` remain valid `completed` outcomes.
- Insufficient visual evidence continues to produce `request_better_image=true` only through the existing controlled path.
- All image bytes, Base64 data, secrets, and private context must remain excluded from prompts, outputs, logs, exceptions, and persisted artifacts.
- No implementation file outside the Recognition Agent folder may be changed unless a failing end-to-end integration test proves a shared-code defect and the project owner explicitly authorizes that separate change.

---

## 4. Sprint 3 scope classification

### 4.1 Already completed — preserve and revalidate, do not rebuild

- Agent objective and responsibilities.
- LangGraph framework selection and implementation.
- Eight-node reasoning workflow and single finalization path.
- GPT-5 mini planner/explainer adapter and two-call budget.
- System prompts and structured-output validation.
- Input and output schemas.
- Image and text validation.
- Ranking, Top-K enforcement, confidence decision, text-image consistency, explanation grounding, and deterministic fallbacks.
- Controlled invalid-input and provider-failure handling.
- `completed`, `needs_agent`, and `failed` result handling.
- Delegation hints and resume protection.
- Recognition-side Global Orchestrator contract.
- Direct Global Orchestrator integration completed on the latest `main`.
- Existing automated test suite and validated implementation baseline.

### 4.2 Remaining Sprint 3 work

- Re-establish the validated baseline on a new branch created from the latest integrated `main`.
- Make production runtime configuration internally consistent.
- Verify GPT-5 mini against the real Azure deployment on the new branch.
- Replace the BioCLIP-2 fixture implementation with a real BioCLIP-2 provider behind the existing classifier protocol.
- Replace fixture-backed GBIF and NCBI providers with live adapters behind the existing taxonomy protocol.
- Preserve offline tests while adding opt-in real-provider integration tests.
- Test `Agent → real services → AgentResult` with real images.
- Revalidate `Global Orchestrator → Recognition → Result → Global Orchestrator` after real-provider integration.
- Validate resilience, privacy, security, provenance, and dependency-failure behaviour.
- Produce Sprint 3 evidence, documentation, demo scenarios, and a clean pull request.

### 4.3 Not applicable to this agent

- Model training or fine-tuning.
- Training-data preparation.
- Saving a custom trained model.
- Qdrant or any retrieval pipeline.
- Similarity search.
- Frontend implementation.
- Creation of a sub-orchestrator.
- Direct agent-to-agent communication.
- Redesign of the validated workflow.
- Deployment/containerization unless separately requested after Sprint 3 completion.

---

## 5. Execution rules for Claude

Claude must execute the phases **strictly in order**. Each phase ends with a hard gate.

For every phase, Claude must:

1. Read this plan completely.
2. Inspect the latest code before proposing changes.
3. Report the current branch, commit SHA, and clean/dirty state.
4. Touch only files necessary for the current phase.
5. Preserve unrelated user and team changes.
6. Add or update tests before declaring the phase complete.
7. Run the phase-specific tests and the complete offline Recognition test suite.
8. Show an exact file-change list and concise diff summary.
9. Report test counts, failures, warnings, and skipped real-service tests separately.
10. Stop after the phase report and wait for explicit approval before starting the next phase.

Claude must not:

- invent a new architecture;
- revive a superseded design;
- modify the LangGraph sequence without a proven defect and explicit approval;
- add Qdrant, vector storage, embeddings, retrieval, collections, similarity, or nearest-neighbour code;
- add a third LLM call, retry an LLM call, or weaken the failed-planner rule;
- let GPT-5 mini change scientific results;
- let taxonomy services influence classification or confidence;
- rename cross-agent input/output fields;
- commit secrets, `.env`, model weights, datasets, raw test photographs, virtual environments, caches, or generated logs;
- silently fall back to a mock while reporting the provider as real;
- edit shared orchestrator or other-agent code during an agent phase;
- commit or push unless the user explicitly requests it.

---

## 6. Credential and `.env` protocol

When a phase needs credentials or environment values, Claude must not guess them and must not place secret values in source code.

Claude must first output exactly this block and then stop:

```text
ENVIRONMENT INPUT REQUIRED

Phase: <phase number and name>
.env path: backend/agents/multimodal_recognition_agent/.env

Required variables:
- VARIABLE_NAME=<placeholder>
  Purpose: <purpose>
  Required/optional: <required or optional>
  Where to obtain it: <provider portal or official source>
  Expected format: <format, never a real secret>

No secret has been written or displayed.
Add the values locally, confirm completion, and I will continue the phase.
```

Rules:

- Claude provides variable **names, placeholders, purpose, format, and acquisition source**, never a real secret.
- Leith adds the real values locally to `.env`.
- `.env.example` contains placeholders only.
- `.env` must remain ignored and untracked.
- Before continuing, Claude may verify that required variables are present, but must never print their values.
- Logs and exceptions may name a missing variable but must never reveal its value.
- If a service requires no API key, Claude must state that clearly instead of inventing one.

---

# 7. Sequential implementation phases

## Phase 0 — Create the Sprint 3 branch and certify the merged baseline

### Objective

Start from the latest validated `main` containing the completed Recognition integration, without carrying unmerged work from the old feature branch.

### Claude actions

1. Fetch repository metadata without changing code.
2. Confirm the working tree is clean. If it is not clean, stop and report every changed file; do not discard or overwrite anything.
3. Switch to `main` and update it using the repository's approved non-destructive procedure.
4. Confirm that the completed Global Orchestrator integration is present.
5. Record the `main` commit SHA.
6. Create and switch to:
   `group-d-recognition-sprint3-production`
7. Confirm that the new branch points exactly to the recorded `main` SHA.
8. Inspect the Recognition directory, shared schema, registry entry, and current integration tests in read-only mode.
9. Run the complete existing offline Recognition test suite.
10. Run any existing orchestrator/Recognition integration tests without modifying shared code.

### Required evidence

- Starting `main` SHA.
- New branch name and SHA.
- Clean status before and after branch creation.
- Recognition registration on port `8005`.
- Existing workflow node sequence.
- Existing test command and exact result.
- Confirmation that no implementation file changed.

### Acceptance gate

- New branch is created from the latest integrated `main`.
- Working tree is clean.
- The pre-existing test baseline passes.
- No code is modified.

### STOP condition

If the baseline fails, stop. Classify each failure as environment, dependency, Recognition regression, or shared integration regression. Do not begin production-provider work until the baseline is restored or explicitly accepted.

---

## Phase 1 — Production configuration and provider-selection hardening

### Objective

Prepare the existing provider boundaries for real execution while preserving the current offline test path and all protected contracts.

### Scope

This phase may change only Recognition-owned configuration, provider factories, environment examples, targeted tests, and the Azure smoke script. It must not yet implement BioCLIP-2 or taxonomy network logic.

### Claude actions

1. Re-audit configuration and provider factories against the current code, not historical documents.
2. Make `RECOGNITION_LLM_TIMEOUT_SECONDS` effective or remove the duplicate ineffective setting in favour of the single effective timeout, without changing the default runtime behaviour unexpectedly.
3. Correct `RECOGNITION_LLM_MAX_CALLS_PER_REQUEST=0` so it does not silently become `2`. The global maximum must remain `2`.
4. Correct the Azure smoke script so it uses the Recognition Agent's own `.env` location.
5. Ensure provider modes are explicit and validated. Unknown modes must fail loudly at startup.
6. Prepare explicit real-provider mode names for the existing interfaces; do not implement their internals in this phase.
7. Ensure tests can inject providers directly and do not need network or credentials.
8. Keep mock/fake implementations available for offline unit tests, but ensure production mode can never silently select them.
9. Ensure provenance always reports the provider and mode that actually executed.
10. Update `.env.example` with placeholders and explanatory comments only.

### Required tests

- Valid and invalid provider-mode selection.
- `0`, `1`, `2`, negative, malformed, and greater-than-2 call-budget values.
- Effective timeout propagation.
- Correct `.env` path used by the smoke script.
- Missing real-provider configuration fails clearly without exposing values.
- Production mode cannot silently instantiate a mock/fake provider.
- Full offline Recognition suite remains green.

### Acceptance gate

- Configuration has one clear source per setting.
- Maximum GPT-5 mini calls remains two.
- Provider mode and provenance cannot disagree.
- No real credential is required for the offline suite.
- No workflow, schema, output key, or orchestrator file changes.

### Deliverable

One focused configuration-hardening diff plus tests and a Phase 1 report.

---

## Phase 2 — Verify GPT-5 mini as the real reasoning brain

### Objective

Confirm the already-implemented Azure GPT-5 mini adapter works from the new Sprint 3 branch with real credentials and preserves all reasoning safeguards.

### Credentials

This phase requires Azure/OpenAI environment values. Claude must use the credential protocol in Section 6 and stop until Leith confirms that `.env` has been populated.

Expected configuration categories include the existing Azure endpoint/base URL, API key, deployment name, API version if the current adapter requires it, reasoning effort, output-token limit, and timeout. Claude must derive the **exact variable names from the current code** and must not rename them unnecessarily.

### Claude actions after credentials are added

1. Verify required variables are present without printing values.
2. Run the complete offline suite first.
3. Run one controlled live smoke request that exercises:
   - planner call;
   - grounded explanation call;
   - configured deployment;
   - structured JSON parsing;
   - two-call accounting.
4. Run controlled cases for:
   - valid plan and explanation;
   - invalid/off-contract plan falling back safely;
   - grounded-explanation rejection and deterministic replacement;
   - provider timeout or unavailable-provider simulation;
   - failed planner forfeiting the explanation call.
5. Confirm no image bytes, Base64, shared context, secret, or taxonomy identifier enters the GPT payload except safe text metadata explicitly allowed by the current code.
6. Record only redacted evidence.

### Acceptance gate

- `plan_source=llm` on the successful live request.
- `explanation_source=llm` on the successful live request.
- Total real GPT-5 mini calls are exactly `2` for the successful path.
- A failed planner spends one call and skips the explainer.
- No scientific candidate, score, rank, confidence decision, GBIF id, or NCBI id is produced or changed by the LLM.
- No secret or image payload appears in logs, prompts recorded for tests, reports, or responses.
- Offline tests still pass without Azure credentials.

### Deliverable

A redacted Azure smoke-test report containing deployment alias, status, call count, sources used, timing, and pass/fail results—never the endpoint or key value.

---

## Phase 3 — Integrate real BioCLIP-2 inference (remote Hugging Face Space)

> **Amended.** The local-inference design is cancelled. The exception for a local
> 2.66 GB TreeOfLife text-embedding artifact was **refused**, so BioCLIP-2 now runs
> remotely on the official public Space. Phases 0-2 remain complete; Phases 5-8 are
> structurally unchanged.

### Objective

Replace fixture-based visual classification in production with **real BioCLIP-2
inference executed on the official public Hugging Face Space**, through the
existing `BioCLIP2Classifier` contract.

### Cancelled — marked N/A

The following are **N/A** and must not be implemented, installed or downloaded:

- local `pybioclip`; local `open_clip`; local PyTorch inference;
- downloading BioCLIP-2 weights;
- downloading or caching TreeOfLife text embeddings;
- `local-dir:` snapshots; `HF_HUB_OFFLINE`;
- the 4.46 GB download; the 6 GB RAM / 8 GB commit preflight;
- the Windows `torch.compile` workaround;
- any local model or embedding cache;
- adding `torch`, `torchvision`, `open_clip_torch` or `pybioclip`.

### Remote design

- Space ID: `imageomics/bioclip-2-demo`
- Base URL: `https://imageomics-bioclip-2-demo.hf.space`
- Model target: `imageomics/bioclip-2`
- Mode: remote open-domain species classification
- Taxonomic rank sent to the Space: always `Species`
- Expected maximum result: Top 5
- No API key is expected for the public Space. If a mandatory credential is
  discovered, stop and report it without requesting or printing its value.

The external Space is the **only** source of visual species candidates. GPT-5 mini,
GBIF, NCBI and user text must never create, replace, promote or rescore a candidate.

### Non-negotiable design constraints

- Use the selected pre-trained BioCLIP-2 model; do not train or fine-tune.
- Implement the provider behind the existing `classify(image, top_k)` interface.
- Do not edit the LangGraph node sequence to accommodate the provider.
- BioCLIP-2 remains the **only source of species candidates**.
- No Qdrant, vector database, persisted embedding, reference collection, similarity
  lookup, or nearest-neighbour logic.
- The remote client must be created lazily: ordinary imports and offline tests
  perform no network request.
- No model weight, embedding or cache artifact may be downloaded or committed.

### Claude actions

1. Add `RemoteBioCLIP2Provider` implementing the current classifier protocol.
2. Use the already-validated image bytes without changing input validation.
3. Create the remote client lazily; always request the `Species` rank.
4. Map returned predictions into the existing typed prediction objects.
5. Enforce finite scores, allowed score range, descending order, distinct taxa, and
   the configured Top-K through the existing ranking contract.
6. Keep scores labelled as ranking/classification values, not calibrated probabilities.
7. Never fabricate candidates beyond what the Space returns (at most five).
8. Ignore the endpoint's sample image and HTML link for scientific purposes; never
   parse the displayed GBIF HTML link as the production taxonomy source, and do not
   retain downloaded sample files beyond the call lifecycle.
9. Apply an explicit bounded timeout; add no unbounded retry loop.
10. Map timeout, queue failure, sleeping/unavailable Space, malformed payload,
    API-contract change and client failure onto the existing controlled
    `CLASSIFICATION_UNAVAILABLE` path; never return an unhandled HTTP 500.
11. Production remote mode must never silently fall back to `MockBioCLIP2Provider`.
12. Keep the fixture-backed classifier only as an injected offline test double.
13. Update provenance so real execution no longer reports a mock provider or mode.
14. Add only the smallest compatible pinned client dependency required for the call.
15. Add focused offline tests plus one opt-in live smoke test.

### Required tests

- Provider protocol conformance.
- Remote provider selected only in remote production mode.
- No network call during import or construction.
- Image and `Species` rank mapped correctly.
- Valid Top-5 response mapping, and requested Top-K slicing without fabrication.
- Descending ordering, deduplication, non-finite/malformed scores, malformed labels.
- Malformed payload, timeout, queue rejection, sleeping Space, API-contract change.
- Controlled `CLASSIFICATION_UNAVAILABLE`; no silent mock fallback.
- Arbitrary valid photographs are sent to the provider rather than looked up by SHA-256.
- Mock mode unchanged; provenance follows the provider that ran.
- No local ML dependency, weight, cache, embedding, Qdrant or similarity symbol introduced.
- Full offline suite passes.

### Acceptance gate

- At least one previously unregistered public animal image produces real BioCLIP-2
  candidates through the remote Space.
- Re-encoding a photograph does not depend on an exact SHA-256 fixture match.
- `recognition_provider`, `recognition_mode` and model version correctly report
  remote execution.
- No local model/embedding artifact is downloaded.
- Existing confidence and safety logic consumes the provider output without workflow
  changes.
- No scientific accuracy claim is made yet; that is evaluated in Phase 5.

### Deliverable

`RemoteBioCLIP2Provider`, one pinned client dependency, tests, and a redacted remote
inference evidence report.

---

## Phase 4 — Integrate live GBIF and NCBI taxonomy services

### Objective

Replace fixture-backed taxonomy enrichment in production with live GBIF and NCBI lookups through the existing taxonomy interfaces.

> **Amended.** GBIF validates BioCLIP's existing Top 5 directly through the official
> GBIF API. NCBI Taxonomy is looked up only for the final selected candidate. The
> complete structured taxonomy is returned inside the existing nested output
> contract, and the seven top-level output keys are unchanged.

### Fixed behaviour

- Taxonomy runs only after classification and confidence.
- Taxonomy annotates existing candidates only.
- **GBIF validates the existing BioCLIP Top 5 directly via the official GBIF API.**
- **NCBI Taxonomy is queried only for the final selected candidate.**
- **The complete structured taxonomy - kingdom, phylum, class, order, family, genus,
  species - is returned inside the existing nested output contract.** The seven
  top-level output keys do not change.
- **GBIF and NCBI never create, reorder, promote or rescore a candidate.**
- By default GPT-5 mini may present the complete validated taxonomy. If the user asks
  for a single rank, it presents that existing validated field only. GPT-5 mini never
  generates or modifies taxonomy.
- **No new blurry-image detection or image-quality subsystem will be added.** Existing
  validation, confidence and `request_better_image` behaviour remain unchanged.
- Taxonomy must not add, delete, reorder, rescore, or promote a candidate.
- A missing identifier stays `null`.
- A service outage degrades taxonomy information but does not crash an otherwise valid recognition request.
- Inconsistent or ambiguous records must be reported, not silently coerced.
- IUCN is out of scope for this agent.

### API checkpoint

Claude must inspect the current taxonomy protocol, then list the exact live-service requirements before implementation.

- For GBIF, state whether the selected official endpoint requires an API key.
- For NCBI, state which identification/contact variables and optional API key are required by the official client policy.
- If any value is required or recommended, use the Section 6 environment block and stop so Leith can populate `.env`.
- Never hard-code service keys, email addresses, endpoints containing secrets, or personal data.

### Claude actions

1. Add live GBIF and NCBI providers implementing the current lookup contract.
2. Use official APIs/SDKs and explicit, bounded timeouts.
3. Use deterministic exact scientific-name handling and explicit synonym/canonical-name reporting supported by the service response.
4. Validate all external responses before constructing internal taxonomy objects.
5. Preserve `available`, `matched`, `identifier`, and `inconsistent_record` distinctions.
6. Preserve missing identifiers as `null`.
7. Contain timeouts, rate limits, malformed payloads, partial responses, and service errors through the current degradation path.
8. Ensure no external error body or request value leaks into user output or logs.
9. Ensure real production mode cannot silently return fixture data.
10. Update provenance and explanation wording so real providers are never described as mocked and degraded providers are disclosed accurately.

### Required tests

- GBIF exact match, synonym/canonical response, no match, missing id, malformed response, timeout, rate limit, and unavailable service.
- NCBI exact match, no match, missing taxid, malformed response, timeout, rate limit, and unavailable service.
- One service succeeds while the other fails.
- Neither service can create or reorder candidates.
- Identifiers are never borrowed from another species.
- Production mode cannot silently use fixture-backed taxonomy.
- Offline tests use injected HTTP/client doubles and require no network.
- Opt-in live smoke tests are clearly marked and skipped when credentials/network are absent.
- Full offline suite passes.

### Acceptance gate

- A real BioCLIP-2 candidate can be enriched with a live GBIF id and/or NCBI taxid.
- Partial and unavailable-service cases complete safely with accurate degraded provenance.
- Missing ids remain `null`.
- Classification, ranking, and confidence are unchanged before and after taxonomy.

### Deliverable

Live GBIF and NCBI adapters, tests, placeholder-only environment documentation, and redacted live-lookup evidence.

---

## Phase 5 — Validate the complete real standalone agent flow

### Objective

Prove that the Recognition Agent works independently with its real providers before revalidating the Global Orchestrator chain.

### Flow under test

`AgentRequest → validation → GPT-5 mini planning → BioCLIP-2 inference → deterministic confidence → GBIF/NCBI enrichment → GPT-5 mini grounded explanation → delegation decision → AgentResult`

### Test-image policy

- Use a small, documented evaluation set of legally usable animal photographs.
- Do not commit raw images unless the repository policy explicitly permits it.
- Prefer local ignored test assets or stable public test references with licence metadata in the evaluation report.
- The evaluation set is for validation only; it is not training or fine-tuning data.
- Record expected taxon, source/licence, outcome, Top-K presence, and observed limitations.

### Required scenarios

1. Clear known species → `completed/identified`.
2. Visually ambiguous case → `completed/uncertain` when supported by real score behaviour.
3. Insufficient or unusable evidence → controlled `not_identified` or validation failure, as appropriate.
4. User text agrees with the top visual candidate.
5. User text conflicts with the visual candidate and cannot promote a result.
6. Valid scientific follow-up → `needs_agent` capability hint.
7. Resume context present → completes without delegating twice.
8. BioCLIP-2 unavailable → controlled failure, never 500.
9. GPT-5 mini unavailable → deterministic fallback according to the current workflow.
10. One or both taxonomy services unavailable → completed result with degraded taxonomy.
11. Corrupt, oversized, unsupported, or remote/path-based image → existing structured validation failure.
12. Malicious instruction/prompt injection → controlled plan and grounded output.

### Evaluation report

Claude must produce a machine-readable test result plus a concise Markdown report containing:

- provider versions/modes;
- image-set description and licence/source metadata;
- expected vs observed Top-1 and Top-K results;
- recognition decision;
- taxonomy availability;
- LLM call count and source;
- latency per stage;
- failure/degradation behaviour;
- no-leak verification;
- limitations and false/uncertain cases.

Do not change confidence thresholds merely to make the demonstration pass. If real score distributions show that thresholds require calibration, report the evidence and stop for explicit approval before changing them.

### Acceptance gate

- The complete real standalone flow works on previously unregistered images.
- All output objects conform to the existing schema and seven-key contract.
- Real providers are accurately disclosed in provenance.
- Required failure paths remain controlled.
- Offline suite remains fully green.
- Any threshold-calibration need is documented rather than guessed.

---

## Phase 6 — Revalidate the already-integrated Global Orchestrator chain

> **STATUS: N/A — The phase was defined as revalidation of an existing
> integration, but that prerequisite does not exist.** Recognition is not
> registered with, routed by, or invoked through the Global Orchestrator.
> There is no integration to revalidate. Nothing in this phase was run, and
> no orchestrator E2E result is claimed anywhere in the Sprint 3 evidence.

### Objective

Verify that replacing external mocks with real providers did not break the existing, already-completed direct Global Orchestrator integration.

This is a **verification phase**, not a new integration design phase.

### Protected boundary

- Do not create a sub-orchestrator.
- Do not change the direct Global Orchestrator architecture.
- Do not add frontend work.
- Start with read-only inspection and tests.
- If a shared-code failure is discovered, record a minimal reproduction and stop for approval before modifying shared files.

### Required end-to-end scenarios

1. Global Orchestrator routes an animal recognition request to `Multimodal` on port `8005`.
2. The complete image-plus-text context reaches Recognition intact.
3. Recognition returns a valid `completed/identified`, `completed/uncertain`, or `completed/not_identified` result.
4. The Global Orchestrator validates and merges the seven output keys into shared context.
5. A scientific follow-up returns `needs_agent`; the Global Orchestrator resolves and calls the relevant registered specialist.
6. The specialist output is merged and Recognition resumes without an infinite delegation loop.
7. Invalid input returns a structured `failed` result and does not break communication.
8. GPT, BioCLIP-2, taxonomy, and specialist failure simulations do not produce broken routing or unhandled HTTP errors.
9. The final orchestrator response is produced without exposing Base64, secrets, internal prompts, or raw provider errors.

### Required verification

- Correct routing.
- Correct task delegation.
- Correct context passing.
- Correct tool/provider execution.
- Correct response format.
- Correct shared-context aggregation.
- Correct resume behaviour.
- Correct error handling.
- No direct Recognition-to-specialist call.
- No broken communication between components.

### Acceptance gate

- End-to-end tests pass against the already-merged architecture.
- No Recognition contract field is renamed or lost.
- No shared code is changed unless separately approved.
- The agent remains independently testable without running the full platform.

### Deliverable

A redacted end-to-end evidence report with request type, route, statuses, target capability, merged output keys, provider modes, timings, and pass/fail results.

---

## Phase 7 — Resilience, security, privacy, and operational quality

### Objective

Make the real-service agent professional and safe under normal failures without altering its scientific decision architecture.

### Claude actions

1. Verify bounded timeouts for every external dependency.
2. Verify no unbounded retry loop exists. Preserve the current no-retry rule for GPT-5 mini.
3. Verify startup behaviour for missing required production configuration.
4. Verify runtime degradation for optional/unavailable taxonomy services.
5. Verify model-load and inference errors are controlled.
6. Add structured, secret-safe operational logs or metrics only if the repository already has an approved pattern.
7. Record useful non-sensitive fields such as provider mode, stage, duration, decision, taxonomy degradation, plan rejection, explanation source, and LLM call count.
8. Never log instruction contents, Base64, image bytes, complete shared context, keys, endpoints containing secrets, or raw external error bodies.
9. Re-run static scope gates proving there is no Qdrant/vector/retrieval/similarity architecture.
10. Re-run secret, weight, raw-image, cache, and generated-file checks before closure.

### Required tests

- External timeout matrix.
- Malformed external payload matrix.
- Partial-service availability matrix.
- Concurrent independent requests share no state.
- Repeated requests do not accumulate the LLM budget.
- Image/context/secret leak prevention.
- Prompt-injection resistance.
- Structured error contract and never-500 guarantee.
- Real/fake/mock provenance accuracy.
- No forbidden dependency or source symbol.
- Complete offline suite.

### Acceptance gate

- All failures are either controlled `failed` outcomes or documented degradations.
- No secret or image data leak is observed.
- Operational evidence is sufficient to diagnose a stage failure without sensitive content.
- Protected architecture and contracts remain unchanged.

---

## Phase 8 — Sprint 3 closure, documentation, demo, and PR readiness

### Objective

Produce complete evidence that the Recognition Agent satisfies all applicable Sprint 3 outcomes.

### Claude actions

1. Run the complete offline Recognition suite and record the exact count.
2. Run opt-in live GPT-5 mini, BioCLIP-2, GBIF, and NCBI smoke/integration tests.
3. Run the real standalone scenario suite from Phase 5.
4. Run the Global Orchestrator end-to-end verification from Phase 6.
5. Inspect the final diff against the Phase 0 `main` SHA.
6. Confirm all changed implementation files belong to Recognition unless separately approved.
7. Confirm `.env`, secrets, raw images, weights, caches, test outputs, and local datasets are not tracked.
8. Update `RECOGNITION_AGENT_FINAL_REFERENCE.md` to describe the code that now exists:
   - real BioCLIP-2 provider;
   - live GBIF/NCBI providers;
   - verified Azure execution;
   - real provider selection and environment contract;
   - current tests and evidence;
   - remaining limitations;
   - no Qdrant or similarity architecture;
   - direct Global Orchestrator integration.
9. Update provider/provenance examples so they do not claim mocks in real mode.
10. Update the agent card only if its current managed-service descriptions inaccurately say that production providers are mocked. Preserve names and shared schemas.
11. Produce the Sprint 3 final report and a short demonstration script.
12. Prepare a clean PR summary, but do not commit, push, or open the PR without explicit user instruction.

### Required final evidence

- Branch name and base `main` SHA.
- Final commit/diff status.
- Exact changed-file list.
- Offline test command and count.
- Live smoke/integration results, redacted.
- Standalone real-image results.
- Global Orchestrator end-to-end results.
- Provider versions and modes.
- Environment variable names only.
- Security/no-leak result.
- Known limitations.
- Sprint 3 traceability matrix.

### Acceptance gate

Sprint 3 is complete for Recognition only when all applicable items in Section 8 are `PASS`, every exception is documented as `N/A` with the fixed justification in this plan, and no unresolved blocker remains.

---

# 8. Final Sprint 3 traceability matrix

| Sprint 3 requirement | Recognition status after this plan | Required evidence |
|---|---|---|
| Prepare training data | **N/A** | Pre-trained BioCLIP-2; no training/fine-tuning in scope |
| Train or fine-tune a model | **N/A** | Fixed architecture decision |
| Evaluate selected model | **PASS** | `docs/PHASE5_CERTIFICATION.md` real-image evaluation matrix |
| Configure model for inference | **PASS** | Remote `imageomics/bioclip-2-demo`, pinned revision, provider-mode tests |
| Integrate model into agent | **PASS** | `Agent → remote BioCLIP-2 → candidates`, verified live |
| Define when agent uses model | **Already PASS; revalidated** | Existing workflow and Phase 3 tests |
| Test Agent → Model → Result | **PASS** | Real standalone scenario suite, seven real images, live providers |
| Pass model result to orchestrator | **NOT IMPLEMENTED — future Global Orchestrator/integration-team task** | The agent returns the validated seven-key `AgentResult`; no orchestrator consumes it yet |
| Define agent objective/responsibilities | **Already PASS** | Current code and final reference |
| Configure agent framework | **Already PASS** | Compiled LangGraph eight-node graph |
| Connect LLM | **PASS** | Live Azure GPT-5 mini, two-call budget, `store=False`, no retry |
| Define prompts/instructions | **Already PASS; revalidated** | Planner/explainer prompts and guards |
| Implement agent tools | **PASS** | Real remote BioCLIP-2, live GBIF, live NCBI |
| Connect required knowledge/API sources | **PASS** | Live GBIF Species API and NCBI Entrez, verified live |
| Retrieval pipeline | **N/A** | No Qdrant/retrieval/similarity by final decision |
| Define input/output schemas | **Already PASS** | Preserved shared contracts |
| Implement reasoning workflow | **Already PASS** | Existing graph and tests |
| Handle invalid inputs/tool failures | **PASS** | Phase 7 timeout, malformed-payload and partial-availability matrices |
| Test agent independently | **PASS** | 1178 offline tests plus opt-in live smokes |
| Finalize workflow | **Already PASS** | Protected node sequence and finalizer |
| Integrate with orchestration layer | **NOT IMPLEMENTED — future Global Orchestrator/integration-team task** | Never implemented in this or any prior sprint; the opening claim in Section 1 was incorrect |
| Create Group D sub-orchestrator | **N/A** | Direct Global Orchestrator architecture is current |
| Implement delegation/context/results | **PARTIAL — agent side READY FOR FUTURE INTEGRATION** | The agent emits `needs_agent` with a capability hint and completes on resume when the helper key is present (verified live, standalone). Routing, context aggregation and specialist invocation are NOT implemented |
| Validate agent-orchestrator communication | **N/A — revalidation prerequisite absent** | No integration exists to validate |
| Frontend | **N/A** | Explicitly outside this plan |

---

# 9. Definition of Done

The Recognition Agent is Sprint 3 complete when all of the following are true:

- [ ] The work is based on a new branch from the latest integrated and validated `main`.
- [ ] The validated eight-node workflow is unchanged unless an explicitly approved defect required a minimal correction.
- [ ] GPT-5 mini is verified live as planner and grounded explainer.
- [ ] The two-call maximum and failed-planner rule are preserved.
- [ ] Production classification uses real BioCLIP-2 inference.
- [ ] No model is trained or fine-tuned.
- [ ] Production taxonomy uses live GBIF and NCBI integrations.
- [ ] BioCLIP-2 remains the only source of species candidates.
- [ ] Taxonomy remains annotation-only.
- [ ] No Qdrant, vector database, retrieval, embedding collection, similarity, or nearest-neighbour architecture exists.
- [ ] Mock/fake providers remain only as explicit offline test doubles and can never be silently used in production mode.
- [ ] Real and degraded execution are accurately represented in provenance.
- [ ] The standalone agent passes real-image, invalid-input, delegation, and dependency-failure scenarios.
- [ ] **NOT APPLICABLE TO THIS SPRINT — future external work.** The direct Global Orchestrator chain passes routing, context, aggregation, delegation, resume, and failure tests. *No such chain exists; this item was written on the incorrect premise that Recognition was already integrated. It is owned by the orchestrator/integration team and is explicitly out of the standalone Recognition Sprint 3 scope.*
- [ ] The seven output keys and all shared schemas remain compatible.
- [ ] No frontend or sub-orchestrator is added.
- [ ] All offline tests pass without network, credentials, model downloads, or external services.
- [ ] Opt-in live integration tests pass with local `.env` values.
- [ ] No secret, Base64, image bytes, raw context, raw provider error, model weight, or local dataset is committed or leaked.
- [x] Documentation reflects the final code and contains no obsolete mock claims for real mode. *Finding P8-F1 — a stale "Sprint 2 classifier" string in `domain/confidence.py`, reachable in real mode — was corrected; the sentence is now provider-neutral and verified live.*
- [ ] Sprint 3 evidence and traceability are complete and presentation-ready.

---

# 10. Required phase-report template

Claude must use this format at the end of every phase:

```markdown
# Phase <N> Report — <Name>

## Result
PASS | BLOCKED | FAIL

## Baseline
- Branch:
- Commit SHA:
- Working tree before:

## Work completed
- ...

## Files changed
- `path`: reason

## Tests
| Command | Result | Count | Notes |
|---|---:|---:|---|
| ... | PASS/FAIL | ... | ... |

## Live services
| Provider | Mode | Result | Calls | Secrets redacted |
|---|---|---:|---:|---:|
| ... | ... | ... | ... | Yes |

## Protected-contract verification
- Workflow sequence unchanged: Yes/No
- Seven output keys unchanged: Yes/No
- Two-call ceiling preserved: Yes/No
- No Qdrant/vector/retrieval/similarity: Yes/No
- No shared files changed: Yes/No
- No secrets or raw images tracked: Yes/No

## Remaining issues
- ...

## Gate decision
- Phase acceptance criteria met: Yes/No
- Safe to start next phase: Yes/No

STOP — waiting for Leith's approval.
```

---

# 11. First command to give Claude

Use the following instruction to begin:

```text
Read SPRINT_3_COMPLETION_PLAN_RECOGNITION_AGENT.md completely and execute Phase 0 only.

The current code on the latest integrated main is authoritative. Preserve the validated Recognition architecture. Do not modify code in Phase 0. Do not introduce Qdrant, vectors, retrieval, embeddings, similarity search, a sub-orchestrator, frontend work, model training, or a new workflow. Report the baseline exactly, apply the Phase 0 acceptance gate, and stop for my approval.
```

