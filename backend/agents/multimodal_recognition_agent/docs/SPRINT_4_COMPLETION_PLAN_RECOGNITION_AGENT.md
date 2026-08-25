# Sprint 4 Completion Plan — Multimodal Species Recognition Agent

**Team:** Group D — Recognition Agent  
**Members:** Leith Saddouri and Chahd  
**Sprint deliverable:** 30 August 2026  
**Working method:** one shared branch, sequential implementation  
**Branch:** `group-d-recognition-sprint4`

---

## 1. Purpose

This document is the execution contract for completing Sprint 4 of the Multimodal Species
Recognition Agent.

It is written for an implementation LLM executing one phase at a time. Every phase defines:

- the exact owner;
- the exact work required;
- the permitted change boundary;
- the required tests and evidence;
- the conditions required to mark the phase `PASS`;
- the point where the LLM must stop.

The Recognition Agent was already implemented and integrated before Sprint 4. Sprint 4 must
evaluate and observe this validated agent. It must not redesign it.

---

## 2. Fixed branch and execution strategy

There is **one Sprint 4 branch only**:

```text
latest origin/main containing the integrated Recognition Agent
└── group-d-recognition-sprint4
    ├── Phase 0: inspect the new integrated baseline
    ├── Phases 1–3: Leith completes Agent Evaluation
    ├── Handoff: Leith → Chahd on the same branch
    ├── Phases 4–6: Chahd completes LangSmith
    └── Phase 7: Leith + Chahd certify Sprint 4
```

### Non-negotiable workflow

1. Create `group-d-recognition-sprint4` from the latest fetched `origin/main` commit where
   Recognition is already integrated.
2. Inspect and test that exact baseline before implementing Sprint 4.
3. Leith works first on the same branch.
4. Leith completes, tests, documents and commits all evaluation phases.
5. Leith hands Chahd the exact clean commit.
6. Chahd verifies Leith's handoff before changing anything.
7. Chahd completes, tests, documents and commits all LangSmith phases on the same branch.
8. Leith and Chahd run the final Sprint 4 certification together.

No parallel implementation branches are created. Chahd must not start while Leith's work is
uncommitted, failing or incomplete.

---

## 3. Sources of truth

Use this authority order whenever sources disagree:

1. Executable code on the exact `origin/main` baseline selected in Phase 0.
2. Tests executed from that baseline and later Sprint 4 commits.
3. `RECOGNITION_AGENT_POST_SPRINT3_REFERENCE.md` as the previous validated reference.
4. This Sprint 4 completion plan.
5. Older Sprint reports only for historical context.

The Phase 0 inspection is required because `main` contains the integrated agent and may be
newer than the post-Sprint-3 reference.

---

## 4. Sprint 4 applicability

| Sprint 4 task | Recognition decision | Owner |
| --- | --- | --- |
| Agent Evaluation | **REQUIRED** | **Leith** |
| RAG Evaluation / RAGAS | **NOT APPLICABLE** | None |
| Agent Traceability with LangSmith | **REQUIRED** | **Chahd** |
| Explainable AI for Trained Models | **NOT APPLICABLE** | None |

### RAG/RAGAS justification

Recognition contains no RAG pipeline, document chunks, vector store, embeddings, Qdrant,
nearest-neighbour retrieval or similarity search. BioCLIP-2 performs image classification;
GBIF and NCBI are structured taxonomy API lookups. RAGAS must not be added.

### XAI justification

Group D did not train or fine-tune BioCLIP-2. The agent uses a pretrained remote model and
does not access internal attention maps or features. SHAP, feature importance and attention
explanations must not be invented.

Candidate rankings, non-probability scores, confidence margins, taxonomy evidence and
provenance may be reported as result evidence, but not as trained-model XAI.

---

## 5. Historical baseline to re-confirm

The post-Sprint-3 audit established:

| Item | Audited state |
| --- | --- |
| Recognition offline suite | 1196 passed, 0 failed, 0 skipped |
| Workflow | LangGraph `StateGraph`, 8 nodes, one final builder |
| Reasoning | GPT-5 mini, maximum 2 calls per request |
| Classification | BioCLIP-2 mock or real remote mode |
| Taxonomy | GBIF + NCBI mock or real mode |
| Code defaults | mock / mock / disabled |
| Output | exactly seven protected keys |
| Evaluation | only seven historical live images |
| Per-stage observability | not implemented |
| RAG / Qdrant | absent |

These are not automatically the Sprint 4 baseline. Phase 0 must establish the exact state of
the latest integrated `main` and record new test counts.

The earlier `manual_test_ui/` belongs to a separate working branch and is outside Sprint 4.
It must not be deleted, modified or silently carried into the new branch.

---

## 6. Protected contracts

No phase may break these rules:

1. Input remains one non-empty instruction plus one image under
   `context["recognition_image"]`.
2. Shared `AgentRequest`, `AgentResult` and `AgentStatus` compatibility remains unchanged.
3. Recognition output contains exactly:
   `species`, `species_id`, `gbif_id`, `ncbi_taxid`, `recognition`,
   `recognition_candidates`, `recognition_provenance`.
4. `finalize` remains the only `AgentResult` builder.
5. BioCLIP-2 remains the only source of species candidates.
6. GPT-5 mini, text, GBIF and NCBI cannot add, promote, reorder or rescore a species.
7. The confidence gate remains deterministic and closes before taxonomy.
8. Text may downgrade but never promote a decision.
9. `score_is_probability` remains `false`.
10. `mock_verified` remains different from `verified`.
11. Missing taxonomy identifiers remain `null`.
12. Provenance describes what actually ran.
13. Maximum agent LLM calls remains 2 per request.
14. Retries remain zero at every external boundary.
15. Production modes never silently use mocks.
16. External calls remain bounded by deadlines.
17. Image bytes, Base64, full context, credentials and `.env` values never enter logs,
    prompts, traces, reports, exceptions or responses.
18. `uncertain` and `not_identified` remain completed outcomes.
19. Recognition never calls another agent directly.
20. No RAG, Qdrant, vector, embedding or similarity code may enter Recognition.
21. LangSmith must not add an output key or change the public schema.

If a phase appears to require violating a protected contract, stop and report a blocker.

---

## 7. Global rules for the implementation LLM

For every phase:

1. Read this entire plan.
2. Read the latest reference and every affected source/test file.
3. Confirm branch, HEAD and working-tree status.
4. List exact files intended for change.
5. Confirm the changes stay inside the current phase.
6. Implement only the current phase.
7. Add focused tests.
8. Run focused tests.
9. Run the complete Recognition offline suite.
10. Run affected integration tests when the phase touches the integrated flow.
11. Report collected, passed, failed, skipped and warnings exactly.
12. Show `git diff --stat` and classify each changed file.
13. Produce the required phase report.
14. Stop. Do not begin the following phase automatically.

### Forbidden work

The LLM must not:

- create parallel Sprint 4 branches;
- start Chahd's phases before Leith's handoff is `PASS`;
- change unrelated agents;
- redesign the Recognition workflow;
- change confidence thresholds using the evaluation dataset;
- change prompts merely to improve scores;
- train or fine-tune a model;
- add RAG, RAGAS, Qdrant, embeddings or similarity search;
- add frontend, dashboard, deployment or containerisation work;
- fix unrelated technical debt;
- modify the separate manual test UI;
- commit `.env`, keys, images, traces, private paths, caches or virtual environments;
- run live services without the explicit live guard;
- label a mock execution as live;
- hide a failed or skipped evaluation case.

---

## 8. Credential and privacy policy

### Live Recognition providers

Live evaluation uses the existing configuration for:

- Azure GPT-5 mini;
- remote BioCLIP-2;
- real GBIF;
- real NCBI.

The LLM must derive exact variable names from current code and never print their values.

### LangSmith

The standard configuration names are:

- `LANGSMITH_TRACING`;
- `LANGSMITH_API_KEY`;
- `LANGSMITH_PROJECT`;
- `LANGSMITH_ENDPOINT` only when the account region requires it;
- `LANGSMITH_WORKSPACE_ID` only when the key requires a workspace.

Tracing disabled must need no key, account or network. Tracing explicitly enabled with
missing required configuration must fail clearly without exposing values.

### Trace privacy

Automatic trace inputs and outputs must be hidden or transformed. Only allowlisted metadata
may be recorded.

Never trace:

- image bytes, Base64 or data URLs;
- private filenames or photographs;
- raw credentials, headers or `.env` values;
- complete Recognition state;
- untrusted external trace headers;
- external exception bodies that may echo requests.

Safe metadata may include node names, durations, provider modes, candidate counts, scientific
labels, decision, status, fixed error code, LLM role/call count, fallback flag, taxonomy
availability and delegation capability.

---

# EXECUTION PHASES

## Phase 0 — Create and inspect the integrated Sprint 4 baseline

**Owner:** Leith  
**Branch:** `group-d-recognition-sprint4`  
**Type:** read-only baseline before Sprint 4 implementation

### Objective

Create the single Sprint 4 branch from the latest integrated `origin/main`, then establish
the exact current Recognition state before evaluation begins.

### Required actions

1. Record the current branch, HEAD and working-tree state.
2. Preserve the separate manual UI work before switching branches.
3. Fetch remote references without merging feature work.
4. Identify the latest `origin/main` commit.
5. Prove that this commit contains:
   - the Recognition Agent;
   - registry entry `Multimodal`;
   - port 8005;
   - current Global Orchestrator integration;
   - current image-transfer path.
6. Create `group-d-recognition-sprint4` directly from that exact commit.
7. Inspect current Recognition architecture, providers, configuration, contracts, tests and
   integration in read-only mode.
8. Run the complete Recognition offline suite.
9. Run all existing Orchestrator-to-Recognition integration tests available on `main`.
10. Record exact test counts and environment limitations.
11. Create a short Phase 0 baseline report inside Recognition documentation.

### Permitted changes

- branch creation;
- Phase 0 report;
- this plan if it is not already present on the branch.

No runtime implementation change is permitted.

### Required evidence

- exact `origin/main` commit;
- branch ancestry proof;
- Recognition integration evidence;
- directory and provider inventory;
- Recognition and integration test commands/results;
- working-tree status;
- comparison with the post-Sprint-3 reference.

### Phase 0 PASS gate

Phase 0 is `PASS` only when:

- the single Sprint 4 branch starts from the latest integrated `origin/main`;
- Recognition integration is confirmed from current code;
- all Recognition tests collected on the new baseline pass;
- existing integration tests pass, or an exact environment blocker is documented;
- the manual UI work is preserved outside the Sprint 4 branch;
- no runtime file changed;
- the exact baseline commit and test counts are recorded.

If the baseline fails, stop. Leith must not begin evaluation.

---

## Phase 1 — Evaluation criteria and dataset specification

**Owner:** Leith  
**Branch:** `group-d-recognition-sprint4`

### Objective

Define a reproducible evaluation set and ground truth for all applicable Sprint 4 evaluation
criteria. The dataset evaluates the agent; it is not training or tuning data.

### Required evaluation criteria

1. Correctness.
2. Relevance to the user query.
3. Task completion.
4. Agent/tool selection.
5. Response consistency.
6. Recognition-specific evidence:
   - Top-1 correctness;
   - expected species present in Top-5;
   - decision correctness;
   - GBIF/NCBI identifier correctness when a reference exists;
   - provenance truthfulness;
   - latency and controlled-failure rate.

### Required dataset

Create at least 30 documented cases:

| Category | Minimum |
| --- | ---: |
| Real animal-image recognition | 20 |
| Ambiguous, poor-quality or non-animal | 5 |
| Invalid-input or dependency-failure | 3 |
| Delegation and resume | 2 |

The 20 real cases must cover at least 10 species, with at least two distinct images per
species where feasible. Include both clear and challenging images.

### Manifest requirements

Each case must contain the fields required by its category:

- stable case id and category;
- instruction;
- local ignored asset/fixture reference;
- public source and licence evidence for real images;
- expected status and decision;
- expected scientific species for labeled cases;
- accepted Top-K labels only when scientifically justified;
- expected GBIF and NCBI identifiers when independently verified;
- expected delegation capability where applicable;
- ambiguity/failure notes;
- offline/live applicability.

Do not commit Base64, raw image bytes, private photographs or private local paths.

### Ground-truth policy

- Ground truth must come from documented source identification and independent review, not
  from the agent's current output.
- Taxonomy identifiers must be checked against official sources.
- Ambiguous cases must not be forced into exact species scoring.
- Non-animal and invalid cases use behavioural expectations, not invented species.
- The evaluation set must not be used to tune thresholds during Sprint 4.

### Required tests

- manifest schema;
- unique ids;
- required fields per category;
- source/licence present for every real image;
- no Base64/data URL/private path committed;
- valid scientific-name format;
- deterministic loading order;
- no silent skipped case.

### Phase 1 PASS gate

Phase 1 is `PASS` only when at least 30 cases satisfy the schema, every real image has legal
source metadata, ground truth was reviewed independently of agent output, all dataset tests
pass and the set is labelled as a bounded Sprint 4 benchmark rather than a general accuracy
dataset.

---

## Phase 2 — Evaluation runner and evaluators

**Owner:** Leith  
**Branch:** `group-d-recognition-sprint4`

### Objective

Implement a bounded runner that evaluates the actual Recognition Agent contract and computes
all required Sprint 4 criteria without changing runtime behaviour.

### Required execution modes

1. **Offline dry run:** injected providers, no credentials, network or sleeps.
2. **Live evaluation:** explicit opt-in guard, real configured providers, sequential
   execution, bounded deadlines and no retry.

Default mode must be offline. Normal tests must never contact live services.

### Deterministic evaluators

Implement evaluators for:

- output schema;
- status;
- decision;
- Top-1 match;
- Top-5 presence;
- GBIF identifier;
- NCBI taxid;
- delegation capability;
- task completion;
- provider provenance;
- tool/agent-selection rules;
- maximum two agent LLM calls when evidence is available;
- zero retry;
- controlled errors.

For relevance and explanation quality, use either:

1. a rubric-based LLM judge receiving only sanitized instruction, safe output and reference
   fields; or
2. a documented human rubric with recorded scores.

The judge must not decide biological correctness when deterministic ground truth exists.
Evaluation calls are outside the agent and must be counted separately from the agent's
two-call limit.

### Consistency protocol

Select at least five representative valid cases and execute each three times in the same
live configuration. Compare status, decision, primary species, candidate order and delegation
target. Do not require exact explanation wording.

### Required outputs

- one result for every case;
- machine-readable results;
- metrics by category and overall;
- latency summary;
- provider modes/model versions;
- failed-case list;
- separate agent/evaluator LLM call counts;
- Markdown report template or generator.

### Required tests

- evaluator unit tests using synthetic outputs;
- correct Top-1/Top-5 calculations;
- correct `uncertain` and `not_identified` handling;
- correct taxonomy-unavailable handling;
- preserved case ordering and ids;
- one recorded result for each failure;
- live mode refuses without explicit opt-in;
- offline mode performs no network;
- no retry;
- output redaction;
- complete offline dry run;
- complete Recognition offline suite.

### Phase 2 PASS gate

Phase 2 is `PASS` only when the offline runner evaluates every selected case, metrics are
tested, live execution is guarded, outputs contain no sensitive material, evaluation calls
are separated from agent calls and all Recognition tests pass.

No minimum accuracy target may be invented. Sprint 4 requires valid measurement and weakness
analysis, not an unsupported performance claim.

---

## Phase 3 — Live evaluation, analysis and Leith handoff

**Owner:** Leith  
**Branch:** `group-d-recognition-sprint4`

### Objective

Run the approved benchmark with real providers, produce the Agent Evaluation deliverables,
then hand a clean, validated commit to Chahd.

### Preflight

1. Record branch and commit.
2. Confirm live provider modes explicitly.
3. Run existing Azure, remote BioCLIP-2 and real taxonomy smoke checks.
4. Confirm real provenance.
5. Confirm sequential execution and deadlines.
6. Confirm no live mode silently uses mocks.

### Required execution

- run every live-compatible case;
- run the consistency subset three times;
- record provider modes and versions;
- record end-to-end latency;
- record agent/evaluator calls separately;
- preserve failures as results;
- calculate every metric.

### Required analysis

Classify:

- wrong classification;
- correct species only in Top-5;
- confidence/decision issue;
- taxonomy issue;
- relevance/explanation issue;
- tool/agent-selection issue;
- consistency issue;
- provider outage/timeout;
- dataset limitation.

Do not modify thresholds, prompts or provider logic during the certified run. Discovered
weaknesses are report findings, not permission for unplanned fixes.

### Required deliverables

- sanitized machine-readable results;
- evaluation report;
- metric table;
- failed-case analysis;
- latency/failure table;
- limitations/no-overclaim statement;
- Phase 3 report.

### Leith handoff requirements

Before Chahd begins:

1. Commit all accepted evaluation code and documentation on the shared branch.
2. Ensure the working tree is clean.
3. Run every Recognition baseline and evaluation test.
4. Record the exact handoff commit.
5. Confirm no `.env`, image, result containing sensitive data, trace or cache is tracked.
6. Provide the Phase 1–3 reports and evaluation results.

### Phase 3 PASS gate

Phase 3 is `PASS` only when:

- every live-compatible case has a result or explicit blocked reason;
- provenance proves what ran;
- all five Sprint 4 criteria are measured;
- weaknesses are classified;
- no sensitive content exists in results;
- all tests pass;
- the work is committed;
- the tree is clean;
- the exact Leith handoff commit is recorded.

Chahd must not start Phase 4 until this gate is `PASS`.

---

## Phase 4 — Chahd handoff verification and LangSmith foundation

**Owner:** Chahd  
**Reviewer:** Leith  
**Branch:** the same `group-d-recognition-sprint4`

### Objective

Verify Leith's completed evaluation baseline, then add the minimal optional LangSmith
foundation and privacy boundary.

### Handoff verification

Before changing code, Chahd must:

1. confirm the exact handoff commit;
2. confirm the working tree is clean;
3. read all Phase 0–3 reports;
4. run the complete Recognition and evaluation test suites;
5. confirm evaluation deliverables exist;
6. stop if the handoff differs from the recorded commit or any test fails.

### LangSmith foundation actions

1. Inspect installed `langgraph`, `langchain-core` and `langsmith` versions.
2. Add/pin only the minimum compatible dependency; do not broadly upgrade the stack.
3. Add explicit standard LangSmith configuration.
4. Keep tracing disabled by default.
5. Ensure disabled mode needs no credentials/network.
6. Create one central Recognition observability boundary.
7. Hide/transform automatic inputs and outputs.
8. Allow only safe metadata.
9. Make trace-export failures non-fatal to Recognition results.
10. Add placeholder-only `.env.example` documentation.

### Permitted changes

- Recognition configuration;
- Recognition-owned observability module(s);
- minimal dependency declaration;
- placeholder environment documentation;
- LangSmith-focused tests/docs.

No workflow instrumentation until Phase 5.

### Required tests

- disabled tracing needs no key/network;
- incomplete enabled configuration fails safely;
- malformed settings are rejected;
- exporter failure does not change `AgentResult`;
- image/secret canaries absent from trace fields/logs;
- imports remain side-effect free;
- two-call/provider behaviour unchanged;
- all baseline and evaluation tests pass.

### Phase 4 PASS gate

Phase 4 is `PASS` only when the handoff is verified, tracing is optional and safe, only
allowlisted metadata can reach the capture double, exporter failure cannot affect the agent,
all tests pass and no public/decision contract changes.

---

## Phase 5 — Recognition workflow and provider tracing

**Owner:** Chahd  
**Reviewer:** Leith  
**Branch:** the same `group-d-recognition-sprint4`

### Objective

Produce one coherent Recognition trace per request, with child spans for workflow nodes, LLM
calls and external providers, without changing agent behaviour.

### Required logical trace

```text
Recognition request
├── validate image and text
├── plan/analyze text
│   └── GPT-5 mini planner when used
├── BioCLIP-2 classification
├── deterministic confidence decision
├── taxonomy validation
│   ├── GBIF
│   └── NCBI
├── explanation
│   └── GPT-5 mini explainer when used
├── delegation decision
└── finalize AgentResult
```

### Required actions

1. Trace the root Recognition execution.
2. Reuse reliable LangGraph tracing and add only missing provider spans.
3. Avoid duplicate roots/spans.
4. Record duration, status, provider mode, safe decision metadata and fixed error codes.
5. Record LLM roles/call count without raw prompts or raw output.
6. Show deterministic fallbacks.
7. Show controlled BioCLIP/GBIF/NCBI failures without external bodies.
8. Keep latency in trace data; do not change the seven output keys.
9. Keep mock/fake/live modes distinct.

### Required scenarios

- identified;
- uncertain;
- not identified;
- invalid input;
- classification unavailable;
- partial taxonomy outage;
- planner fallback;
- explainer fallback;
- `needs_agent`;
- resume contract.

### Required tests

- exactly one root trace;
- expected operations appear once;
- durations are present/non-negative;
- maximum observed agent LLM calls is 2;
- no retry span;
- traced/untraced `AgentResult` equality;
- disabled tracing produces nothing;
- exporter failure cannot fail a request;
- no Base64/image/secret/exception body;
- all baseline, evaluation and LangSmith tests pass.

### Phase 5 PASS gate

Phase 5 is `PASS` only when every required scenario has a correct sanitized trace, trace data
matches actual execution, two-call/no-retry rules remain true, agent output is unchanged and
all accumulated tests pass.

---

## Phase 6 — Trace the already-integrated Orchestrator flow

**Owner:** Chahd  
**Reviewer:** Leith  
**Coordination:** Global Orchestrator owner if a shared file must change  
**Branch:** the same `group-d-recognition-sprint4`

### Objective

Prove trace continuity across the existing integrated flow:

```text
user request
→ orchestrator decision
→ Multimodal selection
→ internal HTTP call
→ Recognition workflow/tools
→ AgentResult
→ orchestrator response handling
```

No RAG span is expected.

### Required actions

1. Inspect/run existing integration tests first.
2. Use an environment that can run affected Orchestrator tests.
3. Propagate trace context only across the trusted internal service call.
4. Never trust browser/public inbound trace context.
5. Strip/ignore untrusted trace headers at the public boundary.
6. Preserve worker payload and Recognition schema.
7. Add the minimum integration coverage for routing, image resolution, Recognition execution
   and response handling under one trace.
8. Cover completed, failed and `needs_agent`.
9. Do not re-integrate Recognition or redesign routing.
10. Do not fix the separate `target_agent` re-resolution seam.
11. Do not change another agent.

### Shared-code rule

Prefer tests only. A shared implementation change requires Leith and Global Orchestrator
owner approval and must be limited to trusted trace propagation.

### Required tests

- Orchestrator selects `Multimodal`;
- image id becomes `recognition_image` only for Recognition;
- one parent trace contains the Recognition child trace;
- no image or credential is recorded;
- completed output preserves seven keys;
- failure remains controlled;
- `needs_agent` follows the existing resolver without a loop;
- an external caller cannot inject trusted trace metadata;
- all affected accumulated suites pass.

### Phase 6 PASS gate

Phase 6 is `PASS` only when a controlled end-to-end test proves trace continuity through the
already-integrated flow, the trust boundary is enforced, no protected contract changes and
all affected tests pass.

If the environment cannot run Orchestrator tests, mark `BLOCKED` with exact evidence. Do not
replace verification with a diagram.

---

## Phase 7 — Final Sprint 4 certification

**Owners:** Leith and Chahd  
**Branch:** `group-d-recognition-sprint4`

### Objective

Verify the complete sequential work, rerun the evaluation with final tracing where required,
and certify Sprint 4 from one clean branch.

### Required actions

1. Confirm the branch contains the recorded Leith handoff followed by Chahd's commits.
2. Confirm working tree is clean.
3. Run every Recognition baseline test.
4. Run every evaluation test.
5. Run every LangSmith/privacy test.
6. Run affected Orchestrator integration tests.
7. Run the final controlled live evaluation or representative certified subset with tracing
   enabled, without replacing the preserved original evaluation results.
8. Confirm trace/results contain no sensitive data.
9. Confirm protected contracts.
10. Produce the final report and updated Recognition reference.

### Required deliverables

1. Phase 0 integrated-baseline report.
2. Evaluation dataset manifest/tests.
3. Evaluation runner/evaluator tests.
4. Sanitized evaluation results and report.
5. LangSmith integration/tests.
6. Sanitized traces for success, failure and delegation.
7. End-to-end trace evidence.
8. Sprint 4 certification report.
9. Updated current-state Recognition reference.

### Final applicability matrix

| Requirement | Final status |
| --- | --- |
| Correctness | `PASS` |
| Relevance | `PASS` |
| Task completion | `PASS` |
| Agent/tool selection | `PASS` |
| Response consistency | `PASS` |
| Evaluation dataset | `PASS` |
| Weakness analysis | `PASS` |
| RAG/RAGAS | `NOT APPLICABLE` with justification |
| LangSmith tracing | `PASS` |
| Failure/latency visibility | `PASS` |
| Integrated trace continuity | `PASS` or evidenced `BLOCKED` only for shared environment |
| Trained-model XAI | `NOT APPLICABLE` with justification |

### Phase 7 PASS gate

Sprint 4 is complete only when:

- every applicable row is `PASS`;
- `N/A` rows use the fixed technical justification;
- the sequential commit/handoff history is clear;
- all accumulated tests pass with exact counts;
- no secret, image payload or private trace data is exposed/tracked;
- final evidence identifies exact commit and provider modes;
- all protected contracts remain true;
- the updated reference matches final code;
- the working tree is clean.

---

## 9. Phase report template

```markdown
# Phase N Report

## Status
PASS | PARTIAL | BLOCKED | FAIL

## Owner
Leith | Chahd | Joint

## Branch and commit
- Branch:
- Starting commit:
- Ending commit/working-tree state:

## Scope executed
- ...

## Files changed
- path — reason

## Tests
- Command:
- Collected:
- Passed:
- Failed:
- Skipped:
- Warnings:

## Evidence
- ...

## Protected checks
- Seven keys unchanged:
- Maximum two agent LLM calls:
- Zero retry:
- No image/secret leak:
- No RAG/Qdrant:

## Blockers/limitations
- ...

## Gate decision
- Criterion: PASS/FAIL
```

---

## 10. Standard prompt for each implementation phase

```text
Read SPRINT_4_COMPLETION_PLAN_RECOGNITION_AGENT.md completely.
Read the latest Recognition current-state reference completely.

Execute Phase N only, on the existing group-d-recognition-sprint4 branch.

Before changing anything:
1. verify branch, HEAD and working tree;
2. verify that the previous phase is PASS;
3. inspect every affected file;
4. list exact files intended for change;
5. confirm the diff stays inside Phase N.

Preserve every protected contract.
Do not create another branch.
Do not begin the next phase.
Do not fix unrelated problems.
Do not modify the separate manual test UI.
Do not expose .env values, API keys, endpoints, image bytes or Base64.
Do not run live services unless Phase N explicitly requires the live guard and credentials
are already configured.

Implement the phase, add required tests, run focused and complete accumulated suites, then
produce the required Phase N report with exact counts and evidence.

Stop after the Phase N gate decision.
```

---

## 11. Official LangSmith references

- https://docs.langchain.com/langsmith/observability-quickstart
- https://docs.langchain.com/langsmith/trace-with-langgraph
- https://docs.langchain.com/langsmith/mask-inputs-outputs
- https://docs.langchain.com/langsmith/distributed-tracing
- https://docs.langchain.com/langsmith/evaluate-llm-application
- https://docs.langchain.com/langsmith/local

---

## 12. Final decision

Sprint 4 is executed sequentially on **one branch created from the latest integrated
`main`**: Leith first inspects and completes Agent Evaluation, Chahd then verifies Leith's
clean handoff and completes LangSmith, and both members finally certify the complete Sprint 4.
RAG/RAGAS, trained-model XAI and unrelated work remain outside scope.
