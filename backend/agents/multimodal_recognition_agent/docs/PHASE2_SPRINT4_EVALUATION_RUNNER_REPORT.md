# Phase 2 Report — Evaluation Runner and Evaluators

**Sprint:** 4 — Multimodal Species Recognition Agent
**Phase:** 2 — evaluation runner and deterministic evaluators
**Owner:** Leith
**Branch:** `group-d-recognition-sprint4`
**Date:** 2026-08-25

No `.env` value, API key, endpoint, image byte, Base64 payload, data URL or
private path appears in this report or in anything Phase 2 added.

---

## Status

**PASS**

A bounded runner executes all 36 benchmark cases through the agent's ordinary
public contract, fifteen deterministic evaluators judge every case, results
serialize safely into a git-ignored directory, a Markdown report renders, live
mode is double-guarded and was not executed, and the complete Recognition
offline suite is green at **1403 passed** with the 1196 baseline untouched.

---

## 1. Preconditions verified

| Check | Result |
| --- | --- |
| Branch | `group-d-recognition-sprint4` |
| HEAD at start | `0a129a82615886e3be213a9a2248975f5e67d1fd` — *"Add Recognition Sprint 4 evaluation benchmark"* |
| Working tree at start | **clean** |
| Phase 1 benchmark committed | **Yes** — manifest, schema, fetcher, README, `.gitignore`, `__init__.py` and the dataset tests all tracked |
| Manifest loaded through its own loader | **Yes** — `manifest_schema.load_manifest()` / `cases()`, never re-parsed by hand |

All accepted values were re-derived from source rather than restated:
`AgentStatus` (`schema.py`), `Decision` (`domain/models.py`), `ErrorCode`
(`domain/errors.py`), `HELPER_OUTPUT_KEYS` (`workflows/state.py`), the seven
output keys and the provenance field names (`workflows/graph.py`), and the
delegation preconditions (`workflows/nodes.py::make_delegation_node`).

**No stop condition was triggered.** The runner reaches the real public contract
(`RecognitionAgent.run(AgentRequest) -> AgentResult`, the same entry point
`api.py` uses), the output contract carries everything the fifteen metrics need,
no Phase 1 schema defect blocked evaluation, and live safety is achieved without
touching runtime code.

---

## 2. Files created and modified

### Created (7)

| Path | Lines | Role |
| --- | ---: | --- |
| `evaluation/runner.py` | 464 | Execution: scripted offline providers, request building, observation, live guard, CLI |
| `evaluation/evaluators.py` | 532 | The fifteen deterministic evaluators and the applicability policy |
| `evaluation/results_schema.py` | 379 | Result models, metric aggregation, and the redaction boundary |
| `evaluation/report.py` | 194 | Markdown rendering |
| `evaluation/rubric.py` | 104 | The documented human rubric (no LLM judge) |
| `evaluation/consistency.py` | 129 | Phase 3 subset selection and comparison rules |
| `tests/test_sprint4_phase2_evaluation_runner.py` | 916 | 126 Phase 2 tests |

### Modified (3)

| Path | Change | Why |
| --- | --- | --- |
| `evaluation/.gitignore` | +9 | Adds `results/`. Generated results must never be committed. |
| `evaluation/__init__.py` | +31 / −? | The Phase 1 docstring said the package "contains no runner". Phase 2 adds one, so the docstring now describes both halves and states the properties that still hold. |
| `tests/test_sprint4_phase1_evaluation_dataset.py` | +35 / −17 | See below. |

**The one Phase 1 test that had to change.**
`test_the_evaluation_package_contains_no_runner` asserted that *no* module in
`evaluation/` referenced `RecognitionAgent` or built an `AgentRequest`. That
encoded "Phase 2 has not started", and Phase 2 starting is precisely what
invalidates it. It was replaced by two narrower tests that keep the durable half
of the property: `manifest_schema.py` and `fetch_assets.py` — the modules that
*describe and fetch* the dataset — still may not call or import the agent, so
loading or fetching the benchmark can never execute it. Phase 1's remaining 79
assertions are unchanged and still pass.

**No Recognition runtime file was created, modified or deleted.** See §10.

---

## 3. Runner architecture

```
manifest.json ──(manifest_schema.load_manifest / cases)──► committed case order
                                                              │
                                     ┌────────────────────────┴───────────────────────┐
                                     │                                                │
                          OFFLINE (default)                                    LIVE (guarded)
                  injected ScriptedClassifier                          no injection at all:
                  injected MockTaxonomyProvider                        RecognitionAgent(config)
                  injected FakeGPT5MiniProvider                        lets the real factories
                  synthetic in-memory image                            choose, and refuse
                                     │                                                │
                                     └────────────────────────┬───────────────────────┘
                                                              ▼
                              AgentRequest(instruction, context)  ← the public contract
                                                              ▼
                                   RecognitionAgent.run(...) -> AgentResult
                                                              ▼
                            observe(): sanitized labels, ids, modes, counts, latency
                                                              ▼
                       evaluate_case(): 15 deterministic evaluators + 2 deferred
                                                              ▼
                        CaseResult ──► RunSummary ──► JSON (results/) + Markdown
```

Key properties:

- **The agent is used, never altered.** The runner imports `RecognitionAgent`,
  builds an `AgentRequest` and reads an `AgentResult`. It sets no prompt, no
  threshold and no provider mode.
- **Injection uses the agent's own documented seam.** `RecognitionAgent.__init__`
  already accepts `classifier`, `taxonomy_provider` and `reasoning_llm`; the
  offline mode passes all three. That is the same seam the agent's own offline
  suite uses, and its docstring says so.
- **One result per case, always.** The per-case body is wrapped: an exception
  records a sanitized class name and still appends a `CaseResult`. A case cannot
  leave the aggregation by failing.
- **Committed order is preserved.** Cases are iterated in manifest order and
  results carry the same ids.

### Offline scenarios — and why a third of them are wrong on purpose

The offline classifier is a scripted table, not a model. Each case is assigned a
scenario by **position within its category**, fixed in advance and independent of
anything the agent says:

| Scenario | Applied to | Drives |
| --- | --- | --- |
| `top1_correct` | every 1st real case; all delegation cases | Top-1 pass, identified decision, delegation firing |
| `top5_only` | every 2nd real case | Top-1 fail **with** Top-5 pass — the branch that only exists on a rank>1 hit |
| `wrong_species` | every 3rd real case | Top-1 and Top-5 fail; `identifier_mismatch_after_species_mismatch` |
| `no_candidates` | every 4th real case; all ambiguous cases | `not_identified`, taxonomy-never-executed N/A branch |
| `classifier_error` | `DEP-01` | Controlled `CLASSIFICATION_UNAVAILABLE` |

If every scripted label were correct, the mismatch half of every evaluator would
never execute and the dry run would prove far less. `test_the_dry_run_deliberately_includes_wrong_answers`
pins that.

---

## 4. Exact dry-run command and proof

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe -m backend.agents.multimodal_recognition_agent.evaluation.runner --mode offline --report
```

```
36 cases evaluated -> results_offline_20260825T191912+0000.json
DRY RUN: infrastructure validation only. Not a measure of recognition accuracy.
report -> results_offline_20260825T191912+0000.md
22 case(s) with at least one failing metric; 68 failing metric outcome(s) in total
```

### All 36 cases produced exactly one result

| Check | Result |
| --- | --- |
| Cases in manifest | **36** |
| Results produced | **36** |
| Distinct case ids in results | **36** |
| Results in committed manifest order | **Yes** |
| Cases executed (agent actually invoked) | **36 / 36** |
| Cases with an execution error | **0** |
| Cases skipped | **0** |
| Cases with a latency sample | **36 / 36** |

Terminal states reached: **30 `completed`, 4 `failed`, 2 `needs_agent`** — all
three of the statuses Recognition can return. Decisions observed: 13
`identified`, 6 `uncertain`, 11 `not_identified`, 6 with no decision (the four
controlled failures and the two escalations, which carry `output=None`).

Per category: 24 `real_recognition`, 5 `ambiguous_or_non_animal`, 4
`invalid_input_or_dependency_failure`, 3 `delegation_resume`.

### These numbers are infrastructure validation, not accuracy

**Every species label in the dry run came from a table in `runner.py`.** No image
was classified by BioCLIP-2; no model saw anything. The scenario each case
received was decided by its position in the manifest, before the run started.

This is stated in three places so it cannot be lost downstream: the results JSON
carries `"dry_run": true` and
`"scores_are_infrastructure_validation_only": true`; the run's `notes` field says
it in prose; and the Markdown report opens with a blockquote banner saying it
again. `test_the_report_renders_and_carries_the_dry_run_warning` pins the banner.

The 32 live-only cases identified in Phase 1 were **not** converted into
biological claims: they ran against scripted providers, and the only thing their
results demonstrate is that the runner and evaluators handle them.

---

## 5. Implemented metrics

All fifteen required evaluators, plus two metrics the manifest declares that
Phase 2 structurally cannot decide.

| # | Metric | What it judges | Dry-run outcome (P / F / N/A / no-ev) |
| ---: | --- | --- | --- |
| 1 | `output_schema` | Seven keys on `completed`; `{error_code, error}` on `failed`; `output=None` + `target_agent` on `needs_agent` | **36 / 0 / 0 / 0** |
| 2 | `status` | Observed terminal status equals the expected one | **36 / 0 / 0 / 0** |
| 3 | `decision` | Decision is inside the case's *accepted* set | 23 / 7 / 6 / 0 |
| 4 | `top1_species` | Normalised exact binomial equality with the primary species | 6 / 18 / 12 / 0 |
| 5 | `top5_species` | Expected species anywhere in the first five ranked labels | 12 / 12 / 12 / 0 |
| 6 | `gbif_identifier` | Observed `gbif_id` equals the officially verified key | 2 / 16 / 18 / 0 |
| 7 | `ncbi_taxid` | Observed `ncbi_taxid` equals the officially verified taxid | 3 / 15 / 18 / 0 |
| 8 | `delegation_capability` | `target_agent` equals the expected capability, and is a real one | 2 / 0 / 34 / 0 |
| 9 | `task_completion` | Reached the required terminal state **carrying its obligatory payload** | **36 / 0 / 0 / 0** |
| 10 | `provenance_truthfulness` | Provenance describes what actually ran | 30 / 0 / 4 / 2 |
| 11 | `agent_tool_selection` | Ranking, sole-source, no-peer-call and refusal rules | 3 / 0 / 33 / 0 |
| 12 | `agent_llm_call_budget` | `reasoning_llm_calls ≤ 2`, when a count exists | 30 / 0 / 0 / 6 |
| 13 | `zero_retry` | One logical classification per request, from an instrumented provider | **36 / 0 / 0 / 0** |
| 14 | `controlled_error_code` | The exact expected `ErrorCode`, and that it is a known one | 4 / 0 / 32 / 0 |
| 15 | `latency` | Captured and non-negative — recorded, never scored against a target | **36 / 0 / 0 / 0** |
| — | `relevance` | Human rubric | 0 / 0 / 7 / 29 |
| — | `response_consistency` | Phase 3 repetitions | 0 / 0 / 5 / 31 |

Every metric's four buckets sum to 36 for every metric — asserted by
`test_metric_totals_account_for_every_case_exactly_once`, which is what
guarantees no case silently drops out of aggregation.

### What the dry-run failures actually are

| Reason code | Count | Meaning |
| --- | ---: | --- |
| `top1_mismatch` | 12 | Scripted `top5_only` and `wrong_species` scenarios |
| `no_primary_species` | 6 | Scripted `no_candidates` |
| `top5_miss` / `no_candidates` | 6 / 6 | Scripted `wrong_species` / `no_candidates` |
| `decision_not_accepted` | 7 | `no_candidates` on cases whose accepted set is `["identified"]` |
| `identifier_mismatch_after_species_mismatch` | 12 + 12 | The species was already wrong upstream, so the identifier follows |
| `identifier_mismatch` | 4 + 3 | Top-1 was right, but the injected **mock** taxonomy fixture had a different or absent id |

That last row is worth naming. It is the evaluator correctly detecting the mock
fixture discrepancy Phase 1 documented — `fixtures/mock_taxonomy.json` carries
GBIF `5219426` for *Panthera tigris* against the verified `5219416`, and
`2433451` for *Loxodonta africana*, which is in fact *Ursus maritimus*'s key —
plus the species the fixture simply does not know. **It was not fixed**, per the
Phase 2 instruction, and it is not an agent defect: the fixture discloses itself
as unverified and is reachable only in mock mode, where the status reported is
`mock_verified` rather than `verified`. It is recorded here as evidence the
identifier evaluators work.

### The reason-code split is deliberate

`identifier_mismatch_after_species_mismatch` exists so Phase 3 weakness analysis
can separate *"the taxonomy lookup is broken"* from *"the classifier named the
wrong animal and taxonomy then faithfully resolved that wrong animal"*. Those are
different findings with different fixes, and collapsing them into one code would
hide the distinction.

---

## 6. N/A policy

Four outcomes exist, and the last two are never counted as passes.

| Outcome | Meaning |
| --- | --- |
| `pass` / `fail` | The metric applied and was judged |
| `not_applicable` | The case never claimed the metric |
| `no_evidence` | The metric applied, but the run produced nothing to judge it by |

Resolution order:

1. Metric listed in the case's `metrics_not_applicable` → **N/A**.
2. Metric listed in `metrics_applicable` → **judged**.
3. Otherwise, metric is a **universal invariant** (`output_schema`, `status`,
   `agent_llm_call_budget`, `zero_retry`, `latency`) → **judged**. A case cannot
   opt out of "return a valid schema" or "do not retry".
4. Otherwise → **N/A**.

Specific rules the plan required:

- **Ambiguous and non-animal cases are never scored for species.** All four
  species metrics return N/A on them, asserted by test.
- **Taxonomy is not penalised when it was unavailable.** If provenance says
  `taxonomy_executed: false` or `taxonomy_degraded: true` and the observed
  identifier is `null`, the metric is **N/A** with reason `taxonomy_unavailable`
  — because leaving it null is exactly what the agent is required to do. Six real
  cases took this branch in the dry run. A *wrong* identifier still fails.
- **Absent evidence is not a pass.** Six cases failed validation before the
  workflow ran, so no `reasoning_llm_calls` existed; the budget metric recorded
  `no_evidence`, not a free pass.

**One evaluator was corrected during implementation.** `delegation_capability`
originally short-circuited on applicability before checking for an *unexpected*
escalation — which meant `DEL-03`, the resume case that declares the metric
inapplicable, would not have caught the agent escalating a second time. That is
the one case where a second escalation matters most. The unexpected-delegation
check now runs first, regardless of what the case declares. The manifest was not
changed.

---

## 7. Live-guard behaviour

Live mode requires **two independent, explicit** signals:

1. `--mode live` on the command line (offline is the default in the CLI *and* in
   the `run()` signature), and
2. `RECOGNITION_EVAL_LIVE=1` in the environment.

Refusal is clear and total:

```
Live evaluation refused: set RECOGNITION_EVAL_LIVE=1 to contact real providers.
Nothing was executed and no request was sent.
```

The CLI exits **2**. Only the exact string `"1"` is accepted — `""`, `"0"`,
`"true"`, `"yes"`, `"TRUE"` and `"1 "` are all refused, each pinned by a
parametrised test.

| Requirement | How it is met |
| --- | --- |
| Explicit CLI selection | `--mode live`; `offline` is the argparse default |
| Dedicated env opt-in | `RECOGNITION_EVAL_LIVE=1`, checked before any agent is built |
| Refuses clearly when missing | `LiveModeRefused` with an explicit message; exit code 2 |
| Uses existing real provider construction | `build_live_agent()` calls `RecognitionAgent(RecognitionConfig.from_env())` and **injects nothing** — asserted by a test that reads the function's source for `classifier=`, `taxonomy_provider=`, `reasoning_llm=` |
| Runs sequentially | A single `for` loop; no threads, no pool |
| Existing bounded deadlines | Untouched — they live in the providers the factories build |
| Zero retry preserved | Untouched; the runner makes one `agent.run` call per case |
| Never silently falls back to mocks | Guaranteed by the agent's own factories, which refuse a mode they cannot honour. The runner adds no fallback path. |
| Never exposes credentials | The runner reads no credential; only exception **class names** are recorded |
| Results only in a git-ignored directory | `evaluation/results/`, ignored by `evaluation/.gitignore:17` |

**Live mode was not executed in Phase 2.** No Azure, BioCLIP-2, GBIF, NCBI or
LangSmith call was made at any point.

---

## 8. Privacy and redaction evidence

Redaction is enforced in one place, `results_schema.py`, by allow-list rather
than by removal:

- `safe_provenance()` copies only the 23 named provenance fields plus a reduced
  `taxonomy_report` (modes, availability, matched flags). Anything else — a new
  key, a credential, a payload — is **dropped**, not sanitised.
- `safe_candidates()` copies only the 8 named candidate fields.
- `sanitize_error()` records `type(exc).__name__` and nothing else, matching the
  agent's own rule that a remote error body can echo the request.

Verified on the real generated artefacts (285 KB JSON, 22 KB Markdown):

| Check | Result |
| --- | --- |
| `data:image`, `;base64,`, `-----BEGIN`, `Authorization` in results JSON | **0 occurrences** |
| Base64-like runs ≥ 200 chars | **0** |
| Absolute paths / drive letters / `/Users/` / `/home/` | **0** |
| `api_key`, `secret`, `password`, `token`, `bearer` | **0** |
| Same four checks on the Markdown report | **0 occurrences** |

Additional tests: `safe_provenance` and `safe_candidates` are each fed a dict
containing a canary data URL and a fake credential and asserted to return only
the allow-listed field; `sanitize_error` is fed an exception whose message
contains a URL and asserted to return only `ValueError`; and
`test_the_image_bytes_sent_to_the_agent_never_reach_the_results` takes the actual
Base64 payload the runner feeds the agent and asserts its first 80 characters
appear nowhere in the serialized run.

Generated results and downloaded images both remain git-ignored:
`assets/` (line 8) and `results/` (line 17) of `evaluation/.gitignore`, the
latter confirmed live by `git check-ignore -v`.

---

## 9. Agent versus evaluator LLM-call accounting

Counted separately, never summed. Summing them would make the agent's two-call
ceiling unverifiable, which is the whole reason the ceiling is auditable at all.

| Counter | Dry-run value |
| --- | ---: |
| Agent LLM calls, total across 36 cases | **60** |
| Agent LLM calls, **maximum observed in one request** | **2** |
| Agent call ceiling | **2** |
| Cases with call evidence | 30 (6 failed validation before the workflow ran) |
| **Evaluator** LLM calls, total | **0** |
| Evaluator judge model used | **false** |

The 60 comes from the offline `FakeGPT5MiniProvider`, which exercises the real
two-call contract (one plan, one grounded explanation) without a network or a
credential — chosen precisely so the budget evaluator has genuine evidence rather
than a disabled provider reporting zero.

**No LLM judge was added.** Phase 2 uses the documented human rubric in
`rubric.py`: three criteria (relevance, explanation quality, task completion
where judgement is required), each a 1–5 scale with written level descriptions,
plus a scoring protocol and a list of hard failures. Every result carries an
unscored `human_review` slot; unscored is reported as unscored, never as zero.
`overrides_deterministic_metrics` is `false` and asserted by test: a human score
can never overturn whether the right species was named. A test also greps the
whole evaluation package for `openai(`, `AzureChatOpenAI`, `ChatCompletion` and
`judge_model` and asserts none appears.

---

## 10. Consistency support (prepared, not executed)

Selection is fixed **now**, before anyone has seen a live result and could be
tempted to choose the five cases that happened to agree.

| Property | Value |
| --- | --- |
| Selected subset | `REC-PLEO-01`, `REC-PLEO-02`, `REC-PTIG-01`, `AMB-01`, `DEL-01` |
| Size | **5** (minimum 5) |
| Selection rule | One case per (category, difficulty) bucket among live-runnable, non-failure cases, in committed manifest order; topped up in manifest order if the buckets alone fall short |
| Repetitions requested per case | **3** |
| Compared | `status`, `decision`, `primary_species`, `candidate_order`, `target_agent` |
| Not compared | `explanation` (phrasing varies by design), `latency_ms`, `clarification_question` |
| Executed in Phase 2 | **No** — `observations: []`, `stable: null` on every result |

`compare()` reports `stable: null` with reason `insufficient_repetitions` when
fewer than two observations exist, so a single run can never be misreported as
stable.

---

## 11. Runtime code untouched

| Check | Evidence |
| --- | --- |
| Files changed under `adapters/`, `domain/`, `workflows/`, plus `config.py`, `validation.py`, `agent.py`, `api.py`, `schema.py`, `text_analysis.py`, `fixtures/`, `requirements.txt` | **none** — `git status --porcelain` on those paths is empty |
| Orchestrator or registry changed | **none** — same check on `backend/orchestrator` and `backend/registry.py` is empty |
| Prompts, thresholds, providers, workflow nodes modified | **No** |
| Mock taxonomy discrepancy fixed | **No** — reported in §5, deliberately left alone |
| LangSmith added | **No** |
| RAG / RAGAS / Qdrant / embeddings / XAI added | **No** |
| Live evaluation run | **No** |
| Baseline suite intact | 1196 pre-existing tests still pass, unchanged |

The runner interacts with the agent only through `RecognitionAgent(...)`,
`AgentRequest(...)` and the returned `AgentResult` — and through the injection
parameters the constructor already exposed for the agent's own offline tests.

---

## 12. Tests — exact commands and counts

Interpreter: `backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe`
— Python 3.11.0, pytest 8.3.2, run from the repository root.

### Command 1 — Phase 1 dataset tests

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe -m pytest backend/agents/multimodal_recognition_agent/tests/test_sprint4_phase1_evaluation_dataset.py -q
```

```
81 passed in 0.21s
```

80 before; the replaced runner test became two narrower tests (§2).

### Command 2 — Phase 2 runner and evaluator tests

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe -m pytest backend/agents/multimodal_recognition_agent/tests/test_sprint4_phase2_evaluation_runner.py -q
```

```
126 passed in 32.02s
```

Collected 126, passed 126, failed 0, skipped 0, xfail 0, exit 0.

### Command 3 — the complete 36-case offline dry run

Shown in §4. 36 cases, 36 results, 0 execution errors, exit 0.

### Command 4 — the complete Recognition offline suite

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe -m pytest backend/agents/multimodal_recognition_agent/tests -q
```

```
1403 passed, 1 warning in 174.83s (0:02:54)
```

| Metric | Phase 0 | Phase 1 | **Phase 2** |
| --- | ---: | ---: | ---: |
| Collected / passed | 1196 | 1276 | **1403** |
| Failed / errors / skipped / xfail | 0 | 0 | **0** |
| Warnings | 1 | 1 | **1** (same FastAPI test-client deprecation) |
| Exit code | 0 | 0 | **0** |

1403 = 1196 baseline + 81 Phase 1 + 126 Phase 2. **The 1196 baseline is intact.**

### Required-test coverage

| # | Required test | Covered by |
| ---: | --- | --- |
| 1 | Evaluator units with synthetic outputs | The whole first half of the file — every evaluator driven by hand-written observations |
| 2 | Top-1 calculation | `test_top1_matches_on_exact_normalised_name`, `..._does_not_match_a_different_species`, `..._never_uses_similarity`, `..._fails_when_no_species_was_named` |
| 3 | Top-5 calculation | `test_top5_finds_the_species_below_rank_one`, `..._misses_when_the_species_is_absent`, `..._only_considers_the_first_five_labels`, `..._fails_cleanly_when_there_are_no_candidates` |
| 4 | Accepted-decision handling | `test_a_decision_inside_the_accepted_set_passes`, `..._outside_the_accepted_set_fails`, `test_an_unknown_decision_value_fails...` |
| 5 | `uncertain` handling | `test_uncertain_is_a_completed_outcome_for_an_ambiguous_case` |
| 6 | `not_identified` handling | `test_not_identified_is_a_completed_outcome_for_an_ambiguous_case` |
| 7 | Ambiguous N/A handling | `test_species_metrics_are_not_applicable_on_ambiguous_cases` (×4), `test_ambiguous_cases_are_never_scored_for_species`, `test_a_confident_identification_of_a_non_animal_image_fails` |
| 8 | Taxonomy-unavailable handling | `test_a_null_identifier_is_not_penalised_when_taxonomy_never_executed`, `..._when_taxonomy_degraded`, `test_a_wrong_identifier_still_fails_when_taxonomy_was_available` |
| 9 | GBIF / NCBI comparisons | `test_gbif_and_ncbi_match_the_verified_identifiers`, `test_an_identifier_mismatch_following_a_species_mismatch_is_labelled_as_such` |
| 10 | Delegation evaluation | 5 tests incl. wrong capability, missing escalation, non-registry target, unexpected re-escalation |
| 11 | Task completion | 4 tests incl. completed-but-unusable and escalated-without-prompt |
| 12 | Provenance | 5 tests incl. probability claim, mock-key misuse, taxonomy contradiction |
| 13 | Two-call limit | `test_the_agent_call_budget_is_judged_against_the_ceiling_of_two` (0/1/2/3), `test_a_missing_call_count_is_no_evidence_rather_than_a_free_pass` |
| 14 | Separate evaluator accounting | `test_evaluator_calls_are_counted_separately_and_are_zero_in_phase_2`, `test_no_llm_judge_exists_anywhere_in_the_evaluation_package`, `test_the_human_rubric_cannot_override_a_deterministic_verdict` |
| 15 | Zero-retry evidence | `test_zero_retry_is_judged_from_the_invocation_count` (0/1/2), `..._without_instrumentation_is_no_evidence` |
| 16 | Controlled errors | 4 tests incl. uncontrolled code and no-failure-at-all |
| 17 | Case ordering and ids | `test_results_preserve_the_committed_manifest_order_and_ids` |
| 18 | Exactly one result per case | `test_exactly_one_result_exists_for_every_case`, `test_evaluate_case_returns_one_outcome_per_metric_for_every_case` |
| 19 | Failed cases retained | `test_failing_cases_appear_in_the_failed_case_list`, `test_a_case_that_raises_still_produces_a_result` |
| 20 | Offline zero network | `test_the_offline_dry_run_opens_no_socket`, plus the socket trap in test 27 |
| 21 | Offline no real sleep | `test_the_offline_dry_run_never_sleeps` |
| 22 | Live refuses without opt-in | `test_live_mode_refuses_without_the_explicit_opt_in`, `test_only_the_exact_opt_in_value_is_accepted` (×6), `test_the_cli_refuses_live_...` |
| 23 | Output redaction | 6 tests incl. the real-payload canary and the blob scan |
| 24 | Machine-readable serialization | `test_the_run_serializes_to_valid_json` |
| 25 | Metric aggregation | `test_metric_totals_account_for_every_case_exactly_once`, `test_metric_totals_by_category_sum_to_the_overall_totals`, `test_the_latency_summary_...`, `test_provider_modes_...` |
| 26 | Markdown report | 5 tests incl. banner presence, per-case coverage, redaction, and banner absence on a live summary |
| 27 | Complete 36-case dry run | `test_the_complete_offline_dry_run_executes_every_case` (sockets and sleep trapped), plus 6 supporting assertions |

### Zero network — proved twice

Beyond the monkeypatched tests, a process-level check disabled
`socket.socket.connect`, `socket.socket.connect_ex`, `socket.create_connection`
and `socket.getaddrinfo`, and replaced `time.sleep` with a raising stub, **before
importing the runner**:

```
cases: 36
all executed: True
execution errors: []
ZERO NETWORK CONFIRMED, ZERO REAL SLEEP CONFIRMED
```

---

## 13. Phase 2 gate checklist

| Gate criterion | Verdict | Basis |
| --- | --- | --- |
| The offline runner evaluates every selected case | **PASS** | 36/36 executed, 36/36 results, 0 errors (§4) |
| One recorded result per case including failures | **PASS** | Asserted by test, including a forced-exception run where all 36 still produced results |
| Manifest ordering and case ids preserved | **PASS** | `test_results_preserve_the_committed_manifest_order_and_ids` |
| Offline is the default, needs no key or `.env` | **PASS** | argparse default and `run()` signature default; runs with all provider env vars deleted |
| Offline performs zero network | **PASS** | Two independent proofs (§12) |
| Offline performs zero real sleeps | **PASS** | `time.sleep` trapped in tests and in the process-level check |
| Metrics are tested | **PASS** | 126 Phase 2 tests, every evaluator driven through pass, fail, N/A and no-evidence branches |
| Live execution is guarded | **PASS** | Two independent signals, exact-value opt-in, exit code 2, no injection in live construction (§7) |
| Live not executed; no external provider contacted | **PASS** | No Azure, BioCLIP-2, GBIF, NCBI or LangSmith call was made |
| Outputs contain no sensitive material | **PASS** | Allow-list redaction, 0 hits on every forbidden pattern across both artefacts (§8) |
| Evaluation calls separated from agent calls | **PASS** | Counted separately, never summed; evaluator total structurally 0 (§9) |
| No minimum accuracy target invented | **PASS** | No metric has a threshold; `latency` is captured rather than scored; the dry run is labelled infrastructure validation in three places |
| Consistency protocol prepared, not executed | **PASS** | 5 cases selected deterministically, 3 repetitions requested, 0 executed (§10) |
| Human rubric documented; no LLM judge added | **PASS** | `rubric.py`; judge greps pinned by test (§9) |
| All Recognition tests pass | **PASS** | 1403 passed, baseline 1196 intact (§12) |
| Runtime behaviour untouched | **PASS** | §11 |

### Decision

**PHASE 2 — PASS.**

No stop condition was triggered. The runner reaches the real public contract, the
output contract supplies everything the fifteen metrics need, the Phase 1
manifest required no schema change, and live safety was achieved entirely through
guards in the runner without modifying a single runtime file.

Three items are handed forward:

1. **The dry-run scores are not accuracy.** Any Phase 3 comparison against them
   is a comparison against a scripted table, not a baseline.
2. **The mock taxonomy discrepancy is still open and still unfixed** (§5). It is
   confined to mock mode; live mode queries real GBIF and NCBI, so Phase 3's
   identifier metrics will exercise the real path for the first time.
3. **`relevance` and `response_consistency` are unscored by design.** Phase 3
   must run the human rubric and the repetition protocol, or report them as
   unscored — neither may be reported as a pass.

No live evaluation was run. No external provider was contacted. Nothing was
committed or pushed. **Phase 3 has not been started.**

---

*Compiled 2026-08-25 on branch `group-d-recognition-sprint4` at
`0a129a82615886e3be213a9a2248975f5e67d1fd`. No Recognition runtime file was
modified. No API key, endpoint, `.env` value, image payload, Base64 data or
private path appears anywhere in this document or in the artefacts it describes.*
