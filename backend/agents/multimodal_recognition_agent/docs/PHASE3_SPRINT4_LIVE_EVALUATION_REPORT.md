# Phase 3 Report — Certified Live Evaluation

**Sprint:** 4 — Multimodal Species Recognition Agent
**Phase:** 3 — live evaluation, analysis and preparation for human approval
**Owner:** Leith
**Branch:** `group-d-recognition-sprint4`
**Commit at run time:** `35372dee229bcdefe548d6a4cb5406aab8defb21`
**Date:** 2026-08-26

No `.env` value, API key, endpoint, header, image byte, Base64 payload, data URL
or private path appears in this report or in any deliverable it describes.

---

## Status

**PASS**

Leith reviewed the human-review worksheet and **approved all 105 proposed rubric
scores without modification** (35 cases × 3 criteria). Those are now recorded
human scores, and the last outstanding item is closed.

| Item | State |
| --- | --- |
| Human rubric | **APPROVED by Leith**, 105/105, without modification |

| Item | State |
| --- | --- |
| Certified primary run | 35/35 executed, 0 execution errors |
| Certified consistency run | 15/15 executions, **5/5 stable** |
| Supplemental explanation capture | 35/35 executed, **35/35 answers captured**, **0 biological differences** from certified |
| Full offline suite | **zero failing tests** |
| The three stale Sprint 3 gates | **fixed** — now ask git instead of walking the filesystem, and are strictly stricter than before (§14) |
| Weakness classification | **corrected** — no case is called an outage without outage evidence (§9) |
| Privacy scan | clean on every artefact |
| Runtime source | untouched |

Three runs are held strictly separate throughout: the **certified primary
deterministic** run, the **certified consistency** run, and the **supplemental
explanation-capture** run. Nothing from the supplemental pass feeds any metric.

---

## 1. Pre-run verification

| Check | Result |
| --- | --- |
| Branch | `group-d-recognition-sprint4` |
| HEAD | `35372dee229bcdefe548d6a4cb5406aab8defb21` |
| Working tree before run | **clean** |
| Assets present / validated | **33 / 33** |
| Assets tracked | **0** |
| Phase 0 / 1 / 2 reports committed | **Yes**, all three PASS |
| Live guard | CLI `--mode live` **and** `RECOGNITION_EVAL_LIVE=1` |
| Offline suite before any live call | 1403 passed (Part 1) |

`.env` was loaded into the run process by a temporary wrapper held **outside the
repository**. `config.py` was not modified. Resolved configuration, safe values
only:

```
llm_mode                          : azure
bioclip_mode                      : remote
taxonomy_mode                     : real
azure_call_budget                 : 2
llm_timeout_seconds               : 20.0
remote_classifier_timeout_seconds : 90.0
taxonomy_timeout_seconds          : 10.0
top_k_species                     : 5
ncbi_tool_present                 : True     (value never read or printed)
ncbi_email_present                : True     (value never read or printed)
thresholds                        : identified>=0.75 margin>=0.08 uncertain>=0.45
preflight                         : PASS - every mode is real, no mock, no disabled
```

The wrapper aborts before the benchmark if any mode resolves to mock or
disabled. It did not abort.

---

## 2. Primary run

| Field | Value |
| --- | ---: |
| Cases expected | **35** |
| Cases executed | **35** |
| Results produced | **35** |
| Execution errors | **0** |
| Blocked / skipped | **0** |
| Excluded | **1** — `DEP-01`, offline-only by manifest declaration |
| Wall time | **290.0 s** |

Ordering and case ids preserved. Exactly one result per case. Nothing retried,
nothing re-run, nothing hidden. `DEP-01` was excluded by passing a filtered
manifest to `runner.run(manifest=...)` — a parameter the committed Phase 2 runner
already exposes, so no runner code was changed.

### Real provider provenance observed

| Field | Observed |
| --- | --- |
| `recognition_provider` | `RemoteBioCLIP2Provider` |
| `recognition_mode` | `remote_bioclip2_open_domain_species` |
| `model_version` | `imageomics/bioclip-2` |
| `remote_space_id` | `imageomics/bioclip-2-demo` |
| `remote_space_revision` | `4768b0d8b743582f6eaa058fde15d303ed073522` |
| `gbif_mode` | `real` |
| `ncbi_mode` | `real` |
| `reasoning_llm_provider` | `azure-gpt-5-mini` |
| `mock_provider_version` | `null` on every case |
| `score_kind` | `bioclip2_remote_zero_shot_ranking_score` |

**No mock, fixture or offline oracle appears anywhere in the run.** No provider
fell back to one.

---

## 3. Deterministic metric table

35 cases. Each metric's four buckets sum to 35, so no case escapes aggregation.

| Metric | Pass | Fail | N/A | No evidence | Pass rate (judged) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `output_schema` | **35** | 0 | 0 | 0 | 35/35 |
| `status` | **35** | 0 | 0 | 0 | 35/35 |
| `task_completion` | **35** | 0 | 0 | 0 | 35/35 |
| `latency` | **35** | 0 | 0 | 0 | 35/35 |
| `provenance_truthfulness` | **30** | 0 | 3 | 2 | 30/30 |
| `agent_llm_call_budget` | **30** | 0 | 0 | 5 | 30/30 |
| `decision` | 26 | 4 | 5 | 0 | 26/30 |
| `top5_species` | 22 | 2 | 11 | 0 | 22/24 |
| `top1_species` | 18 | 6 | 11 | 0 | 18/24 |
| `gbif_identifier` | 18 | 6 | 11 | 0 | 18/24 |
| `ncbi_taxid` | 18 | 6 | 11 | 0 | 18/24 |
| `delegation_capability` | 2 | 0 | 33 | 0 | 2/2 |
| `agent_tool_selection` | 3 | 0 | 32 | 0 | 3/3 |
| `controlled_error_code` | 3 | 0 | 32 | 0 | 3/3 |
| `zero_retry` | 0 | 0 | 0 | **35** | — |
| `relevance` | 0 | 0 | 6 | 29 | — (human rubric) |
| `response_consistency` | 0 | 0 | 4 | 31 | — (measured separately, §7) |

`zero_retry` is `no_evidence` on every case by design: the metric is counted from
an instrumented classifier, which exists only in offline mode. Live retry
behaviour is evidenced instead by the Part 1 Azure smoke — 2 logical calls, 2 HTTP
attempts, `no_transport_retry: True` — and by `SDK_MAX_RETRIES = 0` in the
adapter. Reporting it as "no evidence" rather than "pass" is the honest reading.

### By category

| Category | Cases | Headline |
| --- | ---: | --- |
| `real_recognition` | 24 | Top-1 18/24, Top-5 22/24, decision 20/24 |
| `ambiguous_or_non_animal` | 5 | **5/5 correctly inconclusive**; no species forced |
| `invalid_input_or_dependency_failure` | 3 | **3/3 exact controlled error codes** |
| `delegation_resume` | 3 | **2/2 escalations correct**, resume completed |

---

## 4. Accuracy and Top-5

On the 24 real-recognition cases:

| Measure | Result | Rate |
| --- | --- | ---: |
| **Top-1 correct** | 18 / 24 | **75.0 %** |
| **Expected species in Top-5** | 22 / 24 | **91.7 %** |
| Correct species present but not ranked first | 4 | 16.7 % |
| Species absent from Top-5 entirely | 2 | 8.3 % |

By difficulty: **11/11 clear images** were correct at Top-1 on species where the
agent committed; the six Top-1 misses are concentrated in the challenging half
and in two taxonomic-concept cases described below.

**No minimum accuracy target was set, and none is claimed.** These are measured
figures on a bounded 36-case benchmark. They are not a statement about
BioCLIP-2's accuracy in general, and the sample is far too small to support one.

### The six Top-1 misses, individually

| Case | Expected | Observed | Reading |
| --- | --- | --- | --- |
| `REC-PPAR-02` | *Panthera pardus* | *Neofelis diardi* | Melanistic leopard; rosettes almost invisible. A genuine misclassification, and the only case where the species is absent from Top-5 with a label returned. |
| `REC-UMAR-02` | *Ursus maritimus* | *(none)* | Distant bears among a large gull flock. The agent declined rather than guessed. |
| `REC-LAFR-02` | *Loxodonta africana* | *(none)* | Distant elephants. **Correct species was in Top-5** — a confidence-gate outcome, not a classification failure. |
| `REC-GCAM-01` | *Giraffa camelopardalis* | *(none)* | Giraffe behind a fence. **Correct species in Top-5.** |
| `REC-GCAM-02` | *Giraffa camelopardalis* | *Giraffa reticulata* | **Taxonomic concept difference.** The reticulated giraffe is treated by some authorities as a distinct species and by others as a subspecies of *G. camelopardalis*. Scored as a miss because ground truth is exact-binomial by design. |
| `REC-PROS-02` | *Phoenicopterus roseus* | *Phoenicopterus ruber* | Backlit silhouette; colour information lost. A congener confusion, with the correct species in Top-5. |

`REC-GCAM-02` is a **dataset limitation**, not an agent error: the benchmark's
exact-name rule cannot express "either concept is acceptable". Recorded, not
fixed — Phase 3 may not edit expectations.

---

## 5. Taxonomy results

| Measure | Result |
| --- | --- |
| GBIF identifier correct | 18 / 24 judged |
| NCBI taxid correct | 18 / 24 judged |
| **Taxonomy failures on a correctly identified species** | **0** |
| Cases where taxonomy executed | 30 |
| Cases where taxonomy degraded | **0** |

Every one of the 12 identifier failures carries the reason code
`identifier_mismatch_after_species_mismatch` — meaning the classifier had already
named the wrong species and taxonomy then faithfully resolved *that* species.
**Live GBIF and live NCBI returned the correct identifier on every case where the
species was correct.** The Phase 2 decision to split that reason code from a
plain `identifier_mismatch` is what makes this distinction visible.

This also closes the discrepancy Phase 1 recorded: live GBIF returned **5219416**
for *Panthera tigris*, matching the independently verified ground truth, where the
mock development fixture holds `5219426`. Live mode is correct; the mock fixture
remains unfixed, as instructed.

---

## 6. Delegation and tool selection

| Case | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| `DEL-01` | `needs_agent` → `Genome` | `needs_agent` → `Genome` | **PASS** |
| `DEL-02` | `needs_agent` → `Evolution` | `needs_agent` → `Evolution` | **PASS** |
| `DEL-03` | `completed` (resume key present) | `completed`, *Panthera leo* identified | **PASS** |

`agent_tool_selection` passed on all 3 cases where it applies: candidates
non-increasing and distinct, primary species equal to the top-ranked candidate,
`needs_agent` carrying `output=None`, and no unsupported capability routed
instead of declined. Recognition never called a peer.

---

## 7. Consistency

| Field | Value |
| --- | ---: |
| Cases | 5 — `REC-PLEO-01`, `REC-PLEO-02`, `REC-PTIG-01`, `AMB-01`, `DEL-01` |
| Repetitions each | 3 |
| Executions expected / performed | **15 / 15** |
| **Stable cases** | **5 / 5** |
| Differing fields | none, on any case |

Compared: `status`, `decision`, `primary_species`, `candidate_order`,
`target_agent`. Explanation wording deliberately not compared. The subset was
fixed in Phase 2, before any live result existed. These 15 executions are held
separately from the 35 certified primary results and are not mixed into them.

---

## 8. Latency and failure summary

| Measure | All 35 cases | Workflow cases only (32) |
| --- | ---: | ---: |
| Min | 1.3 ms | 5 738 ms |
| Median | 7 454 ms | 7 937 ms |
| Mean | 8 277 ms | 9 053 ms |
| p90 | 11 828 ms | 11 828 ms |
| p95 | 12 097 ms | — |
| Max | 31 260 ms | 31 260 ms |
| Total | 289 691 ms | — |

The 1.3 ms minimum is an input-validation refusal that never reaches a provider.
The 31 260 ms maximum is the first case of the run — a cold start on the remote
Space. The "workflow only" column excludes the three validation refusals so the
provider-path figures are not flattered by them.

### Provider failure counts

| Provider event | Count |
| --- | ---: |
| Azure planner calls spent | 30 |
| Azure planner results **accepted** | **5** |
| Azure planner results **not accepted** | **25** |
| Azure explainer results accepted | 1 |
| BioCLIP-2 classification failures | **0** |
| Taxonomy degraded | **0** |
| Case execution errors | **0** |

---

## 9. Weakness classification

**Corrected classification.** An earlier version of this table called all 25
planner cases `provider_outage_or_timeout`. That asserted a cause nobody
measured, and it has been withdrawn. `provider_outage_or_timeout` is now reserved
for cases carrying **actual outage evidence** — a controlled unavailability error
code, a degraded taxonomy report, or a transport-level execution error. The run
has **none of those**:

```
CLASSIFICATION_UNAVAILABLE : 0
TAXONOMY_UNAVAILABLE       : 0
taxonomy_degraded          : 0
execution_error            : 0
```

| Kind | Cases | Detail |
| --- | ---: | --- |
| `planner_output_unaccepted` | **25** | A planner call was spent and no plan was accepted; the deterministic plan took over and the explainer was forfeited. `plan_rejected` is **false** on all 25, so this was *not* a schema rejection — the provider returned nothing usable. |
| `explainer_output_unaccepted` | **4** | A second call was spent and the explanation was not accepted as grounded; the deterministic sentence stood. |
| `confidence_decision_issue` | 4 | `REC-UMAR-02`, `REC-LAFR-02`, `REC-GCAM-01`, `REC-AMEL-01` — the gate was more conservative than the case's accepted set. |
| `correct_species_only_in_top5` | 4 | `REC-LAFR-02`, `REC-GCAM-01`, `REC-GCAM-02`, `REC-PROS-02` |
| `wrong_classification` | 2 | `REC-PPAR-02`, `REC-UMAR-02` |
| `provider_outage_or_timeout` | **0** | No outage evidence exists in the run |
| `planner_output_rejected` | **0** | No plan was rejected by `sanitize_plan` |
| `taxonomy_issue` | **0** | No taxonomy failure on a correctly identified species |
| `tool_agent_selection_issue` | **0** | — |
| `consistency_issue` | **0** | 5/5 stable |
| `dataset_limitation` | 1 | `REC-GCAM-02` — see §4 |

**Why the cause cannot be named.** `adapters/reasoning_llm.py` maps timeout, auth
failure, rate limit, empty body, unparseable JSON and schema violation onto the
same `None`, logging only `type(exc).__name__` and never the body — deliberately,
so a provider error cannot echo the request. The evaluation did not capture that
type. So the honest statement is *"a call was spent and nothing usable came
back"*, and `planner_output_unaccepted` says exactly that and no more.

### The dominant finding: the Azure planner under back-to-back load

25 of 30 planner calls were spent without their result being accepted, and the
explainer was accepted only once. The agent's behaviour was **exactly correct**
throughout: provenance reports `plan_source: deterministic`,
`explanation_source: deterministic`, `reasoning_llm_used: false` and
`reasoning_llm_calls: 1`, and the deterministic path produced a complete, valid
answer every time. `task_completion` and `output_schema` are 35/35 precisely
because the fallback works.

The exact provider-side cause is **not recorded**, by design: the adapter logs
only `type(exc).__name__` and never the exception body, so nothing about the
failed request can leak. This is consistent with the capacity note carried since
Phase 0 (§16.16: the shared deployment has previously returned HTTP 429 under
back-to-back load). The same deployment answered the Part 1 smoke check
perfectly — 2 logical calls, 2 HTTP attempts, `plan_source: llm`,
`explanation_source: llm` — when it was the only request in flight.

**Not fixed, and not worked around.** No retry was added, no pacing was
introduced, no threshold moved. It is reported as a capacity finding.

---

## 10. LLM call accounting

| Counter | Value |
| --- | ---: |
| **Agent** LLM calls, total across 35 cases | **35** |
| Agent LLM calls, maximum in any one request | **2** |
| Agent call ceiling | **2** |
| Cases with call evidence | 30 |
| **Evaluator** LLM calls, total | **0** |
| Evaluator judge model used | **false** |

Counted separately, never summed. The ceiling held on every case: 25 cases made
one call (planner spent, explainer forfeited after the planner failed — the
agent's documented rule that a failed planner does not get to retry as an
explainer), and 5 cases made two.

**Zero retry:** the runner issues exactly one `agent.run` per case and never
re-runs. `SDK_MAX_RETRIES = 0` in the Azure adapter; the remote classifier makes
one bounded attempt; NCBI paces rather than retries. Live per-request instrumented
counts are not available, so the metric is recorded as `no_evidence` rather than
claimed.

---

## 11. Human rubric — and one honest gap

Method: the Phase 2 **documented human rubric**. No LLM judge was added.

The worksheet at `evaluation/certified/human_review_worksheet.md` (and `.json`)
carries all 35 cases with, for each: case id, sanitized instruction, a sanitized
restatement of the final answer, a proposed relevance score, a proposed human
task-completion score, short evidence for each, and empty
`leith_approved_score` / `leith_review_note` fields. Every proposed score is
labelled **`provisional_machine_assisted`** and is **not** a recorded human score.

### The explanation gap is closed — by a separate, labelled run

The Phase 2 allow-list did not preserve the explanation text, so the certified
run has none. Rather than re-run certified cases, a **supplemental
explanation-capture pass** was executed and labelled
`supplemental_explanation_capture: true`:

| Field | Value |
| --- | ---: |
| Cases executed | **35 / 35** |
| Sanitized answers captured | **35 / 35 (100 %)** |
| Wall time | 302.7 s |
| **Biological differences from the certified run** | **0** |
| Used for | relevance, explanation quality, human task-completion review |
| Used for any metric or accuracy figure | **never** |

Zero differences means the supplemental pass reproduced every certified status,
decision, primary species and delegation target. That is reassuring, and it is
**not** treated as a second certification — the certified numbers stand on the
primary run alone.

Evaluation-only support was added for this: `sanitize_answer()` in
`results_schema.py` redacts data URLs, base64 blobs, credentials, key blocks,
absolute paths and dotenv assignments, collapses whitespace, and truncates at a
documented **2000 characters** (the same bound the agent's own grounding check
uses). It is covered by 14 unit tests including a canary per redaction class. No
runtime file was touched; the agent was never asked to change what it says.

### Approved human scores

Leith approved all 105 without modification. Reviewer recorded as **Leith**;
`scoring_basis` is now `human_approved` on every row.

| Criterion | Mean | Median | 5 | 4 | ≤3 | ≥4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Relevance | **4.77** | 5 | 27 | 8 | 0 | 35/35 |
| Explanation quality | **4.86** | 5 | 30 | 5 | 0 | 35/35 |
| Human task completion | **4.63** | 5 | 22 | 13 | 0 | 35/35 |

**Human scores did not override any deterministic verdict**, and this was checked
rather than asserted: the deterministic `metric_totals` were serialized before
and after approval and compared — they are **identical**. Six cases carry a human
score of 4 or 5 alongside a **failed** Top-1 (`REC-PPAR-02`, `REC-UMAR-02`,
`REC-LAFR-02`, `REC-GCAM-01`, `REC-GCAM-02`, `REC-PROS-02`). Those failures
stand. A high score there means the agent communicated well about a result that
was still biologically wrong, which is exactly the separation the rubric exists
to preserve.

**Two false-positive heuristics were caught and fixed before these numbers were
produced**, both mine and neither the agent's:

1. My grounding check flagged ordinary sentence openings — "Both taxonomy",
   "Species classification", "Could you" — as ungrounded binomials. It now reuses
   the agent's own `_EXPLANATION_SAFE_WORDS`, extended with the openings the
   *clarification question* uses, since the captured answer includes it.
2. My probability check fired on the bare word "probability", which appears in
   the agent's own sentence *"...a ranking score …, not a calibrated
   probability"* — i.e. it penalised the agent for **denying** a probability
   claim. It now uses the agent's exact phrase list.

Before the fixes, 30 of 35 answers scored 2. After, none scores below 4. The
difference was entirely in my heuristic.

**No human score overrides a deterministic verdict.** The worksheet shows
`deterministic_top1` and `deterministic_decision` alongside each row for context
only; they are not editable there.

---

## 12. Deliverables

| Path | Tracked location | Contents |
| --- | --- | --- |
| `evaluation/results/live_primary_*.json` | **git-ignored** | Raw primary run |
| `evaluation/results/live_consistency_*.json` | **git-ignored** | Raw consistency rounds |
| `evaluation/certified/certified_results.json` | tracked location | Sanitized permanent result, allow-listed fields only |
| `evaluation/certified/consistency_report.json` | tracked location | Consistency verdicts and observations |
| `evaluation/certified/human_review_worksheet.json` | tracked location | Worksheet data |
| `evaluation/certified/human_review_worksheet.md` | tracked location | Worksheet for Leith |
| `evaluation/results/live_supplemental_*.json` | **git-ignored** | Raw supplemental capture run |
| `docs/PHASE3_SPRINT4_LIVE_EVALUATION_REPORT.md` | tracked location | This report |

Nothing is staged or committed.

### Privacy scan of every certified artefact

| Pattern | Files matching |
| --- | ---: |
| `data:image`, `;base64,`, `-----BEGIN`, `Authorization`, `Bearer `, `api_key`, `AZURE_OPENAI`, email addresses | **0** |
| Base64-like runs ≥ 200 chars | **0** |
| Absolute paths, drive letters, `/Users/`, `/home/` | **0** |

Provenance is written through the Phase 2 allow-list, so a new provider field
cannot widen the file on its own. Exceptions contribute only a class name.

---

## 13. One evaluator correction — disclosed in full

The live run exposed a defect in a Phase 2 **evaluator**, not in the agent.

`evaluate_provenance` treated `reasoning_llm_calls > 0` with
`reasoning_llm_used = false` as a provenance contradiction. It is not.
`workflows/state.py:65-66` states plainly: *"`used` is True only when a result was
genuinely accepted."* A call that was spent and then failed leaves exactly that
combination, and the agent is reporting it correctly.

The offline dry run could never have caught this: `FakeGPT5MiniProvider` always
succeeds, so `plan_source` was always `llm` and `used` always `true`.

| Item | Detail |
| --- | --- |
| What changed | The rule in `evaluation/evaluators.py` only |
| What did not change | No runtime file, no prompt, no threshold, no provider, no manifest, no expectation |
| Effect on results | 25 spurious `provenance_truthfulness` failures → **0** genuine failures |
| Was the agent re-run? | **No.** The metric was recomputed from the preserved observations, which already contain the full provenance |
| Regression cover | 4 new tests pin the corrected rule, including the reverse states that *are* impossible (`plan_source: llm` with `used: false`; `used: true` with 0 calls) |

The raw run file retains the original computed metrics; the certified file
records `provenance_metric_recomputed: 25` so the correction is auditable.

---

## 14. Tests after live execution

| Suite | Command | Result |
| --- | --- | --- |
| Phase 1 dataset | `pytest tests/test_sprint4_phase1_evaluation_dataset.py -q` | **81 passed** |
| Phase 2 runner/evaluators | `pytest tests/test_sprint4_phase2_evaluation_runner.py -q` | **130 passed** (126 + 4 new) |
| The three corrected security gates | `pytest tests/test_recognition_only_gate.py tests/test_phase7_resilience_security.py tests/test_phase8_demo_and_closure.py -q` | **337 passed** (324 + 13 new) |
| Complete Recognition offline suite | `pytest backend/agents/multimodal_recognition_agent/tests -q` | **1445 passed, 0 failed**, 1 warning |

Arithmetic is exact: **1196** original + **13** new gate tests + **81** Phase 1 +
**155** Phase 2 = **1445**.

**All 1196 original tests pass.** The three that were failing now pass on the
corrected, stricter rule; one of them was renamed
(`test_13_2_no_model_weight_or_raw_image_is_present_in_the_package` →
`..._is_committed_in_the_package`) to match the contract it always enforced. No
other pre-existing test was modified, and no test was weakened or removed.

The single warning is still FastAPI's test-client deprecation, not Recognition
code.

### The three stale Sprint 3 gates — fixed under authorization

Three gates were failing because they scanned the **filesystem** —
`PACKAGE.rglob("*")` — for image suffixes, excluding only `.venv`,
`__pycache__`, `demo_images` and `.env`. That proxy for "committed" held while
`fixtures/demo_images/` was the only git-ignored image directory. Phase 1 added a
second one, `evaluation/assets/`, and Part 1 filled it with 33 benchmark images.

All three now ask **git**, via one shared helper in `tests/conftest.py`:

```python
def git_tracked_paths(root=PACKAGE_ROOT):   # git ls-files -z -- <root>
def forbidden_tracked(paths, suffixes=FORBIDDEN_TRACKED_SUFFIXES)
```

| Test | Change |
| --- | --- |
| `test_recognition_only_gate.py::test_no_secret_model_weight_or_raw_image_is_committed` | Filesystem walk → `forbidden_tracked(git_tracked_paths(PACKAGE))`; also asserts `.env` is not tracked |
| `test_phase7_resilience_security.py::test_13_2_no_model_weight_or_raw_image_is_committed_in_the_package` | Same substitution; renamed from `..._is_present_in_the_package`, because *committed* was always the contract and *present* never was |
| `test_phase8_demo_and_closure.py::test_no_tracked_file_under_the_package_is_an_image_weight_or_log` | Same substitution, keeping its extra `.csv` suffix |

**The rule got stricter, not weaker.** Under the old walk, anything under a
skipped directory name was exempt: a weight genuinely committed inside
`demo_images/` would have passed. Git-truth does not care where a file sits, only
whether it is tracked. `test_a_forbidden_file_is_caught_even_inside_an_ignored_directory_path`
pins exactly that.

Ten new tests were added alongside:

- `test_a_tracked_forbidden_file_would_fail_the_gate` — parametrised over 9
  suffixes (`.png`, `.JPG`, `.jpeg`, `.webp`, `.pt`, `.safetensors`, `.ckpt`,
  `.bin`, `.log`), proving the gate still bites;
- `test_a_forbidden_file_is_caught_even_inside_an_ignored_directory_path`;
- `test_the_ignored_benchmark_assets_do_not_fail_the_gate` — the 33 real assets
  are on disk, none is tracked, and the gate passes;
- `test_the_benchmark_asset_directory_is_ignored` — `assets/` and `results/` are
  both in `evaluation/.gitignore`;
- `test_no_forbidden_file_is_tracked_anywhere_under_the_package` — stated on its
  own, and refusing to pass vacuously if git reports nothing.

Verified: `git ls-files` under the package, filtered to image suffixes, returns
**0**; `git check-ignore -v` attributes `assets/AMB-01.jpg` to
`evaluation/.gitignore:8`; `git status --short` shows no image staged or tracked.

---

## 15. Limitations and no-overclaim statement

1. **This is a bounded 36-case benchmark.** 24 real-recognition cases across 12
   species. Nothing here supports a general claim about BioCLIP-2's accuracy, or
   about species recognition on any other distribution of images.
2. **75 % Top-1 and 91.7 % Top-5 are measurements of this set**, under one live
   configuration, on one day. They are not a capability claim and no target was
   set for them to meet.
3. **The Azure planner was largely unavailable during the run** (25 of 30 calls
   not accepted). The agent's deterministic fallback carried the run, so these
   results substantially measure the *deterministic* path. A run with a healthy
   planner could differ — in either direction.
4. **All human scores are provisional.** Relevance, explanation quality and
   human task completion are proposed by rule from preserved evidence and
   labelled `provisional_machine_assisted`. None is a recorded human score until
   Leith signs it off.
9. **Explanation text comes from a supplemental run**, not the certified one. It
   reproduced the certified biology exactly (0 differences), but it is a second
   execution on a different day-part and is never treated as certification.
5. **`zero_retry` has no live evidence.** It is asserted by code review and by the
   Part 1 smoke check, not by per-case instrumentation.
6. **One case is a taxonomic-concept disagreement**, not an error
   (`REC-GCAM-02`, §4).
7. **`DEP-01` was not run live**, by manifest declaration; the dependency-failure
   path is covered offline only.
8. **Ground truth is exact-binomial.** A congener or a subspecies-versus-species
   difference scores as a miss with no partial credit, by design.

---

## 16. What Leith is being asked to approve

**Nothing is outstanding.** Leith approved all 105 rubric scores without
modification, closing the last item.

Settled during closure:

1. **Human rubric** — 105/105 approved by Leith, recorded as `human_approved`
   (§11).
2. **Three security gates** — corrected to ask git; strictly stricter than before
   (§14).
3. **Provenance evaluator** — the Phase 2 rule was wrong; the agent was not
   (§13).
4. **Weakness classification** — corrected; nothing is called an outage without
   outage evidence, and the run contains none (§9).
5. **Supplemental capture run** — separate, labelled, feeds no metric (§11).
6. **25 unaccepted planner calls** — reported, not fixed (§9).

---

## 17. Phase 3 gate checklist

| Gate | Verdict | Basis |
| --- | --- | --- |
| Live guard enforced (`--mode live` + `RECOGNITION_EVAL_LIVE=1`) | **PASS** | §1 |
| All modes real before execution; abort otherwise | **PASS** | `azure` / `remote` / `real`, preflight PASS (§1) |
| Every live-compatible case has a result | **PASS** | 35/35, 0 execution errors (§2) |
| Manifest ordering and case ids preserved | **PASS** | §2 |
| No retry; no case re-run to replace a result | **PASS** | one `agent.run` per case; `SDK_MAX_RETRIES = 0` (§10) |
| No mock, fixture or silent fallback | **PASS** | provenance shows real providers throughout (§2) |
| Provenance proves what ran | **PASS** | 30/30 judged, 0 failures (§3) |
| All five Sprint 4 criteria measured | **PASS** | correctness §4, relevance §11, task completion §3/§11, agent-tool selection §6, consistency §7 |
| Consistency protocol executed | **PASS** | 15/15 executions, 5/5 stable (§7) |
| Weaknesses classified | **PASS** | §9, corrected taxonomy of kinds |
| Human rubric applied and approved | **PASS** | 105/105 approved by Leith (§11) |
| Human scores never override deterministic verdicts | **PASS** | metric totals identical before/after approval; 6 failed-Top-1 cases keep their failure (§11) |
| No minimum accuracy target invented | **PASS** | none set; §4 and §15 state this explicitly |
| No sensitive content in any deliverable | **PASS** | §12, all patterns 0 |
| Raw results and images git-ignored | **PASS** | 0 tracked each (§12) |
| Runtime behaviour untouched | **PASS** | §11 (original numbering) and the file list below |
| All tests pass | **PASS** | see §14 |
| Original 1196 baseline intact | **PASS** | all 1196 pass (§14) |

**Phase 3 — PASS.** Not committed; awaiting the commit instruction.

---

*Compiled 2026-08-26 on branch `group-d-recognition-sprint4` at
`35372dee229bcdefe548d6a4cb5406aab8defb21`. No Recognition runtime file was
modified. No prompt, threshold, provider configuration, manifest entry or
benchmark expectation was changed. No case was re-run to replace its result.
Nothing was committed or pushed.*
