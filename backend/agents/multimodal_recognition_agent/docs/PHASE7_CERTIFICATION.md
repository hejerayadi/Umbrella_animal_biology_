# Phase 7 — Resilience, Security, Privacy and Operational Quality

**Status: PASS**

All external waits are bounded, no retry loop exists, concurrent requests share nothing,
LLM budgets reset per request, prompt injection cannot touch the scientific evidence, and
no request produces an HTTP 500.

The three findings from the first pass — P7-F1, P7-F2 and P7-F3 — are **corrected and
verified**, offline and live. The complete offline suite is green with **zero xfails**, and
a live run at DEBUG (the level that previously exposed the Azure endpoint, the deployment
name and the NCBI email) now emits six log lines, all of them the agent's own.

Phase 6 is `N/A` — Global Orchestrator integration does not currently exist.

---

## 1. Baseline

| Item | Value |
| --- | --- |
| Branch | `group-d-recognition-sprint3` |
| HEAD SHA | `83ae275f0208f7475b155228e7a6785ea5bc1fce` |
| HEAD subject | `phase 5` |
| Working tree at start | Clean |
| Offline suite at baseline | **923 passed, 0 failed, 0 skipped, 0 xfailed** |

---

## 2. Files created or changed

**Implementation (authorized scope only):** `api.py` — F1, F2 and F3.
`git diff --stat HEAD` shows **exactly that one file**.

**Created (untracked, none committed):**

- `tests/test_phase7_resilience_security.py`
- `docs/PHASE7_CERTIFICATION.md` — this report
- `docs/phase7_results.json`

**Untouched, as required:** BioCLIP, taxonomy and NCBI rate limiting, reasoning LLM,
LangGraph workflow, state, confidence, ranking, schemas and the seven-key contract, Phase
3–5 tests, Global Orchestrator, registry, frontend and other agents. No unrelated
refactoring. Nothing committed or pushed.

---

## 3. Test counts

| Suite | Before correction | After correction |
| --- | --- | --- |
| Complete offline suite | 1120 passed, **2 xfailed** | **1140 passed, 0 failed, 0 skipped, 0 xfailed** |
| Phase 7 module | 197 passed, 2 xfailed | **217 passed, 0 xfailed** |
| Phase 3 regression | 141 passed | **141 passed** |
| Phase 4 regression | 172 passed | **172 passed** |
| Phase 5 regression | 113 passed | **113 passed** |
| Architecture + no-leak gates | 63 passed | **63 passed** |

Baseline was 923. **Zero xfails remain.** The two strict xfails encoding P7-F1 and P7-F2
became ordinary passing regression tests; the P7-F3 test that documented the gap was
replaced by tests asserting the policy. **No test that discovered a finding was weakened,
skipped or deleted** — each was strengthened into a positive assertion of the fix.

---

## 4. The corrections

### P7-F1 — no exception detail crosses the boundary

`api.py` no longer derives anything from the exception: not `str(exc)`, not `repr(exc)`,
not its args, not a provider error body, not a traceback. It returns a fixed, data-free
message and logs only safe diagnostics.

**Before**

```json
{"status":"failed","output":"Multimodal Recognition Agent error: internal detail CANARYSECRETINEXCEPTION9z"}
```

**After**

```json
{"status":"failed","output":{"error_code":"INTERNAL_ERROR","error":"The Recognition Agent encountered an internal error and could not complete this request. No species was identified. The failure has been logged for the operators of the service."}}
```

**Log after** — stage, exception *class*, and a fresh correlation id:

```
[Recognition] stage=api.execute request_id=<hex> unexpected ValueError; returning a controlled internal error.
```

Two different exceptions now produce byte-identical response text, so the text itself
cannot become a channel for what went wrong. Asserted.

### P7-F2 — the structured failure contract holds on every path

`output` is now always the same `{error_code, error}` dict every other failure returns —
never a string — so a caller can read `output["error_code"]` on every branch.

**On the error code, as the authorization required:** I checked the existing error model
before inventing anything. `ErrorCode` has no generic internal/unexpected member, and
neither the orchestrator nor any other agent defines one. `WORKFLOW_INCOMPLETE` — the only
existing internal literal, emitted by `workflows/graph.py` — was **deliberately not
reused**: it means *"the workflow ran and produced no result"*, a narrower and different
claim than *"an unexpected exception escaped"*. `INTERNAL_ERROR` is therefore a new
literal, following `graph.py`'s own precedent of a bare string, because `domain/errors.py`
is outside this correction's boundary. Its canonical home is the `ErrorCode` enum once
that file may change. A test pins this reasoning so it cannot be quietly undone.

### P7-F3 — the dependency logging policy

`api.py` pins `httpx`, `httpcore`, `urllib3` and `openai` to `WARNING` at service startup,
so they cannot emit INFO/DEBUG request URLs **even when the root logger is at DEBUG**.

| Aspect | Detail |
| --- | --- |
| Mechanism | `logging.getLogger(name).setLevel(logging.WARNING)` |
| Uses the existing approved pattern | Yes — `backend/api.py:29` already quiets `httpx` exactly this way |
| Competing system created | No — the repository has no central `dictConfig`, only per-entry-point `basicConfig` |
| Requests modified | None |
| `NCBI_EMAIL` removed from the NCBI request | No — Entrez usage guidelines require it, and it is still sent |
| Provider behaviour changed | None |
| Real configuration printed | Never |

**Preserved on purpose:** the agent's own `[Recognition]` INFO stage trail, and `WARNING`
and above from the quieted namespaces — quieting must not blind an operator to a real
transport problem. Both asserted.

> **`openai` is a fourth namespace beyond the three you named.** It is included because it
> is the Azure SDK's own logger and can emit request options containing the URL at DEBUG.
> Flagged here rather than added silently; it is trivially removable if you would rather
> keep the policy to exactly three.

---

## 5. Required regression tests — all 15

| # | Requirement | Test |
| --- | --- | --- |
| 1 | Canary does not leak into the HTTP response | `test_p7_f1_an_unexpected_exception_message_never_reaches_the_caller` |
| 2 | Canary does not appear in logs | `test_p7_f1_the_log_records_type_and_correlation_id_but_no_message` |
| 3 | Failure output is a dictionary | `test_p7_f2_the_unexpected_path_returns_the_structured_failure_contract` |
| 4 | Dictionary contains exactly the safe fields | same — `set(output) == {"error_code", "error"}` |
| 5 | Structured, never HTTP 500 | `test_p7_f1_the_never_500_guarantee_still_holds_on_this_path` |
| 6 | Request ID preserved | `test_p7_f1_the_log_records_type_and_correlation_id_but_no_message`, `..._is_fresh_per_request` |
| 7 | Controlled provider failures unchanged | `test_p7_f1_a_recognition_error_is_unaffected_by_the_correction` |
| 8 | Root at DEBUG, `httpx` hides the Azure endpoint | `test_p7_f3_httpx_cannot_expose_the_azure_endpoint_at_root_debug` |
| 9 | `httpcore` hides deployment and URL details | `test_p7_f3_httpcore_cannot_expose_the_deployment_or_url_details` |
| 10 | `urllib3` hides the NCBI email | `test_p7_f3_urllib3_cannot_expose_the_ncbi_email_in_any_encoding` |
| 11 | Raw + `quote` + `quote_plus` forms scanned | same, parametrised over all three; plus `..._defeats_a_naive_leak_scan` |
| 12 | Azure API-key canary never appears | `test_p7_f3_the_azure_api_key_canary_appears_in_no_logger_at_any_level` |
| 13 | Recognition's own stage/error-code logs remain | `test_p7_f3_recognition_error_code_logging_survives_the_policy` |
| 14 | Seven-key successful responses unchanged | `test_p7_f3_a_successful_seven_key_response_is_unchanged_by_the_policy` |
| 15 | No Phase 3–5 regression | 141 / 172 / 113, all unchanged |

---

## 6. Validation run

| # | Step | Result |
| --- | --- | --- |
| 1 | Focused Phase 7 tests | **217 passed, 0 xfailed** |
| 2 | Complete offline suite | **1140 passed, 0 failed, 0 skipped, 0 xfailed** |
| 3 | Phase 3 regression | 141 passed |
| 4 | Phase 4 regression | 172 passed |
| 5 | Phase 5 regression | 113 passed |
| 6 | Architecture + no-leak gates | 63 passed |
| 7 | Bounded normal live smoke | ALL_PASS — see below |
| 8 | Artifact and lingering-process scans | Clean |

### Live smoke, run at DEBUG

Azure GPT-5 mini + remote BioCLIP-2 + live GBIF + live NCBI. No real service was made to
fail; every failure scenario uses injected clients.

| | |
| --- | --- |
| Result | `completed` / `identified` — *Loxodonta africana* |
| GBIF / NCBI | 2435350 / 9785, `verified` |
| Provenance | `gbif_mode: real`, `ncbi_mode: real`, `taxonomy_executed: true` |
| LLM calls | 2, within budget |
| Latency | 30.2 s, inside the 240 s deadline |
| Seven keys | Exact |
| **`third_party_logger_leaks`** | **`[]`** |
| **Total log lines at DEBUG** | **6 — all `[Recognition]`** |
| API key / endpoint / deployment / NCBI email | Absent from response and logs |

That is the same scenario that produced the finding. Before the correction, a DEBUG run
filled the log with `httpcore` and `urllib3` lines carrying the endpoint, the deployment
and `email=…%40…`.

### Artifact and process scan

No secret, weight, image, cache, `.env`, log or temporary file is untracked-and-unignored.
Evaluation images deleted after use. No stray classifier temp file. No lingering process.

---

## 7. Everything from the first pass that still stands

The timeout matrix (10 cases), retry guarantees, the malformed-payload matrix (63 cases
across five sources), the partial-availability matrix, concurrency and isolation (8 workers,
zero crossings), LLM budget isolation (6 × exactly 2 calls), and prompt-injection
resistance (10 instructions, including one where the model *obeyed* the injection and still
changed nothing) are unchanged by these corrections and remain as certified.

---

## 8. Remaining limitations

**The two judgement calls, preserved as they were:**

1. **Opaque model prose is accepted verbatim.** Grounding restricts *species* and
   *probability* claims, not arbitrary text, because the model is the author of the
   wording. This is precisely why the factual disclosure is appended outside the model's
   text, and the candidate evidence is provably unchanged.
2. **Operational logging carries no per-stage duration, taxonomy degradation flag or
   explanation source.** Assessed as sufficient and left unchanged: the failed stage is
   identifiable from the last node log plus the error code, and all three fields are
   present in `recognition_provenance` on the response — no diagnostic information is
   lost, only its location differs.

**Introduced by the corrections:**

3. The dependency logging policy is applied at the **HTTP service entry point**. Code that
   imports `RecognitionAgent` directly rather than serving it — the smoke scripts do — does
   not pass through `api.py`. Covering that would need `agent.py`, outside the boundary.
4. `INTERNAL_ERROR` is a string literal rather than an `ErrorCode` member, because
   `domain/errors.py` was outside the boundary. Its canonical home is the enum.
5. The correlation id is **logged, not returned**. Surfacing it would add a third key to
   the failure output, which the authorization fixed at `{error_code, error}`.

**Pre-existing scope notes:**

6. Timeout tests prove the agent passes a bounded deadline to each client and maps every
   failure correctly; they do not measure real network waits.
7. Concurrency was exercised at 8 workers in one process.
8. The NCBI rate limiter is per-provider, so several agent instances would each hold their
   own budget. Worth knowing before scaling out.

---

## Acceptance gate

| Requirement | Result |
| --- | --- |
| All external waits are bounded | ✅ PASS |
| No retry loop exists | ✅ PASS |
| All failures are controlled or documented degradations | ✅ PASS |
| No HTTP 500 occurs | ✅ PASS |
| Concurrent requests share no state | ✅ PASS |
| LLM budgets reset per request | ✅ PASS |
| No secret, prompt or image data leaks | ✅ **PASS** |
| Prompt injection cannot alter scientific evidence | ✅ PASS |
| Provenance remains accurate | ✅ PASS |
| Operational evidence identifies the failed stage safely | ✅ PASS |
| Protected architecture and seven-key contract unchanged | ✅ PASS |
| Complete offline suite stays fully green | ✅ PASS — 1140 passed, 0 xfailed |

**Expected final state, confirmed:** zero failures; zero Phase 7 xfails; no exception-text
leakage; structured defensive failure; no dependency INFO/DEBUG URL or email leak;
scientific architecture unchanged.

**Phase 7: PASS.**

*Phase 8 not begun. Nothing committed or pushed.*
