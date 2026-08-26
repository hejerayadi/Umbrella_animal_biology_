# Phase 4 Report — LangSmith Foundation

## Status
PASS

## Owner
Chahd

## Branch and commit
- Branch: `group-d-recognition-sprint4`
- Starting commit: `1bb2b55` (Leith's Phase 3 handoff — "Complete Recognition Sprint 4 agent evaluation")
- Ending commit: `91aafb5` — "Sprint 4 Phase 4: LangSmith foundation, config and privacy boundary"
- Working tree: clean, pushed to `origin/group-d-recognition-sprint4`

## Handoff verification (before any change)

1. Confirmed HEAD was exactly `1bb2b55` and the working tree was clean
   (`git status` → "nothing to commit, working tree clean").
2. Confirmed the Phase 0–3 reports were present and committed:
   `PHASE0_SPRINT4_BASELINE_REPORT.md`, `PHASE1_SPRINT4_EVALUATION_DATASET_REPORT.md`,
   `PHASE2_SPRINT4_EVALUATION_RUNNER_REPORT.md`, `PHASE3_SPRINT4_LIVE_EVALUATION_REPORT.md`.
3. Confirmed the evaluation deliverables existed as committed files:
   `tests/test_sprint4_phase1_evaluation_dataset.py`,
   `tests/test_sprint4_phase2_evaluation_runner.py`.
4. Ran the complete Recognition suite on the handoff commit: **1445 passed, 0 failed,
   1 warning** (the warning is the pre-existing `httpx`/`starlette` deprecation notice,
   unrelated to Recognition code).
5. No blocker found. Handoff accepted; Phase 4 implementation began.

## Scope executed

Added the minimal, optional LangSmith tracing foundation and its privacy boundary,
with no workflow instrumentation (reserved for Phase 5) and no change to any public
output key or decision path.

- Inspected installed versions: `langgraph==1.2.10`, `langchain-core==1.5.6`,
  `langsmith==0.11.0` (already present transitively via `langchain-core`).
- Declared `langsmith==0.11.0` explicitly in `requirements.txt` at its already-resolved
  version — no broad dependency upgrade.
- Added `LangSmithConfig` (frozen dataclass, same pattern as `ValidationConfig` /
  `ThresholdConfig`) and a `_langsmith_config()` resolver to `config.py`, wired into
  `RecognitionConfig.from_env()`.
  - `LANGSMITH_TRACING` disabled by default; disabled mode reads nothing else and
    needs no credential or network access.
  - Enabling tracing without `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` raises
    `ConfigError` at startup, naming the missing variables but never echoing a
    configured value.
  - `LANGSMITH_ENDPOINT` and `LANGSMITH_WORKSPACE_ID` remain optional.
- Added `observability.py`: the one central Recognition observability boundary.
  - `RecognitionTracer` — disabled by default; builds no client and makes no network
    call unless `tracing_enabled` is true. Client construction is wrapped in the
    constructor so *any* setup failure (network, SDK, or otherwise) degrades to
    "tracing disabled" rather than raising into `RecognitionAgent` construction.
  - `safe_metadata()` — allowlist-only filter (`ALLOWED_METADATA_KEYS`); anything
    outside the allowlist is dropped, not redacted, so a disallowed key's mere
    presence is never revealed. A second content check drops any value containing
    `data:image`, `base64,`, or `-----BEGIN`, even under an allowlisted key.
  - `record()` never raises: a `safe_metadata` or export failure is logged and
    swallowed, so a trace-export outage can never change an `AgentResult`.
- Added `.env.example` placeholders for all five LangSmith variables, values left
  blank, tracing defaulted to `false`.

## Files changed

- `config.py` — added `LangSmithConfig`, `_langsmith_config()`, wired into
  `RecognitionConfig`.
- `observability.py` — new file; the tracing boundary described above.
- `tests/test_sprint4_phase4_langsmith_foundation.py` — new file; 20 focused tests.
- `.env.example` — added LangSmith placeholder block.
- `requirements.txt` — added explicit `langsmith==0.11.0` pin.

## Tests

- Command: `python -m pytest tests\ -v`
- Collected: 1466
- Passed: 1466
- Failed: 0
- Skipped: 0
- Warnings: 1 (pre-existing `httpx`/`starlette` deprecation notice, unrelated to this
  phase)

Focused Phase 4 tests (`tests/test_sprint4_phase4_langsmith_foundation.py`, 20 tests)
cover: disabled-by-default behaviour; no client/network activity while disabled;
loud failure on incomplete enabled configuration, with no configured value echoed
in the error; the metadata allowlist and its content filter; exporter and client
construction failures being fully contained; and that `RecognitionAgent`
construction and the two-call ceiling are unaffected by the new configuration.

Two tests failed on first run and were corrected during this phase (see Blockers
below); both now pass and no other test was changed.

## Evidence

- `pip show langgraph langchain-core langsmith` confirmed `langsmith==0.11.0` was
  already resolved transitively before this phase began.
- Full suite run before implementation (handoff verification): 1445 passed.
- Full suite run after implementation: 1466 passed (1445 baseline + 20 new Phase 4
  tests, with the 1 pre-existing warning unchanged).

## Protected checks

- Seven keys unchanged: yes — no workflow node or output builder was touched.
- Maximum two agent LLM calls: yes — `test_two_call_ceiling_is_unchanged` and
  `MAX_REASONING_LLM_CALLS_PER_REQUEST` untouched.
- Zero retry: yes — no retry logic added; `observability.py` makes no HTTP call in
  Phase 4 beyond LangSmith client construction, itself gated by explicit opt-in.
- No image/secret leak: yes — `safe_metadata()` allowlist plus forbidden-substring
  filter tested directly; `ConfigError` messages tested to never echo a configured
  credential.
- No RAG/Qdrant: yes — no vector, embedding, or retrieval code introduced.

## Blockers/limitations

Two self-authored tests initially failed and were corrected within this phase
(not blockers to the phase, but recorded for transparency):

1. `test_the_module_imports_no_langsmith_client_at_module_level` used a regex that
   also matched the deliberate lazy `from langsmith import Client` import nested
   inside `_build_client`. Corrected to only match a column-0 (true module-level)
   import.
2. `test_client_construction_failure_leaves_tracer_disabled` monkeypatched
   `_build_client` entirely, which also removed that method's own internal
   try/except — the test was bypassing the very safety net it meant to test. Fixed
   by moving the safety net up into `RecognitionTracer.__init__` as well, so client
   construction failure is caught at both levels (defense in depth), and the test
   now exercises real code paths rather than a hollowed-out replacement.

No other blockers. No workflow instrumentation was added in this phase, by design —
that is Phase 5's scope.

## Gate decision
- Handoff verified before implementation: PASS
- Tracing optional and disabled by default: PASS
- Only allowlisted metadata reaches the capture boundary: PASS
- Exporter/client failure cannot affect the agent: PASS
- All tests pass: PASS
- No public/decision contract change: PASS

**Phase 4: PASS**
