# Phase 6 Report — Trace Continuity with the Global Orchestrator

## Status
PASS (implementation and tests) — **PENDING cross-team review before merge to any shared/deployed branch**

## Owner
Chahd

## Reviewers required
- Leith (Recognition Sprint 4 reviewer, per the sprint plan)
- The Global Orchestrator owner (required in addition to Leith, because this
  phase touches the shared HTTP contract between the Orchestrator and
  Recognition — see "Shared-boundary nature of this change" below)

This report documents completed, tested work. It does not represent sign-off
to merge. Per the plan, Phase 6 changes affecting the Orchestrator/Recognition
boundary require both approvals before landing anywhere shared.

## Branch and commit
- Branch: `group-d-recognition-sprint4`
- Starting commit: `7c3aa0e` — "Sprint 4 Phase 5: Recognition workflow and provider tracing"
- Ending commit: (fill in after `git push` — see Next Steps)
- Working tree: clean once committed

## Environment setup (prerequisite work, recorded for auditability)

Before any Phase 6 code was written, the plan's Required Action #1 and #2
("inspect and run existing integration tests" / "use an environment that can
run affected Orchestrator tests") were carried out in full:

1. Discovered the Orchestrator has no dedicated virtual environment and no
   `requirements.txt` had ever been located for it at first attempt; located
   its actual `requirements.txt` at `backend/orchestrator/requirements.txt`
   after a first empty search.
2. **Incident:** attempted to run the Orchestrator's tests using Recognition's
   own `.venv`. This upgraded packages Recognition pins exactly
   (`langgraph`, `pytest`) as a side effect, and left `langchain-core`
   above what Recognition ships with. Recognition's own suite still passed
   at the bumped versions (1480 passed), but the venv no longer matched
   Recognition's declared `requirements.txt`, so it was rebuilt from scratch
   rather than trusted.
3. Root cause of a related failure: `python -m venv` was invoked with
   whatever `python` was first on PATH at the time, which on this machine
   resolved to a 32-bit Python 3.10 system install rather than the 64-bit
   Python 3.13 both Recognition and the Orchestrator actually require. This
   caused unrelated build failures (`httptools`, `hf-xet`, `ormsgpack`
   attempting to compile from source with no C++/Rust toolchain present).
4. **Fix, now the standing procedure:** a separate, disposable virtual
   environment for Orchestrator diagnostics
   (`Umbrella_animal_biology_/.venv-orchestrator-diag`), built explicitly
   against the correct 64-bit Python 3.13 interpreter, is never used for
   Recognition work, and is safe to delete and rebuild at any time.
5. Recognition's own `.venv` was rebuilt from scratch against
   `requirements.txt` on the correct interpreter and reconfirmed at
   **1480 passed, 0 failed** before Phase 6 code was written.
6. The Orchestrator's own test suite was run in its disposable venv:
   **7 passed, 0 failed** (`backend/orchestrator/tests/test_responder.py`).
   This is the only test file that currently exists in
   `backend/orchestrator/tests/`.

This two-venv separation is now the standing procedure for any future
Orchestrator diagnostic work from this machine.

## Scope executed

Read (not modified) `backend/orchestrator/langgraph/graph.py`,
`backend/orchestrator/langgraph/nodes/resolver_node.py`, and
`backend/orchestrator/langgraph/nodes/worker_node.py` to establish the actual
integration surface before writing any code, per the plan's explicit warning
against touching `capability_resolver`, routing, or dispatch logic.

**Finding:** `worker_node.py` already sends `X-Trace-Id` and `X-Request-Id`
headers on every call to every worker agent, including Recognition
(`_call_agent`, in the `client.post(...)` call). This is pre-existing
Orchestrator behaviour; no Orchestrator file was changed. `resolver_node.py`
and `graph.py` needed no change and were not touched — confirming the plan's
scope boundary held in practice, not just in principle.

**Gap identified:** Recognition's `api.py` read no incoming headers at all;
`execute(request: AgentRequest)` only ever saw the JSON body. The `X-Trace-Id`
the Orchestrator was already sending arrived and was silently dropped.

**Work done, entirely inside Recognition's own package:**

- `api.py` — `execute` now accepts an optional `X-Trace-Id` header
  (`Header(default=None, alias="X-Trace-Id")`) and passes it to
  `RecognitionAgent.run(request, trace_id=...)`. Never validated or parsed;
  absence, an empty string, or an arbitrary/malformed value are all handled
  without affecting the response.
- `agent.py` — `RecognitionAgent.run` accepts an optional `trace_id` keyword
  argument and threads it into `RecognitionWorkflow.run`. Every existing call
  site that omits it (smoke scripts, direct `RecognitionAgent()` use,
  Phase 4/5 tests) is unaffected.
- `workflows/graph.py` — `RecognitionWorkflow.run` accepts the same optional
  `trace_id` and, only when truthy, adds one `trace_id` field to the single
  `finalize` root event Phase 5 already emits. No new event, no new node
  instrumented, no duplicate root — the plan's "add only missing spans, avoid
  duplicate roots" instruction from Phase 5 carries over unchanged into this
  phase.
- `observability.py` — `trace_id` added to `ALLOWED_METADATA_KEYS`. This is
  the only allowlist change; the rest of the Phase 4 privacy boundary is
  untouched.

## Files changed

- `api.py` — accept `X-Trace-Id` header, pass through.
- `agent.py` — accept and forward optional `trace_id`.
- `workflows/graph.py` — accept `trace_id`, attach to the `finalize` event only.
- `observability.py` — allowlist `trace_id`.
- `tests/test_sprint4_phase6_trace_propagation.py` — new file; 9 focused tests.

No file under `backend/orchestrator/` was changed.

## Tests

- Command: `python -m pytest tests\ -v`
- Collected: 1489
- Passed: 1489
- Failed: 0
- Skipped: 0
- Warnings: 1 (pre-existing `httpx`/`starlette` deprecation notice, unrelated to
  this phase)

Focused Phase 6 tests (`tests/test_sprint4_phase6_trace_propagation.py`, 9
tests) cover: a trace id passed directly to `RecognitionAgent.run` reaches only
the `finalize` event; the same is true when the id arrives via the HTTP
`X-Trace-Id` header; a request with no header carries no `trace_id` field
anywhere; the `AgentResult` returned is byte-identical with and without a
trace id; every existing call site that omits the argument is unaffected; an
empty-string trace id is treated as absent; an arbitrary or actively hostile
string (e.g. containing a SQL-injection-shaped or script-injection-shaped
payload) is echoed verbatim into the trace field and never raises, is never
parsed, and never affects the response; the field never leaks onto any event
other than `finalize`; and disabled tracing ignores the argument entirely,
touching no part of the metadata pipeline.

Two of the nine tests (`test_the_api_header_reaches_the_same_finalize_event`,
`test_no_header_means_no_trace_id_field_at_all`) monkeypatch the module-level
`api._agent` singleton to inject a tracer-observable agent for the duration of
one HTTP call through `TestClient`. This is flagged explicitly for review:
`api._agent` is normally built once at import time and never swapped in
production; the test pattern is standard for FastAPI singletons but touches
the same object real traffic uses, which is exactly the shared-boundary
surface this phase is about.

## Evidence

- Direct inspection of `worker_node.py` (quoted above) is the evidence that
  `X-Trace-Id` propagation from the Orchestrator side already exists and needed
  no change.
- Full suite run after implementation: **1489 passed, 0 failed** — 1480
  Recognition baseline (post-Phase-5) + 9 new Phase 6 tests, no regressions.
- No Orchestrator test was re-run after Recognition's changes, because no
  Orchestrator file was touched; there is nothing on that side for Recognition's
  change to have affected.

## Protected checks

- Seven keys unchanged: yes — `test_the_agent_result_is_identical_with_and_without_a_trace_id`
  confirms byte-identical `status` and `output`.
- Maximum two agent LLM calls / zero retry: unaffected — this phase touches
  no LLM call path and adds no retry logic.
- No image/secret leak: yes — `test_a_trace_id_can_never_appear_outside_the_allowlist`
  confirms the new field is scoped to exactly the `finalize` event; the
  Phase 4 allowlist and forbidden-substring filter are otherwise unchanged.
- The trace id itself is never trusted: `test_an_arbitrary_or_malformed_trace_id_never_raises`
  passes a string shaped like a SQL-injection and a script-injection payload
  through the full path and confirms it is echoed verbatim (never
  interpreted, never executed, never used to branch logic) and the request
  still completes normally.
- No RAG/Qdrant: unaffected.
- `capability_resolver` / routing untouched: confirmed by inspection and by
  making zero edits to `resolver_node.py` or `graph.py` under
  `backend/orchestrator/`.

## Shared-boundary nature of this change

This phase differs from Phases 4 and 5 in one important respect: those phases
were entirely internal to Recognition and needed only Leith's review. Phase 6
changes what Recognition *accepts and does with* an HTTP header that the
Orchestrator already sends — that is a statement about the shared contract
between the two services, even though no Orchestrator file was edited.

Per the plan, this requires sign-off from both Leith and whoever owns the
Global Orchestrator, specifically to confirm:
- `X-Trace-Id` is genuinely intended by the Orchestrator team to be optional
  and non-authoritative from Recognition's side (this report assumes so,
  based on `worker_node.py` sending it unconditionally with no apparent
  expectation of a specific response), and
- no other worker agent already relies on Recognition rejecting or requiring
  this header in a way this change could break.

## Blockers/limitations

- The venv/environment issues documented above cost significant time before
  any Phase 6 code was written, but did not block the phase — they are
  recorded here for transparency and because the resulting two-venv
  separation is now reusable infrastructure for the team.
- This phase cannot independently verify true end-to-end trace continuity
  (a single trace ID visible in an actual LangSmith project spanning both
  the Orchestrator's and Recognition's spans) without LangSmith credentials
  and both services running simultaneously with tracing enabled. What is
  verified here is the full chain of custody for the trace ID *within*
  Recognition's own boundary, and the fact that the Orchestrator already
  sends one. A live, credentialed end-to-end check is recommended before
  this is considered fully proven, and is a reasonable next step for
  whoever reviews this on the Orchestrator side.

## Gate decision
- Handoff/environment verified before implementation: PASS
- `capability_resolver` and routing untouched: PASS
- Trace id propagation implemented, tested, and scoped to one event: PASS
- No public/decision contract change: PASS
- All tests pass: PASS
- **Cross-team sign-off (Leith + Orchestrator owner): PENDING**

**Phase 6: PASS on implementation and tests. NOT yet cleared for merge to any
shared or deployed branch pending the two required reviews above.**
