# Phase 5 Report — Recognition Workflow and Provider Tracing

## Status
PASS

## Owner
Chahd

## Branch and commit
- Branch: `group-d-recognition-sprint4`
- Starting commit: `91aafb5` — "Sprint 4 Phase 4: LangSmith foundation, config and privacy boundary"
- Ending commit: (fill in after `git push` — see Next Steps)
- Working tree: clean once committed

## Scope executed

Instrumented the four Recognition workflow nodes that make a provider or LLM
call - `plan_or_analyze_text`, `classify_with_mock_bioclip2`,
`validate_taxonomy_with_mock_gbif_and_ncbi`, and `explain` - plus one root event
per request from `RecognitionWorkflow.run`. No other node was touched: LangGraph
already traces the node-to-node skeleton automatically once LangSmith is
configured, so only the provider-level detail LangGraph cannot see on its own
was added here, per the plan's "reuse reliable LangGraph tracing and add only
missing provider spans" instruction.

Each instrumented node records exactly one sanitized event through the
`RecognitionTracer` built in Phase 4:

| Node | Event fields recorded |
|---|---|
| `plan` | `llm_role="planner"`, `llm_call_count`, `fallback` (true when the deterministic plan was used instead of the model's) |
| `classify` | `bioclip_provider_mode`, `candidate_count` on success, or the fixed `error_code` on a controlled failure - never the exception body |
| `taxonomy` | `taxonomy_provider_mode`, `taxonomy_available` (false when any candidate was degraded) |
| `explain` | `llm_role="explainer"`, `llm_call_count`, `fallback` |
| `finalize` (root) | `status`, and either `decision` (on a completed outcome) or the fixed `error_code` (on a failure) |

Every event carries `duration_ms`, measured from the start of that node's own
work. Tracing is disabled by default: every node factory defaults to a shared,
disabled `_NULL_TRACER`, so a call site or test that predates Phase 5 needed no
change and, with tracing off, does zero extra work - not "sends nothing", but
never even reaches the metadata pipeline.

`RecognitionWorkflow.__init__` builds one `RecognitionTracer` from the
workflow's own `config.langsmith` and passes it into the four node factories
that need it; the other four nodes (`validate`, `confidence`, `delegate`,
`finalize`'s AgentResult builder) are unchanged and untraced, since they make
no provider or LLM call for LangGraph's own tracing to miss.

## Files changed

- `workflows/nodes.py` - added `tracer` parameter (default `_NULL_TRACER`) to
  `make_plan_node`, `make_classify_node`, `make_taxonomy_node`,
  `make_explain_node`; each records one event at the end of its work.
- `workflows/graph.py` - `RecognitionWorkflow.__init__` builds
  `self._tracer = build_tracer(self._config.langsmith)` and wires it into the
  four node factories; `run()` records one root event per request after the
  graph completes.
- `tests/test_sprint4_phase5_workflow_tracing.py` - new file; 14 focused tests.

## Tests

- Command: `python -m pytest tests\ -v`
- Collected: 1480
- Passed: 1480
- Failed: 0
- Skipped: 0
- Warnings: 1 (pre-existing `httpx`/`starlette` deprecation notice, unrelated to
  this phase)

Focused Phase 5 tests (`tests/test_sprint4_phase5_workflow_tracing.py`, 14
tests) cover: disabled tracing never reaching the metadata pipeline; an
identified run producing exactly the five expected node events in order; every
event carrying a non-negative duration; the finalize event correctly reporting
status and decision; a classification failure recording only the fixed error
code, never the exception body; a `not_identified` outcome (no candidates)
being recorded as `completed`, not `failed`; the default (no reasoning LLM)
path being correctly reported as a fallback on both the planner and explainer;
taxonomy correctly reporting its provider mode and availability; classify
correctly reporting provider mode and candidate count; the metadata allowlist
holding under real instrumentation (no instruction, context, image, prompt, or
explanation field ever traced); no retry-count field ever emitted (this agent
performs no retries); traced and untraced runs producing byte-identical
`AgentResult`s; an exporter failure never turning a completed request into a
failed one; and the two-call reasoning LLM ceiling holding with tracing
enabled.

## Evidence

- Full suite run before this phase's edits reached `graph.py` (nodes-only,
  incomplete rollout): 7 of the 14 new tests failed, with the other 1473 tests
  passing - confirming the failures were isolated to Phase 5's own new
  assertions and caused no regression elsewhere.
- Root cause of the 7 failures: `graph.py` had not yet been updated to
  construct a real tracer and pass it into the node factories, so every node
  was still using the shared, disabled `_NULL_TRACER` default. Once `graph.py`
  was updated to build `self._tracer` and wire it through, all 14 passed.
- Full suite run after the complete rollout: **1480 passed, 0 failed**.

## Protected checks

- Seven keys unchanged: yes - `test_traced_and_untraced_results_are_identical`
  asserts byte-identical `status` and `output` whether or not tracing is
  enabled.
- Maximum two agent LLM calls: yes -
  `test_the_two_call_ceiling_still_holds_with_tracing_enabled` confirms
  `reasoning_llm_calls <= 2` with tracing on.
- Zero retry: yes - `test_no_retry_count_is_ever_reported_in_phase_5` confirms
  no `retry_count` field is ever emitted, rather than a hardcoded zero that
  could drift from reality; nothing in this phase adds retry logic.
- No image/secret leak: yes -
  `test_no_traced_event_carries_an_instruction_prompt_or_image_field` checks
  every recorded event against the full set of sensitive field names; the
  Phase 4 allowlist (`safe_metadata`) is exercised unchanged, not bypassed.
- No RAG/Qdrant: yes - no vector, embedding, or retrieval code touched.
- Tracing failure cannot affect the agent:
  `test_an_exporter_failure_cannot_turn_a_success_into_a_failure` confirms a
  `safe_metadata` exception still yields a `COMPLETED` result with the correct
  decision.

## Blockers/limitations

One process issue, not a defect in the design: the four `nodes.py` edits and
the two `graph.py` edits were applied in separate steps, and `graph.py` was
initially left on its pre-Phase-5 version while `nodes.py` had already moved
to the new tracer-aware signatures. Because every node factory's `tracer`
parameter defaults to a disabled tracer, this was not a crash - it silently
under-instrumented every node until `graph.py` was brought up to date. The
first full test run caught it immediately (7 of 14 new tests failed with empty
or missing events), and the fix was to complete the intended `graph.py`
wiring; no `nodes.py` change was needed. Full replacement files were used for
both modules on the second pass specifically to remove this class of
edit-ordering error going forward.

No other blockers. No workflow decision, output key, or provider behaviour was
changed in this phase - only observability was added, and only for the four
nodes LangGraph's own tracing cannot already describe.

## Gate decision
- Only the four provider/LLM-calling nodes instrumented, no duplicate spans: PASS
- Tracing optional and disabled by default, verified at the node level: PASS
- Traced and untraced results are identical: PASS
- Exporter/tracer failure cannot affect the agent: PASS
- All tests pass: PASS
- No public/decision contract change: PASS

**Phase 5: PASS**
