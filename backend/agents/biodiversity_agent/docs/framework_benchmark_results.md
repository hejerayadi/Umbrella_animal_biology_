# Framework choice - LangGraph vs LangChain vs CrewAI

Sprint 2 task 2 asks each group to "select and configure the framework"
and to justify the choice. This note captures the reasoning behind our
pick for the Biodiversity Agent.

## Evaluation criteria

The Biodiversity Orchestrator is a small state machine:

1. Read the ``feature`` field (or a list of features in the parallel case).
2. Optionally normalize the species name via a shared taxonomy service.
3. Dispatch to one or more worker agents.
4. Aggregate their results into a single ``AgentResult``.

We evaluated the three candidates against the shape above, not in the
abstract.

| Criterion                                    | LangGraph | LangChain | CrewAI |
| -------------------------------------------- | :-------: | :-------: | :----: |
| Explicit state machine with conditional edges|    Yes    |     -     |   -    |
| Parallel fan-out primitive                   |    Yes    |     -     |   -    |
| Typed shared state across nodes              |    Yes    |     -     |   -    |
| Ships with async support                     |    Yes    |    Yes    |   -    |
| Role/persona abstractions we need            |    No     |     -     |  Yes   |
| Straightforward to unit-test (pytest)        |    Yes    |    Yes    |   -    |
| Overhead for a 4-worker orchestrator         |    Low    |   Medium  |  High  |

## Decision

We picked **LangGraph** because:

- Our workflow is naturally graph-shaped, not agent-persona-shaped.
- ``add_conditional_edges`` maps 1-to-1 to our "if species missing ->
  fail" and "if worker escalates -> bubble up" rules.
- Parallel branches join deterministically at the ``aggregate`` node,
  which is exactly what our fan-out to multiple workers needs.
- The Trait Discovery group (see ``trait_discovery_agent/``) reached
  the same conclusion for the same reasons, so we keep the codebase
  consistent.

CrewAI is a strong option if the Biodiversity Agent later needs to
delegate free-form reasoning between personas (e.g. "cartographer" and
"ecologist" chatting). It is not needed for a routing + aggregation
pipeline, so we shelved it.

## LLM choice

Azure OpenAI, per the repo-wide standard set in the root ``README.md``.
The wrapper is in ``framework/llm_client.py`` and reads its config from
``AZURE_OPENAI_*`` env vars. It raises ``LLMUnavailable`` when the keys
are absent so the orchestrator can degrade to a deterministic offline
path, which is what the whole test suite exercises.

## Framework validation - the toy agent

``framework/toy_agent_langgraph/`` implements a three-node throwaway
graph (LLM call -> conditional edge -> parallel fan-out and merge) that
proves every LangGraph primitive we depend on works end-to-end before
the real orchestrator is wired. Run it with:

    python -m backend.agents.biodiversity_agent.framework.toy_agent_langgraph
