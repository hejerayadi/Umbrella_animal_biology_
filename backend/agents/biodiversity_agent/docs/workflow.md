# Biodiversity Orchestrator - execution workflow

This document walks through the four execution paths the orchestrator
supports. All four are covered by the pytest suite under ``tests/``.

## LangGraph state machine

```
                 START
                   |
                   v
               [ plan ]
                   |
        (features_to_run empty?)
             /              \
           yes               no
            |                 |
            v                 v
    [fail_no_feature]   [normalize_species]
            |                 |
            |        (needs species and none?)
            |            /            \
            |          yes             no
            |           |               |
            |           v               v
            |   [fail_no_feature]  [dispatch]
            |           |               |
            |           |               v
            |           |         [aggregate]
            |           |               |
            +-----------+---------------+
                        v
                       END
```

## 1. Sequential path

Single feature, single worker.

- Request: ``feature=species_distribution_map, species_name="Loxodonta africana"``.
- ``plan`` -> ``features_to_run = [SPECIES_DISTRIBUTION_MAP]``.
- ``normalize_species`` -> normalizes to scientific name via
  ``services/qdrant_client.py`` (Qdrant online, dict offline).
- ``dispatch`` -> one worker call.
- ``aggregate`` -> single-item result, ``status=COMPLETED``.

## 2. Parallel path

Multiple features in ``context["features"]``.

- Request: ``context={"features": ["species_distribution_map", "migration_analysis"]}, species_name="Loxodonta africana"``.
- ``dispatch`` fans out via ``asyncio.gather`` -> both workers run
  concurrently.
- ``aggregate`` merges: ``output`` is a dict keyed by feature name;
  ``map_url`` is the first non-null, ``migration_route`` and
  ``observation_count`` come from whichever worker produced them.
- Partial failure (one worker FAILED, one COMPLETED) still yields a
  ``COMPLETED`` aggregate - we degrade gracefully.

## 3. Conditional path

Species-scoped feature with no species name.

- Request: ``feature=species_distribution_map`` and no ``species_name``,
  and taxonomy lookup finds nothing in ``context``.
- ``normalize_species`` -> ``normalized_species=None``.
- Conditional edge routes to ``fail_no_feature`` (reused: it emits a
  well-formed FAILED result rather than dispatching a worker with a
  missing input).

## 4. Escalation path

Any worker returns ``NEEDS_AGENT``.

- Request enters normally.
- ``dispatch`` collects results, one of which has
  ``status=NEEDS_AGENT`` and ``target_agent="Trait Discovery Agent"``.
- ``aggregate`` bubbles the first ``NEEDS_AGENT`` upward unchanged:
  same ``target_agent`` and ``prompt_to_target_agent``. The Global
  Scientific Orchestrator is the only component allowed to resolve
  cross-agent dependencies.

## Threading model

The mock workers are synchronous. We wrap each in ``asyncio.to_thread``
inside ``dispatch`` so ``asyncio.gather`` gives us true parallelism on
CPU-bound mocks and real parallelism once the workers switch to async
HTTP calls in Sprint 3.

## Taxonomy service degradation

``SpeciesTaxonomyService.from_env()`` picks the Qdrant backend when
``QDRANT_URL`` and ``QDRANT_API_KEY`` are set, otherwise the offline
dict backend. The orchestrator is agnostic to which backend is live -
the interface is a single ``normalize(query) -> str | None`` call.

## Live demo — Streamlit dashboard

``dashboard.py`` at the root of this package is a manual replacement for
the Global Scientific Orchestrator. It:

1. Accepts a natural-language prompt (FR / EN).
2. Calls the configured LLM (``framework/llm_client.py``) to classify
   the intent - which of the four features to run, and the species /
   region parameters.
3. Builds an ``AgentRequest`` and dispatches it to the same
   ``BiodiversityOrchestrator`` the production wire will call.
4. Streams the LangGraph node executions and shows them as a checklist,
   so a jury can see the routing decision in real time.
5. Embeds the folium map produced by the Species Distribution worker
   directly in the page (via ``st.components.v1.html``).
6. Pretty-prints the full ``AgentResult`` dataclass alongside the map.

The sidebar exposes two focused demos - "Test parallel dispatch" runs
two workers concurrently with ``context["features"]``, and
"Test NEEDS_AGENT escalation" swaps in a worker that always escalates
so the bubble-up path is visible without touching a live cross-agent
integration.

Run it from the repository root:

    streamlit run backend/agents/biodiversity_agent/dashboard.py

Opens at ``http://localhost:8501``.
