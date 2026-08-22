# Biodiversity Agent

## Objective
Analyze and visualize the spatial distribution of animal species, identify biodiversity hotspots, study habitat changes, and support conservation efforts by integrating occurrence and conservation-status data.

## Problem Addressed
Species distributions change continuously because of climate change, habitat destruction, human activities, migration patterns, and extinction risks. These data are scattered across multiple databases (GBIF, IUCN, WorldClim, Movebank) and are difficult to correlate manually.

## Role in the platform
The Biodiversity Agent is a **domain orchestrator**. It exposes a single conversational surface to the Global Scientific Orchestrator and internally routes to four specialized worker agents, each owning one skill end-to-end.

## Internal architecture

```
Global Scientific Orchestrator
             |
             v
  Biodiversity Agent Orchestrator  (this package)
       |         |          |          |
       v         v          v          v
  Species    Habitat    Biodiv.     Migration
  Distrib.   Visualiz.  Hotspots    Analysis
  (GBIF)     (GBIF+     (GBIF+      (Movebank+
             IUCN+      DBSCAN)     GBIF+
             WorldClim)             WorldClim)
       \_________|__________|__________/
                 |
        Shared services layer
     (PostgreSQL, Qdrant taxonomy,
      map/heatmap rendering, cleaning)
```

## Managed workers
- **Species Distribution Agent** — `species_name -> point map` (GBIF)
- **Habitat Visualization Agent** — `species + status -> habitat map` (GBIF, IUCN, WorldClim)
- **Biodiversity Hotspots Agent** — `region -> heatmap + hotspot list` (GBIF + DBSCAN)
- **Migration Analysis Agent** — `species -> route + seasonal pattern` (Movebank, GBIF, WorldClim)

## Cross-agent dependencies
When a worker's request requires biological knowledge outside biodiversity's scope — for example gene-trait associations — this orchestrator returns `AgentStatus.NEEDS_AGENT` with `target_agent = "Trait Discovery Agent"` (or the appropriate one) so the Global Scientific Orchestrator can dispatch the missing information. This orchestrator never calls another top-level agent directly.

## Sprint 2 scope
Per the Sprint 2 brief, the deliverables from this package are:
1. Sub-orchestrator implementation (routing, delegation, aggregation).
2. Agent framework choice (LangGraph, see `docs/framework_benchmark_results.md`) with a toy agent and Azure OpenAI as the LLM.
3. Executable workflows: sequential, parallel, and conditional paths, validated end-to-end.
4. Qdrant collection for species name normalization (`services/qdrant_client.py`).
5. Data ingestion / retrieval pipelines around the Qdrant collection.
6. Integration tests demonstrating all execution modes and escalation.

Real GBIF/IUCN/WorldClim API integration, ML models (DBSCAN, MaxEnt, LSTM) and production RAG are **explicitly out of Sprint 2 scope** — worker agents are served by their `mock.py` fixtures.
