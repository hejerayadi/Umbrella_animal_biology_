# Species Distribution Agent

## Objective
Answer "Where do <species> live?" by producing an interactive point map from GBIF occurrence records.

## Input contract
```
species_name : str                       # e.g. "Loxodonta africana"
feature      : "species_distribution_map"
region       : str = "global"            # optional country / continent filter
time_period  : tuple[int, int] | None    # optional (start_year, end_year)
```

## Output contract
```
species_name       : str
coordinates        : list[tuple[float, float]]
observation_count  : int
map_url            : str
```

## Data source
- **GBIF** (Global Biodiversity Information Facility) — only source used.
  - Endpoint: `https://api.gbif.org/v1/occurrence/search`
  - Required parameters: `scientificName`, `hasCoordinate=true`.
  - Optional: `country`, `year`.

## Workflow (8 steps)
1. Request received from Biodiversity Orchestrator.
2. Normalize species name (Qdrant `species_taxonomy` lookup).
3. Call GBIF occurrence API.
4. Parse JSON records.
5. Clean coordinates (drop `(0, 0)`, deduplicate).
6. Persist raw records to PostgreSQL (Sprint 3+).
7. Render map with folium.
8. Return structured `AgentResult`.

## Failure cases
- Species absent from GBIF -> `status=FAILED`, `output="No occurrences found"`.
- All coordinates invalid -> `status=FAILED`.
- API timeout / 5xx -> `status=FAILED`, orchestrator may retry.

## Confidence scoring
Rough heuristic used until we plug the real backing store:

| observation_count | confidence |
| ----------------- | ---------- |
| >= 1000           | 0.95       |
| >= 100            | 0.80       |
| >= 10             | 0.60       |
| < 10              | 0.30       |

## Sprint 2 scope
Only the `mock.py` is required. `worker.py` is a stub for the real GBIF
integration that will land in Sprint 3+.
