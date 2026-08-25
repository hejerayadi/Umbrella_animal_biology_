# M3 — Biodiversity Hotspots

## Purpose

M1 and M2 answer questions about one species. M3's main question inverts that:
**given a region, which parts of it hold the most biodiversity, and where should
conservation effort go?** It also answers the species-scoped form - where one
species' records concentrate - as a separate mode with its own ranking and its
own wording; see *Two questions, two modes* below. The input is thousands of coordinates across thousands
of species; the output is a small number of named, ranked, defensible hotspots
plus a richness map.

DBSCAN discovers those dense regions without being told how many exist.

Goals, from the design document:

| # | Goal | How it is judged |
|---|---|---|
| G1 | Turn a region into ranked, named hotspots in one question | each cluster carries species count, record count, area and a name |
| G2 | Measure richness rather than sampling effort | `effort_correction` is in every output and states the method |
| G3 | Discover hotspot shape from the data | boundaries come from DBSCAN, not from administrative borders |
| G4 | Rank by urgency, not only by count | selectable index (blocked on an IUCN token — see limitations) |
| G5 | Be reproducible | `parameters_used` is echoed on every run |

## When to use this module

- "Show the biodiversity hotspots of the Congo Basin"
- "Which part of Africa has the most species?"
- "Compare biodiversity between the Amazon and the Congo Basin" (ask twice)
- "Show me a biodiversity heatmap of Southeast Asia"
- "Where is diversity highest in Madagascar?"

## When not to use it

- **Where does one species live** → Species Distribution Agent
- **What kind of place is this, could the species live elsewhere** → Habitat
  Visualization Agent
- **How has this changed over time** → Migration Analysis Agent
- Predicting *undiscovered* hotspots. M3 measures what has been recorded.
- Claiming a low-richness cell is species-poor when its sampling effort is also
  low. That is an absence of looking, not an absence of species.

## The workflow

Sprint 3 task 3 asks for an explicit workflow. M3's is a straight line, and every
step is arithmetic:

```
question -> region + parameters -> GBIF -> clean -> grid -> indices
         -> effort correction -> DBSCAN -> rank -> map -> AgentResult
```

| Question task 3 asks | M3's answer |
|---|---|
| What triggers it? | a question naming a region, routed here by the Biodiversity Orchestrator on `feature=biodiversity_hotspots` |
| What information does it need? | a study area — nothing else is mandatory |
| What does it call? | one service, the GBIF Occurrence API |
| What if that fails? | `FAILED` with the reason. It never raises |
| What does it return? | one `AgentResult`: summary, documented payload, map |
| What if information is insufficient? | `NEEDS_CLARIFICATION` with the options to choose from — never a guessed region |

**There is no LLM in this module, and it needs none.** The question is read by
matching the six documented regions in `workers/common/config.py`, then a gazetteer lookup for anything else; the
answer is composed from the numbers the pipeline just computed. Nothing in the
chain is a judgement call, so nothing has to be generated — which is also why
every figure in the answer is checkable against the payload, and why the module
runs with no API key and at no cost.

## No database, no vector store, no saved model

The design document specifies PostGIS tables, a Redis result cache, a
pre-aggregated grid and a Qdrant knowledge base. None of it is used here:

| Document | Here | Why it is not needed |
|---|---|---|
| PostGIS `h3_richness` table | the grid lives in memory for the run | one region is ~80 rows; a server for 80 rows is overhead |
| Redis result cache | one CSV per region under `cache/` | a file needs no server; a region is ~80 s cold and instant warm |
| `gbif_downloads` archive table | the paged API, cached per region | nothing but that CSV is kept between runs |
| pre-aggregated monthly grids | recomputed per run | the recomputation is milliseconds |
| Qdrant knowledge base | none | there is no free text to search: the method is code, and the findings are the payload |
| a saved fitted model | none | DBSCAN is transductive — see below |

**Why no saved model.** Sprint 3 task 1 asks for the trained model to be saved
and configured for inference. DBSCAN has no `predict()`: it labels the cells it
was fitted on and cannot score a new point later. Saving it would let you reload
labels, not answer a new question — and recomputing them costs milliseconds once
the records are in hand. There is nothing to serve, so nothing is stored. The
settings that produced a run travel in `parameters_used`, which is what
reproducibility actually requires.

## Any place, not a fixed list

The six study areas in ``workers/common/config.py`` come from the design document
and stay authoritative - their bounding boxes are what reproduce the numbers in
the technical report. But a question about Brazil or Borneo should get an answer,
not a list of six regions the asker never mentioned. So a place the table does not
know is resolved through Nominatim, OpenStreetMap's free gazetteer, and the answer
always says where its study area came from (``parameters_used.study_area_source``,
plus a warning naming the resolved place).

Measured, live:

| Question | Study area | Result |
|---|---|---|
| "hotspots in Costa Rica" | Costa Rica, 326,805 km² | 42 cells, 2 clusters, 1,345 species |
| "biodiversity of Borneo" | Borneo / Kalimantan, 1,443,495 km² | 63 cells, 2 clusters, 2,031 species |
| "hotspots in the Serengeti" | Serengeti, 14,094 km² | 13 cells, 1 cluster, 495 species |

Three refusals, each with a reason rather than a shrug:

| Question | Answer |
|---|---|
| "Which part of Africa…" | *Afrika covers about 30,153,621 km² - too large for one study area at this cell size. Name a country or region inside it.* |
| "hotspots in Atlantis" | *Atlantis, Palm Beach County is only about 4 km² - too small to bin into cells.* |
| "Western Ghats" | *OpenStreetMap has only a marker for it, not an outline* |

**Only the top gazetteer result is ever used, deliberately.** Asking for five
results and taking the first one large enough to grid would answer "Atlantis"
with Loire-Atlantique in France - it is the fourth result. A wrong answer with a
map attached is worse than a refusal, so a name that resolves to something
unusable is reported as unusable.

**The documented regions are checked first**, because the gazetteer disagrees with
the document on the ones that matter: "Amazon rainforest" resolves to a small
conservation concession in Peru, not the basin.

Lookups are cached in ``cache/places.json`` and rate-limited to one request per
second, which is what the Nominatim usage policy asks for.

## Two questions, two modes

A region and a species are different questions, and the module answers them
differently rather than pretending one pipeline fits both.

| | Region mode (default) | Species mode |
|---|---|---|
| Question | "hotspots in the Congo Basin" | "hotspots for tigers" |
| Triggered by | a study area | ``species_name``, or a species named in the question |
| Clusters | grid cells, weighted by effort-corrected richness | the occurrence records themselves |
| Ranked by | species count | record count |
| Effort correction | applied | not applicable - see below |
| Called | biodiversity hotspots | occurrence clusters |

**Why species mode is not just the region pipeline with a filter.** With one
species, every grid cell holds richness 1. The diversity indices carry no
information, and the effort correction has nothing to normalise: it divides
richness by observation intensity, and richness is constant. Running the region
pipeline over one species would produce a map of *where people looked*, labelled
as biodiversity. So the species path clusters the points, ranks by record count,
and says in the summary and in ``warnings`` that the result is observation
density - a busy cluster can be one well-watched reserve.

**Names are resolved by GBIF, not guessed.** ``match_species`` tries a common-name
table, then ``/species/match``, then an exact vernacular search, and returns None
rather than a low-confidence guess - at which point the module asks for the
scientific name. A genus is refused: "Panthera" would silently widen the question
to every big cat. The common-name table exists because GBIF's own ranking is
unusable for some names: a vernacular search for "tiger" returns fifty molluscs,
frogs and flies whose *scientific* names contain "tiger" before it reaches
Panthera tigris.

**Subspecies records are kept here, and only here.** 4,552 of the 5,008 GBIF
tiger records are at SUBSPECIES rank (*P. t. tigris*, *P. t. altaica*). They are
tiger records, so the species path keeps them; the region path still excludes
them, because a subspecies would count as an extra species in a richness total.

Measured, live: *Panthera tigris* worldwide - 5,008 records retrieved, 2,674 after
cleaning, 18 clusters at eps 60 km / min_samples 12, silhouette 0.73, busiest
cluster 662 records in India. The scan samples 1,000 points for its silhouette,
because that score is quadratic in pairwise distances: full-data scoring took 19 s
against 6 s sampled, for the same chosen parameters.

## Where the time goes

A question is not slow because of the analysis - it is slow because of GBIF. The
measured split, for a 9,000-record budget:

| Step | Cold | Warm |
|---|---|---|
| paged retrieval (30 requests) | 20-40 s | 0.1 s from the CSV cache |
| clean, grid, indices, effort | 1 s | 1 s |
| eps scan + DBSCAN, twice | 0.5 s | 0.5 s |
| folium map | 2.5 s | 2.5 s |
| **total** | **~30 s** | **~6 s** |

Two things make that acceptable. The pages are fetched concurrently, since each
is an independent offset into the same query - and the pool is deliberately 4,
because a burst gets throttled into a slow lane (measured: 4 workers 3.9 s, 8
workers 17.3 s for the same 30 pages). And the retrieval is cached per region as
one CSV, so only the first question about a study area pays the network cost.

To make every study area warm before a demo, ask each one once - or in Python:

```python
for region in sorted(REGIONS):
    HotspotsWorker().run(AgentRequest(instruction=f"hotspots in {region}",
                                      context={}, region=region))
```

## Asking, rather than guessing

Two cases are answered with a question instead of an analysis, both carrying the
options the caller can choose from:

| The question | What M3 does |
|---|---|
| names no known study area ("which part of Africa…") | asks, listing the six it knows |
| names more than one ("compare the Amazon and the Congo Basin") | asks which one, since it analyses one at a time |

The dashboard renders those options as buttons, so answering is one click and the
run continues to a complete result. Guessing instead would silently answer a
question nobody asked - and with a map attached, that reads as fact.

## Inputs and outputs

Input is `BiodiversityHotspotInput` (`schema.py`). One of `region_name` or `bbox`
is required; everything else has a documented default. The knobs that change the
answer most: `cell_size_km`, `min_records_per_cell`, `normalise_by_effort`,
`eps_km`, `min_samples`, `auto_tune`.

Output is `BiodiversityHotspotOutput`, carrying `ranked_hotspots`, the full
`grid`, `effort_correction`, `quality` (silhouette, noise share, cluster sizes,
reproducibility, the parameter scan), `cleaning_report`, `parameters_used`,
`render_spec`, `citations`, `summary`, `confidence` and `warnings`.

Statuses: `COMPLETED` (a grid and zero or more clusters), `PARTIAL` (records
retrieved but no cell reached the noise floor), `NEEDS_CLARIFICATION` (the region
is unknown — the known ones are listed), `FAILED` (upstream unavailable, or the
area is too large at this cell size). The platform's `AgentStatus` has four
members, and M3's outcomes narrow onto them in [`status.py`](status.py), so the
shared `schema.py` needed no edit.

## Method, and its limitations

**Pipeline.** GBIF count probe → paged retrieval → five cleaning rules →
equal-area binning → richness, Shannon, Simpson, Chao1 per cell → effort
correction → eps scan → DBSCAN (haversine) → ranked hotspots → heatmap.

**Why DBSCAN and not K-Means.** Nobody knows how many hotspots a region has, so a
`k` cannot be supplied. A hotspot following a river or a coast is not convex. And
K-Means forces every point into a cluster, which would make the Sahara a hotspot;
DBSCAN labels sparse cells as noise.

**Three correctness requirements, not tuning choices.** `metric="haversine"`
(degrees are not distances), `eps / 6371` (scikit-learn works in radians),
`algorithm="ball_tree"` (the only one supporting haversine). And no
`StandardScaler`: latitude and longitude are already a physical position, and
standardising them destroys the meaning of the distance.

**Sample weights are normalised by their median.** scikit-learn treats a sample
whose weight is at least `min_samples` as a core sample on its own. Corrected
richness runs far above `min_samples`, so passing it raw — as the design
document's own §4.4 snippet does — makes every cell a core sample: noise becomes
unreachable, DBSCAN degrades into connected components, and the effort correction
stops affecting the labels. `tests/test_m3_pipeline.py` pins this.

**Effort correction is the methodological core.** Uncorrected richness is
biodiversity multiplied by observation intensity. Each cell is compared with the
regional median, damped by a square root, and capped at 3×. All three choices are
conservative on purpose: they under-correct rather than invent a hotspot where
nobody looked.

**Tuning has no labels, so there is no GridSearchCV.** `eps` and `min_samples`
are chosen by an explicit scan over 28 combinations, scored on silhouette with the
noise share constrained to a plausible band. The scan travels in `quality`, so the
choice is visible rather than asserted.

**Limitations, all reported in `warnings`.**

| Missing | Consequence | What it needs |
|---|---|---|
| H3 hexagons | equal-area squares instead; areas equal, shape differs | `pip install h3` |
| WWF ecoregions | hotspots named by country code, not ecoregion | the TEOW shapefile |
| IUCN Red List | no threat-weighted index, no `threatened_count` | a free API token |
| CEPF hotspots | no overlap validation against the published 36 | the CEPF shapefile |
| Bulk Download API | capped at the paged budget, no citable DOI | the asynchronous GBIF path |

One more caveat the worker raises when it applies: the paged path returns the
newest records first, so a capped run over a very large region is a recent
snapshot rather than a sample spread across the requested years.
