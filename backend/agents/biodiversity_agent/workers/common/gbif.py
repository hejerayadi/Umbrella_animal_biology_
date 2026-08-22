"""GBIF access for the biodiversity workers.

Implements the two retrieval mechanics ``Biodiversity Hotspots.pdf`` calls for:

* the **count probe** (doc §4.2 step 2) - one request with ``limit=0`` returns
  only the total, so the workload is known before any data moves and the
  retrieval strategy can be chosen explicitly instead of by timing out;
* **paged retrieval** (doc §4.3, the ``paged`` strategy) - GBIF answers with at
  most 300 records, so offsets are walked until the budget is met.

The asynchronous Download API (doc §3.3.2), which issues a citable DOI, is not
used: it is the >100 000-record path and takes minutes. ``choose_strategy``
still reports when that path *would* be required, so the caller can return
CONTINUE or refuse, exactly as doc Table 12 specifies.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from .config import CACHE_DIR, COMMON_NAMES

SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"

PAGE_SIZE = 300           # GBIF's hard per-request ceiling
# Measured against GBIF, 30 pages of the Kenya query: 4 workers 3.9 s, 8
# workers 17.3 s. Bursts get throttled into a slow lane, so more is slower -
# this is a measured value, not a guess, and raising it will hurt.
PARALLEL_PAGES = 4        # concurrent page requests
PAGED_CEILING = 100_000   # doc Table 12: above this the bulk path is required
REFUSE_CEILING = 20_000_000  # doc Table 12: refuse at this resolution

OCCURRENCE_COLUMNS = [
    "speciesKey", "species", "decimalLatitude", "decimalLongitude",
    "year", "countryCode", "taxonRank",
]

Progress = Callable[[str], None] | None


MATCH_URL = "https://api.gbif.org/v1/species/match"
SPECIES_SEARCH_URL = "https://api.gbif.org/v1/species/search"

# The GBIF Backbone Taxonomy. Without it a name search also returns virus and
# other-dataset entries: "polar bear" ranks a mastadenovirus first.
BACKBONE_DATASET = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"
ANIMALIA_KEY = 1

# GBIF reports its own confidence in a name match. Below this the match is a
# guess, and a guessed species answers about the wrong animal.
MATCH_CONFIDENCE_FLOOR = 90


def _accepted_key(name: str, getter) -> tuple[int, str] | None:
    """A scientific name resolved to its accepted backbone species key."""

    try:
        matched = getter(MATCH_URL, {"name": name, "strict": "false"}).json()
    except Exception:                       # noqa: BLE001 - unavailable is not fatal
        return None
    if (matched.get("rank") == "SPECIES"
            and matched.get("matchType") not in (None, "NONE")
            and int(matched.get("confidence") or 0) >= MATCH_CONFIDENCE_FLOOR
            and matched.get("usageKey")):
        return int(matched["usageKey"]), matched.get("species") or name
    return None


def match_species(name: str, *, fetch=None) -> tuple[int, str] | None:
    """A GBIF species key and canonical name, or None when unresolvable.

    Three passes, cheapest and most reliable first:

    1. the common-name table above, whose value is then resolved by GBIF;
    2. ``/species/match``, which handles scientific names and misspellings;
    3. an exact vernacular search, which covers common names GBIF ranks well
       ("polar bear", "blue whale", "lion").

    Returning None is a real answer - the caller asks for a scientific name
    rather than analysing whichever species happened to rank first.

    ``fetch`` is injectable so the tests run offline.
    """

    if not name or not name.strip():
        return None
    getter = fetch or (lambda url, params: requests.get(url, params=params, timeout=20))
    query = " ".join(name.split()).strip(" ?.!,").lower()

    # Plural forms are what people actually type: "tigers", "polar bears".
    variants = [query]
    if query.endswith("s") and len(query) > 3:
        variants.append(query[:-1])

    for variant in variants:
        if variant in COMMON_NAMES:
            resolved = _accepted_key(COMMON_NAMES[variant], getter)
            if resolved:
                return resolved

    for variant in variants:
        resolved = _accepted_key(variant, getter)
        if resolved:
            return resolved

    # Vernacular, exact only. GBIF's relevance ordering is unusable here, so a
    # candidate counts only if one of its own vernacular names is the query.
    for variant in variants:
        try:
            found = getter(SPECIES_SEARCH_URL,
                           {"q": variant, "qField": "VERNACULAR", "rank": "SPECIES",
                            "datasetKey": BACKBONE_DATASET,
                            "highertaxonKey": ANIMALIA_KEY, "limit": 50}).json()
        except Exception:                   # noqa: BLE001
            continue
        exact = []
        for record in found.get("results", []):
            spoken = {(entry.get("vernacularName") or "").strip().lower()
                      for entry in record.get("vernacularNames", [])}
            canonical = record.get("canonicalName")
            if variant in spoken and canonical:
                # ACCEPTED first: "polar bear" also matches Thalarctos
                # maritimus, a synonym of Ursus maritimus.
                exact.append((0 if record.get("taxonomicStatus") == "ACCEPTED" else 1,
                              canonical))
        for _, canonical in sorted(exact):
            resolved = _accepted_key(canonical, getter)
            if resolved:
                return resolved
    return None


def wkt_polygon(bbox: tuple[float, float, float, float]) -> str:
    """Bounding box as WKT, wound counter-clockwise as GBIF requires."""

    lon_min, lat_min, lon_max, lat_max = bbox
    return (f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, "
            f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))")


def _filters(bbox, species_key=None, taxon_key=None, year_from=None, year_to=None) -> dict:
    params: dict[str, object] = {
        "geometry": wkt_polygon(bbox),
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
    }
    if species_key is not None:
        params["speciesKey"] = species_key
    if taxon_key is not None:
        params["taxonKey"] = taxon_key
    if year_from is not None and year_to is not None:
        params["year"] = f"{year_from},{year_to}"
    return params


def count_probe(bbox, **kwargs) -> tuple[int, str | None]:
    """Total record count under these filters. One sub-second request."""

    params = {**_filters(bbox, **kwargs), "limit": 0}
    try:
        answer = requests.get(SEARCH_URL, params=params, timeout=15)
        return int(answer.json().get("count", 0)), None
    except requests.RequestException as exc:
        return 0, f"count probe unavailable: {exc}"
    except (ValueError, TypeError) as exc:
        return 0, f"count probe returned no usable count: {exc}"


def choose_strategy(estimated_count: int) -> str:
    """Which retrieval path doc Table 12 prescribes for this volume."""

    if estimated_count <= 0:
        return "empty"
    if estimated_count <= PAGED_CEILING:
        return "paged"
    if estimated_count <= REFUSE_CEILING:
        return "download"     # bulk, asynchronous, DOI-issuing - not implemented here
    return "refuse"


def _fetch_page(session: requests.Session, params: dict, offset: int) -> list[dict]:
    """One page of occurrences, retried with backoff.

    A page is a slice of the answer, so losing one silently would change the
    result. Three attempts with a growing pause cover the transient throttling
    GBIF applies to bursts, and only a page that fails all three is reported.
    """

    last: Exception | None = None
    for attempt in range(3):
        try:
            answer = session.get(SEARCH_URL, params={**params, "offset": offset},
                                 timeout=60)
            if answer.status_code != 200:
                raise RuntimeError(f"GBIF answered {answer.status_code}")
            return [{column: record.get(column) for column in OCCURRENCE_COLUMNS}
                    for record in answer.json().get("results", [])]
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1) ** 2)
    raise last if last else RuntimeError("page fetch failed")


def download_occurrences(
    bbox: tuple[float, float, float, float],
    cache_name: str,
    max_records: int,
    *,
    species_key: int | None = None,
    taxon_key: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    progress: Progress = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Stream occurrences through the paged path into a cached CSV.

    Pages are fetched concurrently (see ``PARALLEL_PAGES``), so a full budget
    costs a handful of seconds rather than one round trip per page. The CSV cache
    on top of that makes every later call for the same study area immediate.
    """

    cache_file = CACHE_DIR / cache_name
    if cache_file.exists() and not refresh:
        table = pd.read_csv(cache_file)
        if progress:
            progress(f"cache hit: {len(table):,} records from {cache_name}")
        return table

    params = {**_filters(bbox, species_key=species_key, taxon_key=taxon_key,
                         year_from=year_from, year_to=year_to),
              "limit": PAGE_SIZE}

    total, _ = count_probe(bbox, species_key=species_key, taxon_key=taxon_key,
                           year_from=year_from, year_to=year_to)
    budget = min(total, max_records) if total else max_records
    if progress:
        progress(f"GBIF holds {total:,} records; retrieving {budget:,}")

    offsets = list(range(0, budget, PAGE_SIZE))
    started = time.time()

    # The pages are independent - each is a plain offset into the same query - so
    # fetching them serially spends the whole wall-clock on round trips that could
    # overlap. A small pool collapses that; see PARALLEL_PAGES for why small.
    pages: dict[int, list[dict]] = {}
    failures: list[str] = []

    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(pool_maxsize=PARALLEL_PAGES,
                                             pool_connections=PARALLEL_PAGES))
        with ThreadPoolExecutor(max_workers=PARALLEL_PAGES) as pool:
            futures = {pool.submit(_fetch_page, session, params, offset): offset
                       for offset in offsets}
            done = 0
            for future in as_completed(futures):
                offset = futures[future]
                try:
                    pages[offset] = future.result()
                except Exception as exc:                  # noqa: BLE001 - reported, never raised
                    failures.append(f"offset {offset}: {exc}")
                done += 1
                if progress and (done % 6 == 0 or done == len(offsets)):
                    got = sum(len(page) for page in pages.values())
                    progress(f"  {got:,} records · page {done}/{len(offsets)} "
                             f"· {time.time() - started:.1f}s")

    # Reassemble in offset order, so a cached CSV is byte-identical to what the
    # serial path produced and the "newest first" caveat still holds.
    rows = [record for offset in offsets for record in pages.get(offset, [])]

    if failures and progress:
        progress(f"  {len(failures)} of {len(offsets)} pages failed ({failures[0]}); "
                 f"keeping the rest")

    table = pd.DataFrame(rows, columns=OCCURRENCE_COLUMNS)
    if not table.empty:
        table.to_csv(cache_file, index=False)
    if progress:
        progress(f"retrieved {len(table):,} records in {time.time() - started:.0f}s")
    return table
