"""Turning a place named in a question into a study area.

The six regions in ``config.REGIONS`` come from ``Biodiversity Hotspots.pdf`` and
stay authoritative: their bounding boxes are what reproduce the numbers in the
technical report. Everything else - a country, a province, an island, a national
park - is resolved here through Nominatim, OpenStreetMap's free geocoder.

Three things this must get right, all learned by measuring:

* **A point is not a study area.** "Western Ghats" resolves to a mountain-range
  *node*, whose bounding box is a single coordinate. Gridding it would produce
  one cell. Anything smaller than ``MIN_AREA_KM2`` is refused.
* **The resolved name must travel back.** "Yellowstone" resolves to Yellowstone
  *County, Montana*, not the national park. That is a legitimate reading, but the
  answer has to say which place it used so the reader can correct it.
* **The documented regions win.** "Amazon rainforest" resolves to a small
  conservation concession in Peru, not the basin, so the table is consulted first.

Nominatim asks for at most one request per second and a real User-Agent. Both are
honoured, and every lookup is cached on disk, so a repeated question costs nothing.
"""

from __future__ import annotations

import json
import math
import threading
import time

import requests

from .config import CACHE_DIR, REGIONS, REGION_ALIASES

SEARCH_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy requires an identifying User-Agent; a generic one gets
# blocked, and rightly so.
USER_AGENT = "AnimalBioHub-M3-BiodiversityHotspots/1.0 (academic project)"

# Nominatim classes that describe an area of land. A study area is one of these;
# an embassy, a restaurant or a shop is not. Measured: with English names
# preferred, "turkiye" returns a Turkish Embassy in Brussels (class=office)
# above the country (class=boundary), and a 0 km2 embassy is not an answer.
USABLE_CLASSES = {"boundary", "place", "natural", "landuse", "leisure"}

MIN_REQUEST_INTERVAL = 1.1      # seconds, per the Nominatim usage policy
# With a 1 km cell - the finest the worker will refine to - 25 km2 is still
# 25 cells. Below that there is nothing to cluster, whatever the cell size.
MIN_AREA_KM2 = 25.0
MAX_AREA_KM2 = 30_000_000.0     # larger than Russia-plus is not a study area

_CACHE_FILE = CACHE_DIR / "places.json"
_lock = threading.Lock()
_last_request = 0.0

KM_PER_DEGREE_LAT = 110.57
KM_PER_DEGREE_LON = 111.32


class PlaceNotUsable(LookupError):
    """Resolved to a real place whose bounding box cannot be a study area.

    Three ways that happens, all seen in practice: a marker with no outline
    ("Western Ghats"), something the size of a village ("Atlantis, Florida"), and
    a country that crosses the antimeridian, whose box spans the whole planet
    ("United States", because of Alaska).
    """


# Kept as an alias: the name only described the first of the three cases.
PlaceTooSmall = PlaceNotUsable


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Rough area of a lon/lat box, with the cosine correction for longitude."""

    lon_min, lat_min, lon_max, lat_max = bbox
    mean_lat = math.radians((lat_min + lat_max) / 2)
    height = abs(lat_max - lat_min) * KM_PER_DEGREE_LAT
    width = abs(lon_max - lon_min) * KM_PER_DEGREE_LON * math.cos(mean_lat)
    return abs(height * width)


def _load_cache() -> dict:
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass                    # a cache that cannot be written is not an error


def _documented(name: str) -> tuple[float, float, float, float] | None:
    """The design document's own bounding box for one of its six regions."""

    key = name.strip().lower()
    key = REGION_ALIASES.get(key, key)
    return REGIONS.get(key)


def resolve_place(name: str, *, fetch=None) -> dict | None:
    """A study area for a named place, or None if it cannot be resolved.

    Returns ``{"bbox", "name", "source", "area_km2"}``. ``source`` is
    "design document" for the six documented regions and "OpenStreetMap"
    otherwise, so an answer can say where its study area came from.

    Raises ``PlaceTooSmall`` when the place is real but too small to grid - a
    different answer from "I have never heard of it", and worth saying so.
    """

    if not name or not name.strip():
        return None
    query = " ".join(name.split()).strip(" ?.!,")

    documented = _documented(query)
    if documented:
        key = REGION_ALIASES.get(query.lower(), query.lower())
        return {"bbox": tuple(documented), "name": key, "source": "design document",
                "area_km2": bbox_area_km2(documented)}

    cache = _load_cache()
    cached = cache.get(query.lower())
    if cached == "unresolved":
        return None
    if cached:
        return {**cached, "bbox": tuple(cached["bbox"])}

    global _last_request
    getter = fetch or (lambda url, params, headers: requests.get(
        url, params=params, headers=headers, timeout=20))

    with _lock:
        wait = MIN_REQUEST_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            answer = getter(SEARCH_URL,
                            {"q": query, "format": "json", "limit": 5,
                             # Without this, names come back in the local script:
                             # "grece" resolved to a study area labelled "Ελλάς"
                             # and "tokyo" to "東京都", which a reader who typed
                             # the English name cannot check.
                             "accept-language": "en"},
                            {"User-Agent": USER_AGENT})
            results = answer.json()
        except Exception:               # noqa: BLE001 - unavailable is not fatal
            return None
        finally:
            _last_request = time.time()

    if not results:
        cache[query.lower()] = "unresolved"
        _save_cache(cache)
        return None

    # The first result that describes an area of land, not the first result. Only
    # that one: taking the first *large enough* result instead would answer
    # "Atlantis" with Loire-Atlantique in France, its fourth result.
    first = next((candidate for candidate in results
                  if candidate.get("class") in USABLE_CLASSES), None)
    if first is None:
        cache[query.lower()] = "unresolved"
        _save_cache(cache)
        return None
    try:
        south, north, west, east = (float(value) for value in first["boundingbox"])
    except (KeyError, TypeError, ValueError):
        return None

    bbox = (west, south, east, north)

    # A country that crosses the antimeridian is reported by Nominatim with a box
    # from -180 to +180: the United States (Alaska), Russia, Fiji and New Zealand
    # all come back spanning 360 degrees. The area of that box is 336 million km2
    # - larger than the planet - and gridding it would query the whole world.
    if abs(east - west) > 180.0:
        whole = str(first.get("display_name", query)).split(",")[0].split(";")[0]
        raise PlaceNotUsable(
            f"{whole} crosses the antimeridian, so its bounding box spans the "
            f"whole globe and cannot be one study area. Name a part of it - a "
            f"state, province or region.")

    area = bbox_area_km2(bbox)

    if area < MIN_AREA_KM2:
        marker = ", ".join(
            part.strip() for part in
            str(first.get("display_name", query)).split(",")[:2])
        if area < 1.0:
            raise PlaceNotUsable(
                f"OpenStreetMap has only a marker for {marker}, not an outline, so "
                f"there is no area to grid. Try a wider or administrative area - a "
                f"state, province or country.")
        raise PlaceNotUsable(
            f"{marker} is only about {area:,.0f} km2 - too small to bin into "
            f"cells at this resolution. Name a wider area.")
    if area > MAX_AREA_KM2:
        whole = str(first.get("display_name", query)).split(",")[0].split(";")[0]
        raise PlaceNotUsable(
            f"{whole} covers about {area:,.0f} km2 - too large for one study area "
            f"at this cell size. Name a country or region inside it.")

    # The short name, not the full postal chain ("Serengeti, Mara, Lake Zone,
    # 31611, Tanzania" is not a caption). Nominatim also packs alternates into
    # one field: Africa comes back as
    # "Afrika;<arabic>". The first alternate of the first component is the name.
    display = str(first.get("display_name") or query)
    short = display.split(",")[0].split(";")[0].strip() or query

    record = {"bbox": list(bbox), "name": short, "full_name": display,
              "source": "OpenStreetMap", "area_km2": round(area, 1)}
    cache[query.lower()] = record
    _save_cache(cache)
    return {**record, "bbox": bbox}
