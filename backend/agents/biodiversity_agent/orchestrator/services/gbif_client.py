"""GBIF API client for the Species Distribution worker.

Wraps ``pygbif`` (https://github.com/gbif/pygbif) with the small set of
production concerns the raw library does not handle for us:

- **Retry with exponential back-off** on any transient error (network
  glitch, 5xx, timeout). Three attempts by default with jitter to avoid
  thundering-herd retries.
- **Typed exceptions** — ``GBIFError`` for retriable / operational
  failures, ``GBIFNoResults`` when the query is well-formed but the API
  returned zero records. The worker maps both to ``AgentStatus.FAILED``
  with a clear message instead of leaking pygbif internals upstream.
- **Record normalization** — GBIF's occurrence records carry ~90 fields.
  We keep the six we actually use (lat, lon, country, region, year,
  gbif_id, basisOfRecord) in a dict that matches the shape the folium
  renderer already expects, so the map pipeline stays unchanged.

GBIF is a public, un-authenticated REST API. No API key required.
Endpoint: https://api.gbif.org/v1/occurrence/search
"""

from __future__ import annotations

import random
import time
from typing import Any


class GBIFError(RuntimeError):
    """Raised when GBIF cannot be reached after all retries or returns
    a malformed response. The worker converts this to
    ``AgentStatus.FAILED`` — the orchestrator may retry the whole
    request later."""


class GBIFNoResults(RuntimeError):
    """Raised when GBIF returns zero occurrences for the query. This is
    a valid API response, not an error — the worker converts it into
    ``AgentStatus.FAILED`` with a clear message so the caller knows the
    species is unknown rather than the network being down."""


# ---------- retry helper ----------


def _with_retry(fn, *, max_attempts: int = 3, base_delay: float = 1.0):
    """Call ``fn()`` up to ``max_attempts`` times, sleeping
    ``base_delay * 2**attempt + jitter`` seconds between attempts. Raises
    ``GBIFError`` wrapping the last exception if every attempt fails."""

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # broad — pygbif can raise many things
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise GBIFError(
        f"GBIF call failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


# ---------- record normalization ----------


def _normalize_record(rec: dict) -> dict | None:
    """Convert a raw GBIF occurrence record into our internal
    ``observation`` dict shape. Returns ``None`` if the record is
    missing coordinates (which happens even with ``hasCoordinate=True``
    on the query - GBIF occasionally returns records tagged with the
    flag but without the fields populated)."""

    lat = rec.get("decimalLatitude")
    lon = rec.get("decimalLongitude")
    if lat is None or lon is None:
        return None

    return {
        "lat":     float(lat),
        "lon":     float(lon),
        "country": rec.get("country") or "",
        "region":  rec.get("stateProvince") or rec.get("locality") or "",
        "year":    rec.get("year"),
        "gbif_id": rec.get("key"),
        "basis":   rec.get("basisOfRecord") or "",
    }


# ---------- public API ----------


def search_occurrences(
    scientific_name: str,
    limit: int = 300,
    country: str | None = None,
    year_range: tuple[int, int] | None = None,
) -> tuple[list[dict], int]:
    """Fetch occurrences from GBIF.

    Parameters
    ----------
    scientific_name
        The exact scientific binomial ("Loxodonta africana"). GBIF is
        strict — normalize the user's input via the taxonomy service
        before calling this.
    limit
        Max records to fetch. GBIF caps single-page responses at 300.
        Pagination is intentionally not implemented here — Sprint 3
        scope is one page per query. Add later if needed.
    country
        Optional ISO-3166-1 alpha-2 country code filter (``"KE"``,
        ``"TZ"``, ...).
    year_range
        Optional ``(start, end)`` inclusive year filter.

    Returns
    -------
    ``(observations, total_count)`` where ``observations`` is the list
    of normalized records (may be shorter than ``limit``) and
    ``total_count`` is GBIF's overall count for the query — used by the
    worker to score confidence.

    Raises
    ------
    GBIFNoResults
        Query well-formed but GBIF returned zero records.
    GBIFError
        Network / API failure after retries.
    """

    from pygbif import occurrences as gbif_occ  # lazy import

    kwargs: dict[str, Any] = {
        "scientificName": scientific_name,
        "hasCoordinate":  True,
        "limit":          limit,
    }
    if country:
        kwargs["country"] = country
    if year_range:
        kwargs["year"] = f"{year_range[0]},{year_range[1]}"

    resp = _with_retry(lambda: gbif_occ.search(**kwargs))

    total_count = int(resp.get("count", 0))
    raw_records = resp.get("results") or []
    if not raw_records:
        raise GBIFNoResults(
            f"GBIF returned no occurrences for '{scientific_name}'"
            + (f" in country={country}" if country else "")
            + (f" for years {year_range}" if year_range else "")
        )

    observations = [n for r in raw_records if (n := _normalize_record(r))]
    if not observations:
        # Every record was filtered out for missing coordinates.
        raise GBIFNoResults(
            f"GBIF returned {len(raw_records)} records for '{scientific_name}' "
            "but none had valid coordinates."
        )
    return observations, total_count


def get_common_name(scientific_name: str) -> str | None:
    """Best-effort lookup of the vernacular (common) name via GBIF's
    species name-backbone. Never raises - returns ``None`` if the
    lookup fails so the worker can proceed with an empty common name
    rather than aborting."""

    from pygbif import species as gbif_sp

    try:
        result = _with_retry(lambda: gbif_sp.name_backbone(name=scientific_name))
    except GBIFError:
        return None
    return result.get("vernacularName") or None
