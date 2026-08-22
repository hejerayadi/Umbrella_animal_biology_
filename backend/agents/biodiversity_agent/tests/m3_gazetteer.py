"""A fixed gazetteer for the M3 tests, so the suite never leaves the machine.

The worker resolves any named place through OpenStreetMap. That is the right
behaviour in production and the wrong behaviour in a test suite: it would make
the tests slow, dependent on someone else's uptime, and quietly different when a
gazetteer entry changes. So the tests replace the lookup with this table.

The six documented regions are *not* here - they resolve from
``config.REGIONS`` without any lookup, which is exactly what these tests want to
keep true.
"""

from __future__ import annotations

from backend.agents.biodiversity_agent.workers.common.geocode import (
    PlaceNotUsable,
    bbox_area_km2,
)

# Real bounding boxes, rounded. Enough to be plausible, small enough to read.
PLACES: dict[str, tuple[float, float, float, float]] = {
    "brazil": (-74.0, -33.9, -28.6, 5.3),
    "costa rica": (-86.0, 8.0, -82.5, 11.2),
    "borneo": (108.8, -4.2, 119.3, 7.0),
    "india": (68.0, 6.6, 97.4, 35.7),
    "serengeti": (34.2, -2.5, 35.3, -1.4),
}

# Places that resolve to something real but unusable, which is a different answer
# from "not found" and is reported differently.
TOO_SMALL = {"atlantis": 4.0, "western ghats": 0.0}
TOO_LARGE = {"africa": 30_153_621.0}

# Crosses the antimeridian, so Nominatim reports a 360-degree box.
WRAPS_THE_GLOBE = {"united states", "america", "russia", "new zealand"}


def fake_resolve_place(name: str, **_kwargs) -> dict | None:
    """Stand-in for ``geocode.resolve_place``, with no network."""

    key = (name or "").strip().lower()
    if key in PLACES:
        bbox = PLACES[key]
        return {"bbox": bbox, "name": key.title(), "full_name": f"{key.title()}, test",
                "source": "OpenStreetMap", "area_km2": round(bbox_area_km2(bbox), 1)}
    if key in TOO_SMALL:
        raise PlaceNotUsable(
            f"{key.title()} is only about {TOO_SMALL[key]:,.0f} km2 - too small to "
            f"bin into cells at this resolution. Name a wider area.")
    if key in WRAPS_THE_GLOBE:
        raise PlaceNotUsable(
            f"{key.title()} crosses the antimeridian, so its bounding box spans "
            f"the whole globe and cannot be one study area. Name a part of it - a "
            f"state, province or region.")
    if key in TOO_LARGE:
        raise PlaceNotUsable(
            f"{key.title()} covers about {TOO_LARGE[key]:,.0f} km2 - too large for "
            f"one study area at this cell size. Name a country or region inside it.")
    return None


def patch_gazetteer(monkeypatch) -> None:
    """Point every caller of ``resolve_place`` at the table above."""

    from backend.agents.biodiversity_agent.workers.common import geocode
    from backend.agents.biodiversity_agent.workers.hotspots import worker

    monkeypatch.setattr(geocode, "resolve_place", fake_resolve_place)
    monkeypatch.setattr(worker, "resolve_place", fake_resolve_place)
