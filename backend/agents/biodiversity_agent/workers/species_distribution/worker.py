"""Real GBIF-backed Species Distribution worker (Sprint 3).

Replaces ``SpeciesDistributionMock`` for production use. The
orchestrator picks ``SpeciesDistributionWorker`` as its default worker
for the ``species_distribution_map`` feature. ``SpeciesDistributionMock``
is kept in the same package for tests that need to run offline without
hitting GBIF.

Pipeline (matches the Sprint 1 spec drawn in
``docs/framework_benchmark_results.md``):

1. Read ``species_name`` from the request (already normalized by the
   orchestrator's ``normalize_species`` node via Qdrant taxonomy).
2. Fetch occurrences from GBIF via
   ``orchestrator.services.gbif_client.search_occurrences``.
3. Clean the coordinate list (drop ``(0, 0)``, drop out-of-range,
   deduplicate on ``(lat, lon)`` rounded to 4 decimals).
4. Enrich each kept point with per-record country / region / year
   metadata pulled from the GBIF record itself, plus the vernacular
   name looked up once per species from GBIF's species backbone.
5. Score confidence from GBIF's total ``count`` field.
6. Render the folium map via the shared
   ``orchestrator.services.map_renderer.render_point_map``.
7. Return a ``COMPLETED`` ``AgentResult`` with the structured
   ``SpeciesDistributionOutput`` payload and the map URL.

Every network / data failure downgrades to ``AgentStatus.FAILED`` with
a message the orchestrator can bubble upward — the worker never lets a
raw exception escape.
"""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus
from .schema import SpeciesDistributionOutput


class SpeciesDistributionWorker:
    """Production Species Distribution worker backed by GBIF."""

    def run(self, request: AgentRequest) -> AgentResult:
        species = self._resolve_species_name(request)
        if not species:
            return self._failed("species_name is required")

        # Sanitize the species name. GPT-5-mini sometimes returns compound
        # explanations like ``"African elephants (Loxodonta africana /
        # Loxodonta cyclotis)"`` in the ``species_name`` slot. GBIF cannot
        # search on that. Strip any ``(...)`` gloss and, if the remainder
        # is a ``A / B`` disjunction, pick the first Latin binomial - it
        # is the safer bet (savanna elephant is far more numerous than
        # forest elephant in GBIF, tiger subspecies rank the same way).
        species = self._sanitize_species(species)

        # Lazy imports break the circular dependency between the
        # orchestrator package and its worker packages.
        try:
            from ...orchestrator.services.gbif_client import (
                GBIFError,
                GBIFNoResults,
                get_common_name,
                search_occurrences,
            )
        except ImportError as exc:
            return self._failed(
                f"GBIF client unavailable: {exc}. Install pygbif "
                "(`pip install pygbif`)."
            )

        # 1. Fetch from GBIF ---------------------------------------------------
        try:
            observations, total_count = search_occurrences(
                scientific_name=species,
                limit=300,
                year_range=request.time_period,
            )
        except GBIFNoResults as exc:
            return self._failed(str(exc))
        except GBIFError as exc:
            return self._failed(f"GBIF API error: {exc}")

        # 2. Clean coordinates -------------------------------------------------
        raw_coords = [(o["lat"], o["lon"]) for o in observations]
        cleaned, kept_indices = self._clean_coordinates_indexed(raw_coords)
        if not cleaned:
            return self._failed(
                f"All {len(observations)} GBIF records for '{species}' "
                "had invalid coordinates after cleaning."
            )

        # 3. Enrich metadata for popups ----------------------------------------
        common_name = get_common_name(species) or ""
        kept = [observations[i] for i in kept_indices]
        metadata = [
            {
                "common_name": common_name,
                "country":     obs["country"],
                "region":      obs["region"],
                "year":        obs.get("year"),
            }
            for obs in kept
        ]

        # 4. Score confidence from the real total count ------------------------
        confidence = self._confidence(total_count)

        # 5. Render the folium map --------------------------------------------
        map_url: str | None = None
        try:
            from ...orchestrator.services.map_renderer import render_point_map

            map_url = render_point_map(
                species,
                cleaned,
                metadata=metadata,
                confidence=confidence,
            )
        except Exception:
            # Map rendering must never take the whole request down. The
            # structured payload still contains the coordinates so a
            # frontend can render its own map.
            map_url = None

        # 6. Build the structured payload --------------------------------------
        payload = SpeciesDistributionOutput(
            species_name=species,
            coordinates=cleaned,
            observation_count=total_count,
            map_url=map_url or "",
        )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=map_url,
            observation_count=total_count,
            confidence=confidence,
            source_agents=["Species Distribution Agent"],
        )

    # ---------- helpers ----------

    @staticmethod
    def _sanitize_species(raw: str) -> str:
        """Clean an LLM-supplied species name for GBIF.

        GPT-5-mini has been observed to return, in the ``species_name``
        slot, any of: ``"African elephant"`` (a common name); ``"African
        elephants (Loxodonta africana / Loxodonta cyclotis)"`` (a common
        name with a parenthetical Latin gloss); ``"Loxodonta africana /
        Loxodonta cyclotis"`` (a disjunction of two binomials);
        ``"Panthera tigris (Bengal tiger)"`` (a binomial with a common
        name gloss). GBIF ``search_occurrences`` only accepts a single
        Latin binomial.

        Strategy: prefer any Latin binomial found anywhere in the string
        (``Genus species`` - capitalised genus, lowercase species). If
        several, keep the first. If none, fall back to the string with
        parentheticals stripped and any disjunction split. Never returns
        empty - if sanitising empties the string, the raw input is
        returned so the downstream ``NoResults`` message is meaningful.
        """

        import re

        # A word that starts with a capital + English adjective is not a
        # Latin genus. Kept short: Latin genera are 8k+ words and blocking
        # a real one is worse than accepting an English one as long as the
        # blocked set stays tiny and uncontroversial.
        _NOT_LATIN = {
            "African", "American", "Arctic", "Asian", "Atlantic", "Australian",
            "Bengal", "Black", "Blue", "Brown", "Common", "Eastern",
            "European", "Giant", "Gray", "Great", "Grey", "Indian", "Little",
            "Northern", "Pacific", "Polar", "Red", "Siberian", "Southern",
            "Sumatran", "Western", "White", "Yellow",
        }

        def _binomials(text: str) -> list[str]:
            hits = re.findall(r"\b[A-Z][a-z]+ [a-z]+\b", text)
            return [h for h in hits if h.split()[0] not in _NOT_LATIN]

        # 1. Prefer a Latin binomial inside parens: "African elephants
        # (Loxodonta africana / ...)" -> "Loxodonta africana".
        for paren in re.findall(r"\(([^)]+)\)", raw):
            inner = _binomials(paren)
            if inner:
                return inner[0]

        # 2. Otherwise look outside parens.
        outside = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
        binomials = _binomials(outside)
        if binomials:
            return binomials[0]

        # 3. Fallback: keep the left side of any ``A / B`` disjunction.
        if "/" in outside:
            outside = outside.split("/", 1)[0].strip()
        return outside or raw

    @staticmethod
    def _resolve_species_name(request: AgentRequest) -> str | None:
        if request.species_name:
            return request.species_name.strip() or None
        return (
            request.context.get("species")
            or request.context.get("species_name")
            or None
        )

    @staticmethod
    def _failed(message: str) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            output=message,
            source_agents=["Species Distribution Agent"],
        )

    @staticmethod
    def _clean_coordinates_indexed(
        raw: list[tuple[float, float]],
    ) -> tuple[list[tuple[float, float]], list[int]]:
        seen: set[tuple[float, float]] = set()
        cleaned: list[tuple[float, float]] = []
        kept_indices: list[int] = []
        for idx, (lat, lon) in enumerate(raw):
            if (lat, lon) == (0.0, 0.0):
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            key = (round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append((lat, lon))
            kept_indices.append(idx)
        return cleaned, kept_indices

    @staticmethod
    def _confidence(observation_count: int) -> float:
        if observation_count >= 1000:
            return 0.95
        if observation_count >= 100:
            return 0.80
        if observation_count >= 10:
            return 0.60
        return 0.30
