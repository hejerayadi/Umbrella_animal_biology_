"""M3 - Biodiversity Hotspots: the real worker.

Implements the eight steps of ``Biodiversity Hotspots.pdf`` §4.2 and the status
semantics of §4.5. The worker keeps the platform contract - ``run(request)``
returns an ``AgentResult`` - so the sub-orchestrator needs no change to use it in
place of ``HotspotsMock``.

    1 resolve the study area        5 correct for sampling effort
    2 estimate the workload         6 cluster (DBSCAN)
    3 retrieve occurrences          7 characterise and rank
    4 build the richness grid       8 heatmap and summary

Statuses, per doc Table 14:

    COMPLETED            a grid and zero or more clusters were produced
    PARTIAL              the grid was built but clustering or naming failed
    NEEDS_CLARIFICATION  the region could not be resolved - the known ones are listed
    FAILED               the upstream source is unavailable, or the area is too large
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ...schema import AgentRequest, AgentResult, AgentStatus
from .status import M3Outcome, build_result
from ..common.config import (M3_DEFAULTS, M3_EPS_GRID, REGIONS, SUBSTITUTIONS,
                            WORLD_BBOX,
                            place_candidates, regions_in_text, species_in_text)
from ..common.geocode import bbox_area_km2, resolve_place
from ..common.gbif import (choose_strategy, count_probe, download_occurrences,
                          match_species)
from .pipeline import (clustering, effort, grid, indices, points, ranking,
                      render)
from .pipeline.cleaning import clean_occurrences
from .schema import (
    BiodiversityHotspotInput,
    BiodiversityHotspotOutput,
    Citation,
    ClusteringQuality,
    EffortCorrection,
    HotspotIndex,
    HotspotValidation,
    MapRenderSpec,
    RichnessCell,
)

Progress = Callable[[str], None] | None

# Substitutions that always apply to this module, so the caveats always travel
# with the result instead of being lost (doc §7.3).
_ALWAYS_WARN = ["h3", "ecoregion_name", "iucn_threat", "cepf_validation"]


# Words that mean the asker really did want the whole planet, as opposed to the
# platform default for AgentRequest.region happening to be "global".
_WORLD_WORDS = ("global", "globally", "world", "worldwide", "planet", "everywhere")


def _asks_for_the_world(instruction: str | None) -> bool:
    lowered = (instruction or "").lower()
    return any(word in lowered for word in _WORLD_WORDS)


def _place_warning(params: BiodiversityHotspotInput) -> list[str]:
    """Say so when the study area came from a gazetteer rather than the document.

    "Yellowstone" resolves to Yellowstone County, Montana - a real place, and not
    the park the asker probably meant. The answer cannot detect that, but it can
    show which box it used, which is what lets the reader correct it.
    """

    if params.place_source != "OpenStreetMap":
        return []
    area = ""
    if params.bbox:
        lon_min, lat_min, lon_max, lat_max = params.bbox
        area = (f", about {abs(lat_max - lat_min):.1f} deg by "
                f"{abs(lon_max - lon_min):.1f} deg")
    named = params.place_full_name or params.region_name
    return [f"Study area taken from OpenStreetMap: {named}{area}. If that is not "
            f"the place you meant, name it more precisely."]


def _slug(name: str | None) -> str:
    """A filename fragment from a place or species name.

    Gazetteer names contain characters that are not filenames: "Borneo /
    Kalimantan" turned into a cache path with a directory separator in it, and
    pandas correctly refused to write to a directory that does not exist.
    """

    kept = "".join(character if character.isalnum() else "_"
                   for character in (name or "custom").lower())
    # Collapse runs, so "Borneo / Kalimantan" is borneo_kalimantan rather than
    # borneo___kalimantan.
    return "_".join(part for part in kept.split("_") if part) or "custom"


# A grid needs enough cells to cluster: DBSCAN over eight cells is not
# clustering, it is arithmetic with extra steps. Below this count the cell size
# is refined to reach roughly TARGET_CELLS, which is what makes a small island or
# a national park answerable at all - the documented 42 km cell would put
# Pantelleria (136 km2) inside a single cell.
MIN_CELLS_FOR_CLUSTERING = 30
TARGET_CELLS = 60
FINEST_CELL_KM = 1.0


def _eps_grid_for(cell_size_km: float) -> list[float] | None:
    """The eps candidates to scan, scaled to the grid actually in use.

    eps is a distance between cell centres, so it only means anything relative to
    the cell size. The documented candidates (60-250 km) belong to the documented
    42 km cell; on a refined 10 km grid they all merge everything into one
    cluster, and the scan reports that nothing qualified. Scaling keeps the
    documented ratios - and returns None for the documented cell, so a region
    from the design document scans exactly the documented values.
    """

    documented = float(M3_DEFAULTS["cell_size_km"])
    if abs(cell_size_km - documented) < 0.05:
        return None
    scale = cell_size_km / documented
    return [round(candidate * scale, 1) for candidate in M3_EPS_GRID]



def _as_file_uri(path: str | None) -> str | None:
    """A rendered map's path as a ``file://`` URI, the shape `map_url` promises.

    Every other worker returns ``Path(...).resolve().as_uri()`` (see
    ``orchestrator/services/map_renderer.py``); hotspots returned the raw
    filesystem path, which on Windows is a bare drive path, not a URL a browser
    will load. ``spec.html_path`` itself stays a plain path - the dashboard and
    ``api_m3`` read it as one - so only the external contract changes here.
    """
    if not path:
        return None
    try:
        return Path(path).resolve().as_uri()
    except (ValueError, OSError):
        # Already a URI, or something that is not a local path: hand it back
        # untouched rather than losing the only pointer to the map.
        return path


class AmbiguousRegion(LookupError):
    """More than one known study area was named, so the module asks which.

    A LookupError subclass, so a caller that only knows the base class still
    treats it as "region could not be resolved" rather than crashing.
    """

    def __init__(self, message: str, options: list[str]) -> None:
        super().__init__(message)
        self.options = options


class HotspotsWorker:
    """Region in, ranked hotspots out. No LLM, no network beyond GBIF."""

    def run(self, request: AgentRequest, *, progress: Progress = None) -> AgentResult:
        started = time.time()
        try:
            params = self._parse(request)
        except AmbiguousRegion as exc:
            return build_result(
                M3Outcome.NEEDS_CLARIFICATION,
                output={"message": str(exc), "options": exc.options,
                        "known_regions": sorted(REGIONS)},
                source_agents=["Biodiversity Hotspots Agent"])
        except LookupError as exc:
            # Unresolvable region: list what we do know rather than guessing.
            # M3 asks rather than guessing. The platform has no such status, so
            # this narrows to FAILED with `status_detail` preserved - see
            # workers/hotspots/status.py.
            return build_result(
                M3Outcome.NEEDS_CLARIFICATION,
                output={
                    "message": str(exc),
                    "known_regions": sorted(REGIONS),
                },
                source_agents=["Biodiversity Hotspots Agent"],
            )
        except ValueError as exc:
            return AgentResult(status=AgentStatus.FAILED, output=str(exc),
                               source_agents=["Biodiversity Hotspots Agent"])

        return self.analyse(params, progress=progress, started=started)

    # ------------------------------------------------------------ step 1

    def _parse(self, request: AgentRequest) -> BiodiversityHotspotInput:
        """Resolve the study area from a request. Doc §4.2 step 1.

        The caller may name the region explicitly (``request.region``, which is
        what the orchestrator sets) or leave it in the sentence (which is what a
        chat client does). Both must work, so an unset or unusable ``region``
        falls back to reading the instruction - a lookup, not a model.
        """

        settings = dict(request.context.get("hotspot_params") or {})

        # Species mode, if the caller named one. The platform's own AgentRequest
        # already carries species_name - the orchestrator fills it in for M1 -
        # so a species question needs no new field anywhere upstream.
        wanted = settings.get("species_name") or request.species_name
        if not wanted:
            wanted = species_in_text(request.instruction)
        typed_species = wanted
        if wanted and not settings.get("species_key"):
            resolved = match_species(wanted)
            if resolved is None:
                raise LookupError(
                    f"I could not resolve '{wanted}' to a single species. Try the "
                    f"scientific name, for example 'Panthera tigris'.")
            settings["species_key"], settings["species_name"] = resolved

        # A species is not bound to one of the six study areas: "where do tigers
        # live" is a question about the whole range. Only the region path needs a
        # study area, so this one defaults to the world.
        if (settings.get("species_key") and not settings.get("region_name")
                and not settings.get("bbox")):
            found = regions_in_text(request.instruction)
            if found:
                settings["region_name"] = found[0]
            else:
                # Both names are excluded: the canonical one ("Panthera
                # tigris") does not appear in "hotspots for tigers", so excluding
                # only that left "tigers" to be geocoded - and it resolves to a
                # street in Frederiksberg.
                place = self._study_area_from_text(
                    request.instruction,
                    exclude=[typed_species, settings.get("species_name")])
                if place:
                    settings["bbox"] = place["bbox"]
                    settings["region_name"] = place["name"]
                    settings["place_source"] = place["source"]
                    settings["place_full_name"] = place.get("full_name")
                else:
                    # No place named: a species question is about its whole range.
                    settings["bbox"] = WORLD_BBOX
                    settings["region_name"] = "worldwide"
            return self.build_params(settings)

        if not settings.get("region_name") and not settings.get("bbox"):
            named = (request.region or "").strip().lower()
            if named in ("", "global"):
                # Nothing usable was passed: read the question itself. If it
                # names no known area either, build_params raises LookupError and
                # run() answers NEEDS_CLARIFICATION with the list of known ones.
                found = regions_in_text(request.instruction)
                if len(found) > 1:
                    # "Compare the Amazon and the Congo Basin" is two analyses.
                    # Guessing which one the user meant would silently answer a
                    # question nobody asked, so ask - with the options.
                    raise AmbiguousRegion(
                        "That question names more than one study area, and I "
                        "analyse one at a time.", found)
                if found:
                    named = found[0]
                else:
                    # Not one of the documented six: try the gazetteer, so a
                    # question about Brazil or Borneo gets an answer instead of a
                    # list of six regions the asker never mentioned.
                    place = self._study_area_from_text(request.instruction)
                    if place:
                        settings["bbox"] = place["bbox"]
                        settings["region_name"] = place["name"]
                        settings["place_source"] = place["source"]
                        settings["place_full_name"] = place.get("full_name")
                        return self.build_params(settings)
                if not found and named == "global" and not _asks_for_the_world(
                        request.instruction):
                    # "global" is the platform default for AgentRequest.region,
                    # not a request. Answering "the whole globe is too large" to
                    # "which part of Africa has the most species?" replies to a
                    # question nobody asked, so this falls through to the clearer
                    # "I could not tell which study area you mean".
                    named = ""
            settings["region_name"] = named

        return self.build_params(settings)

    @staticmethod
    def _study_area_from_text(instruction: str | None,
                              exclude: str | list[str | None] | None = None
                              ) -> dict | None:
        """The first place named in a question that a gazetteer can resolve.

        ``place_candidates`` offers more than one reading of the sentence; the
        gazetteer decides between them. ``PlaceTooSmall`` is allowed to escape:
        "that place is 4 km2" is a better answer than "I could not find it".
        """

        text = instruction or ""
        names = [exclude] if isinstance(exclude, str) else list(exclude or [])
        for name in [n for n in names if n]:
            # "hotspots for tigers" would otherwise ask the gazetteer about
            # "tigers", which resolves to a street in Frederiksberg and answers a
            # question about the wrong thing entirely.
            # Longest form first: removing "tiger" before "tigers" leaves an
            # orphan "s", which then gets geocoded as if it were a place name.
            for form in sorted({name, name + "s", name.rstrip("s")},
                               key=len, reverse=True):
                text = re.sub(re.escape(form), " ", text, flags=re.IGNORECASE)

        for candidate in place_candidates(text):
            place = resolve_place(candidate)
            if place:
                return place
        return None

    def build_params(self, settings: dict) -> BiodiversityHotspotInput:
        """Turn a settings dict into the documented input contract.

        Public because the dashboard assembles its own settings from the sidebar
        controls, and both paths must land on the same validated
        ``BiodiversityHotspotInput`` - the validation lives here, once.
        """

        overrides = dict(settings)
        bbox = overrides.pop("bbox", None)
        region = overrides.pop("region_name", None)

        if bbox is None:
            key = (region or "").strip().lower()
            if key == "global":
                # "global" is not a study area M3 can bin at this cell size.
                raise LookupError(
                    "The whole globe is too large to grid at this cell size, so "
                    "I work one study area at a time.")
            if key == "":
                # Nothing to guess from. Saying "global is too large" here would
                # answer a question the user never asked.
                raise LookupError(
                    "I could not tell which study area you mean.")
            if key in REGIONS:
                bbox = REGIONS[key]
                region = key
            else:
                # A place the caller named directly - the orchestrator may pass
                # "brazil" in AgentRequest.region, which is not one of the six.
                place = resolve_place(key)
                if place is None:
                    raise LookupError(
                        f"I could not find a place called '{region}'. Name a "
                        f"country, region, island or park - for example the Congo "
                        f"Basin, Brazil, Borneo or the Serengeti.")
                bbox = place["bbox"]
                region = place["name"]
                overrides.setdefault("place_source", place["source"])
                overrides.setdefault("place_full_name", place.get("full_name"))
        else:
            region = region or "custom bbox"

        defaults = dict(M3_DEFAULTS)
        defaults.update(overrides)

        # The grid has to fit the place, not the other way round. A 42 km cell
        # over a 136 km2 island is one cell; refusing the island would be
        # refusing a real question because of a default.
        area = bbox_area_km2(tuple(bbox))
        cell_km = float(defaults["cell_size_km"])
        if not overrides.get("cell_size_km") and not defaults.get("species_key"):
            if area / (cell_km * cell_km) < MIN_CELLS_FOR_CLUSTERING:
                refined = max(FINEST_CELL_KM,
                              round(math.sqrt(area / TARGET_CELLS), 1))
                if refined < cell_km:
                    coarse = area / (cell_km * cell_km)
                    had = ("would put the whole area inside a single cell"
                           if coarse < 1.5 else
                           f"would leave only {coarse:.0f} cells")
                    defaults["cell_size_note"] = (
                        f"{area:,.0f} km2 is small for the documented "
                        f"{cell_km:.0f} km grid, which {had} - nothing to cluster. "
                        f"The grid was refined to {refined:g} km, about "
                        f"{area / (refined * refined):.0f} cells.")
                    defaults["cell_size_km"] = refined
                    # eps is a distance between cell centres, so the fallback has
                    # to move with the cell: 120 km across a 27 km island is one
                    # cluster by construction.
                    if not overrides.get("eps_km"):
                        ratio = (float(M3_DEFAULTS["eps_km"])
                                 / float(M3_DEFAULTS["cell_size_km"]))
                        defaults["eps_km"] = round(refined * ratio, 1)

        index = defaults.pop("index", HotspotIndex.RICHNESS)
        if isinstance(index, str):
            index = HotspotIndex(index)

        return BiodiversityHotspotInput(
            region_name=region,
            bbox=tuple(bbox),
            taxon_filter=defaults["taxon_filter"],
            year_from=defaults["year_from"],
            year_to=defaults["year_to"],
            cell_size_km=float(defaults["cell_size_km"]),
            min_records_per_cell=int(defaults["min_records_per_cell"]),
            index=index,
            normalise_by_effort=bool(defaults.get("normalise_by_effort", True)),
            eps_km=float(defaults["eps_km"]),
            min_samples=int(defaults["min_samples"]),
            auto_tune=bool(defaults.get("auto_tune", True)),
            top_n=int(defaults["top_n"]),
            include_grid=bool(defaults.get("include_grid", True)),
            max_records=int(defaults["max_records"]),
            # Carried through, not defaulted: a species key set here is what makes
            # analyse() take the species path instead of the grid.
            species_name=defaults.get("species_name"),
            species_key=(int(defaults["species_key"])
                         if defaults.get("species_key") else None),
            place_source=defaults.get("place_source") or "design document",
            place_full_name=defaults.get("place_full_name"),
            cell_size_note=defaults.get("cell_size_note"),
        )

    # ------------------------------------------------------------ steps 2-8

    def analyse(self, params: BiodiversityHotspotInput, *,
                progress: Progress = None, started: float | None = None) -> AgentResult:
        """Run steps 2 to 8. Public so the dashboard can call it directly."""

        started = started or time.time()
        warnings = [SUBSTITUTIONS[key] for key in _ALWAYS_WARN]
        warnings.extend(_place_warning(params))
        if params.cell_size_note:
            warnings.append(params.cell_size_note)

        def say(message: str) -> None:
            if progress:
                progress(message)

        if params.species_key:
            # One species: the grid, the diversity indices and the effort
            # correction all measure nothing, so the species path clusters the
            # records themselves. See _analyse_species.
            return self._analyse_species(params, say=say, started=started)

        # --- step 2: estimate the workload before moving any data
        estimated, probe_error = count_probe(
            params.bbox, taxon_key=params.taxon_key,
            year_from=params.year_from, year_to=params.year_to)
        strategy = choose_strategy(estimated)

        if probe_error:
            # Doc §4.2 step 2: an unavailable probe assumes the safe default.
            warnings.append(f"{probe_error} - assuming the paged path.")
            strategy = "paged"

        if strategy == "refuse":
            # Doc Table 12 refuses above 20 million records, on the assumption
            # that they would all be downloaded. This path never downloads more
            # than max_records, so refusing would turn "hotspots in Costa Rica"
            # - 37 million records - into a failure for no reason. It is answered
            # from a sample instead, and the sample size is stated.
            warnings.append(
                f"GBIF holds about {estimated:,} matching records here; this run "
                f"reads the {params.max_records:,} most recent "
                f"({params.max_records / estimated:.2%}). The hotspots are those of "
                f"that sample, not of the full archive.")

        if strategy == "download":
            warnings.append(
                f"About {estimated:,} records exceed the paged ceiling; the "
                f"asynchronous GBIF Download API (which issues a citable DOI) is not "
                f"implemented, so this run is capped at {params.max_records:,} records "
                f"and is a sample, not a census.")

        # --- step 3: retrieve
        say(f"retrieving occurrences for {params.region_name}")
        slug = _slug(params.region_name)
        raw = download_occurrences(
            params.bbox, f"gbif_{slug}.csv", params.max_records,
            taxon_key=params.taxon_key,
            year_from=params.year_from, year_to=params.year_to,
            progress=progress)

        if raw.empty:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"GBIF returned no records for {params.region_name}.",
                source_agents=["Biodiversity Hotspots Agent"])

        say("applying the cleaning rules")
        clean, cleaning_report = clean_occurrences(raw, params.bbox)
        if len(clean) < len(raw) * 0.5:
            # For one species this is normally rule 3: the same sighting
            # republished by several aggregators, or many records sharing a
            # rounded coordinate within one year. Large and worth stating.
            removed = {row["rule"]: row["removed"] for row in cleaning_report
                       if row.get("removed")}
            warnings.append(
                f"Cleaning kept {len(clean):,} of {len(raw):,} retrieved records. "
                f"Removed: " + ", ".join(f"{rule} ({count:,})"
                                         for rule, count in removed.items()) + ".")
        if clean.empty:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="Every record was removed by the cleaning rules.",
                source_agents=["Biodiversity Hotspots Agent"])

        # A single-year span means the paged path returned only the newest records,
        # so the sample is biased in time. Say so rather than let it pass.
        years = clean["year"].dropna()
        if not years.empty and years.nunique() <= 1:
            warnings.append(
                f"All retrieved records are from {int(years.iloc[0])}: the paged path "
                f"returns the newest first, so this is a recent snapshot rather than "
                f"a sample across {params.year_from}-{params.year_to}.")

        # --- step 4: build the richness grid
        say("binning onto the equal-area grid")
        clean = grid.assign_cells(clean, params.cell_size_km)
        cells, dropped = indices.build_cell_table(
            clean, params.cell_size_km, params.min_records_per_cell)
        say(f"{len(cells):,} cells above the noise floor "
            f"({len(clean):,} records, {clean['speciesKey'].nunique():,} species)")

        if cells.empty:
            # Doc §4.2 step 4: a legitimate outcome, not an error.
            return self._empty_grid_result(params, raw, clean, cleaning_report,
                                           warnings, dropped, started)

        # --- step 5: correct for sampling effort
        say("correcting richness for sampling effort")
        cells, effort_record = effort.correct_for_effort(
            cells, applied=params.normalise_by_effort)

        # --- step 6: cluster
        eps_km, min_samples = params.eps_km, params.min_samples
        scan: list[dict] = []
        tuned = False
        if params.auto_tune:
            say("scanning eps x min_samples")
            eps_km, min_samples, scan, tuned = clustering.tune(
                cells, params.eps_km, params.min_samples,
                eps_grid=_eps_grid_for(params.cell_size_km))
            if not tuned:
                warnings.append(
                    f"The scan found no (eps, min_samples) combination giving at "
                    f"least two clusters inside the plausible noise band, so the "
                    f"fallback was kept: eps {eps_km:g} km, min_samples "
                    f"{min_samples}.")

        say(f"clustering: DBSCAN, eps {eps_km:.0f} km, min_samples {min_samples}")
        model, labels = clustering.run_dbscan(cells, eps_km, min_samples)
        cells["cluster_id"] = [None if label == -1 else int(label) for label in labels]

        quality = clustering.measure_quality(cells, labels, eps_km=eps_km,
                                             min_samples=min_samples)
        quality.tuned = tuned
        quality.scan = scan

        # Reproducibility: doc §2.5 makes identical labels across two runs a
        # success metric, so it is asserted rather than assumed.
        _, labels_again = clustering.run_dbscan(cells, eps_km, min_samples)
        quality.reproducible = bool((labels == labels_again).all())

        # --- step 7: characterise and rank
        clusters, ranked = ranking.build_clusters(
            cells, clean, index=params.index, top_n=params.top_n)
        say(f"ranking {len(clusters)} hotspot(s) by {params.index.value}")

        if params.index is HotspotIndex.THREAT_WEIGHTED:
            warnings.append(
                "index=threat_weighted was requested but no IUCN token is configured; "
                "the ranking fell back to effort-corrected richness.")

        # --- step 8: render spec + summary
        say("drawing the map")
        spec = render.build_render_spec(cells, params.bbox)
        spec.html_path = render.render_folium(
            cells, ranked, slug, cell_size_km=params.cell_size_km,
            effort_applied=effort_record.applied)

        say("assembling the payload")
        payload = self._build_output(
            params, raw, clean, cells, clusters, ranked, effort_record, quality,
            spec, cleaning_report, warnings, started)

        # Nothing is persisted. DBSCAN has no predict(): it labels the cells it was
        # fitted on and cannot score a new point later, so there is no trained
        # artefact worth storing. A repeated question re-clusters from the cached
        # GBIF extract, which takes about a second.

        # One cluster holding everything is not a finding. It happens for a small
        # study area - the whole of Malta comes back as a single dense blob - and
        # an answer that reports "1 hotspot" without saying so implies a
        # discrimination the analysis did not make.
        if len(clusters) == 1 and quality.noise_share < 0.05:
            payload.summary += (
                f" Every one of the {len(cells)} cells fell into that single "
                f"cluster, so at this scale the study area reads as uniformly "
                f"rich rather than having a richest part. Only {len(cells)} cells "
                f"held the minimum {params.min_records_per_cell} records, and "
                f"{len(cells)} cells cannot be separated into distinct areas - a "
                f"larger study area, or denser recording, would be needed.")
            payload.confidence = min(payload.confidence, 0.5)

        # Zero clusters is COMPLETED with an explanation, never an empty heatmap.
        status = AgentStatus.COMPLETED
        if not clusters:
            payload.summary += (
                " No cluster met the density threshold, which is a legitimate result "
                "for a sparsely surveyed region - try a larger eps_km or a coarser cell.")

        return AgentResult(
            status=status,
            output=payload,
            map_url=_as_file_uri(spec.html_path),
            hotspots=[asdict(cluster) for cluster in ranked],
            confidence=payload.confidence,
            source_agents=["Biodiversity Hotspots Agent"],
        )

    # ------------------------------------------------- the species-scoped path

    def _analyse_species(self, params: BiodiversityHotspotInput, *, say,
                         started: float) -> AgentResult:
        """Where one species has been recorded, and where those records cluster.

        Deliberately not the region pipeline. With a single species every grid
        cell holds richness 1, so the indices carry no information and the effort
        correction has nothing to normalise against. What remains meaningful is
        the geography of the records themselves - so this clusters the points and
        ranks by record count, and says plainly that the ranking is observation
        density rather than biodiversity.
        """

        warnings = list(_place_warning(params)) + [
            f"Ranked by occurrence count, not by species richness: with one "
            f"species ({params.species_name}) richness is 1 everywhere. A cluster "
            f"here is where records concentrate, which reflects both the animal "
            f"and where people looked for it.",
        ]

        estimated, probe_error = count_probe(
            params.bbox, species_key=params.species_key,
            year_from=params.year_from, year_to=params.year_to)
        if probe_error:
            warnings.append(f"{probe_error} - assuming the paged path.")

        say(f"retrieving occurrences of {params.species_name}")
        slug = f"{_slug(params.species_name)}_{params.species_key}"
        raw = download_occurrences(
            params.bbox, f"gbif_species_{slug}.csv", params.max_records,
            species_key=params.species_key,
            year_from=params.year_from, year_to=params.year_to,
            progress=say)

        if raw.empty:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"GBIF returned no records for {params.species_name}.",
                source_agents=["Biodiversity Hotspots Agent"])

        say("applying the cleaning rules")
        # Subspecies records are kept: they are records of this species. Measured
        # on Panthera tigris, 4,552 of 5,008 GBIF records are at SUBSPECIES rank,
        # so the strict rule would discard 91% of the tiger data.
        clean, cleaning_report = clean_occurrences(raw, params.bbox,
                                                  allow_infraspecific=True)
        if clean.empty:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(f"Every record of {params.species_name} was removed by the "
                        f"cleaning rules."),
                source_agents=["Biodiversity Hotspots Agent"])

        eps_km, min_samples = params.eps_km, params.min_samples
        scan: list[dict] = []
        tuned = False
        if params.auto_tune:
            say("scanning eps x min_samples")
            eps_km, min_samples, scan, tuned = points.tune_occurrences(
                clean, params.eps_km, params.min_samples)
            if not tuned:
                warnings.append(
                    f"The scan found no (eps, min_samples) combination giving at "
                    f"least two clusters inside the plausible noise band, so the "
                    f"fallback was kept: eps {eps_km:g} km, min_samples "
                    f"{min_samples}.")

        say(f"clustering {len(clean):,} records: DBSCAN, eps {eps_km:.0f} km, "
            f"min_samples {min_samples}")
        _, labels = points.cluster_occurrences(clean, eps_km, min_samples)
        frame = points.as_latlon(clean)
        quality = clustering.measure_quality(frame, labels, eps_km=eps_km,
                                             min_samples=min_samples)
        quality.tuned = tuned
        quality.scan = scan
        _, again = points.cluster_occurrences(clean, eps_km, min_samples)
        quality.reproducible = bool((labels == again).all())

        clusters, ranked = points.build_occurrence_clusters(
            clean, labels, top_n=params.top_n, species_name=params.species_name)
        say(f"ranking {len(clusters)} occurrence cluster(s) by record count")

        say("drawing the map")
        spec = MapRenderSpec(
            type="occurrence_density",
            scale="none",
            classes=0,
            bbox=params.bbox,
            center=(round(float(frame["lat"].mean()), 3),
                    round(float(frame["lon"].mean()), 3)),
            zoom=4,
        )
        spec.html_path = render.render_species_folium(clean, ranked, slug)

        noise = int((labels == -1).sum())
        richest = ranked[0] if ranked else None
        where = ("worldwide" if params.region_name == "worldwide"
                 else f"in {params.region_name}")
        if richest:
            summary = (
                f"{len(clean):,} cleaned records of {params.species_name} {where} "
                f"cluster into {quality.n_clusters} area(s), with {noise:,} isolated "
                f"records left as noise. The busiest is {richest.label} with "
                f"{richest.record_count:,} records over {richest.area_km2:,.0f} km2. "
                f"That is where the species has been observed most, which is not "
                f"the same as where it is most abundant.")
        else:
            summary = (
                f"{len(clean):,} cleaned records of {params.species_name} {where}, "
                f"but none formed a cluster at eps {eps_km:.0f} km with min_samples "
                f"{min_samples}: the records are too scattered. Try a larger eps_km.")

        payload = BiodiversityHotspotOutput(
            region=params.region_name or "worldwide",
            study_area_km2=grid.study_area_km2(params.bbox),
            records_retrieved=int(len(raw)),
            records_analysed=int(len(clean)),
            species_analysed=1,
            cells_analysed=0,                  # no grid on this path
            noise_cells=noise,                 # noise records; the schema field is shared
            clusters=clusters,
            ranked_hotspots=ranked,
            effort_correction=EffortCorrection(
                method="not applied - a single species has no richness to correct",
                applied=False,
                note="Effort correction normalises richness per cell; with one "
                     "species there is nothing to normalise."),
            quality=quality,
            validation=HotspotValidation(),
            parameters_used={
                "mode": "species",
                "species_name": params.species_name,
                "species_key": params.species_key,
                "region": params.region_name,
                "bbox": params.bbox,
                "year_from": params.year_from,
                "year_to": params.year_to,
                "eps_km": eps_km,
                "min_samples": min_samples,
                "auto_tune": params.auto_tune,
                "ranked_by": "record_count",
                "top_n": params.top_n,
                "max_records": params.max_records,
            },
            render_spec=spec,
            citations=[Citation(
                source="GBIF",
                identifier=(f"GBIF Occurrence Search API, speciesKey="
                            f"{params.species_key}, {params.year_from}-{params.year_to}"),
                url="https://www.gbif.org",
                license="per contributing dataset")],
            summary=summary,
            confidence=0.8 if ranked else 0.5,
            cleaning_report=cleaning_report,
            warnings=warnings,
        )

        return build_result(
            M3Outcome.COMPLETED if ranked else M3Outcome.PARTIAL,
            output=payload,
            map_url=_as_file_uri(spec.html_path),
            hotspots=[asdict(cluster) for cluster in ranked],
            confidence=payload.confidence,
            source_agents=["Biodiversity Hotspots Agent"])

    # ------------------------------------------------------------ helpers

    def _empty_grid_result(self, params, raw, clean, cleaning_report, warnings,
                           dropped, started) -> AgentResult:
        warnings.append(
            f"No cell reached the noise floor of {params.min_records_per_cell} records "
            f"({dropped} cells were below it), so no grid could be built.")
        # Records were retrieved and cleaned but no cell cleared the noise floor:
        # real work, incomplete result. PARTIAL narrows to COMPLETED.
        return build_result(
            M3Outcome.PARTIAL,
            output=BiodiversityHotspotOutput(
                region=params.region_name or "custom",
                study_area_km2=round(grid.study_area_km2(params.bbox), 1),
                records_retrieved=int(len(raw)),
                records_analysed=int(len(clean)),
                species_analysed=int(clean["speciesKey"].nunique()),
                cells_analysed=0, noise_cells=0,
                clusters=[], ranked_hotspots=[],
                effort_correction=EffortCorrection(method="none", applied=False),
                quality=ClusteringQuality(n_clusters=0, noise_share=1.0, silhouette=None),
                validation=HotspotValidation(),
                parameters_used=self._parameters(params, params.eps_km, params.min_samples),
                render_spec=render.build_render_spec(clean.head(0), params.bbox),
                citations=self._citations(),
                summary=("Records were retrieved and cleaned, but the region is too "
                         "sparsely surveyed to build a defensible grid."),
                confidence=0.3,
                cleaning_report=cleaning_report,
                warnings=warnings,
            ),
            source_agents=["Biodiversity Hotspots Agent"],
        )

    def _build_output(self, params, raw, clean, cells, clusters, ranked,
                      effort_record, quality, spec, cleaning_report, warnings,
                      started) -> BiodiversityHotspotOutput:
        richest = ranked[0] if ranked else None
        scope = params.taxon_filter.lower()

        # Doc §7.3: the summary must state the taxonomic scope and whether the
        # effort correction was applied. Both are asserted here, not optional.
        if richest:
            summary = (
                f"Across {len(clean):,} cleaned {scope} occurrence records in "
                f"{params.region_name}, {clean['speciesKey'].nunique():,} species were "
                f"recorded in {len(cells):,} equal-area cells of "
                f"{params.cell_size_km:g} km. "
                f"{'After correcting for sampling effort, ' if effort_record.applied else 'Without effort correction, '}"
                f"DBSCAN identified {quality.n_clusters} hotspot cluster(s) and left "
                f"{quality.noise_share:.0%} of cells unassigned. The richest is "
                f"{richest.label} with {richest.species_count:,} species over "
                f"{richest.area_km2:,.0f} km2.")
        else:
            summary = (
                f"Across {len(clean):,} cleaned {scope} records in {params.region_name}, "
                f"{len(cells):,} cells were built but no cluster met the density "
                f"threshold.")

        confidence = self._confidence(quality, effort_record, len(cells))

        return BiodiversityHotspotOutput(
            region=params.region_name or "custom",
            study_area_km2=round(grid.study_area_km2(params.bbox), 1),
            records_retrieved=int(len(raw)),
            records_analysed=int(len(clean)),
            species_analysed=int(clean["speciesKey"].nunique()),
            cells_analysed=int(len(cells)),
            noise_cells=int(cells["cluster_id"].isna().sum()),
            clusters=clusters,
            ranked_hotspots=ranked,
            effort_correction=effort_record,
            quality=quality,
            validation=HotspotValidation(),
            parameters_used=self._parameters(params, quality.eps_km, quality.min_samples),
            render_spec=spec,
            citations=self._citations(),
            summary=summary,
            confidence=confidence,
            cleaning_report=cleaning_report,
            grid=self._cells_to_dataclasses(cells) if params.include_grid else [],
            warnings=warnings,
        )

    @staticmethod
    def _confidence(quality: ClusteringQuality, effort_record: EffortCorrection,
                    n_cells: int) -> float:
        """Confidence follows the evidence: separation, sample size, correction."""

        score = 0.45
        if quality.silhouette is not None:
            score += 0.30 * max(0.0, min(1.0, quality.silhouette))
        if n_cells >= 50:
            score += 0.10
        if effort_record.applied:
            score += 0.10
        if quality.reproducible:
            score += 0.05
        return round(min(score, 0.95), 2)

    @staticmethod
    def _parameters(params: BiodiversityHotspotInput, eps_km: float,
                    min_samples: int) -> dict:
        """Echo every parameter actually used - doc goal G5, reproducibility."""

        return {
            "region_name": params.region_name,
            "study_area_source": params.place_source,
            "bbox": list(params.bbox) if params.bbox else None,
            "taxon_filter": params.taxon_filter,
            "year_from": params.year_from,
            "year_to": params.year_to,
            "cell_size_km": params.cell_size_km,
            "min_records_per_cell": params.min_records_per_cell,
            "index": params.index.value,
            "normalise_by_effort": params.normalise_by_effort,
            "algorithm": params.algorithm.value,
            "eps_km": eps_km,
            "min_samples": min_samples,
            "auto_tune": params.auto_tune,
            "metric": "haversine",
            "top_n": params.top_n,
            "max_records": params.max_records,
        }

    @staticmethod
    def _citations() -> list[Citation]:
        return [Citation(source="GBIF",
                         identifier="GBIF Occurrence Search API",
                         url="https://www.gbif.org",
                         license="per contributing dataset")]

    @staticmethod
    def _cells_to_dataclasses(cells) -> list[RichnessCell]:
        # A nullable integer column round-trips through pandas as float, so an
        # unassigned cell arrives as NaN rather than None. NaN is the only value
        # that is not equal to itself, which is the cheapest way to spot it
        # without importing pandas here.
        def _cluster(value):
            if value is None or value != value:
                return None
            return int(value)

        return [
            RichnessCell(
                cell_id=row.cell_id,
                center=(round(row.lat, 4), round(row.lon, 4)),
                area_km2=row.area_km2,
                species_count=int(row.n_species),
                record_count=int(row.n_records),
                shannon=round(float(row.shannon), 3),
                simpson=round(float(row.simpson), 3),
                chao1_estimate=round(float(row.chao1), 2),
                corrected_richness=round(float(row.corrected_species), 2),
                effort_factor=round(float(row.effort_factor), 3),
                dominant_country=row.country,
                cluster_id=_cluster(row.cluster_id),
            )
            for row in cells.itertuples()
        ]
