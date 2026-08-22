"""Settings shared by the biodiversity workers.

Currently only M3 - Biodiversity Hotspots - has a real pipeline; M1, M2 and M4
are served by their ``mock.py`` fixtures. Every value here is traceable to
``Biodiversity Hotspots.pdf`` or to the Sprint 3 brief, and the source is named
in the comment beside it.
"""

from __future__ import annotations

import re

from pathlib import Path

_HERE = Path(__file__).resolve().parent          # workers/common
_AGENT = _HERE.parent.parent                     # biodiversity_agent

# Downloaded data and rendered maps stay inside this agent folder, so nothing
# outside it is read or written.
CACHE_DIR = _AGENT / "cache"
OUTPUT_DIR = _AGENT / "outputs" / "maps"

for _d in (CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6371.0

# ------------------------------------------------------------------ M3

# Study areas M3 can resolve by name, as (lon_min, lat_min, lon_max, lat_max).
# The design document resolves names through GADM; without that layer loaded,
# these boxes are the documented substitute and the fallback is reported.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "congo basin":    (8.0, -6.0, 30.0, 5.0),
    "amazon":         (-75.0, -15.0, -45.0, 5.0),
    "southeast asia": (95.0, -10.0, 130.0, 20.0),
    "kenya":          (33.9, -4.7, 41.9, 5.5),
    "madagascar":     (43.0, -26.0, 51.0, -11.0),
    "sahara":         (-10.0, 18.0, 30.0, 30.0),
}

M3_DEFAULTS = {
    "taxon_filter": "Animalia",
    "year_from": 1990,                # doc §7.2 default
    "year_to": 2025,
    "cell_size_km": 42.0,             # equal-area square; H3 substitute, see below
    "min_records_per_cell": 10,       # doc Table 13 noise floor
    "eps_km": 120.0,                  # doc Table 13 default, re-tuned per region
    "min_samples": 5,                 # doc Table 13 default
    "top_n": 10,                      # doc §7.2 default
    "max_records": 9000,              # paged path budget, doc §4.3
}

# The scan that replaces GridSearchCV, which cannot score an unlabelled problem.
M3_EPS_GRID = [60, 80, 100, 120, 150, 200, 250]
M3_MIN_SAMPLES_GRID = [3, 5, 8, 12]

# A configuration is admissible only inside this noise band: 0% means nothing
# was rejected (the Sahara would become a hotspot), 100% means nothing found.
M3_NOISE_BAND = (0.05, 0.80)

# ------------------------------------------------------------------ substitutions

# Every documented source that is unavailable here, and what replaces it. The
# worker copies the matching entries into its output ``warnings``, so a caveat
# always travels with the result instead of being lost (doc §7.3).
SUBSTITUTIONS = {
    "h3": "Equal-area sinusoidal squares are used instead of H3 hexagons "
          "(the h3 library is not installed). Cell areas are equal; only the "
          "cell shape differs.",
    "ecoregion_name": "Hotspot names come from the dominant country code, not the "
                      "WWF terrestrial ecoregion layer, which is not loaded. These "
                      "are not official ecoregion names.",
    "iucn_threat": "The IUCN threat-weighted index and threatened_count are not "
                   "computed: no Red List API token is configured.",
    "cepf_validation": "Overlap with the 36 published CEPF hotspots is not computed: "
                       "the CEPF shapefile is not loaded.",
}


# Ways a question can name one of the study areas above. The module has no LLM,
# so reading the question is a lookup: the longest phrase that appears in the
# sentence wins, which is why "southeast asia" must be tried before "asia" would
# ever match. Only unambiguous synonyms belong here - anything vague ("africa",
# "the tropics") is better answered by NEEDS_CLARIFICATION listing what is known.
REGION_ALIASES: dict[str, str] = {
    "congo basin": "congo basin",
    "congo": "congo basin",
    "drc": "congo basin",
    "democratic republic of the congo": "congo basin",
    "southeast asia": "southeast asia",
    "south east asia": "southeast asia",
    "south-east asia": "southeast asia",
    "se asia": "southeast asia",
    "amazon": "amazon",
    "amazonia": "amazon",
    "amazon basin": "amazon",
    "amazon rainforest": "amazon",
    "madagascar": "madagascar",
    "kenya": "kenya",
    "sahara": "sahara",
    "sahara desert": "sahara",
}


def regions_in_text(text: str | None) -> list[str]:
    """Every study area a free-text question names, in the order they appear.

    Longest phrase first, so "southeast asia" is not shadowed by a shorter key
    that is a substring of it, and a matched phrase is blanked out so "amazon"
    inside "amazon basin" cannot be counted twice.
    """

    if not text:
        return []
    haystack = text.lower()
    found: list[tuple[int, str]] = []
    for phrase in sorted(REGION_ALIASES, key=len, reverse=True):
        at = haystack.find(phrase)
        if at >= 0:
            region = REGION_ALIASES[phrase]
            if region not in [r for _, r in found]:
                found.append((at, region))
            haystack = haystack.replace(phrase, " " * len(phrase))
    return [region for _, region in sorted(found)]


def region_in_text(text: str | None) -> str | None:
    """The first study area a question names, or None."""

    found = regions_in_text(text)
    return found[0] if found else None


# The whole planet, for a species question that names no study area: "where do
# tigers live" is about the range, not about one of the six regions.
WORLD_BBOX: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)

# Common names GBIF's own search ranks badly. Measured, not assumed: a
# vernacular search for "tiger" returns fifty molluscs, frogs and flies whose
# *scientific* names contain "tiger" before it reaches Panthera tigris, even
# though GBIF holds "Tiger" as its vernacular name. The values here are
# scientific names, not keys - the key still comes from GBIF, so this table
# cannot go stale against the backbone.
COMMON_NAMES: dict[str, str] = {
    "tiger": "Panthera tigris",
    "lion": "Panthera leo",
    "leopard": "Panthera pardus",
    "snow leopard": "Panthera uncia",
    "jaguar": "Panthera onca",
    "cheetah": "Acinonyx jubatus",
    "polar bear": "Ursus maritimus",
    "brown bear": "Ursus arctos",
    "giant panda": "Ailuropoda melanoleuca",
    "panda": "Ailuropoda melanoleuca",
    "african elephant": "Loxodonta africana",
    "asian elephant": "Elephas maximus",
    "elephant": "Loxodonta africana",
    "gorilla": "Gorilla gorilla",
    "chimpanzee": "Pan troglodytes",
    "orangutan": "Pongo pygmaeus",
    "blue whale": "Balaenoptera musculus",
    "humpback whale": "Megaptera novaeangliae",
    "grey wolf": "Canis lupus",
    "gray wolf": "Canis lupus",
    "wolf": "Canis lupus",
    "white rhinoceros": "Ceratotherium simum",
    "black rhinoceros": "Diceros bicornis",
    "giraffe": "Giraffa camelopardalis",
    "hippopotamus": "Hippopotamus amphibius",
    "koala": "Phascolarctos cinereus",
    "red panda": "Ailurus fulgens",
}


# A binomial: "Panthera tigris". Genus capitalised, species epithet lower case -
# the convention is strict enough to key on.
_BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b")

# Words that start a question and would otherwise read as a genus: "Where are",
# "Show me", "Which part".
_NOT_A_GENUS = {"where", "show", "which", "what", "who", "how", "why", "give",
                "find", "compare", "list", "tell", "hotspots", "biodiversity",
                "species", "richest", "most", "there", "these", "this", "that"}


def species_in_text(text: str | None) -> str | None:
    """The species a question names, or None.

    Common names are matched first and longest-first, so "african elephant" is
    not shadowed by "elephant". Then a binomial, guarded twice: the first word
    must not be a question opener, and the pair must not be a study area - "in
    the Congo basin" is a region, not a genus and species.
    """

    if not text:
        return None
    lowered = text.lower()

    for phrase in sorted(COMMON_NAMES, key=len, reverse=True):
        if phrase in lowered or phrase + "s" in lowered:
            return phrase

    found = _BINOMIAL.search(text)
    if found:
        genus, epithet = found.group(1), found.group(2)
        candidate = f"{genus} {epithet}"
        if genus.lower() in _NOT_A_GENUS:
            return None
        if regions_in_text(candidate):
            return None
        return candidate
    return None


# Prepositions that introduce a place in the questions this module gets asked:
# "hotspots in Brazil", "biodiversity of Borneo", "richest areas across India".
_PLACE_PREPOSITIONS = ("in", "of", "for", "across", "within", "near", "around",
                       "throughout", "inside")

# Words that are not part of a place name.
_PLACE_TAIL_NOISE = {"please", "thanks", "region", "regions", "area", "areas",
                     "now", "today", "map", "data", "records", "hotspots",
                     "biodiversity", "species"}

# A place name ends here. "Which part of Africa has the most species" must yield
# "Africa", not "Africa has the most" - so the phrase is cut at the first verb or
# quantifier, none of which appear inside a place name.
_PLACE_ENDS_AT = {"has", "have", "had", "is", "are", "was", "were", "do", "does",
                  "did", "with", "that", "which", "who", "what", "where", "when",
                  "and", "or", "but", "most", "more", "than", "best", "top",
                  "show", "give", "tell", "find", "compare", "me", "my", "we",
                  "you", "it", "there", "should", "would", "can", "could", "look",
                  "looking", "see", "get"}

# Question openers, when a short message has no preposition at all.
_QUESTION_OPENERS = {"what", "where", "which", "who", "how", "why", "show",
                     "tell", "give", "find", "list", "compare"}


def place_candidates(text: str | None) -> list[str]:
    """Place names a question might contain, best guess first.

    A question can put the place after its first preposition ("hotspots in the
    Democratic Republic of the Congo") or after its last ("hotspots for tigers in
    India"), and no single rule wins both. So both readings are returned and the
    caller tries them against a real gazetteer, which is what settles it. A guess
    that resolves to nothing costs one cached miss; a guess that resolves to the
    wrong continent would cost a wrong answer, which is why nothing here is
    trusted on its own.
    """

    if not text:
        return []
    cleaned = " ".join(text.replace("?", " ").replace(",", " ").split())
    words = cleaned.split()
    lowered = [word.lower() for word in words]

    starts: list[int] = [position + 1 for position, word in enumerate(lowered)
                         if word in _PLACE_PREPOSITIONS]
    if not starts and 1 <= len(words) <= 3:
        starts = [0]                    # a bare name: "Brazil", "Borneo"

    seen: set[str] = set()
    candidates: list[str] = []
    for start in starts:
        phrase: list[str] = []
        for word in words[start:]:
            low = word.lower()
            if low in _PLACE_ENDS_AT:
                break
            if low in _PLACE_TAIL_NOISE:
                continue
            phrase.append(word)

        while phrase and phrase[0].lower() in ("the", "a", "an"):
            phrase.pop(0)
        # After a species name is removed, "hotspots for tigers in India" leaves
        # "in India" - the preposition is not part of the name.
        while phrase and phrase[0].lower() in _PLACE_PREPOSITIONS:
            phrase.pop(0)
        while phrase and phrase[-1].lower() in ("the", "a", "an"):
            phrase.pop()
        # Five words covers "Democratic Republic of the Congo"; beyond that it
        # is a sentence, not a name.
        if not phrase or len(phrase) > 5:
            continue
        # One or two characters is never a place worth asking a gazetteer about,
        # and it is what a botched word removal leaves behind.
        if len(" ".join(phrase)) < 3:
            continue
        if phrase[0].lower() in _QUESTION_OPENERS:
            continue

        candidate = " ".join(phrase).strip(" .!")
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            candidates.append(candidate)
    return candidates


def place_in_text(text: str | None) -> str | None:
    """The most likely place a question names, or None."""

    found = place_candidates(text)
    return found[0] if found else None
